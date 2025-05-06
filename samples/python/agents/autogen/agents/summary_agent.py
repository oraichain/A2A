import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Sequence, override
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken, FunctionCall
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, ThoughtEvent, ToolCallExecutionEvent, ToolCallRequestEvent, TextMessage, StructuredMessage, ToolCallSummaryMessage, ModelClientStreamingChunkEvent
from autogen_agentchat.base import Response, Handoff as HandoffBase
from autogen_core.models import CreateResult, SystemMessage, ChatCompletionClient, AssistantMessage, FunctionExecutionResultMessage
from autogen_core.memory import MemoryContent, MemoryMimeType, ListMemory
from autogen_core.tools import BaseTool
from pydantic import BaseModel
from autogen_core.model_context import (
    ChatCompletionContext,
)

DESCRIPTION = """
You are an expert analyst and trader specializing in decentralized finance (DeFi) analysis, summarization, and trade recommendations. Your task is to analyze and summarize DeFi data collected from MCP tool calls and previous summarization results. You also generate trade recommendations or perform complex analysis. You can provide entry points, stop losses, take profit levels, or safety scores for potential trades based on the summary you have.

Each tool call result contains relevant DeFi metrics, such as protocol names, total value locked (TVL), transaction volumes, yield rates, or other key performance indicators.

    Instructions:
    1. Parse the data within the <tool_call_results></tool_call_results> XML tag.
    2. Identify and extract key DeFi metrics from each tool call result, such as:
    - Protocol or platform names
    - Total value locked (TVL) in USD
    - Transaction volumes or counts
    - Yield rates or APYs
    - Token prices or market data
    - Other relevant DeFi-specific metrics
    3. Provide a detailed summary that includes:
    - An overview of the DeFi landscape based on the data (e.g., dominant protocols, overall TVL trends).
    - Key insights for each tool call result, highlighting significant metrics (e.g., highest TVL, notable yield opportunities, or unusual transaction activity).
    - Comparisons between protocols or metrics where relevant (e.g., TVL growth, yield competitiveness).
    - Any trends, risks, or opportunities in the DeFi ecosystem inferred from the data.
    - Structure the summary clearly with sections or bullet points for readability.
    - If any data is ambiguous or incomplete, note it and provide a best-effort interpretation.
    
    4. Generate trade recommendations or perform complex analysis. You can provide entry points, stop losses, take profit levels, or safety scores for potential trades based on the summary you have. If you can't provide any trade recommendations, just respond with your best effort.
    
    Remember, you are an analyst, summarizer, and a recommender, not a tool caller.
    
    If you are asked to do tool calling or other unrelated tasks, respond with "I'm sorry, I can only do analysis and summarization. Please ask ToolCallAgent to do the tool calling tasks."
"""

SYSTEM_INSTRUCTION = """

You are an expert analyst and trader specializing in decentralized finance (DeFi) analysis, summarization, and trade recommendations. Your task is to analyze and summarize DeFi data collected from MCP tool calls and previous summarization results. You also generate trade recommendations or perform complex analysis. You can provide entry points, stop losses, take profit levels, or safety scores for potential trades based on the summary you have.

The data is delimited by <tool_call_result></tool_call_result> XML tags, with the query.

Each tool call result contains relevant DeFi metrics, such as protocol names, total value locked (TVL), transaction volumes, yield rates, or other key performance indicators.

    Instructions:
    1. Parse the data within the <tool_call_results></tool_call_results> XML tag.
    2. Identify and extract key DeFi metrics from each tool call result, such as:
    - Protocol or platform names
    - Total value locked (TVL) in USD
    - Transaction volumes or counts
    - Yield rates or APYs
    - Token prices or market data
    - Other relevant DeFi-specific metrics
    - Put your discovery, analysis, highlights, and key metrics in <analysis></analysis> XML tag.
    3. Provide a detailed summary that includes:
    - An overview of the DeFi landscape based on the data (e.g., dominant protocols, overall TVL trends).
    - Key insights for each tool call result, highlighting significant metrics (e.g., highest TVL, notable yield opportunities, or unusual transaction activity).
    - Comparisons between protocols or metrics where relevant (e.g., TVL growth, yield competitiveness).
    - Any trends, risks, or opportunities in the DeFi ecosystem inferred from the data.
    - Structure the summary clearly with sections or bullet points for readability.
    - If any data is ambiguous or incomplete, note it and provide a best-effort interpretation.
    - Put your summary in <summary></summary> XML tag.
    
    4. Generate trade recommendations or perform complex analysis. You can provide entry points, stop losses, take profit levels, or safety scores for potential trades based on the summary you have. If you can't provide any trade recommendations, just respond with your best effort.
    
    - Put your trade recommendations in <trade_sentiment></trade_sentiment> XML tag.
    
    Remember, you are an analyst, summarizer, and a recommender, not a tool caller.
    
    If you are asked to do tool calling or other unrelated tasks, respond with only one sentence: "I'm sorry, I can only do analysis and summarization. Please ask ToolCallAgent to do the tool calling tasks."
    
    ## Analysis Best‑Practices
    1. **Source Triangulation:** corroborate important claims with at least two independent sources when possible.
    2. **Bias Awareness:** note significant discrepancies between sources and highlight uncertainties.
    3. **Insight First:** lead summaries with the takeaway, then provide supporting details.
    4. **Data Hygiene:** document data origin, cleaning steps, and assumptions made during analysis.
    5. **Result Validation:** sanity‑check numbers, code outputs, and logic before presenting.
    
Please provide your detailed response in the <response></response> tag.

Below is an example of the data format:

<tool_call_result>
tool_name: "Protocol 1",
arguments: "",
result: "Protocol 1"
</tool_call_result>
<tool_call_result>
tool_name: "Protocol 2",
arguments: "",
result: "Protocol 2"
</tool_call_result>

Below is an example of the response format:

<response>
<analysis>
A decent analysis of the data above.
</analysis>
<summary>
A decent summary of the data above.
</summary>
<trade_sentiment>A decent trade recommendation of the data above.
</trade_sentiment>
</response>
"""

class SummaryAgent(AssistantAgent):
    """
    Summary agent won't use any tools, handoffs, or reflections.
    It will just summarize the tool calls and their results from the memory.
    """
    def __init__(self, *args, **kwargs):
        memory: ListMemory | None = kwargs.get("list_memory", None)
        if memory is None:
            raise ValueError("list_memory is required")
        del kwargs["list_memory"]
        super().__init__(*args, **kwargs, description=DESCRIPTION)
        self._memory = memory
        self._system_messages = [SystemMessage(content=SYSTEM_INSTRUCTION)]
        
    # @staticmethod
    # async def _update_model_context_with_memory(
    #     memory: ListMemory,
    #     model_context: ChatCompletionContext,
    #     agent_name: str,
    # ) -> List[MemoryQueryEvent]:
    #     """
    #     If memory modules are present, update the model context and return the events produced.
    #     """
    #     events: List[MemoryQueryEvent] = []
    #     if memory:
    #         update_context_result = await memory.update_context(model_context)
    #         if update_context_result and len(update_context_result.memories.results) > 0:
    #             memory_query_event_msg = MemoryQueryEvent(
    #                 content=update_context_result.memories.results,
    #                 source=agent_name,
    #             )
    #             events.append(memory_query_event_msg)
    #     return events
    
    @staticmethod
    async def _add_messages_to_context(
        model_context: ChatCompletionContext,
        messages: Sequence[BaseChatMessage],
    ):
        """
        Add incoming messages to the model context.
        """
        added_message_count = 0
        for msg in messages:
                await model_context.add_message(msg.to_model_message())
                added_message_count += 1
        return added_message_count
    
    @override
    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        """
        Process the incoming messages with the assistant agent and yield events/responses as they happen.
        """

        # Gather all relevant state here
        agent_name = self.name
        model_context = self._model_context
        assert self._memory is not None, "Memory is not set"
        memory = self._memory
        system_messages = self._system_messages
        tools = []
        handoff_tools = []
        model_client = self._model_client
        model_client_stream = self._model_client_stream
        output_content_type = self._output_content_type
        format_string = self._output_content_type_format

        # STEP 1: Add new user/handoff messages to the model context
        # SYSTEM_INSTRUCTION.format(tool_call_data=tool_call_data, query=query)
        added_message_count = await SummaryAgent._add_messages_to_context(
            model_context=model_context,
            messages=messages,
        )
        print("Added message count: ", added_message_count)

        # STEP 2: Update model context with any relevant memory
        inner_messages: List[BaseAgentEvent | BaseChatMessage] = []
        # for event_msg in await self._update_model_context_with_memory(
        #     memory=memory,
        #     model_context=model_context,
        #     agent_name=agent_name,
        # ):
        #     inner_messages.append(event_msg)
        #     yield event_msg
        
        # print("Memory of Summary Agent: ", memory.content)

        # STEP 3: Run the first inference
        model_result = None
        try:
            async for inference_output in self._call_llm(
                model_client=model_client,
                model_client_stream=model_client_stream,
                system_messages=system_messages,
                model_context=model_context,
                tools=[],
                handoff_tools=[],
                agent_name=agent_name,
                cancellation_token=cancellation_token,
                output_content_type=output_content_type,
            ):
                if isinstance(inference_output, CreateResult):
                    model_result = inference_output
                elif isinstance(inference_output, ModelClientStreamingChunkEvent):
                    # Streaming chunk event
                    yield inference_output
        except RuntimeError as e:
            print(f"Runtime error in summary agent: {e}")
            yield ModelClientStreamingChunkEvent(source=agent_name, content=f"Runtime error in summary agent: {e}")
        except Exception as e:
            print(f"Error in summary agent: {e}")
            yield ModelClientStreamingChunkEvent(source=agent_name, content=f"Error in summary agent: {e}")

        assert model_result is not None, "No model result was produced."

        # --- NEW: If the model produced a hidden "thought," yield it as an event ---
        if model_result.thought:
            thought_event = ThoughtEvent(content=model_result.thought, source=agent_name)
            yield thought_event
            inner_messages.append(thought_event)

        # STEP 4: Clear the old context and Add the summary message to the context
        # messages = await model_context.get_messages()
        # messages = messages[:-added_message_count] if added_message_count > 0 else messages
        # await model_context.clear()
        
        # # # we clear all tool results since we don't need them anymore after analysing and summarizing the results
        # for msg in messages:
        #     await model_context.add_message(msg)
        await model_context.add_message(
            AssistantMessage(
                content=model_result.content,
                source=agent_name,
                thought=getattr(model_result, "thought", None),
            )
        )
        print("New model context of Summary Agent: ", await model_context.get_messages())
        
        # STEP 6: Process the model output
        # If direct text response (string)
        if isinstance(model_result.content, str):
            if output_content_type:
                content = output_content_type.model_validate_json(model_result.content)
                yield Response(
                    chat_message=StructuredMessage[output_content_type](  # type: ignore[valid-type]
                        content=content,
                        source=agent_name,
                        models_usage=model_result.usage,
                        format_string=format_string,
                    ),
                    inner_messages=inner_messages,
                )
            else:
                yield Response(
                    chat_message=TextMessage(
                        content=model_result.content,
                        source=agent_name,
                        models_usage=model_result.usage,
                    ),
                    inner_messages=inner_messages,
                )
            return