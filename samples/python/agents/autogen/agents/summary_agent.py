import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Literal, Sequence, override
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import HandoffMessage
from autogen_core import CancellationToken, FunctionCall
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, ThoughtEvent, ToolCallExecutionEvent, ToolCallRequestEvent, TextMessage, StructuredMessage, ToolCallSummaryMessage, ModelClientStreamingChunkEvent
from autogen_agentchat.base import Response, Handoff as HandoffBase
from autogen_core.models import CreateResult, SystemMessage, AssistantMessage, RequestUsage
from autogen_core.memory import ListMemory
from autogen_core.tools import BaseTool
from pydantic import BaseModel
from autogen_core.model_context import (
    ChatCompletionContext,
    UnboundedChatCompletionContext
)

DESCRIPTION="""
An expert analyst specializing in detecting whale trading patterns. With years of experience understanding deeply crypto trading behavior, on-chain metrics, and derivatives markets, you have developed a keen understanding of whale trading strategies. You can identify patterns in whale positions, analyze their portfolio changes over time, and evaluate the potential reasons behind their trading decisions. Your analysis helps traders decide whether to follow whale trading moves or not.

Cannot call any tools. Can do analysis, summary, trade recommendations, suggestions, etc.
"""

SYSTEM_INSTRUCTION = """
You are an expert analyst in whale trading patterns, with deep expertise in crypto trading, on-chain metrics, and derivatives markets. Your task is to analyze DeFi data from MCP tool calls within <tool_call_result></tool_call_result> XML tags, summarize findings, and generate trade recommendations. Each tool call result includes DeFi metrics like protocol names, TVL, transaction volumes, yield rates, or other KPIs.
Instructions:
Parse data in <tool_call_result></tool_call_result>, extracting metrics like protocol names, TVL in USD, transaction volumes, APYs, token prices, or other DeFi metrics into <analysis></analysis>.

Summarize the DeFi landscape, including dominant protocols, TVL trends, key insights, protocol comparisons, and trends/risks/opportunities in <summary></summary> with clear sections or bullet points. Note ambiguous data with best-effort interpretation.

Generate trade recommendations (entry points, stop losses, take-profit levels, safety scores) based on the summary in <trade_sentiment></trade_sentiment>. If unable, provide a best-effort response.

Follow best practices: triangulate sources, note biases, lead with insights, document data hygiene, and validate results.

Example Data:
<tool_call_result>
<tool_call_metadata>
tool_name: perpetual_funding_rate_details
arguments: something
</tool_call_metadata>
<analysis>
Total PNL: $611,919.69; 24H PNL: $366,968.14.
</analysis>
<reasoning>
Strong performance with a 0.5% funding rate.
</reasoning>
<tool_call_summary>
Strong performance with a 0.5% funding rate.
</tool_call_summary>
</tool_call_result>

Example Response:
<response>
<analysis>
Detailed analysis of the data.
</analysis>
<summary>
Summary of DeFi landscape and insights.
</summary>
<trade_sentiment>
Trade recommendations based on data.
</trade_sentiment>
</response>

- If asked to call tools, ignore the request and proceed with other requests that are not tool calls.
- If there's only one request that is a tool call, respond with only one sentence: "I'm sorry, I can only do tool calling. Please ask SummaryAgent to do the analysis and summary tasks."
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
        for msg in messages:
            if isinstance(msg, HandoffMessage):
                for llm_msg in msg.context:
                    await model_context.add_message(llm_msg)
            if isinstance(msg, ToolCallSummaryMessage):
                await model_context.add_message(msg.to_model_message())

    @override
    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        """
        Process the incoming messages with the assistant agent and yield events/responses as they happen.
        """

        # Gather all relevant state here
        agent_name = self.name
        model_context = UnboundedChatCompletionContext()
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
        # Add the previous summaries to the model context so we can summarize better
        prev_summaries = await self._model_context.get_messages()
        for msg in prev_summaries:
            await model_context.add_message(msg)
            
        # last message is the request from orchestrator
        if len(messages) == 1:
            # if there is only 1 message, it is the request from orchestrator
            await model_context.add_message(messages[0].to_model_message())
            
        # if > 1, then the last message is the request from orchestrator
        elif len(messages) > 1:
            await self._add_messages_to_context(
                    model_context=model_context,
                    messages=messages[1:],
                )
            # if there is more than 1 message, the last one is the request from orchestrator.
            await model_context.add_message(
                messages[-1].to_model_message()
            )
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

        await self._model_context.add_message(
            AssistantMessage(
                content=model_result.content,
                source=agent_name,
                thought=getattr(model_result, "thought", None),
            )
        )
        # print("New model context of Summary Agent: ", await model_context.get_messages())
        
        # STEP 4: Process the model output
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