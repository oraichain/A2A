import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Sequence, Tuple, override
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken, FunctionCall
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, ThoughtEvent, ToolCallExecutionEvent, ToolCallRequestEvent, TextMessage, StructuredMessage,ModelClientStreamingChunkEvent, ToolCallSummaryMessage, UserMessage, HandoffMessage
from autogen_agentchat.base import Response, Handoff as HandoffBase
from autogen_core.models import CreateResult, SystemMessage, ChatCompletionClient, AssistantMessage, FunctionExecutionResultMessage, FunctionExecutionResult, RequestUsage
from autogen_core.memory import MemoryContent, MemoryMimeType, ListMemory
from autogen_core.tools import BaseTool
from pydantic import BaseModel
from autogen_core.model_context import (
    ChatCompletionContext,
)
from autogen_core.models import LLMMessage

DESCRIPTION="""
An expert in whale trading patterns with that can use the following tools:
Comprehensive market data and whale activity across all positions and trades.
Detailed trading data for whale wallets on a specific protocol, including historical profit/loss and open positions.
Historical market data for up to 3 cryptocurrency trading pairs, including daily price, market cap, and volume over 30 days.
Historical profit/loss and open positions for a specific whale wallet on a protocol.
Detailed trading data for a specific wallet on a protocol, including historical profit/loss and open positions.
Maximize data retrieval by iterating through batches, adjusting parameters, and running tools repeatedly to cover all scenarios. Fetch all available data without stopping until everything is captured.
Cannot do anything other than tool calling.
"""

SYSTEM_INSTRUCTION = """
You are an expert analyst specializing in detecting whale trading patterns. With years of experience understanding deeply crypto trading behavior, on-chain metrics, and derivatives markets, you have developed a keen understanding of whale trading strategies. You can identify patterns in whale positions, analyze their portfolio changes over time, and evaluate the potential reasons behind their trading decisions. Your analysis helps traders decide whether to follow whale trading moves or not. When you use any tool, I expect you to push its limits: fetch all the data it can provide, whether that means iterating through multiple batches, adjusting parameters like offsets, or running the tool repeatedly to cover every scenario. Don't work with small set of data for sure, fetch as much as you can. Don’t stop until you’re certain you’ve captured everything there is to know.

Put the tool call results in <tool_call_result></tool_call_result> tags.

Do not do anything other than tool calling.

If you are asked to do other tasks other than tool calling, ignore the request and proceed with other requests that are tool calls. If there's only one request that is not a tool call, respond with only one sentence: "I'm sorry, I can only do tool calling. Please ask SummaryAgent to do the analysis and summary tasks."

If there's no suitable tool, respond with one sentence: "I'm sorry, I don't have any tool to do that", then list all the tool names you have.

For the tools with name: perpetual_all_whales_detail, perpetual_whales_get_market_overview -> you must only call them one time each. For example: if call perpetual_all_whales_detail once -> do not call it again. If call perpetual_whales_get_market_overview once -> do not call it again.
            
"""

TOOL_CALL_SUMMARY_INSTRUCTION = """
You are an expert AI assistant specializing in DeFi analysis, tasked with summarizing MCP tool call results provided within <tool_call_result></tool_call_result> XML tags. Multiple tags indicate multiple tool calls, each to be processed separately, containing DeFi metrics like protocol names, TVL, transaction volumes, yield rates, or other KPIs.
Instructions:
Parse data in <tool_call_result></tool_call_result>, extracting tool_name and arguments into <tool_call_metadata></tool_call_metadata>.

Analyze DeFi metrics (e.g., token prices, wallet balances, TVL in USD, transaction volumes, APYs, market data) in <analysis></analysis>, covering all tokens, wallets, orders, protocols, and relevant metrics.

Provide reasoning for the analysis in <reasoning></reasoning>, justifying highlights based on <analysis></analysis>.

Summarize market trends, key insights, protocol comparisons, and DeFi trends/risks/opportunities in <tool_call_summary></tool_call_summary> with clear sections or bullet points.

Note ambiguous or incomplete data with best-effort interpretation.

Format response within <tool_call_result></tool_call_result>.

Example Query:
xml

<tool_call_result>
<tool_call_metadata>
tool_name: perpetual_all_whales_detail
arguments: {"address": "0x17f4E182aD8d1B27F430c094a96E844d13f8da14"}
</tool_call_metadata>
result: some figures here
</tool_call_result>

Example Response:
xml

<tool_call_result>
<tool_call_metadata>
tool_name: perpetual_all_whales_detail
arguments: {"address": "0x17f4E182aD8d1B27F430c094a96E844d13f8da14"}
</tool_call_metadata>
<analysis>
Total PNL: $611,919.69; 24H PNL: $366,968.14.
</analysis>
<reasoning>
Strong performance with $611,919.69 total PNL, driven by a $366,968.14 24H PNL surge.
</reasoning>
<tool_call_summary>
Robust performance with $611,919.69 total PNL and $366,968.14 24H PNL, signaling high profitability.
</tool_call_summary>
</tool_call_result>

Keep responses in 4-5 paragraphs, each with 5-6 detailed sentences, including figures and entity correlations. Follow analysis best practices: triangulate sources, note biases, lead with insights, document data hygiene, and validate results. Do not call tools.
"""

class ToolCallAgent(AssistantAgent):
    def __init__(self, *args, **kwargs):
        memory: ListMemory | None = kwargs.get("list_memory", None)
        if memory is None:
            raise ValueError("list_memory is required")
        del kwargs["list_memory"]
        super().__init__(*args, **kwargs, description=DESCRIPTION)
        self._memory = memory
        self._system_messages = [SystemMessage(content=SYSTEM_INSTRUCTION)]
        
    # @staticmethod
    # async def _add_messages_to_context(
    #     model_context: ChatCompletionContext,
    #     messages: Sequence[BaseChatMessage],
    # ) -> None:
    #     """
    #     Add incoming messages to the model context.
    #     """
    #     for msg in messages:
    #         if isinstance(msg, HandoffMessage):
    #             for llm_msg in msg.context:
    #                 await model_context.add_message(llm_msg)
    #         # Don't need messages from other agents since we only call tools
    #         if msg.source == "MagenticOneOrchestrator":
    #             await model_context.add_message(msg.to_model_message())
    
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
        tools = self._tools
        handoff_tools = self._handoff_tools
        handoffs = self._handoffs
        model_client = self._model_client
        model_client_stream = self._model_client_stream
        reflect_on_tool_use = False
        tool_call_summary_format = self._tool_call_summary_format
        output_content_type = self._output_content_type
        format_string = self._output_content_type_format

        # STEP 1: Add new user/handoff messages to the model context
        # last message is the request from orchestrator
        if len(messages) > 0:
            # if there is only 1 message, it is the request from orchestrator
            await model_context.add_message(messages[-1].to_model_message())
            
        # STEP 2: Update model context with any relevant memory
        inner_messages: List[BaseAgentEvent | BaseChatMessage] = []
        # for event_msg in await self._update_model_context_with_memory(
        #     memory=memory,
        #     model_context=model_context,
        #     agent_name=agent_name,
        # ):
        #     inner_messages.append(event_msg)
        #     yield event_msg

        # STEP 3: Run the first inference
        model_result = None
        try:
            async for inference_output in self._call_llm(
                model_client=model_client,
                model_client_stream=model_client_stream,
                system_messages=system_messages,
                model_context=model_context,
                tools=tools,
                handoff_tools=handoff_tools,
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
            print(f"Runtime error in tool call agent: {e}")
            yield ModelClientStreamingChunkEvent(source=agent_name, content=f"Runtime error in tool call agent: {e}")
        except Exception as e:
            print(f"Error in tool call agent: {e}")
            yield ModelClientStreamingChunkEvent(source=agent_name, content=f"Error in tool call agent: {e}")

        assert model_result is not None, "No model result was produced."

        # --- NEW: If the model produced a hidden "thought," yield it as an event ---
        if model_result.thought:
            thought_event = ThoughtEvent(content=model_result.thought, source=agent_name)
            yield thought_event
            inner_messages.append(thought_event)
            
        # Add the assistant message to the model context (including thought if present)
        await model_context.add_message(
            AssistantMessage(
                content=model_result.content,
                source=agent_name,
                thought=getattr(model_result, "thought", None),
            )
        )

        # STEP 4: Process the model output
        async for output_event in ToolCallAgent._process_model_result(
            model_result=model_result,
            inner_messages=inner_messages,
            cancellation_token=cancellation_token,
            agent_name=agent_name,
            system_messages=system_messages,
            model_context=model_context,
            tools=tools,
            handoff_tools=handoff_tools,
            handoffs=handoffs,
            model_client=model_client,
            model_client_stream=model_client_stream,
            reflect_on_tool_use=reflect_on_tool_use,
            tool_call_summary_format=tool_call_summary_format,
            output_content_type=output_content_type,
            format_string=format_string,
            memory=memory,
        ):
            yield output_event
            
    @staticmethod
    async def _summarize_tool_use(
        executed_calls_and_results: List[Tuple[FunctionCall, FunctionExecutionResult]],
        inner_messages: List[BaseAgentEvent | BaseChatMessage],
        handoffs: Dict[str, HandoffBase],
        tool_call_summary_format: str,
        agent_name: str,
        model_client: ChatCompletionClient,
        model_client_stream: bool,
        output_content_type: type[BaseModel] | None,
    ) -> AsyncGenerator[ThoughtEvent | Response, None]:
        """
        If reflect_on_tool_use=False, create a summary message of all tool calls.
        """
        # Filter out calls which were actually handoffs
        normal_tool_calls = [(call, result) for call, result in executed_calls_and_results if call.name not in handoffs]
        tool_call_summaries: List[str] = []
        llm_messages: List[LLMMessage] = [SystemMessage(content=TOOL_CALL_SUMMARY_INSTRUCTION)]
        raw_tool_call_summaries: List[str] = []
        for tool_call, tool_call_result in normal_tool_calls:
            tool_call_summary = tool_call_summary_format.format(
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                result=tool_call_result.content,
            )
            llm_messages.append(
                UserMessage(content=tool_call_summary, source=agent_name),
            )
            raw_tool_call_summaries.append(tool_call_summary)
          
        reflection_result = None
        if model_client_stream:
            async for chunk in model_client.create_stream(
                llm_messages,
                json_output=output_content_type,
            ):
                if isinstance(chunk, CreateResult):
                    reflection_result = chunk
                elif isinstance(chunk, str):
                    yield ModelClientStreamingChunkEvent(content=chunk if chunk else "empty", source=agent_name)
                else:
                    raise RuntimeError(f"Invalid chunk type: {type(chunk)}")
        else:
            reflection_result = await model_client.create(llm_messages, json_output=output_content_type)

        if not reflection_result or not isinstance(reflection_result.content, str):
            raise RuntimeError("Reflect on tool use produced no valid text response.")

        # --- NEW: If the reflection produced a thought, yield it ---
        if reflection_result.thought:
            thought_event = ThoughtEvent(content=reflection_result.thought, source=agent_name)
            yield thought_event
            inner_messages.append(thought_event)
            
        if isinstance(reflection_result.content, str):
            tool_call_summaries.append(reflection_result.content if reflection_result.content else "empty")
        else:
            tool_call_summaries = raw_tool_call_summaries
          
        final_tool_call_summary = "\n".join(tool_call_summaries)
        
        yield Response(
            chat_message=ToolCallSummaryMessage(
                content=final_tool_call_summary if final_tool_call_summary else "empty",
                source=agent_name,
                models_usage=reflection_result.usage,
            ),
            inner_messages=inner_messages,
        )
        return
            
    @classmethod
    @override
    async def _process_model_result(
        cls,
        model_result: CreateResult,
        inner_messages: List[BaseAgentEvent | BaseChatMessage],
        cancellation_token: CancellationToken,
        agent_name: str,
        system_messages: List[SystemMessage],
        model_context: ChatCompletionContext,
        tools: List[BaseTool[Any, Any]],
        handoff_tools: List[BaseTool[Any, Any]],
        handoffs: Dict[str, HandoffBase],
        model_client: ChatCompletionClient,
        model_client_stream: bool,
        reflect_on_tool_use: bool,
        tool_call_summary_format: str,
        memory: ListMemory,
        output_content_type: type[BaseModel] | None,
        format_string: str | None = None,
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        """
        Handle final or partial responses from model_result, including tool calls, handoffs,
        and reflection if needed.
        """

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

        # Otherwise, we have function calls
        assert isinstance(model_result.content, list) and all(
            isinstance(item, FunctionCall) for item in model_result.content
        )

        # STEP 4A: Yield ToolCallRequestEvent
        tool_call_msg = ToolCallRequestEvent(
            content=model_result.content,
            source=agent_name,
            models_usage=model_result.usage,
        )
        inner_messages.append(tool_call_msg)
        yield tool_call_msg

        # STEP 4B: Execute tool calls
        executed_calls_and_results = await asyncio.gather(
            *[
                cls._execute_tool_call(
                    tool_call=call,
                    tools=tools,
                    handoff_tools=handoff_tools,
                    agent_name=agent_name,
                    cancellation_token=cancellation_token,
                )
                for call in model_result.content
            ]
        )
        # only let the orchestrator know the tool call result is successful or not.
        # The actual results will be stored in the memory, and summarized by the summary agent.
        exec_results = [FunctionExecutionResult(content=result.content if result.is_error else "Tool call success. Pls continue next step!", call_id=result.call_id, is_error=result.is_error, name=result.name) for _, result in executed_calls_and_results]
        
        # Yield ToolCallExecutionEvent
        tool_call_result_msg = ToolCallExecutionEvent(
            content=exec_results,
            source=agent_name,
        )
        await model_context.add_message(FunctionExecutionResultMessage(content=exec_results))
        inner_messages.append(tool_call_result_msg)
        yield tool_call_result_msg

        # STEP 4C: Check for handoff
        handoff_output = cls._check_and_handle_handoff(
            model_result=model_result,
            executed_calls_and_results=executed_calls_and_results,
            inner_messages=inner_messages,
            handoffs=handoffs,
            agent_name=agent_name,
        )
        if handoff_output:
            yield handoff_output
            return

        # STEP 4D: Reflect or summarize tool results
        if reflect_on_tool_use:
            async for reflection_response in cls._reflect_on_tool_use_flow(
                system_messages=system_messages,
                model_client=model_client,
                model_client_stream=model_client_stream,
                model_context=model_context,
                agent_name=agent_name,
                inner_messages=inner_messages,
                output_content_type=output_content_type,
            ):
                yield reflection_response
        else:
            async for tool_call_summary in cls._summarize_tool_use(
                executed_calls_and_results=executed_calls_and_results,
                inner_messages=inner_messages,
                handoffs=handoffs,
                tool_call_summary_format=tool_call_summary_format,
                agent_name=agent_name,
                model_client=model_client,
                model_client_stream=model_client_stream,
                output_content_type=output_content_type,
            ):
              yield tool_call_summary
            # memory.add(MemoryContent(content=tool_call_summary.chat_message.model_dump_json(), mime_type=MemoryMimeType.JSON))
            # memory.add(MemoryContent(content=tool_call_summary.inner_messages, mime_type=MemoryMimeType.JSON))