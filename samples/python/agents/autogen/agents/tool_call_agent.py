import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Sequence, Tuple, override
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken, FunctionCall
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, ThoughtEvent, ToolCallExecutionEvent, ToolCallRequestEvent, TextMessage, StructuredMessage,ModelClientStreamingChunkEvent, ToolCallSummaryMessage, UserMessage
from autogen_agentchat.base import Response, Handoff as HandoffBase
from autogen_core.models import CreateResult, SystemMessage, ChatCompletionClient, AssistantMessage, FunctionExecutionResultMessage, FunctionExecutionResult, RequestUsage
from autogen_core.memory import MemoryContent, MemoryMimeType, ListMemory
from autogen_core.tools import BaseTool
from pydantic import BaseModel
from autogen_core.model_context import (
    ChatCompletionContext,
)
from autogen_core.models import LLMMessage

TOOL_CALL_SUMMARY_INSTRUCTION = """
You are an expert AI assistant specializing in decentralized finance (DeFi) analysis. Your task is to summarize MCP tool calls. The MCP query and the result are shared inside <tool_call_result></tool_call_result> XML tag.

Each tool call result contains relevant DeFi metrics, such as protocol names, total value locked (TVL), transaction volumes, yield rates, or other key performance indicators.

    Instructions:
    1. Parse the data within the <tool_call_result></tool_call_result> XML tag.
    2. Provide a deep analysis for all DeFi metrics in the tool call result, such as:
    - If the metrics involves token prices or user balances, funds, orders, or any form of currencies, you must analyze every token, wallet and order involved.
    - Protocol or platform names
    - Total value locked (TVL) in USD
    - Transaction volumes or counts
    - Yield rates or APYs
    - Token prices or market data
    - Other relevant DeFi-specific metrics
    - Put your analysis and highlights in <analysis></analysis> XML tag within <tool_call_result></tool_call_result> XML tag.
    3. Provide a detailed reasoning for the analysis in <reasoning></reasoning> XML tag within <tool_call_result></tool_call_result> XML tag.
    - The reasoning must be based on the analysis and highlights in <analysis></analysis> XML tag.
    - Provide reasons for the analysis and highlights in <reasoning></reasoning> XML tag.
    4. Provide a detailed summary as a conclusion that includes:
    - The market trends based on the data (e.g., dominant protocols, overall TVL trends).
    - Key insights for each tool call result, highlighting significant metrics (e.g., highest TVL, notable yield opportunities, or unusual transaction activity).
    - Comparisons between protocols or metrics where relevant (e.g., TVL growth, yield competitiveness).
    - Any trends, risks, or opportunities in the DeFi ecosystem inferred from the data.
    - Put your analysis and highlights in <tool_call_summary></tool_call_summary> XML tag within <tool_call_result></tool_call_result> XML tag.
    5. Structure the summary clearly with sections or bullet points for readability.
    6. If any data is ambiguous or incomplete, note it and provide a best-effort interpretation.
    7. Format the response within <tool_call_result></tool_call_result> XML tag.
    
    
For example, the query format is:
<tool_call_result>
tool_name: perpetual_all_whales_detail
arguments: {{
    "address": "0x17f4E182aD8d1B27F430c094a96E844d13f8da14"
}}
result: -|\\\\n| Total PNL | $611919.69 |\\\\n| 24H PNL | $366968.14 |\\\\n| 48H PNL | $116166.31 |\\\\n| 7D PNL | $316888.56 |\\\\n| 30D PNL | $1785565.09 |\\\\n| Snapshot Time | 2025-05-06T01:55:52.427Z |\\\\n\\\\n#### Open Positions\\\\n\\\\n| Token | Side | Size | Entry Price | Liquidate Price | Leverage | PNL | Funding Fee |\\\\n|-------|------|------|-------------|----------------|----------|-----|--------------|\\\\n| ETH | LONG | $3436083.00 | $1809.42 | $764.62 | 25x | $-11482.70 | $37.84|\\\\n| HYPE | LONG | $3169212.00 | $19.84 | $6.49 | 5x | $66966.46 | $5751.47|\\\\n| TRUMP | LONG | $2949581.00 | $11.33 | $3.41 | 10x | $-64704.52 | $-11983.35|\\\\n| FARTCOIN | LONG | $2018102.00 | $1.13 | $0.18 | 3x | $-36552.22 | $11633.17|\\\\n| BTC | LONG | $1324833.00 | $93879.60 | $18164.05 | 40x | $5849.93 | $11.90|\\\\n| kPEPE | LONG | $1223299.00 | $0.01 | N/A | 10x | $-881.58 | $163.35|\\\\n\\\\n## Address: 0x17f4E182aD8d1B27F430c094a96E844d13f8da14\\\\n\
</tool_call_result>

Then your response must be in the following format:
<tool_call_result>
tool_name: perpetual_all_whales_detail
arguments: {{
    "address": "0x17f4E182aD8d1B27F430c094a96E844d13f8da14"
}}
<analysis>
Total PNL is $611919.69, which is a 24H PNL of $366968.14.
</analysis>
<reasoning>
The tool call result shows a strong performance with a total PNL of $611919.69. The 24H PNL is $366968.14, indicating a significant increase in profitability over the last 24 hours.
</reasoning>
<tool_call_summary>
Overall, the tool call result shows a strong performance with a total PNL of $611919.69. The 24H PNL is $366968.14, indicating a significant increase in profitability over the last 24 hours.
</tool_call_summary>
</tool_call_result>

Keep the response within 4-5 paragraphs, each having 5-6 sentences. If you could make the paragraphs longer to include more details, do so. Include as many figures and numbers as possible with corolation to the entities involved.

## Analysis Best‑Practices
1. **Source Triangulation:** corroborate important claims with at least two independent sources when possible.
2. **Bias Awareness:** note significant discrepancies between sources and highlight uncertainties.
3. **Insight First:** lead summaries with the takeaway, then provide supporting details.
4. **Data Hygiene:** document data origin, cleaning steps, and assumptions made during analysis.
5. **Result Validation:** sanity‑check numbers, code outputs, and logic before presenting.
"""

class ToolCallAgent(AssistantAgent):
    def __init__(self, *args, **kwargs):
        memory: ListMemory | None = kwargs.get("list_memory", None)
        if memory is None:
            raise ValueError("list_memory is required")
        del kwargs["list_memory"]
        super().__init__(*args, **kwargs)
        self._memory = memory
    
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
        await self._add_messages_to_context(
            model_context=model_context,
            messages=messages,
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
        async for output_event in self._process_model_result(
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
    ) -> AsyncGenerator[Response, None]:
        """
        If reflect_on_tool_use=False, create a summary message of all tool calls.
        """
        # Filter out calls which were actually handoffs
        normal_tool_calls = [(call, result) for call, result in executed_calls_and_results if call.name not in handoffs]
        tool_call_summaries: List[str] = []
        total_request_usage: RequestUsage = RequestUsage(
            prompt_tokens=0,
            completion_tokens=0,
        )
        for tool_call, tool_call_result in normal_tool_calls:
            tool_call_summary = tool_call_summary_format.format(
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                result=tool_call_result.content,
            )
            llm_messages: List[LLMMessage] = [
                SystemMessage(content=TOOL_CALL_SUMMARY_INSTRUCTION),
                UserMessage(content=tool_call_summary, source=agent_name),
            ]
          
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
                # yield thought_event
                inner_messages.append(thought_event)
                
            if isinstance(reflection_result.content, str):
                tool_call_summaries.append(reflection_result.content if reflection_result.content else "empty")
            else:
                tool_call_summaries.append(
                    tool_call_summary
                )
            total_request_usage.prompt_tokens += reflection_result.usage.prompt_tokens
            total_request_usage.completion_tokens += reflection_result.usage.completion_tokens
          
        final_tool_call_summary = "\n".join(tool_call_summaries)
        yield Response(
            chat_message=ToolCallSummaryMessage(
                content=final_tool_call_summary if final_tool_call_summary else "empty",
                source=agent_name,
                models_usage=total_request_usage,
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
        exec_results = [FunctionExecutionResult(content=result.content if result.is_error else "Success. Continue!", call_id=result.call_id, is_error=result.is_error, name=result.name) for _, result in executed_calls_and_results]
        
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