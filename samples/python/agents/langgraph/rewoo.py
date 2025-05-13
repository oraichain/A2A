import asyncio
import logging
import json
import re
import traceback
from typing import Annotated, Literal, Optional, List, Dict, Any, Tuple, Union
import asyncio
import uuid
from langgraph.graph import StateGraph
from langgraph.store.base import BaseStore
from langgraph.graph.state import CompiledStateGraph
from langchain_core.prompts import PromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient, SSEConnection
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, TextContent, ImageContent, EmbeddedResource, TextResourceContents, BlobResourceContents
from langchain_core.tools import StructuredTool, BaseTool
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START
from litellm import acompletion, completion_cost, model_list
from langgraph.types import Checkpointer
from pydantic import BaseModel, ValidationError
from langchain_core.runnables.config import RunnableConfig

logger = logging.getLogger(__name__)

# NOTE: currently we don't use messages as conversation history. Right now it's just a reserve for future use.
# This function prevents overbloating the memory.
def add_with_trim(current_messages: list[dict], new_messages: list[dict]) -> list[dict]:
    condensed_messages = current_messages + new_messages
    # NOTE: why 10? for simplicity, we can change it later
    if len(condensed_messages) > 10:
        # get the last 10 messages
        condensed_messages = current_messages[-10:]
    else:
        condensed_messages = current_messages
    return condensed_messages

def remove_json_comments(json_str):
    # Remove single-line comments (// ...)
    json_str = re.sub(r'//.*?\n', '\n', json_str)
    return json_str

class ModelUsage(BaseModel):
    total_cost: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    
# Define the ReWOO states
class ReWOOState(BaseModel):
    messages: Annotated[list[dict], add_with_trim] # conversation history with reducer to automatically add new messages
    task: str
    plan: List[Dict[str, Any]]
    mcp_results: list[str]
    analysis: str
    model_usage: ModelUsage
    type: Literal["init", "planner", "worker", "solver"]
    
class InitState(BaseModel):
    messages: Annotated[list[dict], add_with_trim] # conversation history with reducer to automatically add new messages
    task: str
    type: Literal["init"] = "init"
    
class PlannerState(BaseModel):
    task: str
    plan: List[Dict[str, Any]]
    model_usage: ModelUsage
    type: Literal["planner"] = "planner"

class WorkerState(BaseModel):
    messages: Annotated[list[dict], add_with_trim] # conversation history with reducer to automatically add new messages
    task: str
    mcp_results: list[str]
    model_usage: ModelUsage
    type: Literal["worker"] = "worker"

class SolverState(BaseModel):
    messages: Annotated[list[dict], add_with_trim] # conversation history with reducer to automatically add new messages
    task: str
    analysis: str
    model_usage: ModelUsage
    type: Literal["solver"] = "solver"

class ReWOOModel(BaseModel):    
    model: str
    api_key: str

class PlannerModel(ReWOOModel):
    type: Literal["planner"] = "planner"

class AnalysisModel(ReWOOModel):
    type: Literal["analysis"] = "analysis"

# Function to check if response matches InitState
def is_init_state(response: Dict) -> InitState | None:
    try:
        return InitState.model_validate(response)
    except ValidationError:
        return None
    
def is_planner_state(response: Dict) -> PlannerState | None:
    try:
        return PlannerState.model_validate(response)
    except ValidationError:
        return None

def is_worker_state(response: Dict) -> WorkerState | None:
    try:
        return WorkerState.model_validate(response)
    except ValidationError:
        return None

def is_solver_state(response: Dict) -> SolverState | None:
    try:
        return SolverState.model_validate(response)
    except ValidationError:
        return None

StructuredResponse = Union[dict, BaseModel]
StructuredResponseSchema = Union[dict, type[BaseModel]]
    
DEFAULT_PLANNER_USER_PROMPT = """
You are an expert analyst specializing in detecting whale trading patterns. With years of experience understanding deeply crypto trading behavior, on-chain metrics, and derivatives markets, you have developed a keen understanding of whale trading strategies. You can identify patterns in whale positions, analyze their portfolio changes over time, and evaluate the potential reasons behind their trading decisions. Your analysis helps traders decide whether to follow whale trading moves or not.

Parse all the active whale token trades in the <active_whale_token_trades></active_whale_token_trades> XML tag.
Include from 4-6 active whale token trades in the <active_whale_token_trades></active_whale_token_trades> XML tag when calling tools that require tokens as arguments.

If there is no <active_whale_token_trades></active_whale_token_trades> XML tag, you can safely ignore it.
"""

DEFAULT_ANALYSIS_USER_PROMPT = """
#### 2. DeFi-Specific Analysis Prompt
This compact prompt guides the LLM to analyze DeFi metrics, aligning with the Markdown structure.

```plaintext
Analyze DeFi metrics from task results, focusing on, but not limited to:
- Protocol metrics: TVL (USD), APYs, transaction volumes, user activity.
- Token metrics: Prices, trading volumes, volatility.
- Wallet metrics: Balances, transaction patterns.
- Market data: Liquidity, funding rates, leverage.

For each <result> tag:
- Extract metrics from "tool_responses" (e.g., TVL, APY, prices).
- Compare protocols, tokens, or wallets across steps.
- Identify trends, patterns, relationships, implications, and recommendations (e.g., TVL growth, yield changes).
"""

def create_tool_to_url_map(
    mcp_server_name_to_tools: Dict[str, List[BaseTool]],
    sse_mcp_server_sessions: Dict[str, SSEConnection]
) -> Dict[str, str]:
    """
    Create a mapping of tool names to their corresponding server URLs.
    
    Args:
        mcp_server_name_to_tools: Dictionary mapping server names to lists of BaseTool objects.
        sse_mcp_server_sessions: Dictionary mapping server names to SSEConnection dictionaries
        with a 'url' key.
    
    Returns:
        Dictionary mapping tool names to server URLs.
    """
    tool_to_url = {}
    
    for server_name, tools in mcp_server_name_to_tools.items():
        if server_name in sse_mcp_server_sessions:
            server_url = sse_mcp_server_sessions[server_name].get('url')
            for tool in tools:
                # Assuming BaseTool has a 'name' attribute
                tool_to_url[tool.name] = server_url
    
    return tool_to_url

# MCP Client: Execute a tool call via SSE
async def call_tool(tool_name: str, args: Dict, sse_server_url: str) -> CallToolResult:
    try:
        tool_result = await asyncio.wait_for(
            execute_call_tool(tool_name=tool_name, args=args, sse_server_url=sse_server_url),
            timeout=60
        )
        return tool_result
    except Exception as e:
        print(f'Tool call to {tool_name} failed: {str(e)}')
        return CallToolResult(
            content=[{'text': f'Tool call to {tool_name} failed: {str(e)}', 'type': 'text'}],
            isError=True
        )

async def execute_call_tool(tool_name: str, args: Dict, sse_server_url: str) -> CallToolResult:
    async with sse_client(sse_server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            try:
                result = await session.call_tool(tool_name, args)
                if hasattr(session, 'close'):
                    await session.close()
                return result
            except Exception as e:
                print(f'Tool call to {tool_name} failed: {str(e)}')
                if hasattr(session, 'close'):
                    await session.close()
                return CallToolResult(
                    content=[TextContent(text=f'Error executing tool: {str(e)}', type='text')],
                    isError=True
                )

def parse_tool_result(tool_result: CallToolResult) -> List[Dict]:
    parsed_response_content: List[Dict] = []
    if isinstance(tool_result.content, list):
        for item in tool_result.content:
            if isinstance(item, TextContent):
                parsed_response_content.append({
                    "type": item.type,
                    "text": item.text
                })
            elif isinstance(item, ImageContent):
                parsed_response_content.append({
                    "type": item.type,
                    "image": item.data,
                    "mime_type": item.mimeType
                })
            elif isinstance(item, EmbeddedResource):
                if isinstance(item.resource, TextResourceContents):
                    parsed_response_content.append({
                        "type": item.type,
                        "data": item.resource.text,
                        "mime_type": item.resource.mimeType
                    })
                elif isinstance(item.resource, BlobResourceContents):
                    parsed_response_content.append({
                        "type": item.type,
                        "data": item.resource.blob,
                        "mime_type": item.resource.mimeType
                    })
                else:
                    raise ValueError(f"Unsupported resource type: {type(item.resource)}")
            else:
                raise ValueError(f"Unsupported content type: {type(item)}")
    return parsed_response_content

def convert_langchain_schema_to_json_schema(schema: Any) -> Dict[str, Any]:
    """
    Convert a LangChain args_schema (Pydantic model or dict) to a JSON schema.
    """
    if schema is None:
        return {"type": "object", "properties": {}, "required": []}
    
    if isinstance(schema, dict):
        # If already a dict, ensure it has required JSON schema fields
        if "type" not in schema:
            schema["type"] = "object"
        if "properties" not in schema:
            schema["properties"] = {}
        if "required" not in schema:
            schema["required"] = []
        return schema
    
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        # Convert Pydantic model to JSON schema
        try:
            return schema.model_json_schema()
        except Exception as e:
            print(f"Error converting Pydantic schema to JSON: {e}")
            return {"type": "object", "properties": {}, "required": []}
    
    raise ValueError(f"Unsupported schema type: {type(schema)}")

def to_anthropic_tools(tools: List[BaseTool]) -> List[Dict[str, Any]]:
    """
    Convert LangChain BaseTool list to Anthropic-compatible tool format.
    """
    formatted_tools = []
    for tool in tools:
        try:
            # Ensure schema is compatible with Anthropic
            schema = tool.args_schema or {
                "type": "object", 
                "properties": {}, 
                "required": []
            }
            if isinstance(schema, dict) and "type" not in schema:
                schema["type"] = "object"
            formatted_tools.append({
                "name": tool.name,
                "description": tool.description or f"Tool: {tool.name}",
                "input_schema": schema
            })
        except Exception as e:
            print(f"Error formatting tool {tool.name}: {e}")
            continue
    
    return formatted_tools

def to_openai_tools(tools: List[BaseTool]) -> List[Dict[str, Any]]:
    """
    Convert LangChain BaseTool list to OpenAI-compatible tool format.
    """
    formatted_tools = []
    for tool in tools:
        try:
            # Convert args_schema to JSON schema
            input_schema = convert_langchain_schema_to_json_schema(tool.args_schema)
            
            # Ensure schema is OpenAI-compatible
            if not isinstance(input_schema, dict) or input_schema.get("type") != "object":
                input_schema = {"type": "object", "properties": {}, "required": []}
            
            # OpenAI expects 'parameters' instead of 'input_schema'
            formatted_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or f"Tool: {tool.name}",
                    "parameters": input_schema
                }
            })
        except Exception as e:
            print(f"Error formatting tool {tool.name} for OpenAI: {e}")
            continue
    
    return formatted_tools

# Function to detect model type and filter tools
def get_formatted_tools(model_name: str, tools: List[BaseTool]) -> List[Dict[str, Any]]:
    """
    Detect the model type (Anthropic or OpenAI) and return tools in the appropriate format.
    
    Args:
        model: The model instance (e.g., ChatAnthropic, ChatOpenAI, or other).
        tools: List of LangChain BaseTool objects.
    
    Returns:
        List of formatted tools compatible with the model.
    
    Raises:
        ValueError: If the model type is not recognized.
    """
    # Method 1: Check instance type
    model_name = model_name.lower()
    if re.search(r"claude", model_name):
        return to_anthropic_tools(tools)
    elif re.search(r"gpt|o1|grok", model_name):
        return to_openai_tools(tools)
    else:
        raise ValueError(f"Could not determine model type for {model_name}. Please ensure it's an Anthropic or OpenAI model.")
    
class ReWooAgent:    
    def __init__(
        self,
        sse_mcp_server_sessions: dict[str, SSEConnection],
        planner_model: PlannerModel,
        analysis_model: AnalysisModel,
        user_prompt: str = DEFAULT_PLANNER_USER_PROMPT,
        user_analysis_prompt: str = DEFAULT_ANALYSIS_USER_PROMPT,
        ):
            self.planner_model = planner_model
            self.analysis_model = analysis_model
            self.sse_mcp_server_sessions = sse_mcp_server_sessions
            self.user_prompt = user_prompt
            self.user_analysis_prompt = user_analysis_prompt
            self.tools: list[BaseTool] = []
            self.mcp_server_name_to_tools: dict[str, list[BaseTool]] = {}
            self.tool_to_url: dict[str, str] = {}
            self.memory_ttl: float = 15 # 15 minutes
            
    # Planner: Generate a structured plan using litellm
    planner_prompt = PromptTemplate(
        input_variables=["tools", "user_prompt"],
        template="""
        {user_prompt}
            
        Create a step-by-step plan in JSON format to accomplish the task provided in <task></task> XML tag.

        Available MCP tool names are provided in <tools></tools> XML tag.
        
        Extract the already gathered information from the user, which is provided in <knowledge_gathered></knowledge_gathered> XML tag, and use it to inform the plan and call tools appropriately. 
        Refrain from calling tools if you already have the information in the <knowledge_gathered></knowledge_gathered> XML tag.
        If there is no <knowledge_gathered></knowledge_gathered> XML tag, you can safely ignore it.
        
        Also, previous task and analysis results are provided in <previous_analysis></previous_analysis> XML tags.
        There can be multiple <previous_analysis></previous_analysis> XML tags. 
        Refrain from calling tools if you already have the information in the <previous_analysis></previous_analysis> XML tags.
        If the previous analysis complements the current plan, use it to inform the plan.
        If there is no <previous_analysis></previous_analysis> XML tag, you can safely ignore it.

        Each step must be a JSON object with a 'description' (string) and a 'tool_calls' array. The description should be at within 2-3 sentences per step. 
        Each tool call is an object with 'name' (string, matching an MCP tool) and 'args' (object).
                
        Example of the input format:
        
        <tools>
        tool1, tool2, tool3
        </tools>
        <task>
        Task description
        <knowledge_gathered>
        Knowledge gathered
        </knowledge_gathered>
        </task>
        <previous_analysis>
        Previous analysis
        </previous_analysis>
        <previous_analysis>
        Previous analysis
        </previous_analysis>

        Example of the output format:
        [
        {{
            "description": "Step 1",
            "tool_calls": [
            {{"name": "list_resources", "args": {{}}}},
            {{"name": "get_info", "args": {{"id": "123"}}}}
            ]
        }},
        {{
            "description": "Step 2",
            "tool_calls": [
            {{"name": "process_data", "args": {{"data": "value"}}}}
            ]
        }}
        ]

        Return only the JSON format, nothing else.
        Do not include comments, notes in the JSON. For example: do not put // in the JSON.
        It must be a valid JSON that can be loaded by Python using json.loads().
        If you cannot come up with a plan, return a plan with 1 step, no tool calls, and a description of what you will do to complete the task.
        Below are the actual task, knowledge gathered, previous analysis, and tools. Proceed and generate the plan. Good luck!
        
        <tools>
        {tools}
        </tools>
        """
    )
                
    # Analysis and Summary Prompts
    analysis_prompt = PromptTemplate(
        input_variables=["user_analysis_prompt"],
        template="""
        
        # Detailed Analysis

        You are an expert AI assistant tasked with analyzing the results of the task provided in <task></task> XML tag.
        Multiple <result></result> XML tags indicate multiple tool calls. Your goal is to deliver a detailed and comprehensive analysis that fully satisfies the main agent's request with key metrics, relationships, and their implications.

        1. Parse JSON data in each <result></result> tag, containing "step_description" and "tool_responses" (with "name", "args", "isError", "response").
        3. Synthesize results to directly address the task's requirements, focusing on essential details and avoiding redundancy.
        4. Identify and highlight key metrics, patterns, and relationships across tool responses. Explain their implications for the task's objectives in a clear, actionable manner.
        5. Structure the response to match the task's requested format (e.g., lists, tables, or summaries) to ensure clarity and completeness.
        6. If the task requires recommendations or insights, provide them in a compact, structured format, including only the requested fields or details.
        7. If errors occur in tool responses, note their impact and provide a best-effort analysis using available data.
        8. Incorporate domain-specific guidance from '{user_analysis_prompt}' to align with the agent's predefined analytical approach.
        9. Use bullet points, numbered lists, or tables for multiple items to enhance readability and reduce token usage.
        10. Conclude with a brief summary of the most critical findings and their relevance to the task's goals.

        Ensure the response is complete, actionable, and tailored to the task, omitting extraneous information.
        
        Also, previous task and analysis results are provided in <previous_analysis></previous_analysis> XML tags. 
        There can be multiple <previous_analysis></previous_analysis> XML tags.
        Use it to inform the analysis. Refrain from calling tools if you already have the information in the <previous_analysis></previous_analysis> XML tags.
        If the previous analysis complements the current analysis, use it to inform the analysis.
        If there is no <previous_analysis></previous_analysis> XML tag, you can safely ignore it.
        
        Example of the input format:
        <task>
        Task description
        </task>
        <previous_analysis>
        Previous analysis
        </previous_analysis>
        <previous_analysis>
        Previous analysis
        </previous_analysis>
        <result>
        Result
        </result>
        <result>
        Result
        </result>
        ```
        """
    )
                
    # MCP Client: Initialize MultiServerMCPClient
    async def initialize_mcp_client(self):
        try:
            async with MultiServerMCPClient(self.sse_mcp_server_sessions) as mcp_client:
                tools = mcp_client.get_tools()
                self.tools = tools
                self.mcp_server_name_to_tools = mcp_client.server_name_to_tools
                self.tool_to_url = create_tool_to_url_map(
                    self.mcp_server_name_to_tools,
                    self.sse_mcp_server_sessions
                )
                if not tools:
                    raise Exception("Failed to retrieve MCP tools")
        except Exception as e:
            print(f"Error initializing MCP client: {e}")
            raise e
        
    async def _store_analysis_into_knowledge_base(self, config: RunnableConfig, store: BaseStore, analysis_content: str) -> None:
        namespace = (config["configurable"]["thread_id"], "analysis_knowledge_base")
        memory_id = config["metadata"]["task_id"]
        await store.aput(namespace=namespace, key=memory_id, value={"analysis": analysis_content}, index=["analysis"], ttl=self.memory_ttl)
        
    # NOTE: trick to save active whale token trades, so that we can use it for analysis or other tool calls
    async def _presave_active_whale_token_trades(self, config: RunnableConfig, store: BaseStore) -> str:
        key = "active_whale_token_trades"
        tool_name = "perp_whales_open_pos_tokens"
        namespace = (config["configurable"]["thread_id"], "analysis_knowledge_base")
        result = await store.aget(namespace, key=key)
        # if None -> we intentionally call the tool to get the active whale token trades
        if result is None:
            tool_result = await call_tool(tool_name, {}, self.tool_to_url[tool_name])
            parsed_tool_result = parse_tool_result(tool_result)
            tokens = parsed_tool_result[0].get("text", "") if len(parsed_tool_result) > 0 else ""
            await store.aput(namespace=namespace, key=key, value={"active_whale_token_trades": tokens}, ttl=self.memory_ttl)
            return json.dumps(f"<active_whale_token_trades>{tokens}</active_whale_token_trades>")
        else:
            return json.dumps(f"<active_whale_token_trades>{result.value.get("active_whale_token_trades", "")}</active_whale_token_trades>")
        
        
    async def _load_previous_analysis_from_knowledge_base(self, task: str, config: RunnableConfig, store: BaseStore) -> str:
        namespace = (config["configurable"]["thread_id"], "analysis_knowledge_base")
        result = await store.asearch(namespace, query=task, limit=5)
        for res in result:
            logger.info(f"Prev analysis score: {res.score}")
            logger.info(f"Prev analysis value: {res.value}")
            logger.info(f"Prev analysis updated_at: {res.updated_at}")
        prev_analysis_messages = [
            {"role": "assistant", "content": [{"text": res.value.get('analysis', ''), "type": "text"}]}
            # NOTE: why 0.4?
            for res in result if res.score and res.score >= 0.4
        ]
        if len(prev_analysis_messages) > 0:
            first_text = prev_analysis_messages[0]["content"][0]["text"]
            prev_analysis_messages[0]["content"][0]["text"] = f"<previous_analysis>\n{first_text}\n"
            last_text = prev_analysis_messages[-1]["content"][0]["text"]
            prev_analysis_messages[-1]["content"][0]["text"] = f"{last_text}\n</previous_analysis>\n"
        return prev_analysis_messages

    async def worker(self, state: PlannerState) -> WorkerState:

        async def process_tool_call(tool_call: Dict) -> Dict:
            """Process a single tool call."""
            try:
                server_url = self.tool_to_url.get(tool_call["name"])
                if not server_url:
                    raise ValueError(f"No server URL for tool {tool_call['name']}")
                
                response = await call_tool(
                    tool_call["name"],
                    tool_call.get("args", {}),
                    server_url
                )
                
                parsed_response_content = parse_tool_result(response)
                return {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}),
                    "response": parsed_response_content,
                    "isError": response.isError
                }
            except Exception as e:
                logger.error(f"Error executing MCP command {tool_call['name']}: {e}\nTraceback: {traceback.format_exc()}")
                return {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}),
                    "error": str(e),
                    "isError": True
                }

        async def process_step(step: Dict):
            """Process a single step, running tool calls concurrently."""
            step_results = {
                "step_description": step["description"],
                "tool_responses": []
            }
            tool_calls = step.get("tool_calls", [])
            if tool_calls:
                tool_results = await asyncio.gather(
                    *(process_tool_call(tool_call) for tool_call in tool_calls),
                    return_exceptions=True
                )
                for result in tool_results:
                    if isinstance(result, Exception):
                        logger.error(f"Tool call failed: {result}\nTraceback: {traceback.format_exc()}")
                        step_results["tool_responses"].append({
                            "name": "unknown",
                            "args": {},
                            "error": str(result),
                            "isError": True
                        })
                    else:
                        step_results["tool_responses"].append(result)
            
            return step_results

        # Run all steps concurrently
        results = await asyncio.gather(
            *(process_step(step) for step in state.plan),
            return_exceptions=True
        )
        
        # Handle any step-level exceptions and split large results
        final_results: List[str] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Step processing failed: {result}\nTraceback: {traceback.format_exc()}")
                error_result = {
                    "name": "unknown",
                    "args": {},
                    "error": str(result),
                    "isError": True
                }
                error_str = f"\n<result>{json.dumps(error_result)}</result>\n"
                final_results.append(error_str)
            else:
                final_results.extend(f"<result>{json.dumps(result)}</result>")

        return {
            "mcp_results": final_results,
            "type": "worker"
        }

    async def planner(self, state: InitState, config: RunnableConfig, *, store: BaseStore) -> PlannerState:
        system_prompt = self.planner_prompt.format(
            tools=json.dumps([tool.name for tool in self.tools]), 
            user_prompt=self.user_prompt
        )
        system_message = {
            "role": "system",
            "content": [
                {
                    "text": system_prompt,
                    "type": "text",
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        }
        task_message = {
            "role": "user",
            "content": [
                {
                    "text": f"<task>{state.task}</task>",
                    "type": "text",
                }
            ]
        }
        previous_analysis_messages = await self._load_previous_analysis_from_knowledge_base(state.task, config, store)
        # NOTE: trick to save active whale token trades, so that we can use it for analysis or other tool calls
        active_whale_token_trades = await self._presave_active_whale_token_trades(config, store)
        logger.info(f"Active whale token trades: {active_whale_token_trades}")
        active_whale_token_trades_message = {"role": "assistant", "content": [{"text": active_whale_token_trades, "type": "text"}]}
        messages = [system_message, task_message, *previous_analysis_messages, active_whale_token_trades_message]
    
        # Format tools for Anthropic
        formatted_tools = get_formatted_tools(self.planner_model.model, self.tools)
        
        # Use litellm for completion
        response = await acompletion(
            model=self.planner_model.model,
            messages=messages,
            api_key=self.planner_model.api_key,
            tools=formatted_tools
        )
        
        # Calculate cost and tokens
        cost = completion_cost(response)
        token_data = response.usage if response.usage else {}
        total_tokens = token_data.get('total_tokens', 0)
        prompt_tokens = token_data.get('prompt_tokens', 0)
        completion_tokens = token_data.get('completion_tokens', 0)
        cache_read_tokens = token_data.get('cache_read_tokens', 0)
        cache_write_tokens = token_data.get('cache_write_tokens', 0)
        
        logger.info(f"Planner response: {response.choices[0].message.content}")
        sanitized_plan = remove_json_comments(response.choices[0].message.content)
        try:
            plan = json.loads(sanitized_plan)
            if not isinstance(plan, list):
                raise ValueError("Plan must be a JSON array")
                
            return {
                "plan": plan,
                "model_usage": {
                    "total_cost": cost,
                    "total_tokens": total_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cache_read_tokens": cache_read_tokens,
                    "cache_write_tokens": cache_write_tokens
                },
                "type": "planner"
            }
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing plan: {e}")
            plan = [{
                "description": "Failed to generate a plan due to the LLM error. Please try again.",
                "tool_calls": []
            }]
        return {
            "plan": plan,
            "type": "planner"
        }
        
    # Solver: Analyze and summarize all worker results using litellm
    async def solver(self, state: WorkerState, config: RunnableConfig, *, store: BaseStore) -> SolverState:
        try:
            # Generate analysis
            analysis_input = self.analysis_prompt.format(
                user_analysis_prompt=self.user_analysis_prompt
            )
            system_message = {
                "role": "system",
                "content": [
                    {
                        "text": analysis_input,
                        "type": "text",
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            }
            task_message = {"role": "user", "content": [{"text": f"<task>{state.task}</task>", "type": "text"}]}
            
            previous_analysis_messages = await self._load_previous_analysis_from_knowledge_base(state.task, config, store)
            messages = [system_message, task_message, *previous_analysis_messages]
            messages.extend([{"role": "assistant", "content": result} for result in state.mcp_results])  
            
            analysis_response = await acompletion(
                model=self.analysis_model.model,
                messages=messages,
                api_key=self.analysis_model.api_key
            )
            analysis_cost = completion_cost(analysis_response)
            analysis_token_data = analysis_response.usage if analysis_response.usage else {}
            
            # Update metrics
            total_cost = state.model_usage.total_cost + analysis_cost
            total_tokens = (
                state.model_usage.total_tokens +
                analysis_token_data.get('total_tokens', 0)
            )
            prompt_tokens = (
                state.model_usage.prompt_tokens +
                analysis_token_data.get('prompt_tokens', 0)
            )
            completion_tokens = (
                state.model_usage.completion_tokens +
                analysis_token_data.get('completion_tokens', 0)
            )
            cache_read_tokens = (
                state.model_usage.cache_read_tokens +
                analysis_token_data.get('cache_read_tokens', 0)
            )
            cache_write_tokens = (
                state.model_usage.cache_write_tokens +
                analysis_token_data.get('cache_write_tokens', 0)
            )
            
            analysis_content = analysis_response.choices[0].message.content
            await self._store_analysis_into_knowledge_base(config, store, analysis_content)
            
            return {
                "analysis": analysis_content,
                "messages": [
                    {"role": "user", "content": [{"text": f"<previous_analysis>\n{state.task}\n", "type": "text"}]},
                    {"role": "assistant", "content": [{"text": f"{analysis_content}\n</previous_analysis>\n", "type": "text"}]}
                ],
                "model_usage": {
                    "total_cost": total_cost,
                    "total_tokens": total_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cache_read_tokens": cache_read_tokens,
                    "cache_write_tokens": cache_write_tokens
                },
                "type": "solver"
            }
        except Exception as e:
            print(f"Error in solver: {e}")
            print(traceback.format_exc())
            return {
                "task": state.task,
                "analysis": "",
                "model_usage": state.model_usage,
                "type": "solver"
            }
        
async def create_rewoo_agent(
    sse_mcp_server_sessions: dict[str, SSEConnection],
    planner_model: PlannerModel,
    analysis_model: AnalysisModel,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    user_prompt: Optional[str] = None,
    user_analysis_prompt: Optional[str] = None,
) -> CompiledStateGraph:
    """Create a ReWOO agent workflow graph.
    
    Args:
        sse_mcp_server_sessions: Dictionary mapping server names to SSE connections
        planner_model: Model for planning steps
        analysis_model: Model for analyzing results
        solver_model: Model for solving/finalizing results
        user_prompt: Optional custom user prompt
        user_analysis_prompt: Optional custom analysis prompt
        
    Returns:
        Compiled workflow graph
    """
    agent = ReWooAgent(
        sse_mcp_server_sessions=sse_mcp_server_sessions,
        planner_model=planner_model,
        analysis_model=analysis_model,
        user_prompt=user_prompt,
        user_analysis_prompt=user_analysis_prompt,
    )
    await agent.initialize_mcp_client()
    
    workflow = StateGraph(ReWOOState, input=InitState, output=SolverState)
    
    # Add nodes
    workflow.add_node("planner", agent.planner)
    workflow.add_node("worker", agent.worker)
    workflow.add_node("solver", agent.solver)

    # Add edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "worker")
    workflow.add_edge("worker", "solver")
    workflow.add_edge("solver", END)

    return workflow.compile(checkpointer=checkpointer, store=store)