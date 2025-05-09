import asyncio
import logging
from operator import add
import json
import traceback
from typing import Annotated, Literal, Optional, List, Dict, Any, Union
import asyncio
import uuid
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.prompts import PromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient, SSEConnection
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, TextContent, ImageContent, EmbeddedResource, TextResourceContents, BlobResourceContents
from langchain_core.tools import StructuredTool, BaseTool
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import END, START
from litellm import acompletion, completion_cost,completion
from langgraph.types import Checkpointer
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

class ModelUsage(BaseModel):
    total_cost: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    
# Define the ReWOO states
class ReWOOState(BaseModel):
    messages: Annotated[List[AnyMessage], add] # conversation history with reducer to automatically add new messages
    task: str
    plan: List[Dict[str, Any]]
    mcp_results: list[str]
    analysis: str
    model_usage: ModelUsage
    type: Literal["init", "planner", "worker", "solver"]
    
class InitState(BaseModel):
    task: str
    type: Literal["init"] = "init"
    
class PlannerState(BaseModel):
    task: str
    plan: List[Dict[str, Any]]
    model_usage: ModelUsage
    type: Literal["planner"] = "planner"

class WorkerState(BaseModel):
    task: str
    mcp_results: list[str]
    model_usage: ModelUsage
    type: Literal["worker"] = "worker"

class SolverState(BaseModel):
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

When you use any tool, I expect you to push its limits: fetch all the data it can provide, whether that means iterating through multiple batches, adjusting parameters like offsets, or running the tool repeatedly to cover every scenario. Don't work with small set of data for sure, fetch as much as you can. Don’t stop until you’re certain you’ve captured everything there is to know.
"""

DEFAULT_ANALYSIS_USER_PROMPT = """
#### 2. DeFi-Specific Analysis Prompt
This compact prompt guides the LLM to analyze DeFi metrics, aligning with the Markdown structure.

```plaintext
Analyze DeFi metrics from task results, focusing on:
- Protocol metrics: TVL (USD), APYs, transaction volumes, user activity.
- Token metrics: Prices, trading volumes, volatility.
- Wallet metrics: Balances, transaction patterns.
- Market data: Liquidity, funding rates, leverage.

For each <result> tag:
- Extract metrics from "tool_responses" (e.g., TVL, APY, prices).
- Compare protocols, tokens, or wallets across steps.
- Identify trends (e.g., TVL growth, yield changes).

Output requirements:
- **Market Overview**: Summarize protocol TVL, volumes, yields, and token performance (30-day trends).
- **Top-Performing Entities**: Detail top protocols or wallets (e.g., TVL, APY, PnL) with strategies.
- **Patterns**: Identify protocol usage or wallet trading patterns (e.g., yield farming, arbitrage).
- **Risk Assessment**: List risks (e.g., smart contract issues, high leverage) with asset-specific ratings.
- **Trend Analysis**: Analyze token price or protocol TVL trends with support/resistance levels.
- **Recommendations**: Suggest DeFi strategies (e.g., staking, liquidity provision) with risk/reward and safety scores.
- **Conclusion**: Highlight key DeFi trends, risks, and opportunities.
- Note step relationships (e.g., protocol data informing token prices).
- Flag incomplete data with assumptions.
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

class ReWooAgent:    
    def __init__(
        self,
        sse_mcp_server_sessions: dict[str, SSEConnection],
        planner_model: PlannerModel,
        analysis_model: AnalysisModel,
        checkpointer: Optional[Checkpointer] = None,
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
            self.checkpointer = checkpointer
            
    # Planner: Generate a structured plan using litellm
    planner_prompt = PromptTemplate(
        input_variables=["task", "tools", "user_prompt"],
        template="""
        {user_prompt}
            
        Create a step-by-step plan in JSON format to accomplish the task: {task}

        Available MCP tools: {tools}

        Each step must be a JSON object with a 'description' (string) and a 'tool_calls' array. The description should be at least 5 to 10 sentences per step. 
        Each tool call is an object with 'name' (string, matching an MCP tool) and 'args' (object).

        Example format:
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

        Return only the JSON array, nothing else."""
    )
                
    # Analysis and Summary Prompts
    analysis_prompt = PromptTemplate(
        input_variables=["task", "user_analysis_prompt"],
        template="""
        
        # Detailed Analysis

        You are an expert AI assistant tasked with analyzing the results of the task '{task}' provided in <result></result> XML tags. Multiple tags indicate multiple tool calls. The user has provided domain-specific analysis instructions in '{user_analysis_prompt}'.

        ## Instructions
        - Parse JSON data in each <result></result> tag, containing "step_description" and "tool_responses" (with "name", "args", "isError", "response").
        - Structure the output in Markdown following the format below, ensuring clarity and depth:
        - **Market Overview**: Summarize current market conditions and key asset performance (if applicable).
        - **Top-Performing Entities**: Analyze key entities (e.g., whales, protocols) with metrics and strategies.
        - **Identified Patterns**: Identify behavioral or operational patterns (e.g., trading strategies).
        - **Risk Assessment**: Evaluate risks with a table or list of risk factors.
        - **Market Trend Analysis**: Analyze trends for key assets or metrics.
        - **Recommendations**: Provide actionable recommendations with rationale, risk/reward, and safety scores.
        - **Conclusion**: Summarize findings, trends, and strategic advice.
        - Explicitly note relationships between steps in the relevant section (e.g., Market Overview or Top-Performing Entities).
        - Flag ambiguous or incomplete data with best-effort interpretation.
        - Ensure the analysis is successful (complete and aligned with instructions) or note failures.

        ## Output Format
        ```markdown
        # Detailed Analysis

        ## 1. Market Overview
        ### Current Market Conditions
        [Summary of market metrics, e.g., TVL, volumes, or positions]
        ### Key Asset Performance
        [Asset-specific metrics, trends, and analysis]

        ## 2. Top-Performing Entities
        [Details of key entities, e.g., protocols or whales, with metrics and strategies]

        ## 3. Identified Patterns
        [Behavioral or operational patterns, e.g., trading or protocol strategies]

        ## 4. Risk Assessment
        [Risk factors with table or list, including asset-specific risks]

        ## 5. Market Trend Analysis
        [Trend analysis for key assets or metrics, e.g., price, TVL]

        ## 6. Recommendations
        [Actionable recommendations with entry, stop loss, take profit, leverage, risk/reward, safety score, timeframe, and rationale]

        ## 7. Conclusion
        [Summary of findings, trends, and strategic advice]

        ## Success
        [true/false with brief explanation]
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
                
                parsed_response_content: List[Dict] = []
                if isinstance(response.content, list):
                    for item in response.content:
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

        async def process_step(step: Dict) -> str:
            """Process a single step, running tool calls concurrently."""
            step_results = {
                "step_description": step["description"],
                "tool_responses": []
            }
            tool_calls = step.get("tool_calls", [])
            if tool_calls:
                # Run all tool calls concurrently
                tool_results = await asyncio.gather(
                    *(process_tool_call(tool_call) for tool_call in tool_calls),
                    return_exceptions=True
                )
                # Collect results, handling any exceptions
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
            
            return f"\n<result>{json.dumps(step_results)}</result>\n"

        # Run all steps concurrently
        results = await asyncio.gather(
            *(process_step(step) for step in state.plan),
            return_exceptions=True
        )
        
        # Handle any step-level exceptions
        final_results: list[str] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Step processing failed: {result}\nTraceback: {traceback.format_exc()}")
                final_results.append(f"\n<result>{json.dumps({"name":"unknown", "args": {}, "error": str(result), "isError": True})}</result>\n")
            else:
                final_results.append(result)

        return {
            "mcp_results": final_results,
            "type": "worker"
        }

    async def planner(self, state: InitState) -> PlannerState:
        prompt = self.planner_prompt.format(
            task=state.task, 
            tools=json.dumps([tool.name for tool in self.tools]), 
            user_prompt=self.user_prompt
        )
    
        # Format tools for Anthropic
        formatted_tools = []
        for tool in self.tools:
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
        
        # Use litellm for completion
        response = await acompletion(
            model=self.planner_model.model,
            messages=[{"role": "user", "content": prompt}],
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
        
        try:
            plan = json.loads(response.choices[0].message.content)
            if not isinstance(plan, list):
                raise ValueError("Plan must be a JSON array")
                
            ai_messages = [
                AIMessage(
                    content=step["description"],
                    tool_calls=[
                        {**tool_call, "id": str(uuid.uuid4())}
                        for tool_call in step["tool_calls"]
                    ]
                )
                for step in plan
            ]
            return {
                "plan": plan,
                "messages": [
                    HumanMessage(content=state.task),
                    *ai_messages
                ],
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
                "description": "Fallback: Query MCP server",
                "tool_calls": [{"name": "list_resources", "args": {}, "id": str(uuid.uuid4())}]
            }]
        return {
            "plan": [],
            "messages": [
                HumanMessage(content=state.task),
            ],
        }
        
    # Solver: Analyze and summarize all worker results using litellm
    async def solver(self, state: WorkerState) -> SolverState:
        try:
            # Generate analysis
            analysis_input = self.analysis_prompt.format(
                task=state.task,
                user_analysis_prompt=self.user_analysis_prompt
            )
            messages = [{"role": "user", "content": analysis_input}]
            messages.extend([{"role": "user", "content": result} for result in state.mcp_results])
            
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
            
            return {
                "analysis": analysis_response.choices[0].message.content,
                "messages": [AIMessage(content=analysis_response.choices[0].message.content)],
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
    user_prompt: Optional[str] = None,
    user_analysis_prompt: Optional[str] = None,
) -> CompiledStateGraph:
    """Create a ReWOO agent workflow graph.
    
    Args:
        sse_mcp_server_sessions: Dictionary mapping server names to SSE connections
        planner_model: Model for planning steps
        analysis_model: Model for analyzing results
        solver_model: Model for solving/finalizing results
        checkpointer: Optional checkpointer for state persistence
        user_prompt: Optional custom user prompt
        user_analysis_prompt: Optional custom analysis prompt
        
    Returns:
        Compiled workflow graph
    """
    agent = ReWooAgent(
        sse_mcp_server_sessions=sse_mcp_server_sessions,
        planner_model=planner_model,
        analysis_model=analysis_model,
        checkpointer=checkpointer,
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

    return workflow.compile()