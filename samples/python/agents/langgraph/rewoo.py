import os
import json
from typing import TypedDict, List, Dict, Any
import asyncio
from langgraph.graph import StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, TextContent, ImageContent, EmbeddedResource, TextResourceContents, BlobResourceContents
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START
from litellm import acompletion, completion_cost

# MCP Client Configuration
MCP_SERVER_URLS = ["http://15.235.225.246:4010/sse"]

# Define the ReWOO state
class ReWOOState(TypedDict):
    task: str
    plan: List[Dict[str, Any]]
    results: List[Dict[str, Any]]
    mcp_tools: Dict[str, Any]
    total_cost: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int

# MCP Client: Initialize MultiServerMCPClient
async def initialize_mcp_client(state: ReWOOState) -> ReWOOState:
    try:
        async with MultiServerMCPClient({'hyperwhales': {
            "url": MCP_SERVER_URLS[0],
            "transport": "sse",
        }}) as mcp_client:
            tools = mcp_client.get_tools()
            if tools:
                return {
                    "mcp_tools": tools,
                    "total_cost": state.get("total_cost", 0.0),
                    "total_tokens": state.get("total_tokens", 0),
                    "prompt_tokens": state.get("prompt_tokens", 0),
                    "completion_tokens": state.get("completion_tokens", 0),
                    "cache_read_tokens": state.get("cache_read_tokens", 0),
                    "cache_write_tokens": state.get("cache_write_tokens", 0)
                }
            raise Exception("Failed to retrieve MCP tools")
    except Exception as e:
        print(f"Error initializing MCP client: {e}")
        return {
            "mcp_tools": {},
            "total_cost": state.get("total_cost", 0.0),
            "total_tokens": state.get("total_tokens", 0),
            "prompt_tokens": state.get("prompt_tokens", 0),
            "completion_tokens": state.get("completion_tokens", 0),
            "cache_read_tokens": state.get("cache_read_tokens", 0),
            "cache_write_tokens": state.get("cache_write_tokens", 0)
        }

# MCP Client: Execute a tool call via SSE
async def call_tool(tool_name: str, args: Dict) -> CallToolResult:
    try:
        tool_result = await asyncio.wait_for(
            execute_call_tool(tool_name=tool_name, args=args),
            timeout=60
        )
        return tool_result
    except Exception as e:
        print(f'Tool call to {tool_name} failed: {str(e)}')
        return CallToolResult(
            content=[{'text': f'Tool call to {tool_name} failed: {str(e)}', 'type': 'text'}],
            isError=True
        )

async def execute_call_tool(tool_name: str, args: Dict) -> CallToolResult:
    async with sse_client(MCP_SERVER_URLS[0]) as (read, write):
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

# Worker: Execute all tool calls across all plan steps
async def worker(state: ReWOOState) -> ReWOOState:
    plan = state["plan"]
    results: list[str] = []
    for step in plan:
        step_results = {
            "step_description": step["description"],
            "tool_responses": []
        }
        for tool_call in step["tool_calls"]:
            try:
                response = await call_tool(tool_call["method"], tool_call.get("params", {}))
                parsed_response_content: list[str] = []
                if isinstance(response.content, list):
                    for item in response.content:
                        if isinstance(item, TextContent):
                            parsed_response_content.append({"type": item.type, "text": item.text})
                        elif isinstance(item, ImageContent):
                            parsed_response_content.append({"type": item.type, "image": item.data, "mime_type": item.mimeType})
                        elif isinstance(item, EmbeddedResource):
                            if isinstance(item.resource, TextResourceContents):
                                parsed_response_content.append({"type": item.type, "data": item.resource.text, "mime_type": item.resource.mimeType})
                            elif isinstance(item.resource, BlobResourceContents):
                                parsed_response_content.append({"type": item.type, "data": item.resource.blob, "mime_type": item.resource.mimeType})
                            else:
                                raise ValueError(f"Unsupported resource type: {type(item.resource)}")
                        else:
                            raise ValueError(f"Unsupported content type: {type(item)}")
                result = {
                    "response": parsed_response_content,
                    "isError": response.isError
                } if response else {"error": "No response received"}
            except Exception as e:
                print(f"Error executing MCP command {tool_call['method']}: {e}")
                result = {"error": str(e), "isError": True}
            step_results["tool_responses"].append({
                "method": tool_call["method"],
                "params": tool_call.get("params", {}),
                **result
            })
        results.append(f"\n<result>{json.dumps(step_results)}</result>\n")
    return {
        "results": results,
        "total_cost": state.get("total_cost", 0.0),
        "total_tokens": state.get("total_tokens", 0),
        "prompt_tokens": state.get("prompt_tokens", 0),
        "completion_tokens": state.get("completion_tokens", 0),
        "cache_read_tokens": state.get("cache_read_tokens", 0),
        "cache_write_tokens": state.get("cache_write_tokens", 0)
    }

# Planner: Generate a structured plan using litellm
planner_prompt = PromptTemplate(
    input_variables=["task", "tools"],
    template="""
    
You are an expert analyst specializing in detecting whale trading patterns. With years of experience understanding deeply crypto trading behavior, on-chain metrics, and derivatives markets, you have developed a keen understanding of whale trading strategies. You can identify patterns in whale positions, analyze their portfolio changes over time, and evaluate the potential reasons behind their trading decisions. Your analysis helps traders decide whether to follow whale trading moves or not. 

When you use any tool, I expect you to push its limits: fetch all the data it can provide, whether that means iterating through multiple batches, adjusting parameters like offsets, or running the tool repeatedly to cover every scenario. Don't work with small set of data for sure, fetch as much as you can. Don’t stop until you’re certain you’ve captured everything there is to know.
    
Create a step-by-step plan in JSON format to accomplish the task: {task}

Available MCP tools: {tools}

Each step must be a JSON object with a 'description' (string) and a 'tool_calls' array. The description should be at least 5 to 10 sentences per step. 
Each tool call is an object with 'method' (string, matching an MCP tool) and 'params' (object).

Example format:
[
  {{
    "description": "Step 1",
    "tool_calls": [
      {{"method": "list_resources", "params": {{}}}},
      {{"method": "get_info", "params": {{"id": "123"}}}}
    ]
  }},
  {{
    "description": "Step 2",
    "tool_calls": [
      {{"method": "process_data", "params": {{"data": "value"}}}}
    ]
  }}
]

Return only the JSON array, nothing else."""
)

async def planner(state: ReWOOState) -> ReWOOState:
    tools: list[StructuredTool] = state.get("mcp_tools", {})
    prompt = planner_prompt.format(task=state["task"], tools=json.dumps([tool.name for tool in tools]))
   
    # Format tools for Anthropic
    formatted_tools = []
    for tool in tools:
        try:
            # Ensure schema is compatible with Anthropic
            schema = tool.args_schema or {"type": "object", "properties": {}, "required": []}
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
        model=os.getenv("LLM_MODEL"),
        messages=[{"role": "user", "content": prompt}],
        api_key=os.getenv("API_KEY"),
        tools=formatted_tools
    )
    
    # Calculate cost and tokens
    cost = completion_cost(response)
    token_data = response.usage if response.usage else {}
    total_cost = state.get("total_cost", 0.0) + cost
    total_tokens = state.get("total_tokens", 0) + token_data.get('total_tokens', 0)
    prompt_tokens = state.get("prompt_tokens", 0) + token_data.get('prompt_tokens', 0)
    completion_tokens = state.get("completion_tokens", 0) + token_data.get('completion_tokens', 0)
    cache_read_tokens = state.get("cache_read_tokens", 0) + token_data.get('cache_read_tokens', 0)
    cache_write_tokens = state.get("cache_write_tokens", 0) + token_data.get('cache_write_tokens', 0)
    
    try:
        plan = json.loads(response.choices[0].message.content)
        if not isinstance(plan, list):
            raise ValueError("Plan must be a JSON array")
        for step in plan:
            if not isinstance(step, dict) or "description" not in step or "tool_calls" not in step:
                raise ValueError("Each step must have 'description' and 'tool_calls'")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing plan: {e}")
        plan = [{
            "description": "Fallback: Query MCP server",
            "tool_calls": [{"method": "list_resources", "params": {}}]
        }]
    return {
        "plan": plan,
        "results": [],
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens
    }

# Analysis and Summary Prompts
analysis_prompt = PromptTemplate(
    input_variables=["task"],
    template="""
    
    You are an expert AI assistant specializing in DeFi analysis, tasked with analyze the results of all steps for the DeFi tasks '{task}' and the results in <result></result> XML tags. Multiple tags indicate multiple tool calls, each to be processed separately, containing DeFi metrics like protocol names, TVL, transaction volumes, yield rates, or other KPIs.
    Instructions:
    Parse data in <result></result>.

    Analyze DeFi metrics (e.g., token prices, wallet balances, TVL in USD, transaction volumes, APYs, market data) in <analysis></analysis> tag, covering all tokens, wallets, orders, protocols, and relevant metrics.

    Provide reasoning for the analysis in <reasoning></reasoning>, justifying highlights based on <analysis></analysis>.

    Give insights about market trends, key insights, protocol comparisons, and DeFi trends/risks/opportunities in <insights></insights> with clear sections or bullet points.

    Note ambiguous or incomplete data with best-effort interpretation.
    
    Lastly, confirm if each analysis is successful or not using <success></success> tag. Inside the tag, you should provide true if the analysis is successful, false otherwise.
    
    If you find the relationships of data between the steps, explicitly mention it in details in <analysis></analysis> tag.
    
    Your result should be put in <analysis_result></analysis_result> tag.

    Example of the results:
    Eg:
    <result>
        {{
            "step_description": "Step 1",
            "tool_responses": [
                {{"method": "list_resources", "params": {{}}, "isError": false, "response": "Successfully retrieved resources from the MCP server."}}
            ]
        }}
    </result>
    <result>
        {{
            "step_description": "Step 2",
            "tool_responses": [
                {{"method": "process_data", "params": {{"data": "value"}}, "isError": false, "response": "Successfully processed data."}}
            ]
        }}
    </result>
    
    Example of the analysis:
    <analysis_result>
    <analysis>
    your analysis result
    </analysis>
    <success>
    true
    </success>
    <reasoning>
    your reasoning
    </reasoning>
    <insights>
    your insights
    </insights>
    </analysis_result>
    """
)

summary_prompt = PromptTemplate(
    input_variables=["task", "analysis"],
    template="""Summarize the results of all steps for the task '{task}':
             Analysis: {analysis}
             Provide a concise summary in JSON format with a single 'summary' field (string). The summary should be within 5-10 sentences, 2-3 paragraphs.
             
             The raw results will also be provided in the <result></result> tags. If you find the relationships of data between the steps and in the analysis, which is put in the <analysis_result></analysis_result> tag, explicitly mention it in details with focus on figures, numbers, etc.
            """
)

# Solver: Analyze and summarize all worker results using litellm
async def solver(state: ReWOOState) -> ReWOOState:
    results: list[str] = state["results"]
    
    # Generate analysis
    analysis_input = analysis_prompt.format(task=state["task"])
    messages = [{"role": "user", "content": analysis_input}]
    messages.extend([{"role": "user", "content": result} for result in results])
    analysis_response = await acompletion(
        model=os.getenv("LLM_MODEL"),
        messages=messages,
        api_key=os.getenv("API_KEY")
    )
    analysis_cost = completion_cost(analysis_response)
    analysis_token_data = analysis_response.usage if analysis_response.usage else {}
    
    # Generate summary
    summary_input = summary_prompt.format(task=state["task"], analysis=analysis_response.choices[0].message.content)
    summary_messages = [{"role": "user", "content": summary_input}]
    summary_messages.extend([{"role": "user", "content": result} for result in results])
    summary_response = await acompletion(
        model=os.getenv("LLM_MODEL"),
        messages=summary_messages,
        api_key=os.getenv("API_KEY")
    )
    summary_cost = completion_cost(summary_response)
    summary_token_data = summary_response.usage if summary_response.usage else {}
    
    # Update metrics
    total_cost = state.get("total_cost", 0.0) + analysis_cost + summary_cost
    total_tokens = state.get("total_tokens", 0) + analysis_token_data.get('total_tokens', 0) + summary_token_data.get('total_tokens', 0)
    prompt_tokens = state.get("prompt_tokens", 0) + analysis_token_data.get('prompt_tokens', 0) + summary_token_data.get('prompt_tokens', 0)
    completion_tokens = state.get("completion_tokens", 0) + analysis_token_data.get('completion_tokens', 0) + summary_token_data.get('completion_tokens', 0)
    cache_read_tokens = state.get("cache_read_tokens", 0) + analysis_token_data.get('cache_read_tokens', 0) + summary_token_data.get('cache_read_tokens', 0)
    cache_write_tokens = state.get("cache_write_tokens", 0) + analysis_token_data.get('cache_write_tokens', 0) + summary_token_data.get('cache_write_tokens', 0)
    
    # Append analysis and summary to results
    final_results = results + [{"analysis": analysis_response.choices[0].message.content, "summary": summary_response.choices[0].message.content}]
    return {
        "results": final_results,
        "total_cost": total_cost,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens
    }

# Define the LangGraph workflow
workflow = StateGraph(ReWOOState)
workflow.add_node("initialize_mcp", initialize_mcp_client)
workflow.add_node("planner", planner)
workflow.add_node("worker", worker)
workflow.add_node("solver", solver)
workflow.add_edge("initialize_mcp", "planner")
workflow.add_edge("planner", "worker")
workflow.add_edge("worker", "solver")
workflow.add_edge("solver", END)
workflow.add_edge(START, "initialize_mcp")
graph = workflow.compile()

# Main function to run the agent
async def run_rewoo_agent(task: str):
    initial_state = ReWOOState(
        task=task,
        plan=[],
        results=[],
        mcp_tools={},
        total_cost=0.0,
        total_tokens=0,
        prompt_tokens=0,
        completion_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0
    )
    final_state = None
    print("Starting streaming workflow...")
    async for chunk in graph.astream(initial_state):
        print("\n--- Stream Chunk ---")
        print(chunk)
        final_state = chunk if chunk else final_state
    
    print("\n--- Streaming Complete ---")
    if final_state:
        print(f"Total Completion Cost: ${final_state.get('total_cost', 0.0):.6f}")
        print(f"Total Tokens: {final_state.get('total_tokens', 0)}")
        print(f"Prompt Tokens: {final_state.get('prompt_tokens', 0)}")
        print(f"Completion Tokens: {final_state.get('completion_tokens', 0)}")
        print(f"Cache Read Tokens: {final_state.get('cache_read_tokens', 0)}")
        print(f"Cache Write Tokens: {final_state.get('cache_write_tokens', 0)}")
    return final_state

# Example usage
if __name__ == "__main__":
    task = """
    You are a Perpetual Whales Agent agent who is an expert analyst specializing in detecting whale trading patterns with years of experience understanding deeply crypto trading behavior, on-chain metrics, and derivatives markets, you have developed a keen understanding of whale trading strategies.

    You can identify patterns in whale positions, analyze their portfolio changes over time, and evaluate the potential reasons behind their trading decisions. Your analysis helps traders decide whether to follow whale trading moves or not.

    Here will be your task, please do it from step by step, one task is done you will able to move to next task. DO NOT use liquidity heatmap tool, function for analyzing:

    - Fetching every whales on some markets
    - Find trading patterns and strategies identified based on latest whales activity, histocial trading pnl
    - Risk assessment of all current positions
    - Analyze market trend based on 30 days of tokens
    - Define short-term trades as many as possible that can be executed with safety scoring and entries, stop loss, take profit, concise description, bias including short-term or long-term trades. The entries should be closest to latest price, stop loss and take profit should be realistic which is not too far from entry.
    
    Identify and extract key DeFi metrics from each tool call result, such as:
    - Protocol or platform names
    - Total value locked (TVL) in USD
    - Transaction volumes or counts
    - Yield rates or APYs
    - Token prices or market data
    - Other relevant DeFi-specific metrics

    Summarize your final report as detailed as possible. Make it from 5 to 10 paragraphs. Remember to be very specific and precise about the metrics and numbers.
    """
    final_state = asyncio.run(run_rewoo_agent(task))
    print("Final State:")
    print(json.dumps(final_state, indent=2, default=lambda o: '<not serializable>'))