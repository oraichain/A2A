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
MCP_SERVER_URLS = ["http://localhost:4000/sse"]

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
        async with MultiServerMCPClient({'oraichain': {
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
    results = []
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
        results.append(step_results)
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
    template="""Create a step-by-step plan in JSON format to accomplish the task: {task}

Available MCP tools: {tools}

Each step must be a JSON object with a 'description' (string) and a 'tool_calls' array.
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
    input_variables=["task", "results"],
    template="""Analyze the results of all steps for the task '{task}':
             Results: {results}
             Provide a detailed analysis in JSON format with the following fields:
             - success: boolean (true if all tool calls succeeded, false if any failed)
             - details: string (explain the outcomes across all steps, including successes and errors)
             - insights: string (key takeaways or implications for the task)
             Example: 
             {{
                 "success": true,
                 "details": "All tools executed successfully across steps, retrieved and processed resources.",
                 "insights": "Resource data is ready for further analysis or storage."
             }}
             Return only the JSON response, nothing else.
            """
)

summary_prompt = PromptTemplate(
    input_variables=["task", "results", "analysis"],
    template="""Summarize the results of all steps for the task '{task}':
             Results: {results}
             Analysis: {analysis}
             Provide a concise summary in JSON format with a single 'summary' field (string).
             Example: {{
                 'summary': 'Successfully retrieved and processed resource data from the MCP server.'
             }}
             Return only the JSON response, nothing else.
            """
)

# Solver: Analyze and summarize all worker results using litellm
async def solver(state: ReWOOState) -> ReWOOState:
    results = state["results"]
    results_str = json.dumps(results)
    
    # Generate analysis
    analysis_input = analysis_prompt.format(task=state["task"], results=results_str)
    analysis_response = await acompletion(
        model=os.getenv("LLM_MODEL"),
        messages=[{"role": "user", "content": analysis_input}],
        api_key=os.getenv("API_KEY")
    )
    analysis_cost = completion_cost(analysis_response)
    analysis_token_data = analysis_response.usage if analysis_response.usage else {}
    
    try:
        analysis = json.loads(analysis_response.choices[0].message.content)
    except json.JSONDecodeError:
        analysis = {
            "success": False,
            "details": "Failed to parse analysis response",
            "insights": "Unable to derive insights due to parsing error"
        }
    
    # Generate summary
    summary_input = summary_prompt.format(task=state["task"], results=results_str, analysis=json.dumps(analysis))
    summary_response = await acompletion(
        model=os.getenv("LLM_MODEL"),
        messages=[{"role": "user", "content": summary_input}],
        api_key=os.getenv("API_KEY")
    )
    summary_cost = completion_cost(summary_response)
    summary_token_data = summary_response.usage if summary_response.usage else {}
    
    try:
        summary = json.loads(summary_response.choices[0].message.content)
    except json.JSONDecodeError:
        summary = {"summary": "Failed to generate summary due to parsing error"}
    
    # Update metrics
    total_cost = state.get("total_cost", 0.0) + analysis_cost + summary_cost
    total_tokens = state.get("total_tokens", 0) + analysis_token_data.get('total_tokens', 0) + summary_token_data.get('total_tokens', 0)
    prompt_tokens = state.get("prompt_tokens", 0) + analysis_token_data.get('prompt_tokens', 0) + summary_token_data.get('prompt_tokens', 0)
    completion_tokens = state.get("completion_tokens", 0) + analysis_token_data.get('completion_tokens', 0) + summary_token_data.get('completion_tokens', 0)
    cache_read_tokens = state.get("cache_read_tokens", 0) + analysis_token_data.get('cache_read_tokens', 0) + summary_token_data.get('cache_read_tokens', 0)
    cache_write_tokens = state.get("cache_write_tokens", 0) + analysis_token_data.get('cache_write_tokens', 0) + summary_token_data.get('cache_write_tokens', 0)
    
    # Append analysis and summary to results
    final_results = results + [{"analysis": analysis, "summary": summary}]
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
    task = "What's the ORAI balance of these wallets: orai1f6q9wjn8qp3ll8y8ztd8290vtec2yxyx0wnd0d, orai179dea42h80arp69zd779zcav5jp0kv04zx4h09, orai1f7wcl8drgvyvhzylu54gphul0st2x87kdn6g6k, orai1unpv9tsw7d27n7wym83z4lajh4pt252jsvwgvf, orai1qv5jn7tueeqw7xqdn5rem7s09n7zletrsnc5vq, orai12ru3276mkzuuay6vhmg3t6z9hpvrsnpljq7v75, orai1azu0pge4yx6j6sd0tn8nz4x9vj7l9kga8y3arf"
    final_state = asyncio.run(run_rewoo_agent(task))
    print("Final State:")
    print(json.dumps(final_state, indent=2, default=lambda o: '<not serializable>'))