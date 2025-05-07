import os
import json
from typing import TypedDict, List, Dict, Any
import asyncio
from langgraph.graph import StateGraph
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, TextContent, ImageContent, EmbeddedResource, TextResourceContents, BlobResourceContents
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START
from litellm import completion_cost

# MCP Client Configuration
MCP_SERVER_URLS = ["http://localhost:4000/sse"]  # Replace with your MCP server URLs

# Define the ReWOO state
class ReWOOState(TypedDict):
    task: str
    plan: List[Dict[str, Any]]  # Each step contains a list of tool calls
    results: List[Dict[str, Any]]  # Includes tool responses for all steps
    mcp_tools: Dict[str, Any]  # Renamed from mcp_capabilities to match your code

# Initialize the Anthropic model
llm = ChatAnthropic(model_name=os.getenv("LLM_MODEL"), api_key=os.getenv("API_KEY"))

# MCP Client: Initialize MultiServerMCPClient
async def initialize_mcp_client(state: ReWOOState) -> ReWOOState:
    try:
        async with MultiServerMCPClient({'oraichain': {
            # make sure you start your weather server on port 8000
            "url": MCP_SERVER_URLS[0],
            "transport": "sse",
        }}) as mcp_client:
            tools = mcp_client.get_tools()
            if tools:
                return {"mcp_tools": tools}
            raise Exception("Failed to retrieve MCP tools")
    except Exception as e:
        print(f"Error initializing MCP client: {e}")
        return {"mcp_tools": {}}

# MCP Client: Execute a tool call via SSE
async def call_tool(tool_name: str, args: Dict) -> CallToolResult:
    """Call a tool on the MCP server with automatic reconnection on failure."""
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

    # Loop through all steps in the plan
    for step in plan:
        step_results = {
            "step_description": step["description"],
            "tool_responses": []
        }

        # Execute each tool call in the step
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

    return {"results": results}

# Planner: Generate a structured plan
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
    print("task: ", state["task"])
    prompt = planner_prompt.format(task=state["task"], tools=json.dumps([tool.name for tool in tools]))
    agent_with_tools = llm.bind_tools(tools)
    response = await agent_with_tools.ainvoke(prompt)
    try:
        plan = json.loads(response.content)
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
    return {"plan": plan, "results": []}

# Analysis and Summary Prompt
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

# Solver: Analyze and summarize all worker results
async def solver(state: ReWOOState) -> ReWOOState:
    results = state["results"]
    results_str = json.dumps(results)

    # Generate analysis for all steps
    analysis_input = analysis_prompt.format(
        task=state["task"],
        results=results_str
    )
    analysis_response = await llm.ainvoke(analysis_input)
    try:
        analysis = json.loads(analysis_response.content)
    except json.JSONDecodeError:
        analysis = {
            "success": False,
            "details": "Failed to parse analysis response",
            "insights": "Unable to derive insights due to parsing error"
        }

    # Generate summary for all steps
    summary_input = summary_prompt.format(
        task=state["task"],
        results=results_str,
        analysis=json.dumps(analysis)
    )
    summary_response = await llm.ainvoke(summary_input)
    try:
        summary = json.loads(summary_response.content)
    except json.JSONDecodeError:
        summary = {"summary": "Failed to generate summary due to parsing error"}

    # Append analysis and summary to results
    final_results = results + [{"analysis": analysis, "summary": summary}]
    return {
        "results": final_results
    }

# Define the LangGraph workflow
workflow = StateGraph(ReWOOState)

# Add nodes
workflow.add_node("initialize_mcp", initialize_mcp_client)
workflow.add_node("planner", planner)
workflow.add_node("worker", worker)
workflow.add_node("solver", solver)

# Define edges
workflow.add_edge("initialize_mcp", "planner")
workflow.add_edge("planner", "worker")
workflow.add_edge("worker", "solver")
workflow.add_edge("solver", END)
workflow.add_edge(START, "initialize_mcp")

# Compile the graph
graph = workflow.compile()

# Main function to run the agent
async def run_rewoo_agent(task: str):
    initial_state = ReWOOState(
        task=task,
        plan=[],
        results=[],
        mcp_tools={}
    )
    final_state = await graph.ainvoke(initial_state)
    return final_state

# Example usage
if __name__ == "__main__":
    import asyncio
    task = "What's the ORAI balance of these wallets: orai1f6q9wjn8qp3ll8y8ztd8290vtec2yxyx0wnd0d, orai179dea42h80arp69zd779zcav5jp0kv04zx4h09, orai1f7wcl8drgvyvhzylu54gphul0st2x87kdn6g6k, orai1unpv9tsw7d27n7wym83z4lajh4pt252jsvwgvf, orai1qv5jn7tueeqw7xqdn5rem7s09n7zletrsnc5vq, orai12ru3276mkzuuay6vhmg3t6z9hpvrsnpljq7v75, orai1azu0pge4yx6j6sd0tn8nz4x9vj7l9kga8y3arf"
    final_state = asyncio.run(run_rewoo_agent(task))
    print("Final State:")
    print(json.dumps(final_state, indent=2, default=lambda o: '<not serializable>'))
    
    