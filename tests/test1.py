"""MCP HTTP client example using MCP SDK."""

import asyncio
import sys
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client


def print_items(name: str, result: Any) -> None:
    """Print items with formatting.

    Args:
        name: Category name (tools/resources/prompts)
        result: Result object containing items list
    """
    print("", f"Available {name}:", sep="\n")
    items = getattr(result, name)
    if items:
        for item in items:
            print(" *", item)
    else:
        print("No items available")


async def run_tool_with_session(server_url: str, tool_name: str, arguments: Dict[str, Any]) -> Tuple[str, Any]:
    """Create a session and run a single tool call.
    
    Args:
        server_url: The server URL
        tool_name: The name of the tool to call
        arguments: Arguments for the tool call
        
    Returns:
        Tuple of (tool_name, result)
    """
    try:
        async with sse_client(server_url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                print(f"Session initialized for {tool_name}")
                result = await session.call_tool(tool_name, arguments=arguments)
                return (tool_name, result)
    except Exception as e:
        return (tool_name, f"Error: {e}")


async def main(server_url: str):
    """Connect to MCP server, list capabilities, and run concurrent tool calls.

    Args:
        server_url: Full URL to SSE endpoint (e.g. http://localhost:8000/sse)
    """
    if urlparse(server_url).scheme not in ("http", "https"):
        print("Error: Server URL must start with http:// or https://")
        sys.exit(1)

    try:
        # First connect to get available tools
        print("Connecting to server to get available tools...")
        async with sse_client(server_url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                # print("="*100)
                # print("Connected to MCP server at", server_url)
                # print("="*100)
                # print_items("tools", (await session.list_tools()))
                # print("="*100)
        
        # Define tool calls
        # tool_calls = [
        #     ("perpetual_whales_get_token_price_history", {"exchange": "Binance", "symbol": "ETHUSDT", "type": "spot", "period": "1d", "limit": 30}),
        #     ("perpetual_whales_get_token_price_history", {"exchange": "Binance", "symbol": "BTCUSDT", "type": "spot", "period": "1d", "limit": 30}),
        #     ("perpetual_whales_get_token_price_history", {"exchange": "Binance", "symbol": "SOLUSDT", "type": "spot", "period": "1d", "limit": 30}),
        #     ("perpetual_whales_get_token_price_history", {"exchange": "Binance", "symbol": "BNBUSDT", "type": "spot", "period": "1d", "limit": 30}),
        #     ("perpetual_whales_get_token_price_history", {"exchange": "Binance", "symbol": "ADAUSDT", "type": "spot", "period": "1d", "limit": 30}),
        #     ("perpetual_whales_get_whale_detail", {"protocol": "GMX_V2", "address": "0xdB16BB1E9208c46fa0cD1d64FD290D017958f476"}),
        #     ("perpetual_whales_get_whale_detail", {"protocol": "GMX_V2", "address": "0xecb6e5E2658106e9242bF9704C9eF4398d2dd938"}),
        #     ("perpetual_whales_get_whale_detail", {"protocol": "GMX_V2", "address": "0x6B75d8AF000000e20B7a7DDf000Ba900b4009A80"}),
        #     ("perpetual_whales_get_whale_detail", {"protocol": "GMX_V2", "address": "0x3E8aef5934908770B11346A4A08b1703E5Ae51E8"}),
        #     ("perpetual_whales_get_whale_detail", {"protocol": "GMX_V2", "address": "0x7B9b4930946dB8E4d123D1aFc5c2Ab8a066F70C4"})
        # ]
        
        tool_calls = [
            ("coingecko_top_tokens", {"page": 1, "perPage": 10})
        ] * 100
        
        # Run 10 tool calls concurrently, each with its own session
        print(f"Running {len(tool_calls)} concurrent tool calls with separate sessions...")
        start_time = asyncio.get_event_loop().time()
        
        results = await asyncio.gather(*[
            run_tool_with_session(server_url, tool_name, arguments) 
            for tool_name, arguments in tool_calls
        ])
        
        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time
        
        # Print results
        for i, (tool_name, result) in enumerate(results):
            print("="*100)
            print(f"Result {i+1} ({tool_name}):")
            print(result)
        
        print("="*100)
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Average time per call: {total_time/len(tool_calls):.2f} seconds")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main(server_url = "http://15.235.225.246:8083/sse"))