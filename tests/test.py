"""MCP HTTP client example using MCP SDK."""

import asyncio
import sys
from typing import Any
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

# FunctionCall(id='call_cWjW3T2R09f1VSGg6OJt9v8I', arguments='{\"protocol\": \"GMX_V2\", \"address\": \"0xecb6...2b00\"}', name='perpetual_whales_get_whale_detail'),

async def main(server_url: str):
    """Connect to MCP server and list its capabilities.

    Args:
        server_url: Full URL to SSE endpoint (e.g. http://localhost:8000/sse)
    """
    if urlparse(server_url).scheme not in ("http", "https"):
        print("Error: Server URL must start with http:// or https://")
        sys.exit(1)

    try:
        async with sse_client(server_url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                print("="*100)
                print("Connected to MCP server at", server_url)
                print("="*100)
                print_items("tools", (await session.list_tools()))
                print("="*100)
                result = await session.call_tool("perpetual_whales_get_token_price_history", arguments={"exchange": "Binance", "symbol": "ETHUSDT,BTCUSDT", "type": "spot", "period": "1d", "limit": 30})
                print(result)
                print("="*100)
                result = await session.call_tool("perpetual_whales_get_whale_detail", arguments={"protocol": "GMX_V2", "address": "0xdB16BB1E9208c46fa0cD1d64FD290D017958f476"})
                print(result)
    except Exception as e:
        print(f"Error connecting to server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main(server_url = "http://15.235.225.246:4010/sse"))