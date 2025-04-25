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

#  name='coingecko_top_tokens' description='Fetch tokens by market cap from CoinGecko with pagination support.\n\n  This tool fetches detailed information about cryptocurrencies including:\n  - Market cap rank\n  - Symbol\n  - Name\n  - Current price in USD\n  - Market capitalization\n  - 24h trading volume\n  - 24h price change percentage\n  - 7d price change percentage\n\n  Parameters:\n  - page: Page number to fetch (default: 1)\n  - perPage: Number of tokens per page (default: 100, max: 100)\n\n  Returns data in JSON format with token information sorted by market cap.\n  ' inputSchema={'type': 'object', 'properties': {'page': {'type': 'number'}, 'perPage': {'type': 'number'}}, 'required': ['page', 'perPage'], 'additionalProperties': False, '$schema': 'http://json-schema.org/draft-07/schema#'}
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
                tasks = [
                    session.call_tool("coingecko_top_tokens", arguments={"page": 1, "perPage": 10}),
                    session.call_tool("coingecko_top_tokens", arguments={"page": 2, "perPage": 10}),
                    session.call_tool("coingecko_top_tokens", arguments={"page": 3, "perPage": 10}),
                ]*30
                results = await asyncio.gather(*tasks)
                for result in results:
                    print(result)
                print("="*100)
                # result = await session.call_tool("perpetual_whales_get_whale_detail", arguments={"protocol": "GMX_V2", "address": "0xdB16BB1E9208c46fa0cD1d64FD290D017958f476"})
                # print(result)
    except Exception as e:
        print(f"Error connecting to server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main(server_url = "http://15.235.225.246:8083/sse"))