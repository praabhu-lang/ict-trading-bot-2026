import asyncio
from fastmcp import Client

async def test_mcp():
    # Connect to the local HTTP MCP server
    async with Client("http://127.0.0.1:8000/mcp") as client:
        print("[TEST] Calling check_positions_tool via MCP...")
        result = await client.call_tool("check_positions_tool", {})
        print(f"[TEST RESULT]: {result}")

if __name__ == "__main__":
    asyncio.run(test_mcp())