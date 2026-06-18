"""
Convert MCP tool definitions to LLM (Qwen) tool schema format.

This eliminates manual duplication of tool schemas.
Tools are defined once in MCP server, fetched dynamically by agent.
"""

from typing import List, Dict, Any
from mcp.client.stdio import stdio_client


def mcp_tool_to_llm_schema(mcp_tool) -> Dict[str, Any]:
    """
    Convert MCP Tool object to LLM tool schema format.
    
    MCP Tool format:
        Tool(
            name="search",
            description="...",
            inputSchema={...}
        )
    
    LLM format (Qwen, Claude, etc.):
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "...",
                "parameters": {...}
            }
        }
    """
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema,
        }
    }


async def fetch_tools_from_mcp(mcp_session) -> List[Dict[str, Any]]:
    """
    Fetch all tools from MCP server and convert to LLM schema format.
    
    This is the single source of truth - no manual schema duplication!
    
    Args:
        mcp_session: Active MCP ClientSession
    
    Returns:
        List of tool schemas ready for tokenizer.apply_chat_template()
    """
    # Get all tools from MCP server
    mcp_tools_response = await mcp_session.list_tools()
    #print(mcp_tools_response)
    #exit()
    
    # Convert each tool to LLM format
    llm_tools = [
        mcp_tool_to_llm_schema(tool)
        for tool in mcp_tools_response.tools
    ]
    
    return llm_tools


def print_tool_schemas(tools: List[Dict[str, Any]]):
    """Pretty print tool schemas for debugging"""
    import json
    for tool in tools:
        print(json.dumps(tool, indent=2))
        print("\n" + "="*60 + "\n")




async def main():
    server_params = StdioServerParameters(
        command="python3",
        args=["./tools.py"]
    )
    
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize with no arguments
            await session.initialize()
            print("Connected to MCP server!")
            
            #tools = await session.list_tools()
            #print(f"Available tools: {[t.name for t in tools.tools]}")
            tools = await fetch_tools_from_mcp(session)
            print_tool_schemas(tools)

if __name__ == '__main__':
    import asyncio
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp.client.session import ClientSession
    from mcp.types import ClientCapabilities, Implementation
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp.client.session import ClientSession
    #from mcp.types import ClientInfo, Implementation
    asyncio.run(main())
    