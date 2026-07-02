from pathlib import Path
import json
import sys
import argparse
from typing import Any
import asyncio

from mcp.server import Server, InitializationOptions
from mcp.types import Tool, TextContent, ServerCapabilities
import mcp.server.stdio

from ddgs import DDGS
import ujson


class ToolManager:
    """Manages search tools"""
    
    def __init__(self):
        pass
    
    def search(self, query_list: list) -> str:
        """Perform DuckDuckGo search"""
        results = []
        
        # Handle both list and single string input
        if isinstance(query_list, str):
            query_list = [query_list]
        
        for query in query_list:
            try:
                with DDGS() as ddgs:
                    hits = list(ddgs.text(query, safesearch="moderate", max_results=5))
                
                if hits:
                    results.append(f"### Query: {query}")
                    for i, h in enumerate(hits):
                        results.append(f"{i+1}. {h['title']} - {h['body']} ({h['href']})")
                else:
                    results.append(f"### Query: {query}\nNo results found.")
            except Exception as e:
                results.append(f"### Query: {query}\nError: {str(e)}")
        
        return "\n".join(results) if results else "No results found."



def setup_server():
    """Create and configure MCP server"""
    server = Server("search-tools")
    tool_manager = ToolManager()
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="search",
                description="DuckDuckGo web search. Use it when you need external knowledge.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query_list": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "One or more fully-formed semantic search queries.",
                        }
                    },
                    "required": ["query_list"],
                },
            ),
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            
            if name == "search":
                query_list = arguments.get("query_list", [])
                if not query_list:
                    return [TextContent(type="text", text="Error: query_list parameter required")]
                result = tool_manager.search(query_list)
                return [TextContent(type="text", text=result)]
            
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
        
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    return server


async def main():
    
    server = setup_server()
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        options = InitializationOptions(
            server_name="search-tools",
            server_version="1.0.0",
            capabilities=ServerCapabilities()
        )
        
        await server.run(read_stream, write_stream, options)


if __name__ == "__main__":
    asyncio.run(main())