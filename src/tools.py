from pathlib import Path
import json
import sys
import argparse
from typing import Any
import asyncio

from mcp.server import Server, InitializationOptions
from mcp.types import Tool, TextContent, ServerCapabilities
import mcp.server.stdio

from langchain_community.retrievers import BM25Retriever
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ddgs import DDGS
import ujson


class ToolManager:
    """Manages both retriever and search tools"""
    
    def __init__(self, wiki_path: str):
        self.wiki_docs = self.load_wiki(wiki_path)
        self.retriever = self._setup_retriever()
    
    def load_wiki(self, path: str):
        """Load Wikipedia documents from JSONL file"""
        wiki = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_id, line in enumerate(f):
                line = line.strip()
                if not line.startswith('{'):
                    continue
                try:
                    wiki.append(ujson.loads(line))
                except ujson.JSONDecodeError:
                    continue
                if line_id == 500000:
                    break
        print(f"Loaded {len(wiki)} records", file=sys.stderr)
        return wiki
    
    def _setup_retriever(self):
        """Initialize BM25 retriever from loaded docs"""
        source_docs = [
            Document(page_content=doc["contents"], metadata={"id": doc["id"]})
            for doc in self.wiki_docs
        ]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        
        wiki_docs_split = text_splitter.split_documents(source_docs)
        return BM25Retriever.from_documents(wiki_docs_split, k=5)
    
    def retrieve(self, query: str) -> str:
        """Retrieve relevant documents using BM25"""
        docs = self.retriever.invoke(query)
        result = "\nRetrieved Wikipedia documents:\n" + "".join([
            f"\n\nWikipedia document {str(i)}\n" + doc.page_content
            for i, doc in enumerate(docs)
        ])
        return result
    
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



def setup_server(wiki_path: str):
    """Create and configure MCP server"""
    server = Server("search-tools")
    tool_manager = ToolManager(wiki_path)
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="retriever",
                description="Uses semantic search to retrieve relevant articles from Wikipedia.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to perform. This should be a query related to answering the question. Use the affirmative form rather than a question.",
                        }
                    },
                    "required": ["query"],
                },
            ),
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
            if name == "retriever":
                query = arguments.get("query", "")
                if not query:
                    return [TextContent(type="text", text="Error: query parameter required")]
                result = tool_manager.retrieve(query)
                return [TextContent(type="text", text=result)]
            
            elif name == "search":
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
    parser = argparse.ArgumentParser(description="Party Planning Tools MCP Server")
    script_dir = Path(__file__).parent.parent
    #print('here', script_dir)
    #exit()
    parser.add_argument(
        "--wiki-path",
        default=f"{script_dir}/data/wiki/wiki-18.jsonl",
        help="Path to Wikipedia JSONL file"
    )
    args = parser.parse_args()
    
    server = setup_server(args.wiki_path)
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        options = InitializationOptions(
            server_name="search-tools",
            server_version="1.0.0",
            capabilities=ServerCapabilities()
        )
        #await server.run(read_stream, write_stream, {
        #    "server_name": "search-tools",
        #    "server_version": "1.0.0"
        #})
        await server.run(read_stream, write_stream, options)


if __name__ == "__main__":
    asyncio.run(main())