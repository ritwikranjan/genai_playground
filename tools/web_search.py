#!/usr/bin/env python
"""
Web Search MCP Tool

This tool provides web search capabilities using DuckDuckGo.
It runs as an MCP server and can be used by the playground CLI.
"""

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from duckduckgo_search import DDGS


# Create the MCP server
server = Server("web-search")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search_web",
            description="Search the web using DuckDuckGo. Returns relevant web results for the given query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_news",
            description="Search for recent news articles using DuckDuckGo News.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The news topic to search for"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of news articles to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "search_web":
        return await search_web(arguments)
    elif name == "search_news":
        return await search_news(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def search_web(arguments: dict[str, Any]) -> list[TextContent]:
    """Perform a web search using DuckDuckGo."""
    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 5)
    
    if not query:
        return [TextContent(type="text", text="Error: No search query provided")]
    
    try:
        # Run the synchronous DDGS in a thread pool
        def do_search():
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                return results
        
        results = await asyncio.get_event_loop().run_in_executor(None, do_search)
        
        if not results:
            return [TextContent(type="text", text=f"No results found for: {query}")]
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append({
                "rank": i,
                "title": result.get("title", ""),
                "url": result.get("href", ""),
                "snippet": result.get("body", "")
            })
        
        return [TextContent(
            type="text",
            text=json.dumps({"query": query, "results": formatted_results}, indent=2)
        )]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Search error: {str(e)}")]


async def search_news(arguments: dict[str, Any]) -> list[TextContent]:
    """Search for news articles using DuckDuckGo News."""
    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 5)
    
    if not query:
        return [TextContent(type="text", text="Error: No search query provided")]
    
    try:
        # Run the synchronous DDGS in a thread pool
        def do_search():
            with DDGS() as ddgs:
                results = list(ddgs.news(query, max_results=max_results))
                return results
        
        results = await asyncio.get_event_loop().run_in_executor(None, do_search)
        
        if not results:
            return [TextContent(type="text", text=f"No news found for: {query}")]
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append({
                "rank": i,
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "source": result.get("source", ""),
                "date": result.get("date", ""),
                "snippet": result.get("body", "")
            })
        
        return [TextContent(
            type="text",
            text=json.dumps({"query": query, "news_results": formatted_results}, indent=2)
        )]
        
    except Exception as e:
        return [TextContent(type="text", text=f"News search error: {str(e)}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
