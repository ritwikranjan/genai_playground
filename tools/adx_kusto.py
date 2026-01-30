#!/usr/bin/env python
"""
Azure Data Explorer (ADX/Kusto) MCP Tool

This tool provides Azure Data Explorer query capabilities.
It runs as an MCP server and can be used by the playground CLI.

This tool wraps the Azure Kusto SDK to provide:
- List clusters
- List databases
- List tables
- Get table schema
- Execute KQL queries
- Sample data from tables

Authentication is handled via Azure Identity (DefaultAzureCredential).
"""

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from azure.identity import DefaultAzureCredential
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import KustoServiceError

# Create the MCP server
server = Server("adx-kusto")


def get_kusto_client(cluster_uri: str) -> KustoClient:
    """Create a Kusto client with Azure Identity authentication."""
    credential = DefaultAzureCredential()
    kcsb = KustoConnectionStringBuilder.with_azure_token_credential(
        cluster_uri, credential
    )
    return KustoClient(kcsb)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available ADX tools."""
    return [
        Tool(
            name="adx_list_databases",
            description="List all databases in an Azure Data Explorer cluster.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_uri": {
                        "type": "string",
                        "description": "The URI of the Azure Data Explorer cluster (e.g., https://mycluster.westus.kusto.windows.net)"
                    }
                },
                "required": ["cluster_uri"]
            }
        ),
        Tool(
            name="adx_list_tables",
            description="List all tables in a specific Azure Data Explorer database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_uri": {
                        "type": "string",
                        "description": "The URI of the Azure Data Explorer cluster"
                    },
                    "database": {
                        "type": "string",
                        "description": "The name of the database"
                    }
                },
                "required": ["cluster_uri", "database"]
            }
        ),
        Tool(
            name="adx_get_table_schema",
            description="Get the schema (columns and types) of a specific table in Azure Data Explorer.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_uri": {
                        "type": "string",
                        "description": "The URI of the Azure Data Explorer cluster"
                    },
                    "database": {
                        "type": "string",
                        "description": "The name of the database"
                    },
                    "table": {
                        "type": "string",
                        "description": "The name of the table"
                    }
                },
                "required": ["cluster_uri", "database", "table"]
            }
        ),
        Tool(
            name="adx_sample_data",
            description="Get a sample of data from a table in Azure Data Explorer.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_uri": {
                        "type": "string",
                        "description": "The URI of the Azure Data Explorer cluster"
                    },
                    "database": {
                        "type": "string",
                        "description": "The name of the database"
                    },
                    "table": {
                        "type": "string",
                        "description": "The name of the table"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of rows to return (default: 10)",
                        "default": 10
                    }
                },
                "required": ["cluster_uri", "database", "table"]
            }
        ),
        Tool(
            name="adx_query",
            description="Execute a KQL (Kusto Query Language) query against an Azure Data Explorer database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cluster_uri": {
                        "type": "string",
                        "description": "The URI of the Azure Data Explorer cluster"
                    },
                    "database": {
                        "type": "string",
                        "description": "The name of the database to query"
                    },
                    "query": {
                        "type": "string",
                        "description": "The KQL query to execute"
                    }
                },
                "required": ["cluster_uri", "database", "query"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "adx_list_databases":
        return await adx_list_databases(arguments)
    elif name == "adx_list_tables":
        return await adx_list_tables(arguments)
    elif name == "adx_get_table_schema":
        return await adx_get_table_schema(arguments)
    elif name == "adx_sample_data":
        return await adx_sample_data(arguments)
    elif name == "adx_query":
        return await adx_query(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


def execute_query_sync(cluster_uri: str, database: str, query: str) -> list[dict]:
    """Execute a KQL query synchronously and return results as a list of dicts."""
    client = get_kusto_client(cluster_uri)
    response = client.execute(database, query)
    
    results = []
    for row in response.primary_results[0]:
        row_dict = {}
        for i, col in enumerate(response.primary_results[0].columns):
            row_dict[col.column_name] = row[i]
        results.append(row_dict)
    
    return results


async def adx_list_databases(arguments: dict[str, Any]) -> list[TextContent]:
    """List all databases in a cluster."""
    cluster_uri = arguments.get("cluster_uri", "")
    
    if not cluster_uri:
        return [TextContent(type="text", text="Error: cluster_uri is required")]
    
    try:
        def do_query():
            return execute_query_sync(cluster_uri, "", ".show databases")
        
        results = await asyncio.get_event_loop().run_in_executor(None, do_query)
        
        databases = [r.get("DatabaseName", r.get("Name", "")) for r in results]
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "cluster_uri": cluster_uri,
                "databases": databases,
                "count": len(databases)
            }, indent=2)
        )]
        
    except KustoServiceError as e:
        return [TextContent(type="text", text=f"Kusto error: {str(e)}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def adx_list_tables(arguments: dict[str, Any]) -> list[TextContent]:
    """List all tables in a database."""
    cluster_uri = arguments.get("cluster_uri", "")
    database = arguments.get("database", "")
    
    if not cluster_uri:
        return [TextContent(type="text", text="Error: cluster_uri is required")]
    if not database:
        return [TextContent(type="text", text="Error: database is required")]
    
    try:
        def do_query():
            return execute_query_sync(cluster_uri, database, ".show tables")
        
        results = await asyncio.get_event_loop().run_in_executor(None, do_query)
        
        tables = [r.get("TableName", r.get("Name", "")) for r in results]
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "cluster_uri": cluster_uri,
                "database": database,
                "tables": tables,
                "count": len(tables)
            }, indent=2)
        )]
        
    except KustoServiceError as e:
        return [TextContent(type="text", text=f"Kusto error: {str(e)}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def adx_get_table_schema(arguments: dict[str, Any]) -> list[TextContent]:
    """Get the schema of a table."""
    cluster_uri = arguments.get("cluster_uri", "")
    database = arguments.get("database", "")
    table = arguments.get("table", "")
    
    if not cluster_uri:
        return [TextContent(type="text", text="Error: cluster_uri is required")]
    if not database:
        return [TextContent(type="text", text="Error: database is required")]
    if not table:
        return [TextContent(type="text", text="Error: table is required")]
    
    try:
        def do_query():
            query = f".show table {table} schema as json"
            return execute_query_sync(cluster_uri, database, query)
        
        results = await asyncio.get_event_loop().run_in_executor(None, do_query)
        
        # Parse the schema JSON
        schema_info = []
        if results:
            schema_json = results[0].get("Schema", "{}")
            if isinstance(schema_json, str):
                schema_data = json.loads(schema_json)
            else:
                schema_data = schema_json
            
            columns = schema_data.get("OrderedColumns", [])
            for col in columns:
                schema_info.append({
                    "name": col.get("Name", ""),
                    "type": col.get("CslType", col.get("Type", ""))
                })
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "cluster_uri": cluster_uri,
                "database": database,
                "table": table,
                "columns": schema_info,
                "column_count": len(schema_info)
            }, indent=2)
        )]
        
    except KustoServiceError as e:
        return [TextContent(type="text", text=f"Kusto error: {str(e)}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def adx_sample_data(arguments: dict[str, Any]) -> list[TextContent]:
    """Get sample data from a table."""
    cluster_uri = arguments.get("cluster_uri", "")
    database = arguments.get("database", "")
    table = arguments.get("table", "")
    limit = arguments.get("limit", 10)
    
    if not cluster_uri:
        return [TextContent(type="text", text="Error: cluster_uri is required")]
    if not database:
        return [TextContent(type="text", text="Error: database is required")]
    if not table:
        return [TextContent(type="text", text="Error: table is required")]
    
    try:
        def do_query():
            query = f"{table} | take {limit}"
            return execute_query_sync(cluster_uri, database, query)
        
        results = await asyncio.get_event_loop().run_in_executor(None, do_query)
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "cluster_uri": cluster_uri,
                "database": database,
                "table": table,
                "sample_data": results,
                "row_count": len(results)
            }, indent=2, default=str)
        )]
        
    except KustoServiceError as e:
        return [TextContent(type="text", text=f"Kusto error: {str(e)}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def adx_query(arguments: dict[str, Any]) -> list[TextContent]:
    """Execute a KQL query."""
    cluster_uri = arguments.get("cluster_uri", "")
    database = arguments.get("database", "")
    query = arguments.get("query", "")
    
    if not cluster_uri:
        return [TextContent(type="text", text="Error: cluster_uri is required")]
    if not database:
        return [TextContent(type="text", text="Error: database is required")]
    if not query:
        return [TextContent(type="text", text="Error: query is required")]
    
    try:
        def do_query():
            return execute_query_sync(cluster_uri, database, query)
        
        results = await asyncio.get_event_loop().run_in_executor(None, do_query)
        
        return [TextContent(
            type="text",
            text=json.dumps({
                "cluster_uri": cluster_uri,
                "database": database,
                "query": query,
                "results": results,
                "row_count": len(results)
            }, indent=2, default=str)
        )]
        
    except KustoServiceError as e:
        return [TextContent(type="text", text=f"Kusto error: {str(e)}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
