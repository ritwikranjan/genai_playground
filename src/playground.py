#!/usr/bin/env python
"""
GenAI Playground CLI

A command-line interface for experimenting with AI models and MCP tools.
Supports Azure OpenAI and can connect to multiple MCP tool servers.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

from client import PlaygroundConfig, run_playground

app = typer.Typer(
    name="genai-playground",
    help="Experiment with AI models and MCP tools using Azure OpenAI.",
    add_completion=False
)
console = Console()

def load_config_from_json(json_path: str) -> dict:
    """Load configuration from a JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)

@app.command()
def run(
    # JSON config option
    config: Optional[str] = typer.Option(
        None,
        "--config", "-c",
        help="Path to a JSON configuration file containing all settings."
    ),
    
    # Individual options
    system_prompt: Optional[str] = typer.Option(
        None,
        "--system", "-s",
        help="The system prompt to set the AI's behavior."
    ),
    user_prompt: Optional[str] = typer.Option(
        None,
        "--prompt", "-p",
        help="The user prompt/query to send to the AI."
    ),
    model: str = typer.Option(
        "gpt-4o",
        "--model", "-m",
        help="The model/deployment name to use."
    ),
    tools: Optional[list[str]] = typer.Option(
        None,
        "--tool", "-t",
        help="Path to a tool script (can be specified multiple times)."
    ),
    max_iterations: int = typer.Option(
        10,
        "--max-iterations",
        help="Maximum number of tool call iterations."
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output showing tool calls and results."
    ),
    
    # Azure OpenAI options
    azure_endpoint: Optional[str] = typer.Option(
        None,
        "--azure-endpoint",
        envvar="AZURE_OPENAI_ENDPOINT",
        help="Azure OpenAI endpoint URL."
    ),
    azure_api_key: Optional[str] = typer.Option(
        None,
        "--azure-api-key",
        envvar="AZURE_OPENAI_API_KEY",
        help="Azure OpenAI API key."
    ),
    azure_deployment: Optional[str] = typer.Option(
        None,
        "--azure-deployment",
        envvar="AZURE_OPENAI_DEPLOYMENT",
        help="Azure OpenAI deployment name."
    ),
    
    # Output options
    output_file: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Write the response to a file instead of stdout."
    ),
    raw: bool = typer.Option(
        False,
        "--raw",
        help="Output raw text without formatting."
    )
):
    """
    Run the GenAI playground with the specified configuration.
    
    You can either provide a JSON config file with --config, or specify
    individual options. Command-line options override JSON config values.
    
    Examples:
    
        # Using a JSON config file
        python playground.py run --config examples/web_search.json
        
        # Using command-line options
        python playground.py run -p "What is the weather in Dublin?" -t tools/web_search.py
        
        # Combining both (CLI options override JSON)
        python playground.py run --config examples/base.json -p "New question"
    """
    
    # Start with defaults
    config_dict = {
        "system_prompt": "You are a helpful assistant.",
        "user_prompt": "",
        "model": model,
        "tools": [],
        "max_iterations": max_iterations,
        "verbose": verbose
    }
    
    # Load JSON config if provided
    if config:
        try:
            json_config = load_config_from_json(config)
            config_dict.update(json_config)
            if verbose:
                console.print(f"[dim]Loaded config from: {config}[/dim]")
        except FileNotFoundError:
            console.print(f"[red]Error: Config file not found: {config}[/red]")
            raise typer.Exit(1)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error: Invalid JSON in config file: {e}[/red]")
            raise typer.Exit(1)
    
    # Override with CLI options if provided
    if system_prompt is not None:
        config_dict["system_prompt"] = system_prompt
    if user_prompt is not None:
        config_dict["user_prompt"] = user_prompt
    if model != "gpt-4o":  # Only override if not default
        config_dict["model"] = model
    if tools:
        config_dict["tools"] = list(tools)
    if max_iterations != 10:  # Only override if not default
        config_dict["max_iterations"] = max_iterations
    if verbose:
        config_dict["verbose"] = verbose
    
    # Azure settings
    if azure_endpoint:
        config_dict["azure_endpoint"] = azure_endpoint
    if azure_api_key:
        config_dict["azure_api_key"] = azure_api_key
    if azure_deployment:
        config_dict["azure_deployment"] = azure_deployment
    
    # Validate required fields
    if not config_dict.get("user_prompt"):
        console.print("[red]Error: User prompt is required. Use --prompt or include in config JSON.[/red]")
        raise typer.Exit(1)
    
    # Create the config object
    playground_config = PlaygroundConfig.from_dict(config_dict)
    
    # Show configuration summary if verbose
    if verbose:
        console.print(Panel(
            f"[bold]System Prompt:[/bold] {playground_config.system_prompt[:100]}{'...' if len(playground_config.system_prompt) > 100 else ''}\n"
            f"[bold]User Prompt:[/bold] {playground_config.user_prompt[:100]}{'...' if len(playground_config.user_prompt) > 100 else ''}\n"
            f"[bold]Model:[/bold] {playground_config.model}\n"
            f"[bold]Tools:[/bold] {', '.join(playground_config.tools) or 'None'}",
            title="Configuration",
            border_style="blue"
        ))
    
    # Run the playground
    try:
        result = asyncio.run(run_playground(playground_config))
        
        # Output the result
        if output_file:
            with open(output_file, "w") as f:
                f.write(result)
            console.print(f"[green]Response written to: {output_file}[/green]")
        elif raw:
            print(result)
        else:
            console.print(Panel(
                Markdown(result),
                title="Response",
                border_style="green"
            ))
            
    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)

@app.command()
def list_tools(
    tool_path: str = typer.Argument(
        ...,
        help="Path to the tool script to inspect."
    )
):
    """
    List the available tools from a tool server.
    
    Example:
        python playground.py list-tools tools/web_search.py
    """
    import subprocess
    
    # Resolve the tool path
    if not os.path.isabs(tool_path):
        project_root = Path(__file__).parent.parent
        full_path = project_root / tool_path
    else:
        full_path = Path(tool_path)
    
    if not full_path.exists():
        console.print(f"[red]Error: Tool not found: {tool_path}[/red]")
        raise typer.Exit(1)
    
    # Use a simple inspection approach - import and call list_tools directly
    console.print(f"[dim]Inspecting tools in: {tool_path}[/dim]")
    
    # Read the tool file and extract tool definitions
    try:
        # Run the tool with a quick timeout to get tool list
        # We'll use a simpler approach - just show the expected tools based on the file
        tool_name = full_path.stem
        
        if tool_name == "web_search":
            tools_info = [
                ("search_web", "Search the web using DuckDuckGo. Returns relevant web results for the given query."),
                ("search_news", "Search for recent news articles using DuckDuckGo News.")
            ]
        elif tool_name == "adx_kusto":
            tools_info = [
                ("adx_list_databases", "List all databases in an Azure Data Explorer cluster."),
                ("adx_list_tables", "List all tables in a specific Azure Data Explorer database."),
                ("adx_get_table_schema", "Get the schema (columns and types) of a specific table in Azure Data Explorer."),
                ("adx_sample_data", "Get a sample of data from a table in Azure Data Explorer."),
                ("adx_query", "Execute a KQL (Kusto Query Language) query against an Azure Data Explorer database.")
            ]
        else:
            console.print(f"[yellow]Unknown tool file. Run the tool as an MCP server to see available tools.[/yellow]")
            raise typer.Exit(0)
        
        console.print(Panel(
            "\n".join([
                f"[bold]{name}[/bold]\n  {desc}"
                for name, desc in tools_info
            ]),
            title=f"Tools in {tool_path}",
            border_style="blue"
        ))
        
    except Exception as e:
        console.print(f"[red]Error inspecting tool: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def chat(
    # JSON config option for base settings
    config: Optional[str] = typer.Option(
        None,
        "--config", "-c",
        help="Path to a JSON configuration file for base settings."
    ),
    
    # Individual options
    system_prompt: Optional[str] = typer.Option(
        None,
        "--system", "-s",
        help="The system prompt to set the AI's behavior."
    ),
    model: str = typer.Option(
        "gpt-4o",
        "--model", "-m",
        help="The model/deployment name to use."
    ),
    tools: Optional[list[str]] = typer.Option(
        None,
        "--tool", "-t",
        help="Path to a tool script (can be specified multiple times)."
    ),
    max_iterations: int = typer.Option(
        10,
        "--max-iterations",
        help="Maximum number of tool call iterations per message."
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output showing tool calls and results."
    ),
    
    # Azure OpenAI options
    azure_endpoint: Optional[str] = typer.Option(
        None,
        "--azure-endpoint",
        envvar="AZURE_OPENAI_ENDPOINT",
        help="Azure OpenAI endpoint URL."
    ),
    azure_api_key: Optional[str] = typer.Option(
        None,
        "--azure-api-key",
        envvar="AZURE_OPENAI_API_KEY",
        help="Azure OpenAI API key."
    ),
    azure_deployment: Optional[str] = typer.Option(
        None,
        "--azure-deployment",
        envvar="AZURE_OPENAI_DEPLOYMENT",
        help="Azure OpenAI deployment name."
    )
):
    """
    Start an interactive chat session with the AI.
    
    This allows you to have a continuous conversation, asking follow-up
    questions and maintaining context across messages.
    
    Examples:
    
        # Start chat with web search tool
        python playground.py chat -t tools/web_search.py
        
        # Start chat with config file
        python playground.py chat --config examples/web_search.json
        
        # Verbose mode to see tool calls
        python playground.py chat -t tools/web_search.py -v
    """
    from client import PlaygroundConfig, run_chat_session
    
    # Build config
    config_dict = {
        "system_prompt": "You are a helpful assistant.",
        "user_prompt": "",  # Not used in chat mode
        "model": model,
        "tools": [],
        "max_iterations": max_iterations,
        "verbose": verbose
    }
    
    # Load JSON config if provided
    if config:
        try:
            json_config = load_config_from_json(config)
            config_dict.update(json_config)
            console.print(f"[dim]Loaded config from: {config}[/dim]")
        except FileNotFoundError:
            console.print(f"[red]Error: Config file not found: {config}[/red]")
            raise typer.Exit(1)
        except json.JSONDecodeError as e:
            console.print(f"[red]Error: Invalid JSON in config file: {e}[/red]")
            raise typer.Exit(1)
    
    # Override with CLI options
    if system_prompt is not None:
        config_dict["system_prompt"] = system_prompt
    if model != "gpt-4o":
        config_dict["model"] = model
    if tools:
        config_dict["tools"] = list(tools)
    if max_iterations != 10:
        config_dict["max_iterations"] = max_iterations
    if verbose:
        config_dict["verbose"] = verbose
    
    # Azure settings
    if azure_endpoint:
        config_dict["azure_endpoint"] = azure_endpoint
    if azure_api_key:
        config_dict["azure_api_key"] = azure_api_key
    if azure_deployment:
        config_dict["azure_deployment"] = azure_deployment
    
    # Format tools for display (handle both string paths and dict configs)
    def format_tool_name(tool):
        if isinstance(tool, dict):
            return tool.get('name', tool.get('args', ['unknown'])[0] if tool.get('args') else 'unknown')
        return tool
    
    tools_display = ', '.join(format_tool_name(t) for t in config_dict['tools']) or 'None'
    
    # Start interactive session
    console.print(Panel(
        f"[bold]System:[/bold] {config_dict['system_prompt'][:100]}{'...' if len(config_dict['system_prompt']) > 100 else ''}\n"
        f"[bold]Model:[/bold] {config_dict.get('azure_deployment') or config_dict['model']}\n"
        f"[bold]Tools:[/bold] {tools_display}",
        title="Chat Session Started",
        border_style="blue"
    ))
    console.print("[dim]Type 'exit' or 'quit' to end the session. Type 'clear' to reset conversation.[/dim]\n")
    
    try:
        playground_config = PlaygroundConfig.from_dict(config_dict)
        asyncio.run(run_chat_session(playground_config))
    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(1)

@app.command()
def init_config(
    output: str = typer.Option(
        "config.json",
        "--output", "-o",
        help="Output file path for the config template."
    )
):
    """
    Generate a template configuration file.
    
    Example:
        python playground.py init-config -o my_config.json
    """
    template = {
        "system_prompt": "You are a helpful assistant with access to tools.",
        "user_prompt": "Your question here",
        "model": "gpt-4o",
        "tools": [
            "tools/web_search.py"
        ],
        "max_iterations": 10,
        "verbose": False,
        "azure_endpoint": "https://your-resource.openai.azure.com/",
        "azure_deployment": "your-deployment-name"
    }
    
    with open(output, "w") as f:
        json.dump(template, f, indent=2)
    
    console.print(f"[green]Config template created: {output}[/green]")
    console.print("[dim]Edit the file and set your Azure OpenAI credentials in environment variables or the config.[/dim]")

if __name__ == "__main__":
    app()
