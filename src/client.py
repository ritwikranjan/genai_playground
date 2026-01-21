"""
GenAI Playground Client

This module handles the core logic for:
- Connecting to MCP tool servers
- Integrating with Azure OpenAI
- Orchestrating the tool execution loop
"""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional, AsyncGenerator

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

@dataclass
class PlaygroundConfig:
    """Configuration for the playground."""
    system_prompt: str = "You are a helpful assistant."
    user_prompt: str = ""
    model: str = "gpt-4o"
    tools: list[str] = field(default_factory=list)
    max_iterations: int = 10
    verbose: bool = False
    
    # Azure OpenAI settings (can be overridden)
    azure_endpoint: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_api_version: str = "2024-02-15-preview"
    azure_deployment: Optional[str] = None

    @classmethod
    def from_json(cls, json_path: str) -> "PlaygroundConfig":
        """Load configuration from a JSON file."""
        with open(json_path, "r") as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: dict) -> "PlaygroundConfig":
        """Create configuration from a dictionary."""
        return cls(**data)

@dataclass
class ToolServer:
    """Represents a connected MCP tool server."""
    name: str
    session: ClientSession
    tools: list[dict]


def get_python_executable() -> str:
    """Get the Python executable path, preferring the virtual environment."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if we're in a virtual environment (Windows)
    venv_python = os.path.join(project_root, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        return venv_python
    
    # Check for Unix-style venv
    venv_python_unix = os.path.join(project_root, ".venv", "bin", "python")
    if os.path.exists(venv_python_unix):
        return venv_python_unix
    
    # Fall back to current Python
    return sys.executable


def create_openai_client(config: PlaygroundConfig) -> AzureOpenAI:
    """Create the Azure OpenAI client."""
    endpoint = config.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = config.azure_api_key or os.getenv("AZURE_OPENAI_API_KEY")
    api_version = config.azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    if not endpoint:
        raise ValueError("Azure OpenAI endpoint not configured. Set AZURE_OPENAI_ENDPOINT environment variable or provide azure_endpoint in config.")
    if not api_key:
        raise ValueError("Azure OpenAI API key not configured. Set AZURE_OPENAI_API_KEY environment variable or provide azure_api_key in config.")
    
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version
    )


def find_uvx_executable() -> str:
    """Find the uvx executable, checking common locations."""
    import shutil
    
    # First check if uvx is in PATH
    uvx_path = shutil.which("uvx")
    if uvx_path:
        return uvx_path
    
    # Check in the current virtual environment
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Windows venv
    venv_uvx = os.path.join(project_root, ".venv", "Scripts", "uvx.exe")
    if os.path.exists(venv_uvx):
        return venv_uvx
    
    # Unix venv
    venv_uvx_unix = os.path.join(project_root, ".venv", "bin", "uvx")
    if os.path.exists(venv_uvx_unix):
        return venv_uvx_unix
    
    # Fall back to just "uvx" and hope it's in PATH
    return "uvx"

def parse_tool_config(tool_spec: str | dict) -> tuple[str, list[str], dict[str, str]]:
    """Parse a tool specification into command, args, and env.
    
    Supports:
    - String path to a Python file: "tools/web_search.py"
    - Dict with command/args/env: {"command": "uvx", "args": ["azure-kusto-mcp"], "env": {...}}
    - String starting with "uvx:": "uvx:azure-kusto-mcp"
    """
    if isinstance(tool_spec, dict):
        command = tool_spec.get("command", "python")
        args = tool_spec.get("args", [])
        env = tool_spec.get("env", {})
        # Resolve uvx path
        if command == "uvx":
            command = find_uvx_executable()
        return command, args, env
    
    # String specification
    if tool_spec.startswith("uvx:"):
        # uvx:<package-name>
        package = tool_spec[4:]
        return find_uvx_executable(), [package], {}
    
    # Assume it's a Python script path
    return "python", [tool_spec], {}

@asynccontextmanager
async def connect_tool_server(tool_spec: str | dict, verbose: bool = False) -> AsyncGenerator[ToolServer, None]:
    """Connect to a tool server as an async context manager.
    
    Args:
        tool_spec: Either a path to a Python script, a "uvx:<package>" string,
                   or a dict with command/args/env keys.
        verbose: Whether to print connection info.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    command, args, env = parse_tool_config(tool_spec)
    
    # If it's a Python script path, resolve it
    if command == "python" and args:
        python_exe = get_python_executable()
        command = python_exe
        tool_path = args[0]
        if not os.path.isabs(tool_path):
            tool_path = os.path.join(project_root, tool_path)
        if not os.path.exists(tool_path):
            raise FileNotFoundError(f"Tool not found: {tool_path}")
        args = [tool_path]
    
    # Merge environment variables
    full_env = {**os.environ, **env}
    
    # Determine server name for display
    if isinstance(tool_spec, dict):
        server_name = tool_spec.get("name", args[0] if args else "unknown")
    elif tool_spec.startswith("uvx:"):
        server_name = tool_spec[4:]
    else:
        server_name = os.path.basename(tool_spec)
    
    server_params = StdioServerParameters(
        command=command,
        args=args,
        env=full_env if env else None
    )
    
    # Use proper async context managers
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            # Get available tools
            tools_response = await session.list_tools()
            
            # Convert MCP tools to OpenAI function format
            openai_tools = []
            for tool in tools_response.tools:
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}}
                    }
                }
                openai_tools.append(openai_tool)
            
            tool_server = ToolServer(
                name=server_name,
                session=session,
                tools=openai_tools
            )
            
            if verbose:
                print(f"Connected to tool server: {tool_server.name}")
                print(f"  Available tools: {[t['function']['name'] for t in openai_tools]}")
            
            yield tool_server


async def call_tool(server: ToolServer, tool_name: str, arguments: dict) -> str:
    """Call a tool on its corresponding server."""
    try:
        result = await server.session.call_tool(tool_name, arguments)
        
        # Extract text content from the result
        if result.content:
            texts = []
            for content in result.content:
                if hasattr(content, 'text'):
                    texts.append(content.text)
            return "\n".join(texts) if texts else "Tool returned no content"
        return "Tool returned no content"
        
    except Exception as e:
        return f"Tool error: {str(e)}"


async def run_conversation(
    openai_client: AzureOpenAI,
    messages: list[dict],
    tool_servers: list[ToolServer],
    tool_to_server: dict[str, ToolServer],
    all_tools: list[dict],
    deployment: str,
    max_iterations: int,
    verbose: bool = False
) -> str:
    """Run a single conversation turn with the model."""
    iteration = 0
    final_response = ""
    
    while iteration < max_iterations:
        iteration += 1
        
        if verbose:
            print(f"\n--- Iteration {iteration} ---")
        
        # Call the model
        try:
            if all_tools:
                response = openai_client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    tools=all_tools,
                    tool_choice="auto"
                )
            else:
                response = openai_client.chat.completions.create(
                    model=deployment,
                    messages=messages
                )
        except Exception as e:
            return f"OpenAI API error: {str(e)}"
        
        assistant_message = response.choices[0].message
        
        # Check if the model wants to call tools
        if assistant_message.tool_calls:
            # Add assistant message with tool calls to history
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })
            
            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                if verbose:
                    print(f"Calling tool: {tool_name}")
                    print(f"  Arguments: {json.dumps(arguments, indent=2)}")
                
                # Call the tool
                server = tool_to_server.get(tool_name)
                if server:
                    tool_result = await call_tool(server, tool_name, arguments)
                else:
                    tool_result = f"Error: Unknown tool '{tool_name}'"
                
                if verbose:
                    print(f"  Result: {tool_result[:200]}..." if len(tool_result) > 200 else f"  Result: {tool_result}")
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
        else:
            # No tool calls - we have our final response
            final_response = assistant_message.content or ""
            break
    
    if not final_response and iteration >= max_iterations:
        final_response = "Maximum iterations reached without a final response."
    
    return final_response


async def run_playground(config: PlaygroundConfig) -> str:
    """Run the playground with a single prompt."""
    openai_client = create_openai_client(config)
    deployment = config.azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT") or config.model
    
    # Build the initial messages
    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": config.user_prompt}
    ]
    
    if not config.tools:
        # No tools - just call the model directly
        return await run_conversation(
            openai_client, messages, [], {}, [], deployment, config.max_iterations, config.verbose
        )
    
    # Connect to tool servers using nested context managers
    async def run_with_tools(tool_paths: list[str], index: int = 0, 
                            servers: list[ToolServer] = None, 
                            tool_map: dict = None,
                            all_tools: list = None):
        if servers is None:
            servers = []
        if tool_map is None:
            tool_map = {}
        if all_tools is None:
            all_tools = []
            
        if index >= len(tool_paths):
            # All tools connected, run the conversation
            return await run_conversation(
                openai_client, messages, servers, tool_map, all_tools,
                deployment, config.max_iterations, config.verbose
            )
        
        # Connect next tool
        tool_path = tool_paths[index]
        try:
            async with connect_tool_server(tool_path, config.verbose) as server:
                servers.append(server)
                all_tools.extend(server.tools)
                for tool in server.tools:
                    tool_map[tool['function']['name']] = server
                
                return await run_with_tools(tool_paths, index + 1, servers, tool_map, all_tools)
        except Exception as e:
            print(f"Warning: Failed to connect to tool server '{tool_path}': {e}")
            return await run_with_tools(tool_paths, index + 1, servers, tool_map, all_tools)
    
    return await run_with_tools(config.tools)


async def run_chat_session(config: PlaygroundConfig, on_message: callable = None):
    """Run an interactive chat session. 
    
    Args:
        config: The playground configuration
        on_message: Optional callback for handling messages (for testing)
    """
    openai_client = create_openai_client(config)
    deployment = config.azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT") or config.model
    
    # Conversation history
    messages = [
        {"role": "system", "content": config.system_prompt}
    ]
    
    async def chat_loop(servers: list[ToolServer], tool_map: dict, all_tools: list):
        """The main chat loop."""
        nonlocal messages
        
        while True:
            # Get user input
            try:
                if on_message:
                    user_input = on_message()
                else:
                    # Ensure prompt is displayed immediately
                    print("You: ", end="", flush=True)
                    user_input = input().strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSession ended.")
                break
            
            if not user_input:
                continue
            
            if user_input.lower() in ('exit', 'quit'):
                print("Session ended.")
                break
            
            if user_input.lower() == 'clear':
                messages = [{"role": "system", "content": config.system_prompt}]
                print("Conversation cleared.\n")
                continue
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Run conversation
            response = await run_conversation(
                openai_client, messages, servers, tool_map, all_tools,
                deployment, config.max_iterations, config.verbose
            )
            
            # Add assistant response to history
            messages.append({"role": "assistant", "content": response})
            
            print(f"\nAssistant: {response}\n")
    
    if not config.tools:
        # No tools - just run the chat loop
        await chat_loop([], {}, [])
        return
    
    # Connect to all tools using nested context managers
    async def connect_and_chat(tool_paths: list[str], index: int = 0,
                               servers: list[ToolServer] = None,
                               tool_map: dict = None,
                               all_tools: list = None):
        if servers is None:
            servers = []
        if tool_map is None:
            tool_map = {}
        if all_tools is None:
            all_tools = []
        
        if index >= len(tool_paths):
            # All tools connected, start chat
            print("")  # Blank line before prompt
            await chat_loop(servers, tool_map, all_tools)
            return
        
        # Connect next tool
        tool_path = tool_paths[index]
        
        # Show connection status
        if isinstance(tool_path, dict):
            tool_name = tool_path.get('name', tool_path.get('args', ['unknown'])[0] if tool_path.get('args') else 'unknown')
        elif isinstance(tool_path, str) and tool_path.startswith("uvx:"):
            tool_name = tool_path[4:]
        else:
            tool_name = os.path.basename(str(tool_path)) if isinstance(tool_path, str) else str(tool_path)
        
        print(f"Connecting to {tool_name}...", end="", flush=True)
        
        try:
            async with connect_tool_server(tool_path, verbose=True) as server:
                servers.append(server)
                all_tools.extend(server.tools)
                for tool in server.tools:
                    tool_map[tool['function']['name']] = server
                
                await connect_and_chat(tool_paths, index + 1, servers, tool_map, all_tools)
        except Exception as e:
            print(f"Warning: Failed to connect to tool server '{tool_path}': {e}")
            await connect_and_chat(tool_paths, index + 1, servers, tool_map, all_tools)
    
    await connect_and_chat(config.tools)
