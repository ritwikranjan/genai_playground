"""
GenAI Playground Client V2 - Using GitHub Copilot SDK

This module provides the same interface as client.py but uses
the github-copilot-sdk for agent orchestration instead of
manual MCP/OpenAI integration.

Uses the simpler CopilotOptions API with mcp_servers for native MCP support.
"""

import json
import os
import sys
from pathlib import Path

from rich.console import Console

# Import the PlaygroundConfig from client.py to ensure compatibility
from client import PlaygroundConfig

# Import the Copilot SDK
try:
    from copilot import CopilotClient
    from copilot.generated.session_events import SessionEventType
except ImportError:
    raise ImportError(
        "github-copilot-sdk is not installed. "
        "Install it with: pip install github-copilot-sdk"
    )

console = Console()


def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def _build_mcp_servers_config(config: PlaygroundConfig, verbose: bool = False) -> dict:
    """
    Build the MCP servers configuration from PlaygroundConfig.
    
    Returns a dict[str, MCPLocalServerConfig] suitable for SessionConfig.mcp_servers.
    
    MCPLocalServerConfig fields:
    - tools: list[str] - REQUIRED, list of tools to include (use ["*"] for all)
    - command: str - REQUIRED, the command to run
    - args: list[str] - REQUIRED, command arguments
    - env: dict[str, str] - optional, environment variables
    - type: "local" | "stdio" - optional
    - timeout: int - optional
    - cwd: str - optional, working directory
    """
    mcp_servers = {}
    project_root = _get_project_root()
    
    if not config.tools:
        return mcp_servers
    
    for i, tool_spec in enumerate(config.tools):
        if isinstance(tool_spec, dict):
            server_name = tool_spec.get("name", f"tool_{i}")
            
            # Get the command 
            command = tool_spec.get("command", "python")
            
            # Resolve args paths to absolute if they're Python files
            args = tool_spec.get("args", [])
            resolved_args = []
            for arg in args:
                if arg.endswith('.py') and not os.path.isabs(arg):
                    resolved_path = str(project_root / arg)
                    if os.path.exists(resolved_path):
                        resolved_args.append(resolved_path)
                    else:
                        resolved_args.append(arg)
                else:
                    resolved_args.append(arg)
            
            # Build MCPLocalServerConfig with required 'tools' field
            server_config = {
                "tools": tool_spec.get("tools_filter", ["*"]),  # Include specified tools from this MCP server
                "command": command,
                "args": resolved_args,
                "cwd": str(project_root),  # Set working directory to project root
            }
            
            # Add environment variables if specified
            if tool_spec.get("env"):
                server_config["env"] = tool_spec["env"]
            
            mcp_servers[server_name] = server_config
            
            if verbose:
                console.print(f"[dim]  MCP Server '{server_name}': {command} {' '.join(resolved_args)}[/dim]")
            
        elif isinstance(tool_spec, str):
            # Tool is a path to a Python script - resolve to absolute path
            server_name = os.path.basename(tool_spec).replace(".py", "")
            
            # Resolve relative paths to absolute
            if not os.path.isabs(tool_spec):
                tool_path = str(project_root / tool_spec)
            else:
                tool_path = tool_spec
            
            # Verify the file exists
            if not os.path.exists(tool_path):
                console.print(f"[yellow]Warning: Tool file not found: {tool_path}[/yellow]")
            
            mcp_servers[server_name] = {
                "tools": ["*"],  # Include all tools
                "command": sys.executable,
                "args": [tool_path],  # Use absolute path
                "cwd": str(project_root),  # Set working directory to project root
            }
            
            if verbose:
                console.print(f"[dim]  MCP Server '{server_name}': {sys.executable} {tool_path}[/dim]")
    
    return mcp_servers


async def run_playground(config: PlaygroundConfig) -> str:
    """
    Run the playground with a single prompt using Copilot SDK.
    
    This matches the signature of client.run_playground().
    """
    # Build MCP servers config
    if config.verbose:
        console.print("[cyan]Building MCP servers config...[/cyan]")
    mcp_servers = _build_mcp_servers_config(config, verbose=config.verbose)
    
    if config.verbose:
        console.print(f"[cyan]MCP Servers config:[/cyan]\n{json.dumps(mcp_servers, indent=2, default=str)}")
    
    # Initialize the Copilot Client
    client_options = {
        "log_level": "debug" if config.verbose else "error"
    }
    
    client = CopilotClient(client_options)
    
    await client.start()
    
    try:
        # Create session with MCP servers
        session_config = {
            "mcp_servers": mcp_servers,
            "streaming": config.stream,
        }
        if config.system_prompt:
            session_config["system_message"] = {"content": config.system_prompt}
        
        session = await client.create_session(session_config)
        
        # Collect the response
        final_response = ""
        
        def handle_event(event):
            nonlocal final_response
            if event.type == SessionEventType.ASSISTANT_MESSAGE:
                content = getattr(event.data, 'content', '') if event.data else ''
                final_response = content
                if config.verbose:
                    console.print(f"[green]Assistant:[/green] {content}")
        
        session.on(handle_event)
        
        # Send the user prompt
        await session.send_and_wait({
            "prompt": config.user_prompt,
            "mode": "agent",
            "attachments": []
        }, timeout=300)
        
        await session.destroy()
        
        return final_response
        
    finally:
        await client.stop()


async def run_chat_session(config: PlaygroundConfig, on_message: callable = None):
    """
    Run an interactive chat session using Copilot SDK.
    
    This matches the signature of client.run_chat_session().
    
    Args:
        config: The playground configuration
        on_message: Optional callback for handling messages (for testing)
    """
    # Build MCP servers config - passed directly to session, not written to ~/.copilot/
    if config.verbose:
        console.print("[cyan]Building MCP servers config...[/cyan]")
    mcp_servers = _build_mcp_servers_config(config, verbose=config.verbose)
    
    console.print("🤖 Initializing Copilot Agent...")
    
    # Initialize the Copilot Client
    client_options = {
        "log_level": "error"  # Suppress verbose CLI logging
    }
    
    client = CopilotClient(client_options)
    
    # Start the client
    await client.start()
    
    # Create session with MCP servers passed directly (no file write needed)
    session_config = {
        "mcp_servers": mcp_servers,
        # Enable streaming for responses
        "streaming": bool(config.stream),
    }
    # Note: We don't restrict available_tools here because the tool names from
    # local MCP servers (e.g., adx_list_databases, adx_query) won't match
    # a generic filter like "Azure-kusto". Let all tools be available.
    if config.system_prompt:
        session_config["system_message"] = {"content": config.system_prompt}
    
    session = await client.create_session(session_config)
    
    # Set up event handler with streaming for response and reasoning
    response_text = []
    is_streaming_response = False
    is_streaming_reasoning = False
    
    def handle_event(event):
        nonlocal is_streaming_response, is_streaming_reasoning
        event_data = event.data
        
        # Streaming reasoning delta
        if event.type == SessionEventType.ASSISTANT_REASONING_DELTA:
            if config.show_reasoning:
                delta = getattr(event_data, 'delta_content', '') if event_data else ''
                if delta:
                    if not is_streaming_reasoning:
                        print("\n💭 ", end="", flush=True)
                        is_streaming_reasoning = True
                    print(delta, end="", flush=True)
        
        # End of reasoning (complete reasoning block)
        elif event.type == SessionEventType.ASSISTANT_REASONING:
            if is_streaming_reasoning:
                print()  # Newline after reasoning
                is_streaming_reasoning = False
        
        # Streaming response delta
        elif event.type == SessionEventType.ASSISTANT_MESSAGE_DELTA:
            delta = getattr(event_data, 'delta_content', '') if event_data else ''
            if delta:
                if not is_streaming_response:
                    print("\nCopilot: ", end="", flush=True)
                    is_streaming_response = True
                print(delta, end="", flush=True)
                response_text.append(delta)
        
        # Full message (fallback if not streaming)
        elif event.type == SessionEventType.ASSISTANT_MESSAGE:
            content = getattr(event_data, 'content', '') if event_data else ''
            if content and not response_text:  # Only use if we didn't get deltas
                response_text.append(content)
        
        # Tool execution events
        elif event.type == SessionEventType.TOOL_EXECUTION_START:
            # Reset streaming state when tool starts
            if is_streaming_response:
                print()  # Newline
                is_streaming_response = False
            if is_streaming_reasoning:
                print()
                is_streaming_reasoning = False
            tool_name = getattr(event_data, 'tool_name', 'unknown') if event_data else 'unknown'
            
            # Always show tool execution to verify it's actually happening
            print(f"\n🔧 [TOOL CALL] {tool_name}")
            
            if config.verbose:
                # Show tool parameters in verbose mode
                tool_params = getattr(event_data, 'arguments', None)
                tool_input = getattr(event_data, 'input', None)
                if tool_input:
                    print(f"   📥 Input: {tool_input}")
                if tool_params:
                    if isinstance(tool_params, dict):
                        params_str = json.dumps(tool_params, indent=2, default=str)
                    else:
                        params_str = str(tool_params)
                    print(f"   📋 Parameters: {params_str}")
        
        elif event.type == SessionEventType.TOOL_EXECUTION_COMPLETE:
            # Always show tool completion to verify actual execution
            print(f"   ✅ [TOOL COMPLETE]")
            if config.verbose:
                # Show tool result in verbose mode
                result = getattr(event_data, 'result', None)
                if result:
                    result_str = str(result)[:500]  # Truncate long results
                    if len(str(result)) > 500:
                        result_str += "..."
                    print(f"   📤 Result: {result_str}")
                else:
                    print("   ⚠️ No result returned from tool")
        
        # Session idle - end of turn
        elif event.type == SessionEventType.SESSION_IDLE:
            if is_streaming_response:
                print("\n")  # Newline after response
                is_streaming_response = False
            if is_streaming_reasoning:
                print()
                is_streaming_reasoning = False
        
        # Error handling
        elif event.type == SessionEventType.SESSION_ERROR:
            error_msg = getattr(event_data, 'message', 'Unknown error') if event_data else 'Unknown error'
            console.print(f"\n[red]❌ Error: {error_msg}[/red]")
    
    session.on(handle_event)
    
    try:
        console.print("✅ Connected. Copilot SDK v2 enabled.")
        if config.show_reasoning:
            console.print("   (Reasoning display on)")
        console.print("   Type 'exit' or 'quit' to end, 'clear' to reset session.\n")
        
        while True:
            try:
                if on_message:
                    user_input = on_message()
                else:
                    console.print("You: ", end="")
                    user_input = input().strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nSession ended.")
                break
            
            if not user_input:
                continue
            
            if user_input.lower() in ('exit', 'quit'):
                console.print("Session ended.")
                break
            
            if user_input.lower() == 'clear':
                # Destroy and recreate session
                await session.destroy()
                session = await client.create_session(session_config)
                session.on(handle_event)
                response_text.clear()
                console.print("Session reset.\n")
                continue
            
            try:
                # Clear response text for new message
                response_text.clear()
                is_streaming_response = False
                is_streaming_reasoning = False
                
                # Send with extended timeout (1 hour for complex queries)
                await session.send_and_wait({
                    "prompt": user_input,
                    "mode": "agent",
                    "attachments": []
                }, timeout=3600)
                
                # Ensure final newline after response completes
                print()
                
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]\n")
        
        # Clean up session
        await session.destroy()
        
    finally:
        await client.stop()
