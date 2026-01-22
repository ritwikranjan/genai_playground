# GenAI Playground

A Python CLI for experimenting with AI models and MCP (Model Context Protocol) tools using Azure OpenAI.

## Features

- **Azure OpenAI Integration**: Uses the Azure OpenAI SDK for model inference
- **MCP Tool Support**: Connect multiple MCP tool servers for extended capabilities
- **Flexible Configuration**: Use JSON config files or command-line arguments
- **Streaming Responses**: See AI responses appear token-by-token in real-time
- **Reasoning Display**: View the model's thinking process (for reasoning models like o1, o3, gpt-5)
- **Built-in Tools**:
  - **Web Search**: Search the web using DuckDuckGo (no API key required)
  - **Azure Data Explorer**: Query Kusto databases with KQL

## Project Structure

```file
genai_playground/
├── src/
│   ├── playground.py       # CLI entry point
│   └── client.py           # Core orchestration logic
├── tools/
│   ├── web_search.py       # Web search MCP server
│   └── adx_kusto.py        # Azure Data Explorer MCP server
├── examples/
│   ├── web_search.json     # Example: Web search configuration
│   ├── adx_query.json      # Example: ADX query configuration
│   └── multi_tool.json     # Example: Multiple tools configuration
├── requirements.txt        # Python dependencies
├── .env.template           # Environment variables template
└── README.md               # This file
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy the environment template and fill in your Azure OpenAI credentials:

```bash
cp .env.template .env
```

Edit `.env` with your values:

- `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL
- `AZURE_OPENAI_API_KEY`: Your API key
- `AZURE_OPENAI_DEPLOYMENT`: Your deployment name (e.g., gpt-4o)

## Usage

### Basic Command

```bash
# Activate virtual environment first
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Run with a simple prompt (no tools)
python src/playground.py run -p "What is the capital of France?"

# Run with web search tool
python src/playground.py run -p "What are the latest AI news?" -t tools/web_search.py -v

# Run with a config file
python src/playground.py run --config examples/web_search.json
```

### Interactive Chat Mode

Start a continuous conversation session where you can ask follow-up questions:

```bash
# Start chat with web search tool
python src/playground.py chat -t tools/web_search.py

# Start chat with verbose output to see tool calls
python src/playground.py chat -t tools/web_search.py -v

# Chat with a config file for base settings
python src/playground.py chat --config examples/web_search.json
```

**Chat commands:**

- Type your message and press Enter to send
- Type `clear` to reset the conversation history
- Type `exit` or `quit` to end the session

### CLI Options

```bash
python src/playground.py run --help

Options:
  -c, --config PATH           Path to JSON configuration file
  -s, --system TEXT           System prompt
  -p, --prompt TEXT           User prompt (required)
  -m, --model TEXT            Model/deployment name [default: gpt-4o]
  -t, --tool PATH             Tool script path (can use multiple times)
  --max-iterations INTEGER    Max tool call iterations [default: 10]
  -v, --verbose               Show detailed output
  --azure-endpoint TEXT       Azure OpenAI endpoint
  --azure-api-key TEXT        Azure OpenAI API key
  --azure-deployment TEXT     Azure OpenAI deployment name
  -o, --output PATH           Write response to file
  --raw                       Output raw text without formatting
```

### List Available Tools

```bash
python src/playground.py list-tools tools/web_search.py
```

### Generate Config Template

```bash
python src/playground.py init-config -o my_config.json
```

## Configuration File Format

```json
{
  "system_prompt": "You are a helpful assistant.",
  "user_prompt": "Your question here",
  "model": "gpt-4o",
  "tools": [
    "tools/web_search.py",
    "tools/adx_kusto.py"
  ],
  "max_iterations": 10,
  "verbose": true,
  "stream": true,
  "show_reasoning": true,
  "reasoning_effort": null,
  "azure_endpoint": "https://your-resource.openai.azure.com/",
  "azure_deployment": "gpt-4o"
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|

| `system_prompt` | string | "You are a helpful assistant." | The system prompt for the conversation |
| `user_prompt` | string | "" | The initial user prompt (for single-shot mode) |
| `model` | string | "gpt-4o" | The model name |
| `tools` | array | [] | List of MCP tool configurations |
| `max_iterations` | int | 10 | Maximum tool call iterations |
| `verbose` | bool | false | Enable verbose output |
| `stream` | bool | true | Enable streaming responses |
| `show_reasoning` | bool | true | Display reasoning content from reasoning models |
| `reasoning_effort` | string | null | Reasoning effort level: `"low"`, `"medium"`, or `"high"` |
| `azure_endpoint` | string | - | Azure OpenAI endpoint URL (overrides env) |
| `azure_deployment` | string | - | Deployment name (overrides env) |

## Streaming & Reasoning

### Streaming Responses

By default, chat mode streams responses token-by-token in real-time, giving you immediate feedback as the AI generates its response.

When streaming is enabled, you'll see a status line when chat starts:

```text
[Streaming enabled, reasoning display on]
Type 'exit' or 'quit' to end, 'clear' to reset conversation.

You: What is the capital of France?
```

## Tools

This playground supports **any MCP-compatible tool**. You have full flexibility to:

1. **Use built-in tools** we've included in this repo
2. **Create your own custom MCP servers** for internal/proprietary tools
3. **Use pre-built MCP servers** from the community via `uvx`

### Option 1: Built-in Tools

#### Web Search (`tools/web_search.py`)

A simple example of a custom MCP server we created. Provides web search capabilities using DuckDuckGo:

- `search_web`: Search the web for a query
- `search_news`: Search for recent news articles

No API key required!

**Usage:**

```bash
# As a command-line argument
python src/playground.py run -p "What's the latest AI news?" -t tools/web_search.py

# Or in a config file
{
  "tools": ["tools/web_search.py"]
}
```

### Option 2: Create Your Own MCP Server (In-House Tools)

You can create custom MCP servers for any internal tool, API, or data source. Here's an example of how we built the ADX (Azure Data Explorer) tool:

#### Example: Custom ADX Tool (`tools/adx_kusto.py`)

This is a custom MCP server we created to query Kusto databases:

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder

server = Server("adx-kusto")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="execute_kql",
            description="Execute a KQL query against Azure Data Explorer",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "KQL query to execute"},
                    "database": {"type": "string", "description": "Database name"}
                },
                "required": ["query", "database"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "execute_kql":
        # Connect to ADX and execute query
        result = execute_query(arguments["database"], arguments["query"])
        return [TextContent(type="text", text=result)]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**Usage:**

```bash
python src/playground.py run -p "Show me recent errors" -t tools/adx_kusto.py -v
```

**When to create custom MCP servers:**

- Proprietary APIs or internal services
- Custom data sources
- Tools requiring specific authentication
- Specialized business logic

### Option 3: Use Pre-Built MCP Servers with `uvx`

The MCP ecosystem has many pre-built servers available on PyPI. You can use them directly with `uvx` (no local installation required).

**Prerequisites:** Install `uv` with `pip install uv` or see [uv docs](https://docs.astral.sh/uv/)

#### Example: Azure Kusto MCP (from PyPI)

Instead of our custom tool, you can use the official `azure-kusto-mcp` package:

```json
{
  "tools": [
    {
      "name": "azure-kusto-mcp",
      "command": "uvx",
      "args": ["azure-kusto-mcp"],
      "env": {
        "KUSTO_SERVICE_URI": "https://your-cluster.kusto.windows.net"
      }
    }
  ]
}
```

#### Example: Other Community MCP Servers

You can use any MCP server published to PyPI:

```json
{
  "tools": [
    {
      "name": "filesystem",
      "command": "uvx",
      "args": ["mcp-server-filesystem", "/path/to/allowed/dir"]
    },
    {
      "name": "github",
      "command": "uvx",
      "args": ["mcp-server-github"],
      "env": {
        "GITHUB_TOKEN": "your-token"
      }
    },
    {
      "name": "sqlite",
      "command": "uvx",
      "args": ["mcp-server-sqlite", "path/to/database.db"]
    }
  ]
}
```

**Finding MCP Servers:**

- Browse [PyPI for MCP servers](https://pypi.org/search/?q=mcp-server)
- Check the [MCP Servers Repository](https://github.com/modelcontextprotocol/servers)
- Search for `mcp-server-*` or `*-mcp` packages

### Summary: Choosing Your Approach

| Approach | When to Use | Example |
|----------|-------------|---------|

| **Built-in tools** | Quick start, common use cases | `tools/web_search.py` |
| **Custom MCP server** | Internal APIs, proprietary data, custom logic | `tools/adx_kusto.py` |
| **Pre-built via uvx** | Community tools, standard integrations | `uvx azure-kusto-mcp` |

You can mix and match all three approaches in a single configuration:

```json
{
  "tools": [
    "tools/web_search.py",
    "tools/my_custom_tool.py",
    {
      "name": "github",
      "command": "uvx",
      "args": ["mcp-server-github"],
      "env": { "GITHUB_TOKEN": "..." }
    }
  ]
}
```

## Creating Custom Tools

Create a new Python file in the `tools/` directory following the MCP server pattern:

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("my-tool")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="my_function",
            description="Description of what it does",
            inputSchema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "..."}
                },
                "required": ["param1"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "my_function":
        # Your logic here
        return [TextContent(type="text", text="Result")]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Examples

### Web Search Example

```bash
python src/playground.py run \
  -s "You are a research assistant. Cite your sources." \
  -p "What is the Model Context Protocol?" \
  -t tools/web_search.py \
  -v
```

### ADX Query Example

```bash
python src/playground.py run \
  -s "You are a data analyst. Help query Kusto data." \
  -p "Show me the tables in my database" \
  -t tools/adx_kusto.py \
  -v
```

### Using Config File

```bash
python src/playground.py run --config examples/web_search.json
```

## Understanding Iterations

When using tools, the playground operates in **iterations**. Each iteration is one round-trip with the AI model:

```txt
Iteration 1: User prompt → Model decides to call tool(s) → Returns tool calls
             ↓
             CLI executes tools, sends results back
             ↓
Iteration 2: Model receives results → Calls more tools OR provides final response
             ↓
             (repeat until final response or max_iterations)
```

**Why multiple iterations?**

- The model may call multiple tools in sequence
- It might refine searches based on initial results
- Complex queries may require gathering info from different sources

**Tips to reduce iterations:**

- Use specific, detailed prompts
- In system prompt, instruct the model to "make 1-2 searches maximum then provide answer"
- Set appropriate `max_iterations` (default: 10)

## License

MIT
