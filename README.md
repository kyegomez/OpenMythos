# OpenMythos - Enhanced Hermes-Style Agent

A production-ready agent system inspired by [Hermes Agent](https://github.com/NousResearch/hermes-agent), featuring three-layer memory, context compression, auto-evolution, and a complete tool ecosystem.

## Features

### Memory System
- **Three-Layer Memory**: Working (30min), Short-Term (7 days), Long-Term (365 days)
- **Automatic Decay**: Importance-based memory decay
- **Memory Manager**: BuiltinFirst + OneExternal rules

### Context Compression
- **Six Strategies**: NONE, SUMMARIZE, REFERENCE, PRIORITY, HYBRID, SELECTIVE
- **Adaptive Selection**: Auto-selects based on context pressure
- **Caching**: Prompt caching for efficiency

### Auto-Evolution
- **Pattern Detection**: Detects repeated successful approaches
- **Skill Generation**: Automatically creates skills from patterns
- **Periodic Reminders**: Memory nudges for forgotten insights

### Tool System
- **Central Registry**: Thread-safe tool registration
- **MCP Support**: Connect to Model Context Protocol servers
- **Safe Execution**: Timeout, rate limiting, approval workflows
- **Built-in Tools**: File, Web, Terminal operations

### CLI Interface
- **Interactive Mode**: Full interactive terminal interface
- **Single Prompt Mode**: Run single prompts from command line
- **Multi-Provider**: OpenAI, Anthropic, OpenRouter, Ollama support

## Installation

```bash
pip install -e .
```

## Quick Start

### CLI Usage

```bash
# Interactive mode
python -m open_mythos.cli.main

# Single prompt
python -m open_mythos.cli.main "Fix the bug in auth.py"

# Specify model
python -m open_mythos.cli.main --model gpt-4 "Hello"
```

### Python API

```python
from open_mythos import EnhancedHermes

# Create enhanced agent
hermes = EnhancedHermes()
hermes.initialize_tools()

# Memory
hermes.remember("User prefers TDD", layer="working")
hermes.learn_skill("tdd", "# TDD Skill\n...")

# Execute tools
result = hermes.execute_tool("read_file", {"path": "/tmp/test.txt"})

# Context compression
context = hermes.get_context(max_tokens=4000)

# Record task for evolution
hermes.record_task(
    task="Fixed auth bug",
    success=True,
    approach="Used TDD + mocking",
    quality_score=0.9
)

# Check for new skills
new_skills = hermes.check_new_skills()
```

## Architecture

```
open_mythos/
├── memory/                    # Three-layer memory system
│   ├── three_layer_memory.py  # Core memory implementation
│   └── memory_manager.py      # Memory orchestration
├── context/                   # Context compression
│   └── context_engine.py      # Compression strategies
├── evolution/                 # Auto-evolution
│   ├── auto_evolution.py      # Pattern detection
│   └── evolution_core.py      # Evolution orchestration
├── tools/                     # Tool ecosystem
│   ├── registry.py           # Tool registry
│   ├── execution.py           # Safe execution
│   ├── mcp/                  # MCP client
│   └── builtins/             # Built-in tools
├── cli/                      # CLI interface
│   ├── main.py              # CLI entry
│   ├── agent_loop.py        # Agent orchestration
│   ├── provider.py           # LLM providers
│   └── formatter.py         # Output formatting
├── integration/              # OpenMythos integration
│   └── mythos_integration.py
└── enhanced_hermes.py        # Main entry point
```

## Configuration

### Environment Variables

```bash
# Provider
MYTHOS_PROVIDER=openai          # openai, anthropic, openrouter, ollama
MYTHOS_MODEL=gpt-4             # Model name
MYTHOS_BASE_URL=               # Custom API URL (optional)

# API Keys
OPENAI_API_KEY=                # OpenAI key
ANTHROPIC_API_KEY=             # Anthropic key
```

### Tool Registry

Register tools with the central registry:

```python
from open_mythos.tools import register_tool

def my_handler(arg1: str, arg2: int) -> str:
    return f"{arg1}: {arg2}"

register_tool(
    name="my_tool",
    toolset="custom",
    schema={
        "name": "my_tool",
        "parameters": {...}
    },
    handler=my_handler,
    description="Does something useful",
    emoji="[CUT]",
    danger_level=0  # 0=safe, 1=elevated, 2=dangerous
)
```

### MCP Servers

Connect to MCP servers:

```python
from open_mythos.tools import MCPClient, MCPServerConfig, TransportType

client = MCPClient()

# Stdio server
client.add_server(MCPServerConfig(
    name="filesystem",
    transport=TransportType.STDIO,
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
))

# HTTP server
client.add_server(MCPServerConfig(
    name="api",
    transport=TransportType.HTTP,
    url="https://my-mcp-server.com/mcp"
))

# Connect and discover tools
await client.connect_all()
```

## Development

### Run Tests

```bash
pytest open_mythos/tests/ -v
```

### Project Structure

```
open_mythos/
├── tests/                     # Test suite
├── docs/                      # Documentation
├── open_mythos/              # Main package
│   ├── memory/               # Memory system
│   ├── context/              # Context compression
│   ├── evolution/            # Auto-evolution
│   ├── tools/                # Tool ecosystem
│   ├── cli/                  # CLI interface
│   └── integration/          # OpenMythos integration
```

## CLI Commands

| Command | Description |
|---------|-------------|
| /exit, /quit | Exit the CLI |
| /clear | Clear conversation history |
| /help | Show help |
| /model [name] | Show/change model |
| /memory | Show memory status |
| /stats | Show statistics |
| /tools | List available tools |

## Credits

Inspired by [Hermes Agent](https://github.com/NousResearch/hermes-agent) by NousResearch.

## License

MIT
