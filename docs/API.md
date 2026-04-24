# OpenMythos API Reference

## EnhancedHermes

Main entry point for the enhanced agent system.

```python
from open_mythos import EnhancedHermes

hermes = EnhancedHermes()
```

### Methods

#### Memory Operations

##### `hermes.remember(content, layer="working", importance=0.5, tags=None)`

Add a memory to the specified layer.

**Parameters:**
- `content` (str): Memory content
- `layer` (str): "working", "short_term", or "long_term"
- `importance` (float): 0.0 to 1.0
- `tags` (List[str]): Optional tags

##### `hermes.recall(query, limit=10)`

Search memories across all layers.

**Parameters:**
- `query` (str): Search query
- `limit` (int): Max results

**Returns:** List of matching memory entries

##### `hermes.learn_skill(skill_name, content, metadata=None)`

Learn a new skill in long-term memory.

**Parameters:**
- `skill_name` (str): Skill identifier
- `content` (str): SKILL.md content
- `metadata` (Dict): Optional metadata

##### `hermes.list_skills()`

List all learned skills.

**Returns:** List of skill names

#### Context Operations

##### `hermes.add_to_context(role, content, importance=0.5)`

Add a message to conversation context.

**Parameters:**
- `role` (str): "user", "assistant", or "system"
- `content` (str): Message content
- `importance` (float): 0.0 to 1.0

##### `hermes.get_context(max_tokens=4000)`

Get compressed context for prompts.

**Parameters:**
- `max_tokens` (int): Max tokens to return

**Returns:** List of message dicts

##### `hermes.check_context_pressure(max_tokens)`

Check context compression pressure.

**Returns:** Dict with pressure metrics

##### `hermes.compress_context(strategy=None)`

Manually compress context.

**Parameters:**
- `strategy` (str): "summarize", "reference", "priority", "hybrid", or "selective"

#### Evolution Operations

##### `hermes.record_task(task, success, approach, quality_score=0.5, duration_seconds=0, tools_used=None, skills_used=None)`

Record a task outcome for evolution.

**Parameters:**
- `task` (str): Task description
- `success` (bool): Whether task succeeded
- `approach` (str): Approach used
- `quality_score` (float): 0.0 to 1.0
- `duration_seconds` (float): Task duration
- `tools_used` (List[str]): Tools invoked
- `skills_used` (List[str]): Skills invoked

##### `hermes.check_new_skills()`

Check if patterns suggest new skills.

**Returns:** List of new skill suggestions

##### `hermes.get_reminders()`

Get pending periodic reminders.

**Returns:** List of reminder messages

#### Tool Operations

##### `hermes.initialize_tools(builtin_only=True)`

Initialize the tool system.

**Parameters:**
- `builtin_only` (bool): Only load built-in tools

##### `hermes.execute_tool(tool_name, args)`

Execute a registered tool.

**Parameters:**
- `tool_name` (str): Tool name
- `args` (Dict): Tool arguments

**Returns:** ExecutionResult

##### `hermes.get_available_tools()`

List all available tools.

**Returns:** List of tool names

##### `hermes.list_toolsets()`

List all toolsets and their tools.

**Returns:** Dict of toolset info

#### Self-Check Operations

##### `hermes.run_self_check(force=False)`

Run the self-check system.

**Parameters:**
- `force` (bool): Force recheck

**Returns:** Dict with check results

##### `hermes.get_system_status()`

Get human-readable system status.

**Returns:** Status string

#### Lifecycle

##### `hermes.on_turn_start(message)`

Called at turn start.

##### `hermes.on_turn_end(message, response)`

Called at turn end.

##### `hermes.on_session_end(messages)`

Called at session end.

##### `hermes.get_full_stats()`

Get all system statistics.

**Returns:** Dict with stats

---

## Memory System

### ThreeLayerMemorySystem

```python
from open_mythos.memory import ThreeLayerMemorySystem

system = ThreeLayerMemorySystem()
```

#### Layers

- `MemoryLayer.WORKING`: Current session, 50 entries, 4000 tokens, 30min TTL
- `MemoryLayer.SHORT_TERM`: Recent sessions, 500 entries, 7 day TTL
- `MemoryLayer.LONG_TERM`: Persistent, 10000 entries, 365 day TTL

#### Methods

##### `write_to_layer(layer, content, importance, tags, source)`

Write to a specific layer.

##### `search_unified(query)`

Search across all layers.

##### `get_context_for_prompt(max_tokens)`

Get formatted context for prompts.

##### `get_stats()`

Get memory statistics.

### MemoryManager

```python
from open_mythos.memory import MemoryManager

manager = MemoryManager(memory_system)
```

#### Rules

- **BuiltinFirst**: ThreeLayerMemory cannot be removed
- **OneExternal**: Only one external provider at a time

---

## Context Engine

```python
from open_mythos.context import ContextEngine, CompressionStrategy

engine = ContextEngine()
```

### Compression Strategies

| Strategy | Description |
|----------|-------------|
| `NONE` | No compression |
| `SUMMARIZE` | Summarize old messages |
| `REFERENCE` | Keep references, drop content |
| `PRIORITY` | Keep high-importance messages |
| `HYBRID` | Combine strategies |
| `SELECTIVE` | Smart selection |

### Methods

##### `add_message(role, content, importance=0.5)`

Add a message.

##### `compress(strategy=None, max_tokens=None)`

Compress context.

##### `check_pressure(max_tokens)`

Check compression pressure.

##### `get_context(max_tokens)`

Get context with optional compression.

---

## Tool System

### ToolRegistry

```python
from open_mythos.tools import registry, register_tool
```

##### `register_tool(name, toolset, schema, handler, ...)`

Register a tool.

##### `registry.get_all_tool_names()`

List all tools.

##### `registry.dispatch(tool_name, args)`

Execute a tool.

### ToolExecutor

```python
from open_mythos.tools import ToolExecutor, ExecutionPolicy

executor = ToolExecutor(ExecutionPolicy(
    max_timeout_seconds=120,
    max_result_size=100000,
    max_concurrent=5
))
```

##### `execute(tool_name, args)`

Execute a tool synchronously.

##### `execute_async(tool_name, args)`

Execute a tool asynchronously.

### MCP Client

```python
from open_mythos.tools import MCPClient, MCPServerConfig, TransportType

client = MCPClient()
client.add_server(MCPServerConfig(
    name="my_server",
    transport=TransportType.STDIO,
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
))
await client.connect_all()
```

---

## CLI

### MythosCLI

```python
from open_mythos.cli import MythosCLI

cli = MythosCLI()
cli.run()
```

### Commands

| Command | Description |
|---------|-------------|
| `/exit`, `/quit` | Exit CLI |
| `/clear` | Clear history |
| `/help` | Show help |
| `/model [name]` | Show/change model |
| `/memory` | Memory status |
| `/stats` | Statistics |
| `/tools` | List tools |

### LLM Providers

```python
from open_mythos.cli import create_provider, ProviderConfig

# OpenAI
provider = create_provider("openai", ProviderConfig(
    model="gpt-4",
    api_key="..."
))

# Anthropic
provider = create_provider("anthropic", ProviderConfig(
    model="claude-3",
    api_key="..."
))

# OpenRouter
provider = create_provider("openrouter", ProviderConfig(
    model="anthropic/claude-3",
    api_key="..."
))

# Ollama (local)
provider = create_provider("ollama", ProviderConfig(
    model="llama2",
    base_url="http://localhost:11434"
))
```

---

## Integration

### MythosIntegration

```python
from open_mythos.integration import MythosIntegration

integration = MythosIntegration()
```

##### `load_mythos_skills()`

Load skills from ~/.hermes/skills.

##### `save_skill(name, content)`

Save a skill.

##### `create_skill_from_evolution(name, content, metadata)`

Create skill from evolution output.

##### `get_integrated_context(hermes, max_tokens)`

Get combined context from multiple sources.
