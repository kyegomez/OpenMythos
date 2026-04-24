# Changelog

All notable changes to OpenMythos will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-01

### Added

#### Core Memory System
- ThreeLayerMemorySystem - Three-layer memory (Working, ShortTerm, LongTerm)
- MemoryManager - Memory management with BuiltinFirst + OneExternal rules
- MemoryEntry - Memory entry with importance decay
- MemoryLayer - Layer enum (WORKING, SHORT_TERM, LONG_TERM)

#### Context Management
- ContextEngine - Context management with compression
- Six compression strategies: NONE, SUMMARIZE, REFERENCE, PRIORITY, HYBRID, SELECTIVE
- Context pressure monitoring and automatic compression
- Caching support for repeated queries

#### Auto-Evolution System
- AutoEvolution - Continuous improvement through task analysis
- PatternDetector - Detects recurring patterns in approaches
- SkillImprover - Tracks skill performance and suggests improvements
- PeriodicReminderSystem - Scheduled skill reviews
- TaskOutcomeTracker - Records task success/failure for analysis

#### Tool Ecosystem
- ToolRegistry - Thread-safe tool registration and discovery
- ToolExecutor - Safe tool execution with timeout/rate limiting
- MCPClient - Model Context Protocol client for external servers
- Built-in tools:
  - File tools: read_file, write_file, search_files, list_dir
  - Web tools: web_search, web_extract
  - Math tools: add, subtract, multiply, divide, sqrt, power, abs, round, min, max, clamp
  - DateTime tools: now, parse, add, diff, format, validate
  - JSON tools: parse, stringify, get, set, merge, validate, pretty, minify

#### CLI Interface
- MythosCLI - Interactive terminal interface
- AIAgentLoop - Async agent loop with token budget
- Multi-provider support: OpenAI, Anthropic, OpenRouter, Ollama
- ANSI color formatting with markdown rendering
- Built-in commands: /exit, /clear, /help, /model, /memory, /stats, /tools

#### Integration
- MythosIntegration - Bridge to existing OpenMythos systems
- SkillMigration - Migrate skills between systems
- Skill versioning and metadata support

#### Persistence
- SQLiteStore - ACID-compliant SQLite storage
- JSONStore - Simple JSON file storage
- Skill persistence with metadata

#### Web Dashboard
- MythosDashboard - FastAPI-based monitoring interface
- Real-time stats: memory, context, tools, evolution
- HTML dashboard with auto-refresh

#### Testing & Benchmarks
- Comprehensive test suite with pytest
- Performance benchmarks for memory, context, tools
- Test coverage for core modules

#### Project Configuration
- pyproject.toml - Modern Python project setup
- setup.py - pip installation support
- requirements.txt - Dependency management
- Code quality tools: black, ruff, mypy

### Features

1. **Three-Layer Memory**
   - Working (30 min TTL) - Current session
   - Short-Term (7 days TTL) - Recent context
   - Long-Term (365 days TTL) - Persistent knowledge

2. **Context Compression**
   - Adaptive compression based on pressure
   - Six compression strategies
   - Cache for repeated queries

3. **Auto-Evolution**
   - Pattern detection in approaches
   - Auto-generation of skills from patterns
   - Periodic skill review reminders

4. **Tool Safety**
   - Path traversal protection
   - Rate limiting
   - Timeout enforcement
   - Sandboxed execution

5. **Multi-Provider Support**
   - OpenAI (GPT-4, GPT-3.5)
   - Anthropic (Claude)
   - OpenRouter (Mixed models)
   - Ollama (Local models)

## [0.1.0] - 2024-01-01

### Added
- Initial release
- Basic memory system prototype
- Tool execution framework
- CLI interface
