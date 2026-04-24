# Contributing to OpenMythos

We welcome contributions! Here's how you can help.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/open-mythos.git
   cd open-mythos
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```
4. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### Code Style

We use several tools to maintain code quality:

```bash
# Format code with black
black open_mythos/

# Lint with ruff
ruff check open_mythos/

# Type check with mypy
mypy open_mythos/
```

### Running Tests

```bash
# Run all tests
pytest open_mythos/tests/ -v

# Run with coverage
pytest --cov=open_mythos open_mythos/tests/

# Run specific test file
pytest open_mythos/tests/test_memory.py -v
```

### Running Benchmarks

```bash
# Run performance benchmarks
python benchmarks/run_benchmarks.py
```

## Project Structure

```
open_mythos/
├── memory/           # Three-layer memory system
├── context/          # Context compression
├── evolution/       # Auto-evolution
├── tools/           # Tool ecosystem
│   └── builtins/    # Built-in tools
├── cli/             # CLI interface
├── integration/     # Integration layer
├── web/             # Web dashboard
├── persistence/     # Storage backends
├── tests/           # Test suite
├── benchmarks/      # Performance benchmarks
└── examples/        # Example scripts
```

## Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our code style:
   - Use type hints where possible
   - Add docstrings to public functions/classes
   - Keep functions small and focused

3. Add tests for your changes:
   ```bash
   # Add tests to appropriate test file
   pytest open_mythos/tests/ -v
   ```

4. Run the full quality check:
   ```bash
   black open_mythos/
   ruff check open_mythos/ --fix
   mypy open_mythos/
   pytest open_mythos/tests/ -v
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```

6. Push and create a pull request

## Pull Request Guidelines

- Fill out the PR template completely
- Reference any related issues
- Ensure all tests pass
- Update documentation if needed
- Keep changes focused and atomic

## Adding New Tools

To add a new built-in tool:

1. Create or edit `open_mythos/tools/builtins/your_tool.py`
2. Implement the tool function
3. Register with `register_tool()` decorator
4. Add the tool to `__all__` in builtins `__init__.py`
5. Add tests

Example:
```python
from open_mythos.tools.registry import register_tool, tool_result

def my_tool(arg1: str, arg2: int) -> str:
    """Description of what the tool does."""
    return tool_result({"result": f"{arg1} {arg2}"})

register_tool(
    name="my_tool",
    toolset="custom",
    schema={
        "name": "my_tool",
        "description": "What it does",
        "parameters": {
            "type": "object",
            "properties": {
                "arg1": {"type": "string"},
                "arg2": {"type": "integer"}
            },
            "required": ["arg1", "arg2"]
        }
    },
    handler=my_tool,
    description="Short description",
    emoji="[✨]"
)
```

## Adding New Providers

To add a new LLM provider:

1. Create `open_mythos/cli/providers/your_provider.py`
2. Inherit from `LLMProvider` base class
3. Implement required methods:
   - `complete()` - Synchronous completion
   - `complete_async()` - Async completion (optional)
4. Register in `open_mythos/cli/provider.py` factory

## Documentation

- Update `README.md` for user-facing changes
- Add docstrings to new public APIs
- Update `docs/API.md` if needed

## Questions?

Feel free to open an issue for questions or discussion.
