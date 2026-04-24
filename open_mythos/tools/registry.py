"""
Tool Registry - Hermes-Style Central Tool Registry

Each tool file calls registry.register() at module level to declare:
- Schema: Tool input parameters (JSON Schema)
- Handler: Function to execute
- Toolset: Grouping (core, file, web, mcp, etc.)
- Check function: Availability validation

Import Chain (Circular-Import Safe):
    registry.py (no imports from other tool files)
        ↑
    tools/*.py (import from registry at module level)
        ↑
    tool_execution.py (imports registry + triggers discovery)
"""

import ast
import inspect
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum


@dataclass
class ToolEntry:
    """
    Metadata container for a registered tool.
    
    Uses __slots__ for memory efficiency.
    """
    name: str
    toolset: str
    schema: Dict[str, Any]
    handler: Callable
    check_fn: Optional[Callable[[], bool]] = None
    requires_env: Optional[List[str]] = None
    is_async: bool = False
    description: str = ""
    emoji: str = "⚡"
    max_result_size_chars: Optional[int] = None
    danger_level: int = 0  # 0=safe, 1=elevated, 2=dangerous
    
    def is_available(self) -> bool:
        """Check if tool is available."""
        if self.check_fn is None:
            return True
        try:
            return self.check_fn()
        except Exception:
            return False


class Toolset(Enum):
    """Built-in toolsets."""
    CORE = "core"
    FILE = "file"
    WEB = "web"
    TERMINAL = "terminal"
    MCP = "mcp"
    DELEGATION = "delegation"
    MEMORY = "memory"
    CUSTOM = "custom"


class ToolRegistry:
    """
    Singleton registry for all tools.
    
    Thread-safe operations with RLock.
    
    Usage:
        from tools.registry import registry, register_tool, tool_result, tool_error
        
        # Register a tool
        register_tool(
            name="read_file",
            toolset=Toolset.FILE,
            schema={
                "name": "read_file",
                "description": "Read file contents",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            },
            handler=read_file_impl,
            description="Read contents of a file",
            emoji="📄"
        )
        
        # Get tool info
        schema = registry.get_schema("read_file")
        
        # Execute tool
        result = registry.dispatch("read_file", {"path": "/tmp/test.txt"})
    """
    
    _instance: Optional["ToolRegistry"] = None
    _lock = threading.RLock()
    
    def __new__(cls) -> "ToolRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance
    
    def _init(self) -> None:
        """Initialize registry state."""
        self._tools: Dict[str, ToolEntry] = {}
        self._toolset_checks: Dict[str, Callable[[], bool]] = {}
        self._toolset_aliases: Dict[str, str] = {}
        self._tool_to_toolset: Dict[str, str] = {}
        self._lock = threading.RLock()
    
    def register(
        self,
        name: str,
        toolset: str,
        schema: Dict[str, Any],
        handler: Callable,
        check_fn: Optional[Callable[[], bool]] = None,
        requires_env: Optional[List[str]] = None,
        is_async: bool = False,
        description: str = "",
        emoji: str = "⚡",
        max_result_size_chars: Optional[int] = None,
        danger_level: int = 0
    ) -> None:
        """
        Register a tool.
        
        Args:
            name: Tool name (unique identifier)
            toolset: Toolset name (core, file, web, etc.)
            schema: JSON Schema for tool input
            handler: Function to execute
            check_fn: Optional availability check
            requires_env: Required environment variables
            is_async: Handler is async function
            description: Human-readable description
            emoji: Emoji icon
            max_result_size_chars: Max result size limit
            danger_level: 0=safe, 1=elevated, 2=dangerous
        """
        with self._lock:
            # Check for shadowing (cross-type overwrites rejected)
            existing = self._tools.get(name)
            if existing:
                # MCP-to-MCP is allowed (for server refresh)
                if existing.toolset == "mcp" and toolset == "mcp":
                    pass  # Allow overwrite
                else:
                    print(f"Tool '{name}' already registered as {existing.toolset}, {toolset} registration rejected")
                    return
            
            # Create entry
            entry = ToolEntry(
                name=name,
                toolset=toolset,
                schema=schema,
                handler=handler,
                check_fn=check_fn,
                requires_env=requires_env,
                is_async=is_async,
                description=description,
                emoji=emoji,
                max_result_size_chars=max_result_size_chars,
                danger_level=danger_level
            )
            
            self._tools[name] = entry
            self._tool_to_toolset[name] = toolset
            
            # Register toolset check function if not present
            if toolset not in self._toolset_checks:
                self._toolset_checks[toolset] = lambda: True  # Default: always available
    
    def deregister(self, name: str) -> None:
        """Remove a tool from registry."""
        with self._lock:
            if name in self._tools:
                toolset = self._tools[name].toolset
                del self._tools[name]
                del self._tool_to_toolset[name]
                
                # Clean up toolset check if no tools remain
                if not any(e.toolset == toolset for e in self._tools.values()):
                    self._toolset_checks.pop(toolset, None)
    
    def dispatch(self, name: str, args: dict, **kwargs) -> str:
        """
        Execute a tool by name.
        
        Returns JSON string result.
        All exceptions are caught and returned as error format.
        """
        entry = self._tools.get(name)
        if not entry:
            return tool_error(f"Tool '{name}' not found")
        
        if not entry.is_available():
            return tool_error(f"Tool '{name}' is not available")
        
        try:
            # Execute handler
            if entry.is_async:
                import asyncio
                result = asyncio.run(entry.handler(**args, **kwargs))
            else:
                result = entry.handler(**args, **kwargs)
            
            # Handle None result
            if result is None:
                result = {"success": True}
            
            # Ensure string
            if not isinstance(result, str):
                import json
                return json.dumps(result)
            return result
            
        except Exception as e:
            return tool_error(str(e))
    
    def get_entry(self, name: str) -> Optional[ToolEntry]:
        """Get tool entry by name."""
        return self._tools.get(name)
    
    def get_all_tool_names(self) -> List[str]:
        """Get sorted list of all tool names."""
        return sorted(self._tools.keys())
    
    def get_tool_names_for_toolset(self, toolset: str) -> List[str]:
        """Get tools in a specific toolset."""
        return sorted([
            name for name, ts in self._tool_to_toolset.items()
            if ts == toolset
        ])
    
    def get_registered_toolset_names(self) -> List[str]:
        """Get all toolset names."""
        return sorted(set(self._tool_to_toolset.values()))
    
    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get raw schema dict for a tool."""
        entry = self._tools.get(name)
        return entry.schema if entry else None
    
    def get_toolson(self, name: str) -> Optional[str]:
        """Get toolset for a tool."""
        return self._tool_to_toolset.get(name)
    
    def get_emoji(self, name: str, default: str = "⚡") -> str:
        """Get emoji for a tool."""
        entry = self._tools.get(name)
        return entry.emoji if entry else default
    
    def get_max_result_size(self, name: str, default: Optional[int] = None) -> Optional[int]:
        """Get max result size for a tool."""
        entry = self._tools.get(name)
        return entry.max_result_size_chars if entry else default
    
    def get_tool_to_toolset_map(self) -> Dict[str, str]:
        """Get {tool_name: toolset_name} mapping."""
        return self._tool_to_toolset.copy()
    
    def is_toolset_available(self, toolset: str) -> bool:
        """Check if a toolset is available."""
        if toolset not in self._toolset_checks:
            return True
        try:
            return self._toolset_checks[toolset]()
        except Exception:
            return False
    
    def check_toolset_requirements(self) -> Dict[str, bool]:
        """Returns {toolset: available_bool} for every toolset."""
        return {
            ts: self.is_toolset_available(ts)
            for ts in set(self._tool_to_toolset.values())
        }
    
    def get_available_toolsets(self) -> Dict[str, Dict[str, Any]]:
        """Get toolset metadata for UI display."""
        result = {}
        for ts in set(self._tool_to_toolset.values()):
            tools = self.get_tool_names_for_toolset(ts)
            result[ts] = {
                "available": self.is_toolset_available(ts),
                "tool_count": len(tools),
                "tools": tools[:5]  # Preview
            }
        return result
    
    def check_tool_availability(self, quiet: bool = False) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Check all tools availability.
        
        Returns (available_toolsets, unavailable_info).
        """
        available = []
        unavailable = []
        
        for name, entry in self._tools.items():
            if entry.is_available():
                available.append(name)
            else:
                unavailable.append({
                    "name": name,
                    "toolset": entry.toolset,
                    "reason": "check_fn failed"
                })
        
        return available, unavailable
    
    def discover_builtin_tools(self, tools_dir: Optional[str] = None) -> List[str]:
        """
        Discover and import built-in tool modules.
        
        Uses AST parsing to detect registry.register() calls.
        Excludes __init__.py, registry.py, mcp_tool.py.
        """
        import os
        import importlib.util
        
        discovered = []
        tools_path = tools_dir or os.path.join(os.path.dirname(__file__))
        
        for filename in os.listdir(tools_path):
            if not filename.endswith(".py"):
                continue
            if filename in ("__init__.py", "registry.py", "mcp_tool.py", "execution.py"):
                continue
            
            module_name = filename[:-3]
            module_path = os.path.join(tools_path, filename)
            
            # Use AST to check for register() calls
            try:
                with open(module_path, "r") as f:
                    tree = ast.parse(f.read())
                
                has_register = False
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name) and node.func.id == "register":
                            has_register = True
                            break
                
                if has_register:
                    # Import the module to trigger registration
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        discovered.append(module_name)
                        
            except Exception as e:
                if not quiet:
                    print(f"Failed to discover {module_name}: {e}")
        
        return discovered


# Singleton instance
registry = ToolRegistry()


# Convenience functions

def tool_error(message: str, **extra) -> str:
    """Return JSON error string."""
    import json
    data = {"error": message, **extra}
    return json.dumps(data)


def tool_result(data: Any = None, **kwargs) -> str:
    """Return JSON result string."""
    import json
    if data is None:
        data = {}
    if kwargs:
        data = {**data, **kwargs}
    if isinstance(data, str):
        return data
    return json.dumps(data)


def register_tool(
    name: str,
    toolset: str,
    schema: Dict[str, Any],
    handler: Callable,
    check_fn: Optional[Callable[[], bool]] = None,
    requires_env: Optional[List[str]] = None,
    is_async: bool = False,
    description: str = "",
    emoji: str = "⚡",
    max_result_size_chars: Optional[int] = None,
    danger_level: int = 0
) -> None:
    """Convenience wrapper for registry.register()."""
    registry.register(
        name=name,
        toolset=toolset,
        schema=schema,
        handler=handler,
        check_fn=check_fn,
        requires_env=requires_env,
        is_async=is_async,
        description=description,
        emoji=emoji,
        max_result_size_chars=max_result_size_chars,
        danger_level=danger_level
    )


def get_registry() -> ToolRegistry:
    """Get the singleton registry instance."""
    return registry
