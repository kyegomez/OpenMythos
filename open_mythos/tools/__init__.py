"""
Tools System - Hermes-Style Tool Registry and Execution

Provides:
- ToolRegistry: Central registry for all tools
- ToolExecutor: Safe tool execution with timeout/rate limits
- MCPClient: Model Context Protocol support
- Builtin tools: File, Web, Terminal operations

Usage:
    from tools import registry, executor, MCPClient
    
    # Discover built-in tools
    registry.discover_builtin_tools()
    
    # Execute a tool
    result = executor.execute("read_file", {"path": "/tmp/test.txt"})
    
    # Add MCP server
    client = MCPClient()
    client.add_server(MCPServerConfig(name="filesystem", transport=TransportType.STDIO, ...))
"""

from .registry import (
    ToolRegistry,
    ToolEntry,
    Toolset,
    registry,
    register_tool,
    tool_result,
    tool_error,
    get_registry,
)
from .execution import (
    ToolExecutor,
    ExecutionResult,
    ExecutionPolicy,
    ExecutionStatus,
    RateLimiter,
    ApprovalCallback,
    ToolExecutionMonitor,
)
from .mcp import MCPClient, MCPServerConfig, MCPTool, MCPToolRegistry, TransportType

__all__ = [
    # Registry
    "ToolRegistry",
    "ToolEntry",
    "Toolset",
    "registry",
    "register_tool",
    "tool_result",
    "tool_error",
    "get_registry",
    # Execution
    "ToolExecutor",
    "ExecutionResult",
    "ExecutionPolicy",
    "ExecutionStatus",
    "RateLimiter",
    "ApprovalCallback",
    "ToolExecutionMonitor",
    # MCP
    "MCPClient",
    "MCPServerConfig",
    "MCPTool",
    "MCPToolRegistry",
    "TransportType",
]
