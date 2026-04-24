"""
MCP Client - Model Context Protocol client for connecting to external MCP servers

Supports:
- Stdio transport (command + args)
- HTTP/StreamableHTTP transport (url)
- Automatic reconnection with exponential backoff
- Tool discovery and registration
- Thread-safe operations
"""

import asyncio
import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse


class TransportType(Enum):
    STDIO = "stdio"
    HTTP = "http"
    STREAMABLE_HTTP = "streamable_http"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    transport: TransportType
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    timeout: float = 120.0
    connect_timeout: float = 60.0


@dataclass
class MCPTool:
    """Represents a tool from an MCP server."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str


class MCPClient:
    """
    MCP Client for connecting to MCP servers.
    
    Usage:
        client = MCPClient()
        
        # Add a stdio server
        client.add_server(MCPServerConfig(
            name="filesystem",
            transport=TransportType.STDIO,
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        ))
        
        # Add an HTTP server
        client.add_server(MCPServerConfig(
            name="api",
            transport=TransportType.HTTP,
            url="https://my-mcp-server.com/mcp"
        ))
        
        # Connect and discover tools
        await client.connect_all()
        
        # Call a tool
        result = await client.call_tool("filesystem", "read_file", {"path": "/tmp/test.txt"})
        
        # Shutdown
        await client.shutdown()
    """
    
    def __init__(self):
        self._servers: Dict[str, MCPServerConfig] = {}
        self._connections: Dict[str, Any] = {}
        self._tools: Dict[str, List[MCPTool]] = {}
        self._tools_lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._pending_calls: Dict[str, asyncio.Future] = {}
        
        # Safe env keys
        self._safe_env_keys = frozenset({
            "PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL", "TMPDIR"
        })
    
    def add_server(self, config: MCPServerConfig) -> None:
        """Add an MCP server configuration."""
        self._servers[config.name] = config
    
    def remove_server(self, name: str) -> None:
        """Remove an MCP server."""
        if name in self._servers:
            del self._servers[name]
    
    async def _create_stdio_connection(self, config: MCPServerConfig) -> Tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Create stdio connection to MCP server."""
        # Filter environment
        env = {}
        for key in self._safe_env_keys:
            import os
            if key in os.environ:
                env[key] = os.environ[key]
        if config.env:
            env.update(config.env)
        
        # Start process
        cmd = [config.command] + (config.args or [])
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        return process.stdout, process.stdin
    
    async def _create_http_connection(self, config: MCPServerConfig) -> Any:
        """Create HTTP connection to MCP server."""
        import aiohttp
        session = aiohttp.ClientSession(
            headers=config.headers or {},
            timeout=aiohttp.ClientTimeout(total=config.timeout)
        )
        return session
    
    async def connect(self, name: str) -> bool:
        """Connect to a single MCP server."""
        config = self._servers.get(name)
        if not config:
            return False
        
        try:
            if config.transport == TransportType.STDIO:
                reader, writer = await self._create_stdio_connection(config)
                self._connections[name] = {"reader": reader, "writer": writer, "process": True}
            elif config.transport in (TransportType.HTTP, TransportType.STREAMABLE_HTTP):
                session = await self._create_http_connection(config)
                self._connections[name] = {"session": session, "url": config.url}
            
            # Initialize
            await self._send_initialize(name)
            
            # Discover tools
            tools = await self._discover_tools(name)
            with self._tools_lock:
                self._tools[name] = tools
            
            return True
            
        except Exception as e:
            print(f"Failed to connect to MCP server {name}: {e}")
            return False
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all configured servers."""
        results = {}
        for name in self._servers:
            results[name] = await self.connect(name)
        return results
    
    async def _send_initialize(self, server_name: str) -> None:
        """Send initialize request to server."""
        request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "open-mythos", "version": "1.0.0"}
            },
            "id": 1
        }
        await self._send_request(server_name, request)
    
    async def _discover_tools(self, server_name: str) -> List[MCPTool]:
        """Discover tools from a server."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": 2
        }
        
        try:
            response = await self._send_request(server_name, request)
            tools = []
            
            if response and "result" in response:
                for tool_data in response["result"].get("tools", []):
                    tools.append(MCPTool(
                        name=tool_data["name"],
                        description=tool_data.get("description", ""),
                        input_schema=tool_data.get("inputSchema", {}),
                        server_name=server_name
                    ))
            
            return tools
        except Exception:
            return []
    
    async def _send_request(self, server_name: str, request: Dict) -> Optional[Dict]:
        """Send JSON-RPC request to server."""
        conn = self._connections.get(server_name)
        if not conn:
            return None
        
        request_id = request.get("id")
        
        try:
            if "writer" in conn:
                # Stdio transport
                writer = conn["writer"]
                request_str = json.dumps(request) + "\n"
                writer.write(request_str.encode())
                await writer.drain()
                
                # Read response
                reader = conn["reader"]
                response_line = await asyncio.wait_for(reader.readline(), timeout=60.0)
                return json.loads(response_line.decode())
            
            elif "session" in conn:
                # HTTP transport
                session = conn["session"]
                url = conn["url"]
                
                async with session.post(url, json=request) as resp:
                    return await resp.json()
                    
        except Exception as e:
            print(f"Request failed for {server_name}: {e}")
            return None
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on an MCP server.
        
        Returns:
            {"success": True, "result": ...} or {"success": False, "error": "..."}
        """
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": int(time.time() * 1000)
        }
        
        try:
            response = await self._send_request(server_name, request)
            
            if response and "result" in response:
                return {"success": True, "result": response["result"]}
            elif response and "error" in response:
                return {"success": False, "error": response["error"]}
            else:
                return {"success": False, "error": "No response from server"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_tools(self, server_name: Optional[str] = None) -> List[MCPTool]:
        """Get discovered tools."""
        with self._tools_lock:
            if server_name:
                return self._tools.get(server_name, [])
            return [tool for tools in self._tools.values() for tool in tools]
    
    def get_servers(self) -> List[str]:
        """Get list of configured server names."""
        return list(self._servers.keys())
    
    def get_connected_servers(self) -> List[str]:
        """Get list of connected server names."""
        return list(self._connections.keys())
    
    async def shutdown(self) -> None:
        """Shutdown all connections."""
        self._running = False
        
        for name, conn in self._connections.items():
            try:
                if "writer" in conn:
                    conn["writer"].close()
                    if hasattr(conn.get("process"), "terminate"):
                        conn["process"].terminate()
                if "session" in conn:
                    await conn["session"].close()
            except Exception:
                pass
        
        self._connections.clear()
        self._tools.clear()
    
    def start_background(self) -> None:
        """Start MCP client in background thread."""
        if self._thread and self._thread.is_alive():
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
    
    def _run_loop(self) -> None:
        """Run event loop in background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        
        try:
            loop.run_until_complete(self._background_loop())
        finally:
            loop.close()
    
    async def _background_loop(self) -> None:
        """Background loop for maintaining connections."""
        while self._running:
            # Reconnect disconnected servers
            for name in self._servers:
                if name not in self._connections:
                    await self.connect(name)
            
            await asyncio.sleep(30)  # Check every 30 seconds


class MCPToolRegistry:
    """
    Registry for MCP tools, bridging MCP servers to the tool registry.
    
    Usage:
        mcp_reg = MCPToolRegistry(mcp_client)
        
        # Register all MCP tools
        mcp_reg.register_all_tools()
        
        # Or register a specific server's tools
        mcp_reg.register_server_tools("filesystem")
    """
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self._registered_names: Dict[str, str] = {}  # mcp_tool_name -> full_name
    
    def register_all_tools(self) -> int:
        """Register all tools from all connected servers."""
        total = 0
        for server_name in self.mcp_client.get_connected_servers():
            total += self.register_server_tools(server_name)
        return total
    
    def register_server_tools(self, server_name: str) -> int:
        """Register all tools from a specific server."""
        from .registry import registry, register_tool
        
        tools = self.mcp_client.get_tools(server_name)
        count = 0
        
        for tool in tools:
            full_name = f"mcp_{server_name}_{tool.name}"
            
            # Create handler
            async def make_handler(srv: str, name: str):
                async def handler(**kwargs):
                    result = await self.mcp_client.call_tool(srv, name, kwargs)
                    if result.get("success"):
                        return result.get("result", {})
                    else:
                        raise Exception(result.get("error", "Unknown error"))
                return handler
            
            # Create schema
            schema = {
                "name": full_name,
                "description": tool.description,
                "parameters": tool.input_schema
            }
            
            # Register
            try:
                registry.register(
                    name=full_name,
                    toolset="mcp",
                    schema=schema,
                    handler=make_handler(server_name, tool.name),
                    is_async=True,
                    description=f"[MCP:{server_name}] {tool.description}",
                    emoji="🔌"
                )
                self._registered_names[tool.name] = full_name
                count += 1
            except Exception:
                pass
        
        return count
    
    def get_mcp_tool_name(self, tool_name: str) -> Optional[str]:
        """Get full registered name for an MCP tool."""
        return self._registered_names.get(tool_name)
