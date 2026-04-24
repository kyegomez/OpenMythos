"""
Enhanced Hermes System - Full Integration

Integrates all enhancements:
1. Three-Layer Memory System
2. Context Compression Engine
3. Auto-Evolution System
4. Self-Check System
5. Tool Registry & Execution
6. MCP Client

Usage:
    from enhanced_hermes import EnhancedHermes
    
    hermes = EnhancedHermes()
    
    # Initialize tools
    hermes.initialize_tools()
    
    # Add MCP server
    hermes.add_mcp_server(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    )
    
    # Execute tool
    result = hermes.execute_tool("read_file", {"path": "/tmp/test.txt"})
    
    # Memory operations
    hermes.remember("User prefers TDD", layer="working")
    hermes.learn_skill("tdd", "# TDD Skill\n...")
    
    # Context compression
    context = hermes.get_context(max_tokens=4000)
    
    # Record task for evolution
    hermes.record_task(
        task="Fixed auth bug",
        success=True,
        approach="Used TDD approach",
        quality_score=0.9
    )
"""

from typing import Any, Dict, List, Optional

from .memory import (
    ThreeLayerMemorySystem,
    MemoryManager,
    MemoryLayer,
)
from .context import ContextEngine, CompressionStrategy
from .evolution import AutoEvolution, TaskOutcome
from .config.self_monitor import SelfCheckSystem
from .tools import (
    ToolExecutor,
    ExecutionResult,
    ExecutionPolicy,
    MCPClient,
    MCPServerConfig,
    MCPToolRegistry,
    TransportType,
    registry,
)


class EnhancedHermes:
    """
    Enhanced Hermes System with all improvements.
    
    Combines:
    - Three-layer memory
    - Context compression
    - Auto-evolution
    - Self-check
    - Tool execution
    - MCP integration
    """
    
    def __init__(self):
        # Memory system
        self.memory = ThreeLayerMemorySystem()
        self.memory_manager = MemoryManager(self.memory)
        
        # Context engine
        self.context_engine = ContextEngine()
        
        # Auto-evolution
        self.evolution = AutoEvolution(self.memory_manager)
        
        # Self-check system
        self.self_check = SelfCheckSystem()
        
        # Tool system
        self.tool_executor = ToolExecutor()
        self.mcp_client = MCPClient()
        self.mcp_registry: Optional[MCPToolRegistry] = None
        
        # Session info
        self._session_id: Optional[str] = None
        self._turn_count = 0
        self._tools_initialized = False
    
    # =========================================================================
    # Tool Operations
    # =========================================================================
    
    def initialize_tools(self, builtin_only: bool = True) -> None:
        """
        Initialize tool system.
        
        Args:
            builtin_only: If True, only load built-in tools (no MCP)
        """
        # Discover built-in tools
        registry.discover_builtin_tools()
        
        if not builtin_only:
            # Initialize MCP
            self.mcp_registry = MCPToolRegistry(self.mcp_client)
        
        self._tools_initialized = True
    
    def add_mcp_server(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        url: Optional[str] = None,
        env: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Add an MCP server.
        
        Args:
            name: Server name
            command: Command to run (for stdio transport)
            args: Command arguments
            url: Server URL (for HTTP transport)
            env: Environment variables
            
        Returns:
            True if server was added successfully
        """
        if not self.mcp_client:
            return False
        
        if url:
            config = MCPServerConfig(
                name=name,
                transport=TransportType.HTTP,
                url=url
            )
        else:
            config = MCPServerConfig(
                name=name,
                transport=TransportType.STDIO,
                command=command,
                args=args or [],
                env=env
            )
        
        self.mcp_client.add_server(config)
        return True
    
    async def connect_mcp_servers(self) -> Dict[str, bool]:
        """Connect to all configured MCP servers."""
        if not self.mcp_client:
            return {}
        
        results = await self.mcp_client.connect_all()
        
        # Register MCP tools
        if self.mcp_registry:
            self.mcp_registry.register_all_tools()
        
        return results
    
    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> ExecutionResult:
        """
        Execute a tool.
        
        Args:
            tool_name: Name of tool to execute
            args: Tool arguments
            
        Returns:
            ExecutionResult with status, result, timing
        """
        if not self._tools_initialized:
            self.initialize_tools()
        
        result = self.tool_executor.execute(tool_name, args)
        
        # Record for monitoring
        self.tool_executor.rate_limiter.record(tool_name)
        
        return result
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return registry.get_all_tool_names()
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a tool."""
        return registry.get_schema(tool_name)
    
    def list_toolsets(self) -> Dict[str, Dict[str, Any]]:
        """Get all toolsets and their tools."""
        return registry.get_available_toolsets()
    
    # =========================================================================
    # Memory Operations
    # =========================================================================
    
    def remember(
        self,
        content: str,
        layer: str = "working",
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> None:
        """Add a memory."""
        layer_enum = MemoryLayer(layer)
        self.memory.write_to_layer(
            layer_enum, content, importance, tags or [], "session"
        )
    
    def recall(self, query: str, limit: int = 10) -> List[Any]:
        """Recall memories matching query."""
        from .memory import MemoryQuery
        query_obj = MemoryQuery(
            text=query,
            layers=list(MemoryLayer),
            limit=limit
        )
        return self.memory.search_unified(query_obj)
    
    def learn_skill(self, skill_name: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Learn a new skill."""
        self.memory.long_term.add_skill(skill_name, content, metadata)
    
    def get_skill(self, skill_name: str) -> Optional[Any]:
        """Get a skill."""
        return self.memory.long_term.get_skill(skill_name)
    
    def list_skills(self) -> List[str]:
        """List all skills."""
        return self.memory.long_term.list_skills()
    
    # =========================================================================
    # Context Operations
    # =========================================================================
    
    def add_to_context(self, role: str, content: str, importance: float = 0.5) -> None:
        """Add a message to conversation context."""
        self.context_engine.add_message(role, content, importance)
    
    def get_context(self, max_tokens: int = 4000) -> List[Dict[str, Any]]:
        """Get compressed context for prompt."""
        pressure = self.context_engine.check_pressure(max_tokens)
        if pressure.should_compress:
            self.context_engine.compress(max_tokens=int(max_tokens * 0.9))
        return self.context_engine.get_context(max_tokens)
    
    def check_context_pressure(self, max_tokens: int) -> Dict[str, Any]:
        """Check context pressure."""
        pressure = self.context_engine.check_pressure(max_tokens)
        return {
            "current_tokens": pressure.current_tokens,
            "max_tokens": pressure.max_tokens,
            "pressure_ratio": pressure.pressure_ratio,
            "warnings": pressure.warnings,
            "should_compress": pressure.should_compress,
            "recommended_strategy": pressure.recommended_strategy.value
        }
    
    def compress_context(self, strategy: Optional[str] = None) -> Dict[str, Any]:
        """Manually compress context."""
        strat = CompressionStrategy(strategy) if strategy else None
        result = self.context_engine.compress(strategy=strat)
        return {
            "original_count": result.original_count,
            "compressed_count": result.compressed_count,
            "compression_ratio": result.compression_ratio,
            "strategy": result.strategy.value
        }
    
    # =========================================================================
    # Evolution Operations
    # =========================================================================
    
    def record_task(
        self,
        task: str,
        success: bool,
        approach: str,
        quality_score: float = 0.5,
        duration_seconds: float = 0,
        tools_used: Optional[List[str]] = None,
        skills_used: Optional[List[str]] = None
    ) -> None:
        """Record a task outcome for evolution."""
        outcome = TaskOutcome.SUCCESS if success else TaskOutcome.FAILED
        self.evolution.record_outcome(
            task=task,
            outcome=outcome,
            approach=approach,
            quality_score=quality_score,
            duration_seconds=duration_seconds,
            tools_used=tools_used,
            skills_invoked=skills_used
        )
    
    def check_new_skills(self) -> List[Dict[str, Any]]:
        """Check if any new skills should be created."""
        new_skills = self.evolution.check_for_new_skill()
        return [
            {"name": t.name, "description": t.description, "confidence": t.confidence, "content": c}
            for t, c in new_skills
        ]
    
    def get_reminders(self) -> List[str]:
        """Get any pending reminders."""
        reminders = self.evolution.check_reminders()
        return [r.message for r in reminders]
    
    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        return self.evolution.get_stats()
    
    # =========================================================================
    # Self-Check Operations
    # =========================================================================
    
    def run_self_check(self, force: bool = False) -> Dict[str, Any]:
        """Run self-check system."""
        return self.self_check.run_full_check(force=force)
    
    def get_system_status(self) -> str:
        """Get human-readable system status."""
        return self.self_check.get_status_summary()
    
    # =========================================================================
    # Lifecycle
    # =========================================================================
    
    def on_turn_start(self, message: str) -> None:
        """Called at turn start."""
        self._turn_count += 1
        self.memory_manager.on_turn_start(message)
    
    def on_turn_end(self, message: str, response: str) -> None:
        """Called at turn end."""
        self.memory_manager.on_turn_end(message, response)
    
    def on_session_end(self, messages: List[Dict]) -> None:
        """Called at session end."""
        self.memory_manager.on_session_end(messages)
    
    def get_memory_context(self, max_tokens: int = 2000) -> str:
        """Get formatted memory context for prompt."""
        return self.memory.get_context_for_prompt(max_tokens)
    
    def get_full_stats(self) -> Dict[str, Any]:
        """Get all system statistics."""
        return {
            "memory": self.memory.get_stats(),
            "context": self.context_engine.get_stats(),
            "evolution": self.evolution.get_stats(),
            "tools": {
                "total": len(registry.get_all_tool_names()),
                "toolsets": registry.get_registered_toolset_names(),
                "mcp_servers": self.mcp_client.get_servers() if self.mcp_client else []
            },
            "turns": self._turn_count
        }
    
    async def shutdown(self) -> None:
        """Shutdown all systems."""
        # Shutdown MCP
        if self.mcp_client:
            await self.mcp_client.shutdown()
        
        # Shutdown tool executor
        self.tool_executor.shutdown()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.tool_executor.shutdown()
        except Exception:
            pass
