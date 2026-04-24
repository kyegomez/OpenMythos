"""
OpenMythos - Enhanced Hermes-Style Agent System
"""

__version__ = "1.0.0"
__author__ = "OpenMythos Team"

# Memory system
try:
    from open_mythos.memory.three_layer_memory import (
        ThreeLayerMemorySystem,
        MemoryEntry,
        MemoryLayer,
    )
    from open_mythos.memory.memory_manager import MemoryManager
    from open_mythos.memory.three_layer_memory import MemoryQuery

    _HAS_MEMORY = True
except ImportError:
    _HAS_MEMORY = False

# Context system
try:
    from open_mythos.context.context_engine import (
        ContextEngine,
        ContextMessage,
        CompressionStrategy,
        CompressionResult,
    )
    _HAS_CONTEXT = True
except ImportError:
    _HAS_CONTEXT = False
    ContextEngine = None
    ContextMessage = None
    CompressionStrategy = None
    CompressionResult = None

# Evolution system
try:
    from open_mythos.evolution.auto_evolution import (
        AutoEvolution,
        TaskOutcome,
        TaskRecord,
        SkillTemplate,
        PatternDetector,
        SkillImprover,
        PeriodicReminder,
    )
    from open_mythos.evolution.evolution_core import EvolutionCore

    _HAS_EVOLUTION = True
except ImportError:
    _HAS_EVOLUTION = False
    AutoEvolution = None
    TaskOutcome = None
    TaskRecord = None
    SkillTemplate = None
    PatternDetector = None
    SkillImprover = None
    PeriodicReminder = None
    EvolutionCore = None

# Tool system
try:
    from open_mythos.tools.registry import (
        ToolRegistry,
        register_tool,
        tool_result,
        tool_error,
    )
    from open_mythos.tools.execution import (
        ToolExecutor,
        ExecutionPolicy,
        RateLimiter,
        ExecutionResult,
    )
    from open_mythos.tools.mcp.client import MCPClient, MCPServerConfig
    from open_mythos.tools.builtins import register_all_builtin_tools

    _HAS_TOOLS = True
except ImportError:
    _HAS_TOOLS = False

# CLI system
try:
    from open_mythos.cli.main import MythosCLI
    from open_mythos.cli.agent_loop import AIAgentLoop, AgentConfig
    from open_mythos.cli.provider import (
        LLMProvider,
        ProviderConfig,
        OpenAIProvider,
        AnthropicProvider,
        OpenRouterProvider,
        OllamaProvider,
    )
    from open_mythos.cli.formatter import Formatter, Color

    _HAS_CLI = True
except ImportError:
    _HAS_CLI = False

# Integration
try:
    from open_mythos.integration.mythos_integration import (
        MythosIntegration,
        IntegrationConfig,
        SkillMigration,
    )

    _HAS_INTEGRATION = True
except ImportError:
    _HAS_INTEGRATION = False

# Enhanced main class
try:
    from open_mythos.enhanced_hermes import EnhancedHermes

    _HAS_ENHANCED = True
except ImportError:
    _HAS_ENHANCED = False
    EnhancedHermes = None

# Web dashboard
try:
    from open_mythos.web.dashboard import (
        MythosDashboard,
        HTMLDashboard,
        DashboardData,
    )

    _HAS_WEB = True
except ImportError:
    _HAS_WEB = False
    MythosDashboard = None
    HTMLDashboard = None
    DashboardData = None

# Persistence
try:
    from open_mythos.persistence import (
        SQLiteStore,
        JSONStore,
        SQLiteSkillStore,
        SkillStore,
        get_default_memory_store,
        get_default_skill_store,
    )

    _HAS_PERSISTENCE = True
except ImportError:
    _HAS_PERSISTENCE = False

# Config
try:
    from open_mythos.config.self_monitor import SelfMonitor

    _HAS_CONFIG = True
except ImportError:
    _HAS_CONFIG = False
    SelfMonitor = None

__all__ = [
    "__version__",
    # Memory
    "ThreeLayerMemorySystem",
    "MemoryEntry",
    "MemoryLayer",
    "MemoryManager",
    "MemoryQuery",
    # Context
    "ContextEngine",
    "ContextMessage",
    "CompressionStrategy",
    "CompressionResult",
    # Evolution
    "AutoEvolution",
    "TaskOutcome",
    "TaskRecord",
    "SkillTemplate",
    "PatternDetector",
    "SkillImprover",
    "PeriodicReminder",
    "EvolutionCore",
    # Tools
    "ToolRegistry",
    "register_tool",
    "tool_result",
    "tool_error",
    "ToolExecutor",
    "ExecutionPolicy",
    "RateLimiter",
    "ExecutionResult",
    "MCPClient",
    "MCPServerConfig",
    "register_all_builtin_tools",
    # CLI
    "MythosCLI",
    "AIAgentLoop",
    "AgentConfig",
    "LLMProvider",
    "ProviderConfig",
    "OpenAIProvider",
    "AnthropicProvider",
    "OpenRouterProvider",
    "OllamaProvider",
    "Formatter",
    "Color",
    # Integration
    "MythosIntegration",
    "IntegrationConfig",
    "SkillMigration",
    # Main
    "EnhancedHermes",
    # Web
    "MythosDashboard",
    "HTMLDashboard",
    "DashboardData",
    # Persistence
    "SQLiteStore",
    "JSONStore",
    "SQLiteSkillStore",
    "SkillStore",
    "get_default_memory_store",
    "get_default_skill_store",
    # Config
    "SelfMonitor",
]
