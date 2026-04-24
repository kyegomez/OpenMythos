"""
OpenMythos Schema-First Configuration System

Provides JSON Schema-driven configuration with Pydantic validation,
YAML/JSON file support, and backward compatibility with dataclass-based MythosConfig.

Key Features:
- Schema-first configuration with Pydantic
- YAML/JSON file support
- Backward compatible MythosConfig
- Metadata-driven loop depth
- Inference lineage tracking
- Data contracts and SLA enforcement
- Quality-aware routing
- MCP server integration
- Self-check and self-evolution system
"""

from .schema import (
    # Core configs
    ModelConfig,
    LoopConfig,
    MoEConfig,
    AttentionConfig,
    DataContractConfig,
    OpenMetadataIntegrationConfig,
    MythosEnhancedConfig,
    MythosConfig,
)
from .loader import ConfigLoader, load_config
from .metadata_driven import (
    OpenMetadataClient,
    MetadataComplexityPredictor,
    MetadataFeaturesExtractor,
    MetadataDrivenLoopDepth,
    HeuristicComplexityPredictor,
    BudgetAwareDepthSelector,
)
from .lineage import (
    InferenceLineageTracker,
    LineageContext,
    LineageAnalytics,
    InferenceStatus,
    LineageNodeType,
    LineageEdgeType,
    InferenceMetrics,
    LineageNode,
    LineageEdge,
)
from .contracts import (
    InferenceContract,
    ContractEnforcer,
    ContractManager,
    ContractViolation,
    ContractViolationReport,
    ContractStatus,
    ViolationSeverity,
    SLAConstraint,
    QualityStandard,
    MonitoringConfig,
    DefaultContracts,
    DegradationStrategies,
)
from .routing import (
    QualityAwareRouter,
    QualityScoreCalculator,
    QualitySignal,
    QualityLevel,
    RoutingStrategy,
    RoutingDecision,
    MCPModelContext,
    MCPToolsRegistry,
)
from .self_monitor import (
    SelfCheckSystem,
    ModuleIntegrityChecker,
    HealthMonitor,
    SelfHealer,
    PerformanceOptimizer,
    AutoEvolution,
    HealthStatus,
    IssueSeverity,
    HealthCheckResult,
    Issue,
    # Convenience functions
    get_self_check_system,
    run_quick_check,
    auto_optimize_config,
    print_status,
)

__all__ = [
    # Schema
    "ModelConfig",
    "LoopConfig",
    "MoEConfig",
    "AttentionConfig",
    "DataContractConfig",
    "OpenMetadataIntegrationConfig",
    "MythosEnhancedConfig",
    "MythosConfig",
    # Loader
    "ConfigLoader",
    "load_config",
    # Metadata-driven depth
    "OpenMetadataClient",
    "MetadataComplexityPredictor",
    "MetadataFeaturesExtractor",
    "MetadataDrivenLoopDepth",
    "HeuristicComplexityPredictor",
    "BudgetAwareDepthSelector",
    # Lineage tracking
    "InferenceLineageTracker",
    "LineageContext",
    "LineageAnalytics",
    "InferenceStatus",
    "LineageNodeType",
    "LineageEdgeType",
    "InferenceMetrics",
    "LineageNode",
    "LineageEdge",
    # Contracts
    "InferenceContract",
    "ContractEnforcer",
    "ContractManager",
    "ContractViolation",
    "ContractViolationReport",
    "ContractStatus",
    "ViolationSeverity",
    "SLAConstraint",
    "QualityStandard",
    "MonitoringConfig",
    "DefaultContracts",
    "DegradationStrategies",
    # Routing
    "QualityAwareRouter",
    "QualityScoreCalculator",
    "QualitySignal",
    "QualityLevel",
    "RoutingStrategy",
    "RoutingDecision",
    "MCPModelContext",
    "MCPToolsRegistry",
    # Self-check system
    "SelfCheckSystem",
    "ModuleIntegrityChecker",
    "HealthMonitor",
    "SelfHealer",
    "PerformanceOptimizer",
    "AutoEvolution",
    "HealthStatus",
    "IssueSeverity",
    "HealthCheckResult",
    "Issue",
    "get_self_check_system",
    "run_quick_check",
    "auto_optimize_config",
    "print_status",
]

__version__ = "1.0.0"
