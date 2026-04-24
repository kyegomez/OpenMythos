"""
Quality-Aware Routing and MCP Model Context Module

This module implements:
1. Quality-Aware Routing (QAR) - Routes inference based on data quality metrics
2. MCP Model Context (MCP-MC) - MCP server integration for AI assistants

QAR uses OpenMetadata quality signals to route requests to appropriate
model configurations, ensuring SLA compliance while optimizing resource usage.

MCP-MC provides a Model Context Protocol server that exposes OpenMythos
capabilities to AI assistants like Claude, Copilot, etc.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import defaultdict

import torch
import torch.nn.functional as F


# =============================================================================
# Quality-Aware Routing Enums and Data Classes
# =============================================================================

class QualityLevel(Enum):
    """Quality levels for routing decisions."""
    HIGH = "high"      # Maximum quality, ignore SLA
    MEDIUM = "medium"  # Balanced quality and SLA
    LOW = "low"        # SLA priority, accept lower quality
    BEST_EFFORT = "best_effort"  # Minimal resources


class RoutingStrategy(Enum):
    """Routing strategies."""
    QUALITY_FIRST = "quality_first"
    LATENCY_FIRST = "latency_first"
    BALANCED = "balanced"
    COST_OPTIMIZED = "cost_optimized"
    EXPERT_LOAD_BALANCED = "expert_load_balanced"


@dataclass
class QualitySignal:
    """Quality signal from OpenMetadata."""
    source: str  # e.g., "openmetadata", "manual", "inferred"
    completeness: float = 1.0  # [0, 1]
    validity: float = 1.0
    accuracy: float = 1.0
    consistency: float = 1.0
    freshness: float = 1.0
    overall_score: float = 1.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "completeness": self.completeness,
            "validity": self.validity,
            "accuracy": self.accuracy,
            "consistency": self.consistency,
            "freshness": self.freshness,
            "overall_score": self.overall_score,
            "timestamp": self.timestamp,
        }


@dataclass
class RoutingDecision:
    """Routing decision for an inference request."""
    request_id: str
    quality_level: QualityLevel
    strategy: RoutingStrategy
    loop_depth: int
    attention_type: str
    batch_size: int
    expert_config: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "quality_level": self.quality_level.value,
            "strategy": self.strategy.value,
            "loop_depth": self.loop_depth,
            "attention_type": self.attention_type,
            "batch_size": self.batch_size,
            "expert_config": self.expert_config,
            "metadata": self.metadata,
        }


# =============================================================================
# Quality Score Calculator
# =============================================================================

class QualityScoreCalculator:
    """
    Calculates composite quality scores from multiple signals.
    
    Supports:
    - Weighted aggregation
    - Temporal decay
    - Missing signal handling
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        decay_factor: float = 0.95
    ):
        """
        Args:
            weights: Weights for each quality dimension
            decay_factor: Temporal decay factor for older signals
        """
        self.weights = weights or {
            "completeness": 0.2,
            "validity": 0.2,
            "accuracy": 0.3,
            "consistency": 0.15,
            "freshness": 0.15,
        }
        self.decay_factor = decay_factor
    
    def calculate(
        self,
        signals: List[QualitySignal],
        current_time: Optional[float] = None
    ) -> float:
        """
        Calculate composite quality score.
        
        Args:
            signals: List of quality signals
            current_time: Current timestamp for decay calculation
            
        Returns:
            Composite quality score [0, 1]
        """
        if not signals:
            return 0.5  # Default medium score
        
        current_time = current_time or time.time()
        
        # Aggregate signals with temporal decay
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for signal in signals:
            # Compute decay based on age
            age_hours = (current_time - signal.timestamp) / 3600
            decay = self.decay_factor ** age_hours
            
            # Weighted aggregation
            for key, weight in self.weights.items():
                value = getattr(signal, key, 1.0)
                weighted_sum += value * weight * decay
                weight_sum += weight * decay
        
        if weight_sum == 0:
            return 0.5
        
        return weighted_sum / weight_sum
    
    def get_level(self, score: float) -> QualityLevel:
        """Map score to quality level."""
        if score >= 0.85:
            return QualityLevel.HIGH
        elif score >= 0.65:
            return QualityLevel.MEDIUM
        elif score >= 0.4:
            return QualityLevel.LOW
        else:
            return QualityLevel.BEST_EFFORT


# =============================================================================
# Quality-Aware Router
# =============================================================================

class QualityAwareRouter:
    """
    Routes inference requests based on quality signals.
    
    Uses OpenMetadata quality metrics to determine:
    - Appropriate loop depth
    - Attention mechanism
    - Batch size
    - Expert configuration
    
    Usage:
        router = QualityAwareRouter(
            om_client=om_client,
            min_depth=4,
            max_depth=16
        )
        
        decision = router.route(
            task_context={"asset_fqn": "warehouse.sales.data"},
            strategy=RoutingStrategy.BALANCED
        )
    """
    
    def __init__(
        self,
        om_client: Optional["OpenMetadataClient"] = None,
        min_depth: int = 4,
        max_depth: int = 16,
        max_batch_size: int = 32,
        quality_calculator: Optional[QualityScoreCalculator] = None
    ):
        """
        Args:
            om_client: OpenMetadata client for quality signals
            min_depth: Minimum loop depth
            max_depth: Maximum loop depth
            max_batch_size: Maximum batch size
            quality_calculator: Custom quality calculator
        """
        self.om_client = om_client
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_batch_size = max_batch_size
        self.quality_calculator = quality_calculator or QualityScoreCalculator()
        
        # Routing statistics
        self._stats = {
            "total_requests": 0,
            "by_quality_level": defaultdict(int),
            "by_strategy": defaultdict(int),
            "avg_depth": 0.0,
        }
    
    def route(
        self,
        task_context: Dict[str, Any],
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        available_budget_ms: Optional[float] = None
    ) -> RoutingDecision:
        """
        Make routing decision for an inference request.
        
        Args:
            task_context: Context including asset FQN, metadata, etc.
            strategy: Routing strategy to use
            available_budget_ms: Available time budget
            
        Returns:
            Routing decision
        """
        request_id = str(uuid.uuid4())
        
        # Get quality signals
        signals = self._get_quality_signals(task_context)
        
        # Calculate quality score
        quality_score = self.quality_calculator.calculate(signals)
        quality_level = self.quality_calculator.get_level(quality_score)
        
        # Determine configuration based on strategy and quality
        config = self._determine_config(
            quality_level=quality_level,
            strategy=strategy,
            quality_score=quality_score,
            available_budget_ms=available_budget_ms,
            task_context=task_context
        )
        
        # Update stats
        self._update_stats(config["loop_depth"], quality_level, strategy)
        
        return RoutingDecision(
            request_id=request_id,
            quality_level=quality_level,
            strategy=strategy,
            loop_depth=config["loop_depth"],
            attention_type=config["attention_type"],
            batch_size=config["batch_size"],
            expert_config=config["expert_config"],
            metadata={
                "quality_score": quality_score,
                "num_signals": len(signals),
            }
        )
    
    def _get_quality_signals(self, task_context: Dict[str, Any]) -> List[QualitySignal]:
        """Get quality signals from various sources."""
        signals = []
        
        # 1. From task context directly
        if "quality_signals" in task_context:
            for sig_data in task_context["quality_signals"]:
                signals.append(QualitySignal(**sig_data))
        
        # 2. From OpenMetadata
        if self.om_client and "asset_fqn" in task_context:
            try:
                quality_metrics = self.om_client.get_quality_metrics(
                    task_context["asset_fqn"]
                )
                signals.append(QualitySignal(
                    source="openmetadata",
                    completeness=quality_metrics.get("completeness", 1.0),
                    validity=quality_metrics.get("validity", 1.0),
                    accuracy=quality_metrics.get("accuracy", 1.0),
                    consistency=quality_metrics.get("consistency", 1.0),
                    freshness=quality_metrics.get("freshness", 1.0),
                    overall_score=quality_metrics.get("overall_score", 1.0),
                ))
            except Exception:
                pass
        
        # 3. Inferred from metadata
        if "metadata" in task_context:
            inferred = self._infer_quality_from_metadata(task_context["metadata"])
            if inferred:
                signals.append(inferred)
        
        return signals
    
    def _infer_quality_from_metadata(self, metadata: Dict[str, Any]) -> Optional[QualitySignal]:
        """Infer quality signal from table metadata."""
        # Use test coverage and tier as proxy for quality
        test_coverage = metadata.get("test_coverage", 0.0)
        tier = metadata.get("tier_level", 3)
        
        if test_coverage == 0 and tier == 0:
            return None
        
        # Infer quality score
        quality = (test_coverage * 0.7) + ((tier / 3.0) * 0.3)
        
        return QualitySignal(
            source="inferred",
            completeness=quality,
            validity=quality,
            accuracy=quality,
            consistency=quality,
            freshness=metadata.get("freshness_score", 1.0),
            overall_score=quality,
        )
    
    def _determine_config(
        self,
        quality_level: QualityLevel,
        strategy: RoutingStrategy,
        quality_score: float,
        available_budget_ms: Optional[float],
        task_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine routing configuration."""
        
        # Base depth from quality level
        if quality_level == QualityLevel.HIGH:
            base_depth = self.max_depth
        elif quality_level == QualityLevel.MEDIUM:
            base_depth = (self.min_depth + self.max_depth) // 2
        elif quality_level == QualityLevel.LOW:
            base_depth = self.min_depth + 2
        else:
            base_depth = self.min_depth
        
        # Adjust for strategy
        if strategy == RoutingStrategy.LATENCY_FIRST:
            base_depth = min(base_depth, 8)
        elif strategy == RoutingStrategy.QUALITY_FIRST:
            base_depth = max(base_depth, 12)
        elif strategy == RoutingStrategy.COST_OPTIMIZED:
            base_depth = min(base_depth, self.min_depth + 4)
        
        # Adjust for budget
        if available_budget_ms is not None:
            max_depth_from_budget = int(available_budget_ms / 5.0)  # 5ms per loop
            base_depth = min(base_depth, max(self.min_depth, max_depth_from_budget))
        
        # Attention type
        if quality_level == QualityLevel.HIGH:
            attn_type = "mla"
        else:
            attn_type = "gqa"
        
        # Batch size
        if strategy == RoutingStrategy.LATENCY_FIRST:
            batch_size = min(8, self.max_batch_size)
        elif strategy == RoutingStrategy.QUALITY_FIRST:
            batch_size = min(16, self.max_batch_size)
        else:
            batch_size = min(32, self.max_batch_size)
        
        # Expert config
        if quality_level == QualityLevel.HIGH:
            expert_config = {
                "n_experts": 64,
                "n_experts_per_tok": 4,
                "capacity_factor": 1.5,
            }
        elif quality_level == QualityLevel.MEDIUM:
            expert_config = {
                "n_experts": 32,
                "n_experts_per_tok": 4,
                "capacity_factor": 1.25,
            }
        else:
            expert_config = {
                "n_experts": 16,
                "n_experts_per_tok": 2,
                "capacity_factor": 1.0,
            }
        
        return {
            "loop_depth": base_depth,
            "attention_type": attn_type,
            "batch_size": batch_size,
            "expert_config": expert_config,
        }
    
    def _update_stats(
        self,
        depth: int,
        quality_level: QualityLevel,
        strategy: RoutingStrategy
    ) -> None:
        """Update routing statistics."""
        self._stats["total_requests"] += 1
        self._stats["by_quality_level"][quality_level.value] += 1
        self._stats["by_strategy"][strategy.value] += 1
        
        n = self._stats["total_requests"]
        old_avg = self._stats["avg_depth"]
        self._stats["avg_depth"] = old_avg + (depth - old_avg) / n
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            **self._stats,
            "by_quality_level": dict(self._stats["by_quality_level"]),
            "by_strategy": dict(self._stats["by_strategy"]),
        }


# =============================================================================
# MCP Model Context Server
# =============================================================================

class MCPModelContext:
    """
    Model Context Protocol server for OpenMythos.
    
    Exposes OpenMythos capabilities to AI assistants via MCP protocol.
    AI assistants can:
    - Query model configuration
    - Submit inference requests
    - Get inference lineage
    - Check contract status
    
    Usage:
        mcp_server = MCPModelContext(
            config=config,
            tracker=tracker,
            router=router
        )
        mcp_server.start(port=8080)
    """
    
    def __init__(
        self,
        config: "MythosEnhancedConfig",
        tracker: Optional["InferenceLineageTracker"] = None,
        router: Optional["QualityAwareRouter"] = None,
        enforcer: Optional["ContractEnforcer"] = None
    ):
        """
        Args:
            config: OpenMythos configuration
            tracker: Optional lineage tracker
            router: Optional quality-aware router
            enforcer: Optional contract enforcer
        """
        self.config = config
        self.tracker = tracker
        self.router = router
        self.enforcer = enforcer
        self._running = False
        self._request_handlers: Dict[str, Callable] = {}
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """Register default MCP request handlers."""
        self._request_handlers = {
            "get_config": self._handle_get_config,
            "get_capabilities": self._handle_get_capabilities,
            "route_inference": self._handle_route_inference,
            "get_lineage": self._handle_get_lineage,
            "check_contract": self._handle_check_contract,
            "get_stats": self._handle_get_stats,
        }
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming MCP request.
        
        Args:
            request: MCP request with method and params
            
        Returns:
            MCP response
        """
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        if method not in self._request_handlers:
            return {
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                },
                "id": request_id
            }
        
        try:
            result = self._request_handlers[method](params)
            return {
                "result": result,
                "id": request_id
            }
        except Exception as e:
            return {
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                },
                "id": request_id
            }
    
    def _handle_get_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_config request."""
        detailed = params.get("detailed", False)
        
        if detailed:
            return self.config.model_dump()
        else:
            return {
                "model_dim": self.config.model.dim,
                "n_heads": self.config.model.n_heads,
                "n_experts": self.config.moe.n_experts,
                "max_loop_depth": self.config.loop.max_depth,
                "attention_type": self.config.attention.attn_type,
            }
    
    def _handle_get_capabilities(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_capabilities request."""
        return {
            "features": [
                "quality_aware_routing",
                "metadata_driven_depth",
                "inference_lineage",
                "contract_enforcement",
            ],
            "enhancements": {
                "p0_multiscale_loop": self.config.enhancements.p0.multiscale_loop_enabled,
                "p1_flash_mla": self.config.enhancements.p1.flash_mla_enabled,
                "p2_cross_layer_kv": self.config.enhancements.p2.cross_layer_kv_enabled,
                "p3_hierarchical": self.config.enhancements.p3.hierarchical_loops_enabled,
            },
            "openmetadata": {
                "enabled": self.config.openmetadata.enabled,
                "endpoint": self.config.openmetadata.endpoint,
            },
        }
    
    def _handle_route_inference(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle route_inference request."""
        if self.router is None:
            return {"error": "Router not configured"}
        
        task_context = params.get("task_context", {})
        strategy_str = params.get("strategy", "balanced")
        
        try:
            strategy = RoutingStrategy(strategy_str)
        except ValueError:
            strategy = RoutingStrategy.BALANCED
        
        decision = self.router.route(task_context, strategy)
        return decision.to_dict()
    
    def _handle_get_lineage(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_lineage request."""
        if self.tracker is None:
            return {"error": "Tracker not configured"}
        
        task_id = params.get("task_id")
        if task_id:
            lineage = self.tracker.get_lineage(task_id)
            if lineage is None:
                return {"error": f"Lineage not found: {task_id}"}
            return lineage
        
        # Return all lineages (paginated)
        limit = params.get("limit", 10)
        offset = params.get("offset", 0)
        
        all_tasks = list(self.tracker._lineage_graphs.keys())
        tasks = all_tasks[offset:offset+limit]
        
        return {
            "total": len(all_tasks),
            "lineages": [
                self.tracker.get_lineage(tid)
                for tid in tasks
            ]
        }
    
    def _handle_check_contract(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle check_contract request."""
        if self.enforcer is None:
            return {"error": "Enforcer not configured"}
        
        contract_name = params.get("contract_name", self.enforcer.contract.name)
        
        if contract_name != self.enforcer.contract.name:
            return {"error": f"Contract not found: {contract_name}"}
        
        stats = self.enforcer.get_stats()
        return {
            "contract_name": contract_name,
            "status": "healthy" if stats["violation_rate"] < 0.1 else "warning",
            "violation_rate": stats["violation_rate"],
            "total_inferences": stats["total_inferences"],
        }
    
    def _handle_get_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_stats request."""
        stats = {}
        
        if self.tracker:
            stats["lineage"] = self.tracker.get_stats()
        
        if self.router:
            stats["routing"] = self.router.get_stats()
        
        if self.enforcer:
            stats["contract"] = self.enforcer.get_stats()
        
        return stats
    
    def start(self, host: str = "localhost", port: int = 8080) -> None:
        """Start MCP server."""
        # In a real implementation, this would start an HTTP/WebSocket server
        self._running = True
        self._host = host
        self._port = port
    
    def stop(self) -> None:
        """Stop MCP server."""
        self._running = False
    
    @property
    def is_running(self) -> bool:
        return self._running


# =============================================================================
# MCP Tools Registry
# =============================================================================

class MCPToolsRegistry:
    """
    Registry of MCP tools available to AI assistants.
    
    Tools are functions that can be called by AI assistants
    to interact with OpenMythos.
    """
    
    def __init__(self, mcp_context: MCPModelContext):
        self.mcp_context = mcp_context
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register default MCP tools."""
        self._tools = {
            "openmythos_get_config": {
                "name": "openmythos_get_config",
                "description": "Get current OpenMythos configuration",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "detailed": {
                            "type": "boolean",
                            "description": "Return detailed configuration"
                        }
                    }
                },
                "handler": lambda params: self.mcp_context.handle_request({
                    "method": "get_config",
                    "params": params,
                    "id": "1"
                })
            },
            "openmythos_route": {
                "name": "openmythos_route",
                "description": "Route an inference request with quality-aware routing",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "asset_fqn": {
                            "type": "string",
                            "description": "OpenMetadata asset FQN"
                        },
                        "strategy": {
                            "type": "string",
                            "enum": ["quality_first", "latency_first", "balanced", "cost_optimized"],
                            "description": "Routing strategy"
                        }
                    },
                    "required": ["asset_fqn"]
                },
                "handler": lambda params: self.mcp_context.handle_request({
                    "method": "route_inference",
                    "params": params,
                    "id": "2"
                })
            },
            "openmythos_lineage": {
                "name": "openmythos_lineage",
                "description": "Get inference lineage for a task",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Task ID to get lineage for"
                        }
                    },
                    "required": ["task_id"]
                },
                "handler": lambda params: self.mcp_context.handle_request({
                    "method": "get_lineage",
                    "params": params,
                    "id": "3"
                })
            },
            "openmythos_stats": {
                "name": "openmythos_stats",
                "description": "Get OpenMythos statistics",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                },
                "handler": lambda params: self.mcp_context.handle_request({
                    "method": "get_stats",
                    "params": params,
                    "id": "4"
                })
            },
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["input_schema"]
            }
            for tool in self._tools.values()
        ]
    
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool by name."""
        if name not in self._tools:
            raise ValueError(f"Tool not found: {name}")
        
        return self._tools[name]["handler"](arguments)
    
    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable
    ) -> None:
        """Register a custom tool."""
        self._tools[name] = {
            "name": name,
            "description": description,
            "input_schema": input_schema,
            "handler": handler
        }
