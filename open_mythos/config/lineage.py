"""
Inference Lineage Tracker Module

Tracks the model's inference path, similar to how OpenMetadata tracks data lineage.
Records loop states, state transitions, expert routing decisions, and attention patterns
for debugging, analysis, and governance.

This enables:
- Full traceability of model reasoning
- Impact analysis (what if we changed loop depth?)
- Performance debugging
- Compliance and audit trails
"""

import hashlib
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import defaultdict

import torch
import torch.nn.functional as F


# =============================================================================
# Enums and Data Classes
# =============================================================================

class LineageNodeType(Enum):
    """Types of nodes in the inference lineage graph."""
    INPUT = "input"
    OUTPUT = "output"
    LOOP = "loop"
    EXPERT = "expert"
    TRANSITION = "transition"
    MERGE = "merge"
    CONDITION = "condition"


class LineageEdgeType(Enum):
    """Types of edges in the inference lineage graph."""
    TRANSFORM = "transform"
    ROUTE = "route"
    MERGE = "merge"
    CONTROL = "control"
    ATTENTION = "attention"


class InferenceStatus(Enum):
    """Status of an inference task."""
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class LineageNode:
    """A node in the inference lineage graph."""
    id: str
    type: LineageNodeType
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


@dataclass  
class LineageEdge:
    """An edge in the inference lineage graph."""
    id: str
    from_node: str
    to_node: str
    type: LineageEdgeType
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "type": self.type.value,
            "metadata": self.metadata,
            "weight": self.weight
        }


@dataclass
class InferenceMetrics:
    """Metrics for a single inference."""
    task_id: str
    latency_ms: float
    loop_depth_used: int
    memory_used_mb: float
    num_tokens: int
    expert_distribution: Dict[str, float] = field(default_factory=dict)
    attention_entropy: float = 0.0
    output_length: int = 0
    status: InferenceStatus = InferenceStatus.COMPLETED
    error_message: Optional[str] = None


# =============================================================================
# Hash Utilities
# =============================================================================

def hash_tensor(tensor: torch.Tensor, max_size: int = 1000) -> str:
    """Create a short hash of a tensor for identification."""
    if tensor is None:
        return "none"
    
    # Flatten and take a subset for speed
    flat = tensor.flatten()[:max_size]
    
    # Simple hash based on statistics
    mean_val = flat.mean().item()
    std_val = flat.std().item()
    min_val = flat.min().item()
    max_val = flat.max().item()
    
    hash_str = f"{mean_val:.4f}_{std_val:.4f}_{min_val:.4f}_{max_val:.4f}"
    return hashlib.md5(hash_str.encode()).hexdigest()[:12]


def compute_attention_entropy(attn_weights: torch.Tensor) -> float:
    """
    Compute entropy of attention weights.
    
    Higher entropy = more diffuse attention (harder task)
    Lower entropy = more focused attention (easier task)
    """
    if attn_weights is None or attn_weights.numel() == 0:
        return 0.0
    
    # Flatten and normalize
    flat = attn_weights.flatten()
    flat = F.relu(flat)  # Remove negative values
    total = flat.sum()
    if total == 0:
        return 0.0
    
    probs = flat / total
    
    # Compute entropy: -sum(p * log(p))
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    
    # Normalize by max possible entropy
    n = probs.numel()
    max_entropy = torch.log(torch.tensor(n)).item()
    if max_entropy > 0:
        entropy = entropy / max_entropy
    
    return entropy


def compute_state_difference(t1: torch.Tensor, t2: torch.Tensor) -> Dict[str, float]:
    """Compute difference metrics between two states."""
    if t1 is None or t2 is None:
        return {"l2_diff": 0.0, "cosine_sim": 1.0, "max_diff": 0.0}
    
    diff = (t1 - t2)
    
    return {
        "l2_diff": diff.norm().item(),
        "cosine_sim": F.cosine_similarity(
            t1.flatten().unsqueeze(0), 
            t2.flatten().unsqueeze(0)
        ).item(),
        "max_diff": diff.abs().max().item(),
        "mean_diff": diff.abs().mean().item()
    }


# =============================================================================
# Inference Lineage Tracker
# =============================================================================

class InferenceLineageTracker:
    """
    Tracks inference lineage for OpenMythos models.
    
    Records:
    - Loop iterations and state transitions
    - Expert routing decisions
    - Attention patterns
    - Control flow decisions (ACT halting)
    
    Usage:
        tracker = InferenceLineageTracker()
        
        with tracker.track_inference("task-123", input_shape=(1, 100, 2048)) as ctx:
            # Run inference
            for loop_idx in range(depth):
                hidden = model(hidden, ...)
                
                # Record loop state
                ctx.record_loop(
                    loop_idx=loop_idx,
                    hidden_state=hidden,
                    depth=depth,
                    expert_weights=expert_weights,
                    attention_weights=attn_weights
                )
        
        # Get lineage graph
        lineage = tracker.get_lineage("task-123")
    """
    
    def __init__(
        self,
        max_history: int = 1000,
        store_attention: bool = False,
        store_expert_weights: bool = True,
        om_client: Optional["OpenMetadataClient"] = None
    ):
        """
        Args:
            max_history: Maximum number of inferences to store
            store_attention: Whether to store full attention weights
            store_expert_weights: Whether to store expert routing weights
            om_client: Optional OpenMetadata client for publishing lineage
        """
        self.max_history = max_history
        self.store_attention = store_attention
        self.store_expert_weights = store_expert_weights
        self.om_client = om_client
        
        # Storage
        self._lineage_graphs: Dict[str, Dict[str, Any]] = {}
        self._metrics: Dict[str, InferenceMetrics] = {}
        self._active_tracking: Dict[str, Any] = {}
        
        # Statistics
        self._stats = {
            "total_inferences": 0,
            "completed": 0,
            "failed": 0,
            "avg_depth": 0.0,
        }
    
    def track_inference(
        self,
        task_id: Optional[str] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "LineageContext":
        """
        Start tracking an inference.
        
        Returns a context manager.
        """
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        return LineageContext(
            tracker=self,
            task_id=task_id,
            input_shape=input_shape,
            metadata=metadata or {}
        )
    
    def _start_tracking(
        self,
        task_id: str,
        input_shape: Optional[Tuple[int, ...]],
        metadata: Dict[str, Any]
    ) -> None:
        """Internal method to start tracking."""
        self._active_tracking[task_id] = {
            "task_id": task_id,
            "input_shape": input_shape,
            "metadata": metadata,
            "start_time": time.time(),
            "nodes": [],
            "edges": [],
            "node_counter": 0,
            "edge_counter": 0,
            "loop_states": [],
            "expert_decisions": [],
            "attn_entropies": [],
        }
        
        # Add input node
        self._add_node(
            task_id,
            LineageNodeType.INPUT,
            "input",
            metadata={
                "shape": str(input_shape) if input_shape else "unknown",
                **metadata
            }
        )
    
    def _end_tracking(
        self,
        task_id: str,
        output_shape: Optional[Tuple[int, ...]] = None,
        status: InferenceStatus = InferenceStatus.COMPLETED,
        error: Optional[str] = None
    ) -> None:
        """Internal method to end tracking."""
        if task_id not in self._active_tracking:
            return
        
        ctx = self._active_tracking[task_id]
        elapsed_ms = (time.time() - ctx["start_time"]) * 1000
        
        # Add output node
        self._add_node(
            task_id,
            LineageNodeType.OUTPUT,
            "output",
            metadata={
                "shape": str(output_shape) if output_shape else "unknown",
                "elapsed_ms": elapsed_ms,
                "status": status.value,
                "error": error
            }
        )
        
        # Create lineage graph
        self._lineage_graphs[task_id] = {
            "task_id": task_id,
            "nodes": ctx["nodes"],
            "edges": ctx["edges"],
            "metadata": {
                "start_time": ctx["start_time"],
                "end_time": time.time(),
                "elapsed_ms": elapsed_ms,
                "status": status.value,
                "num_loops": len(ctx["loop_states"]),
            }
        }
        
        # Create metrics
        avg_depth = len(ctx["loop_states"]) if ctx["loop_states"] else 0
        avg_attn_entropy = sum(ctx["attn_entropies"]) / len(ctx["attn_entropies"]) if ctx["attn_entropies"] else 0.0
        
        self._metrics[task_id] = InferenceMetrics(
            task_id=task_id,
            latency_ms=elapsed_ms,
            loop_depth_used=avg_depth,
            memory_used_mb=0.0,  # Would need to track this
            num_tokens=0,  # Would need to track this
            expert_distribution=self._aggregate_expert_dist(ctx["expert_decisions"]),
            attention_entropy=avg_attn_entropy,
            status=status
        )
        
        # Update stats
        self._stats["total_inferences"] += 1
        if status == InferenceStatus.COMPLETED:
            self._stats["completed"] += 1
        else:
            self._stats["failed"] += 1
        
        # Running average of depth
        n = self._stats["total_inferences"]
        old_avg = self._stats["avg_depth"]
        self._stats["avg_depth"] = old_avg + (avg_depth - old_avg) / n
        
        # Cleanup
        del self._active_tracking[task_id]
        
        # Publish to OpenMetadata if configured
        if self.om_client is not None:
            self._publish_lineage(task_id)
    
    def _add_node(
        self,
        task_id: str,
        node_type: LineageNodeType,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a node to the current tracking context."""
        if task_id not in self._active_tracking:
            return None
        
        ctx = self._active_tracking[task_id]
        node_id = f"node_{ctx['node_counter']}"
        ctx["node_counter"] += 1
        
        node = LineageNode(
            id=node_id,
            type=node_type,
            name=name,
            metadata=metadata or {}
        )
        ctx["nodes"].append(node)
        
        return node_id
    
    def _add_edge(
        self,
        task_id: str,
        from_node: str,
        to_node: str,
        edge_type: LineageEdgeType,
        metadata: Optional[Dict[str, Any]] = None,
        weight: float = 1.0
    ) -> None:
        """Add an edge to the current tracking context."""
        if task_id not in self._active_tracking:
            return
        
        ctx = self._active_tracking[task_id]
        edge_id = f"edge_{ctx['edge_counter']}"
        ctx["edge_counter"] += 1
        
        edge = LineageEdge(
            id=edge_id,
            from_node=from_node,
            to_node=to_node,
            type=edge_type,
            metadata=metadata or {},
            weight=weight
        )
        ctx["edges"].append(edge)
    
    def record_loop(
        self,
        task_id: str,
        loop_idx: int,
        hidden_state: torch.Tensor,
        depth: int,
        expert_weights: Optional[Dict[str, float]] = None,
        attention_weights: Optional[torch.Tensor] = None,
        act_halted: bool = False,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a single loop iteration.
        
        Call this inside the inference loop.
        """
        if task_id not in self._active_tracking:
            return
        
        ctx = self._active_tracking[task_id]
        
        # Store loop state
        state_hash = hash_tensor(hidden_state)
        ctx["loop_states"].append({
            "loop_idx": loop_idx,
            "hash": state_hash,
            "norm": hidden_state.norm().item(),
            "depth": depth,
            "act_halted": act_halted
        })
        
        # Compute attention entropy
        if attention_weights is not None:
            entropy = compute_attention_entropy(attention_weights)
            ctx["attn_entropies"].append(entropy)
        
        # Store expert decisions
        if expert_weights is not None and self.store_expert_weights:
            ctx["expert_decisions"].append(expert_weights)
        
        # Add loop node
        loop_node_id = self._add_node(
            task_id,
            LineageNodeType.LOOP,
            f"loop_{loop_idx}",
            metadata={
                "loop_index": loop_idx,
                "depth": depth,
                "state_hash": state_hash,
                "hidden_norm": hidden_state.norm().item(),
                "act_halted": act_halted,
                **(additional_metadata or {})
            }
        )
        
        # Get previous node
        prev_node_id = ctx["nodes"][-2].id if len(ctx["nodes"]) > 1 else None
        
        if prev_node_id:
            # Add edge from previous to current
            self._add_edge(
                task_id,
                prev_node_id,
                loop_node_id,
                LineageEdgeType.TRANSFORM,
                metadata={
                    "loop_index": loop_idx,
                    "depth": depth,
                    "state_diff": compute_state_difference(None, hidden_state)  # Would need prev state
                }
            )
        
        # If ACT halted, add condition node
        if act_halted:
            self._add_node(
                task_id,
                LineageNodeType.CONDITION,
                f"halt_condition_{loop_idx}",
                metadata={
                    "reason": "act_threshold_reached",
                    "loop_index": loop_idx
                }
            )
    
    def record_expert_routing(
        self,
        task_id: str,
        loop_idx: int,
        token_id: int,
        selected_experts: List[int],
        routing_weights: List[float]
    ) -> None:
        """Record expert routing decision for a token."""
        if task_id not in self._active_tracking:
            return
        
        ctx = self._active_tracking[task_id]
        
        # Add expert nodes
        for expert_id, weight in zip(selected_experts, routing_weights):
            expert_node_id = self._add_node(
                task_id,
                LineageNodeType.EXPERT,
                f"expert_{expert_id}",
                metadata={
                    "expert_id": expert_id,
                    "weight": weight,
                    "loop_index": loop_idx,
                    "token_id": token_id
                }
            )
            
            # Add routing edge
            loop_node = ctx["nodes"][-1]
            self._add_edge(
                task_id,
                loop_node.id,
                expert_node_id,
                LineageEdgeType.ROUTE,
                metadata={
                    "expert_id": expert_id,
                    "weight": weight,
                    "loop_index": loop_idx
                },
                weight=weight
            )
    
    def _aggregate_expert_dist(self, decisions: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate expert distribution across all loops."""
        if not decisions:
            return {}
        
        aggregated = defaultdict(float)
        total = 0.0
        
        for decision in decisions:
            for expert_id, weight in decision.items():
                aggregated[expert_id] += weight
                total += weight
        
        if total > 0:
            for expert_id in aggregated:
                aggregated[expert_id] /= total
        
        return dict(aggregated)
    
    def get_lineage(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get lineage graph for a task."""
        if task_id in self._lineage_graphs:
            return self._lineage_graphs[task_id]
        return None
    
    def get_metrics(self, task_id: str) -> Optional[InferenceMetrics]:
        """Get metrics for a task."""
        return self._metrics.get(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            **self._stats,
            "stored_inferences": len(self._lineage_graphs),
            "active_tracking": len(self._active_tracking)
        }
    
    def export_lineage(self, task_id: str, format: str = "dict") -> Any:
        """
        Export lineage in various formats.
        
        Args:
            task_id: Task ID
            format: "dict", "json", or "dot"
            
        Returns:
            Exported lineage in specified format
        """
        lineage = self.get_lineage(task_id)
        if lineage is None:
            return None
        
        if format == "dict":
            return lineage
        
        elif format == "json":
            import json
            return json.dumps(lineage, indent=2, default=str)
        
        elif format == "dot":
            return self._to_dot_graph(lineage)
        
        return None
    
    def _to_dot_graph(self, lineage: Dict[str, Any]) -> str:
        """Convert lineage to DOT format for Graphviz."""
        lines = ["digraph lineage {", '  rankdir="TB";']
        
        for node in lineage["nodes"]:
            label = f'{node["name"]}\n{node["type"]}'
            lines.append(f'  {node["id"]} [label="{label}", shape=box];')
        
        for edge in lineage["edges"]:
            lines.append(
                f'  {edge["from_node"]} -> {edge["to_node"]} '
                f'[label="{edge["type"]}", weight={edge["weight"]}];'
            )
        
        lines.append("}")
        return "\n".join(lines)
    
    def _publish_lineage(self, task_id: str) -> None:
        """Publish lineage to OpenMetadata."""
        if self.om_client is None:
            return
        
        lineage = self.get_lineage(task_id)
        if lineage is None:
            return
        
        # Build payload for OpenMetadata lineage API
        payload = {
            "entityType": "modelInference",
            "entityId": task_id,
            "lineage": {
                "nodes": [n.to_dict() for n in lineage["nodes"]],
                "edges": [e.to_dict() for e in lineage["edges"]]
            },
            "metadata": lineage["metadata"]
        }
        
        # Would make API call here
        # self.om_client.lineage.add(task_id, payload)


class LineageContext:
    """
    Context manager for tracking an inference.
    
    Usage:
        tracker = InferenceLineageTracker()
        
        with tracker.track_inference("task-123", input_shape=(1, 100, 2048)) as ctx:
            # Run inference
            hidden = model(input_ids)
            ctx.record_loop(loop_idx=0, hidden_state=hidden, depth=4)
            ...
    """
    
    def __init__(
        self,
        tracker: InferenceLineageTracker,
        task_id: str,
        input_shape: Optional[Tuple[int, ...]],
        metadata: Dict[str, Any]
    ):
        self.tracker = tracker
        self.task_id = task_id
        self.input_shape = input_shape
        self.metadata = metadata
    
    def __enter__(self) -> "LineageContext":
        self.tracker._start_tracking(self.task_id, self.input_shape, self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.tracker._end_tracking(
                self.task_id,
                status=InferenceStatus.FAILED,
                error=str(exc_val)
            )
        else:
            self.tracker._end_tracking(self.task_id, status=InferenceStatus.COMPLETED)
    
    def record_loop(
        self,
        loop_idx: int,
        hidden_state: torch.Tensor,
        depth: int,
        expert_weights: Optional[Dict[str, float]] = None,
        attention_weights: Optional[torch.Tensor] = None,
        act_halted: bool = False,
        **kwargs
    ) -> None:
        """Record a loop iteration."""
        self.tracker.record_loop(
            self.task_id,
            loop_idx,
            hidden_state,
            depth,
            expert_weights,
            attention_weights,
            act_halted,
            kwargs
        )
    
    def record_expert_routing(
        self,
        loop_idx: int,
        token_id: int,
        selected_experts: List[int],
        routing_weights: List[float]
    ) -> None:
        """Record expert routing."""
        self.tracker.record_expert_routing(
            self.task_id,
            loop_idx,
            token_id,
            selected_experts,
            routing_weights
        )


# =============================================================================
# Lineage Analytics
# =============================================================================

class LineageAnalytics:
    """
    Analytics on inference lineage.
    
    Provides insights like:
    - Average loop depth by task type
    - Expert usage patterns
    - Attention distribution analysis
    """
    
    def __init__(self, tracker: InferenceLineageTracker):
        self.tracker = tracker
    
    def get_depth_distribution(self, bins: int = 10) -> Dict[str, Any]:
        """Get distribution of loop depths."""
        depths = []
        
        for metrics in self.tracker._metrics.values():
            if metrics.status == InferenceStatus.COMPLETED:
                depths.append(metrics.loop_depth_used)
        
        if not depths:
            return {"bins": [], "counts": []}
        
        import numpy as np
        hist, edges = np.histogram(depths, bins=bins)
        
        return {
            "bins": edges.tolist(),
            "counts": hist.tolist(),
            "mean": np.mean(depths),
            "std": np.std(depths),
            "min": int(np.min(depths)),
            "max": int(np.max(depths))
        }
    
    def get_expert_usage(self) -> Dict[str, float]:
        """Get expert usage distribution."""
        aggregated = defaultdict(float)
        total_inferences = 0
        
        for metrics in self.tracker._metrics.values():
            if metrics.status == InferenceStatus.COMPLETED:
                total_inferences += 1
                for expert_id, weight in metrics.expert_distribution.items():
                    aggregated[expert_id] += weight
        
        if total_inferences > 0:
            for expert_id in aggregated:
                aggregated[expert_id] /= total_inferences
        
        return dict(aggregated)
    
    def get_attention_entropy_stats(self) -> Dict[str, float]:
        """Get attention entropy statistics."""
        entropies = []
        
        for metrics in self.tracker._metrics.values():
            if metrics.attention_entropy > 0:
                entropies.append(metrics.attention_entropy)
        
        if not entropies:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        import numpy as np
        return {
            "mean": float(np.mean(entropies)),
            "std": float(np.std(entropies)),
            "min": float(np.min(entropies)),
            "max": float(np.max(entropies))
        }
    
    def get_loop_efficiency(self) -> float:
        """
        Calculate loop efficiency.
        
        Lower ratio = more direct paths (higher efficiency)
        Higher ratio = more loops per inference
        """
        total_depth = 0
        total_inferences = 0
        
        for metrics in self.tracker._metrics.values():
            if metrics.status == InferenceStatus.COMPLETED:
                total_depth += metrics.loop_depth_used
                total_inferences += 1
        
        if total_inferences == 0:
            return 0.0
        
        return total_depth / total_inferences
