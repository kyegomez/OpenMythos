"""
Inference Data Contract Module

Defines SLA and quality guarantees for model inference.
Implements contract checking, violation reporting, and alerting.

This module provides:
1. Contract definition with SLA and quality constraints
2. Contract enforcement during inference
3. Violation detection and reporting
4. Degradation strategies when contracts are at risk
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from collections import defaultdict

import torch


# =============================================================================
# Enums and Data Classes
# =============================================================================

class ViolationSeverity(Enum):
    """Severity level of a contract violation."""
    LOW = "low"           # Minor issue, informational
    MEDIUM = "medium"     # Should be addressed
    HIGH = "high"        # Requires immediate attention
    CRITICAL = "critical" # SLA breached


class ContractStatus(Enum):
    """Overall contract status."""
    HEALTHY = "healthy"           # All metrics within bounds
    WARNING = "warning"           # Some metrics approaching limits
    DEGRADED = "degraded"         # Contract at risk
    VIOLATED = "violated"         # Contract breached


@dataclass
class SLAConstraint:
    """SLA constraint definition."""
    name: str
    max_latency_ms: float = 100.0
    max_memory_mb: int = 4096
    min_throughput_tokens_per_sec: float = 50.0
    max_batch_size: int = 32
    max_queue_time_ms: float = 50.0


@dataclass
class QualityStandard:
    """Quality standard definition."""
    name: str
    min_accuracy: float = 0.85
    max_confidence_variance: float = 0.1
    min_attention_coverage: float = 0.7
    min_expert_diversity: float = 0.3
    max_perplexity: float = 100.0


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration."""
    name: str
    alert_on_latency_p95: bool = True
    alert_on_quality_drop: bool = True
    latency_p95_threshold_ms: float = 150.0
    quality_drop_threshold: float = 0.05
    check_interval_seconds: int = 60
    log_level: str = "INFO"


@dataclass
class InferenceContract:
    """
    Complete inference data contract.
    
    Defines SLA, quality, and monitoring for model inference.
    """
    name: str
    version: str = "1.0"
    
    # SLA constraints
    sla: SLAConstraint = field(default_factory=lambda: SLAConstraint(name="default"))
    
    # Quality standards
    quality: QualityStandard = field(default_factory=lambda: QualityStandard(name="default"))
    
    # Monitoring config
    monitoring: MonitoringConfig = field(default_factory=lambda: MonitoringConfig(name="default"))
    
    # Additional metadata
    description: Optional[str] = None
    owner: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceContract":
        """Create from dictionary."""
        return cls(
            name=data.get("name", "default"),
            version=data.get("version", "1.0"),
            sla=SLAConstraint(**data.get("sla", {})),
            quality=QualityStandard(**data.get("quality", {})),
            monitoring=MonitoringConfig(**data.get("monitoring", {})),
            description=data.get("description"),
            owner=data.get("owner"),
            tags=data.get("tags", [])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "sla": {
                "max_latency_ms": self.sla.max_latency_ms,
                "max_memory_mb": self.sla.max_memory_mb,
                "min_throughput_tokens_per_sec": self.sla.min_throughput_tokens_per_sec,
                "max_batch_size": self.sla.max_batch_size,
                "max_queue_time_ms": self.sla.max_queue_time_ms,
            },
            "quality": {
                "min_accuracy": self.quality.min_accuracy,
                "max_confidence_variance": self.quality.max_confidence_variance,
                "min_attention_coverage": self.quality.min_attention_coverage,
                "min_expert_diversity": self.quality.min_expert_diversity,
                "max_perplexity": self.quality.max_perplexity,
            },
            "monitoring": {
                "alert_on_latency_p95": self.monitoring.alert_on_latency_p95,
                "alert_on_quality_drop": self.monitoring.alert_on_quality_drop,
                "latency_p95_threshold_ms": self.monitoring.latency_p95_threshold_ms,
                "quality_drop_threshold": self.monitoring.quality_drop_threshold,
                "check_interval_seconds": self.monitoring.check_interval_seconds,
                "log_level": self.monitoring.log_level,
            },
            "description": self.description,
            "owner": self.owner,
            "tags": self.tags,
        }


@dataclass
class ContractViolation:
    """A single contract violation."""
    dimension: str
    constraint_name: str
    expected: str
    actual: str
    severity: ViolationSeverity
    timestamp: float = field(default_factory=time.time)
    task_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "constraint_name": self.constraint_name,
            "expected": self.expected,
            "actual": self.actual,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "task_id": self.task_id,
        }


@dataclass
class ContractViolationReport:
    """Report of all violations for an inference."""
    contract_name: str
    task_id: str
    violations: List[ContractViolation] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0
    
    @property
    def has_critical(self) -> bool:
        return any(v.severity == ViolationSeverity.CRITICAL for v in self.violations)
    
    @property
    def has_high(self) -> bool:
        return any(v.severity == ViolationSeverity.HIGH for v in self.violations)
    
    @property
    def severity(self) -> ViolationSeverity:
        if self.has_critical:
            return ViolationSeverity.CRITICAL
        if self.has_high:
            return ViolationSeverity.HIGH
        if self.violations:
            return ViolationSeverity.MEDIUM
        return ViolationSeverity.LOW
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_name": self.contract_name,
            "task_id": self.task_id,
            "violations": [v.to_dict() for v in self.violations],
            "has_violations": self.has_violations,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
        }


# =============================================================================
# Default Contracts
# =============================================================================

class DefaultContracts:
    """Predefined contract templates."""
    
    @staticmethod
    def low_latency() -> InferenceContract:
        """Low latency contract for real-time applications."""
        return InferenceContract(
            name="low-latency",
            description="Contract for real-time inference with strict latency requirements",
            sla=SLAConstraint(
                name="low-latency-sla",
                max_latency_ms=50.0,
                max_memory_mb=2048,
                min_throughput_tokens_per_sec=100.0,
                max_batch_size=8,
            ),
            quality=QualityStandard(
                name="low-latency-quality",
                min_accuracy=0.80,  # Slightly relaxed for speed
                min_attention_coverage=0.5,
            ),
            monitoring=MonitoringConfig(
                name="low-latency-monitoring",
                alert_on_latency_p95=True,
                latency_p95_threshold_ms=75.0,
            )
        )
    
    @staticmethod
    def high_accuracy() -> InferenceContract:
        """High accuracy contract for batch processing."""
        return InferenceContract(
            name="high-accuracy",
            description="Contract for batch processing with maximum accuracy",
            sla=SLAConstraint(
                name="high-accuracy-sla",
                max_latency_ms=500.0,
                max_memory_mb=8192,
                min_throughput_tokens_per_sec=10.0,
                max_batch_size=64,
            ),
            quality=QualityStandard(
                name="high-accuracy-quality",
                min_accuracy=0.95,
                min_attention_coverage=0.85,
                min_expert_diversity=0.5,
            ),
            monitoring=MonitoringConfig(
                name="high-accuracy-monitoring",
                alert_on_quality_drop=True,
                quality_drop_threshold=0.02,
            )
        )
    
    @staticmethod
    def balanced() -> InferenceContract:
        """Balanced contract for general use."""
        return InferenceContract(
            name="balanced",
            description="Balanced contract for general inference workloads",
            sla=SLAConstraint(
                name="balanced-sla",
                max_latency_ms=100.0,
                max_memory_mb=4096,
                min_throughput_tokens_per_sec=50.0,
                max_batch_size=32,
            ),
            quality=QualityStandard(
                name="balanced-quality",
                min_accuracy=0.85,
                min_attention_coverage=0.7,
                min_expert_diversity=0.3,
            ),
            monitoring=MonitoringConfig(
                name="balanced-monitoring",
                alert_on_latency_p95=True,
                alert_on_quality_drop=True,
                latency_p95_threshold_ms=150.0,
                quality_drop_threshold=0.05,
            )
        )


# =============================================================================
# Contract Enforcer
# =============================================================================

class ContractEnforcer:
    """
    Enforces inference contracts.
    
    Validates that inference meets SLA and quality requirements.
    Implements degradation strategies when contracts are at risk.
    
    Usage:
        enforcer = ContractEnforcer(contract=DefaultContracts.balanced())
        
        # Pre-inference check
        if not enforcer.pre_inference_check(task_context):
            # Apply degradation
            depth = enforcer.apply_degradation(base_depth)
        
        # Post-inference validation
        report = enforcer.post_inference_validate(metrics)
        if report.has_violations:
            enforcer.handle_violations(report)
    """
    
    def __init__(
        self,
        contract: InferenceContract,
        on_violation: Optional[Callable[[ContractViolationReport], None]] = None
    ):
        """
        Args:
            contract: Contract to enforce
            on_violation: Callback when violations occur
        """
        self.contract = contract
        self.on_violation = on_violation
        
        # History for tracking
        self._violation_history: List[ContractViolationReport] = []
        self._metrics_history: List[Dict[str, Any]] = []
        self._max_history = 1000
    
    def pre_inference_check(
        self,
        task_context: Dict[str, Any],
        estimated_input_size: Optional[int] = None
    ) -> bool:
        """
        Pre-inference validation.
        
        Estimates whether inference can meet contract and returns
        whether to proceed or apply degradation.
        
        Args:
            task_context: Context about the inference task
            estimated_input_size: Estimated input size in tokens
            
        Returns:
            True if inference can proceed, False if degradation needed
        """
        # Estimate latency based on input size and loop depth
        estimated_depth = task_context.get("estimated_depth", self.contract.sla.max_latency_ms / 5)
        estimated_latency = estimated_depth * 5.0  # 5ms per loop iteration
        
        if estimated_input_size:
            estimated_latency += estimated_input_size * 0.1  # 0.1ms per token
        
        # Check if we can meet SLA
        if estimated_latency > self.contract.sla.max_latency_ms:
            return False
        
        return True
    
    def post_inference_validate(
        self,
        metrics: "InferenceMetrics"
    ) -> ContractViolationReport:
        """
        Post-inference validation.
        
        Validates actual metrics against contract.
        
        Args:
            metrics: Actual inference metrics
            
        Returns:
            Violation report
        """
        violations = []
        
        # Latency check
        if metrics.latency_ms > self.contract.sla.max_latency_ms:
            violations.append(ContractViolation(
                dimension="latency",
                constraint_name="max_latency",
                expected=f"< {self.contract.sla.max_latency_ms}ms",
                actual=f"{metrics.latency_ms:.2f}ms",
                severity=self._latency_severity(metrics.latency_ms)
            ))
        
        # Throughput check
        if metrics.num_tokens > 0 and metrics.latency_ms > 0:
            throughput = metrics.num_tokens / (metrics.latency_ms / 1000)
            if throughput < self.contract.sla.min_throughput_tokens_per_sec:
                violations.append(ContractViolation(
                    dimension="throughput",
                    constraint_name="min_throughput",
                    expected=f"> {self.contract.sla.min_throughput_tokens_per_sec} tok/s",
                    actual=f"{throughput:.2f} tok/s",
                    severity=ViolationSeverity.MEDIUM
                ))
        
        # Attention coverage check
        if metrics.attention_entropy > 0:
            # Higher entropy = lower coverage
            coverage = 1.0 - metrics.attention_entropy
            if coverage < self.contract.quality.min_attention_coverage:
                violations.append(ContractViolation(
                    dimension="quality",
                    constraint_name="min_attention_coverage",
                    expected=f"> {self.contract.quality.min_attention_coverage}",
                    actual=f"{coverage:.3f}",
                    severity=ViolationSeverity.HIGH
                ))
        
        # Expert diversity check
        if metrics.expert_distribution:
            diversity = self._compute_expert_diversity(metrics.expert_distribution)
            if diversity < self.contract.quality.min_expert_diversity:
                violations.append(ContractViolation(
                    dimension="quality",
                    constraint_name="min_expert_diversity",
                    expected=f"> {self.contract.quality.min_expert_diversity}",
                    actual=f"{diversity:.3f}",
                    severity=ViolationSeverity.MEDIUM
                ))
        
        report = ContractViolationReport(
            contract_name=self.contract.name,
            task_id=metrics.task_id,
            violations=violations
        )
        
        # Store history
        self._violation_history.append(report)
        if len(self._violation_history) > self._max_history:
            self._violation_history.pop(0)
        
        # Callback
        if report.has_violations and self.on_violation:
            self.on_violation(report)
        
        return report
    
    def _latency_severity(self, latency_ms: float) -> ViolationSeverity:
        """Determine latency violation severity."""
        ratio = latency_ms / self.contract.sla.max_latency_ms
        if ratio > 2.0:
            return ViolationSeverity.CRITICAL
        elif ratio > 1.5:
            return ViolationSeverity.HIGH
        elif ratio > 1.0:
            return ViolationSeverity.MEDIUM
        return ViolationSeverity.LOW
    
    def _compute_expert_diversity(self, distribution: Dict[str, float]) -> float:
        """Compute expert diversity (entropy-based)."""
        if not distribution:
            return 0.0
        
        # Normalize
        total = sum(distribution.values())
        if total == 0:
            return 0.0
        
        probs = [v / total for v in distribution.values()]
        
        # Compute entropy
        import math
        entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        
        # Normalize by max entropy (uniform distribution)
        n = len(probs)
        max_entropy = math.log(n)
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def apply_degradation(
        self,
        base_depth: int,
        reason: str = "contract_risk"
    ) -> int:
        """
        Apply degradation strategy to reduce latency.
        
        Args:
            base_depth: Original planned depth
            reason: Reason for degradation
            
        Returns:
            Degraded depth
        """
        # Reduce depth proportionally to SLA ratio
        max_allowed_depth = int(self.contract.sla.max_latency_ms / 5.0)
        degraded_depth = min(base_depth, max(self.contract.sla.max_batch_size, max_allowed_depth))
        
        return degraded_depth
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enforcement statistics."""
        total = len(self._violation_history)
        violated = sum(1 for r in self._violation_history if r.has_violations)
        
        severity_counts = defaultdict(int)
        for report in self._violation_history:
            severity_counts[report.severity.value] += 1
        
        return {
            "contract_name": self.contract.name,
            "total_inferences": total,
            "violated_inferences": violated,
            "violation_rate": violated / total if total > 0 else 0.0,
            "severity_distribution": dict(severity_counts),
        }


# =============================================================================
# Degradation Strategies
# =============================================================================

class DegradationStrategies:
    """Predefined degradation strategies for contract violations."""
    
    @staticmethod
    def reduce_depth(current_depth: int, reduction_factor: float = 0.5) -> int:
        """Reduce loop depth."""
        return max(4, int(current_depth * reduction_factor))
    
    @staticmethod
    def reduce_batch_size(current_batch: int, reduction_factor: float = 0.5) -> int:
        """Reduce batch size."""
        return max(1, int(current_batch * reduction_factor))
    
    @staticmethod
    def skip_quality_checks(current_enabled: bool) -> bool:
        """Skip quality checks to reduce overhead."""
        return False
    
    @staticmethod
    def use_faster_attention(current_type: str) -> str:
        """Switch to faster attention mechanism."""
        if current_type == "mla":
            return "gqa"
        return "flash"
    
    @staticmethod
    def limit_experts(current_n: int) -> int:
        """Reduce number of experts."""
        return max(1, current_n // 2)


# =============================================================================
# Contract Manager
# =============================================================================

class ContractManager:
    """
    Manages multiple inference contracts.
    
    Provides contract selection, switching, and monitoring
    across different workloads.
    """
    
    def __init__(self):
        self._contracts: Dict[str, InferenceContract] = {}
        self._enforcers: Dict[str, ContractEnforcer] = {}
        self._default_contract: Optional[str] = None
        self._violation_callbacks: List[Callable] = []
    
    def register_contract(
        self,
        contract: InferenceContract,
        set_as_default: bool = False
    ) -> None:
        """Register a contract."""
        self._contracts[contract.name] = contract
        self._enforcers[contract.name] = ContractEnforcer(
            contract,
            on_violation=self._handle_violation
        )
        
        if set_as_default or self._default_contract is None:
            self._default_contract = contract.name
    
    def get_contract(self, name: str) -> Optional[InferenceContract]:
        """Get contract by name."""
        return self._contracts.get(name)
    
    def get_enforcer(self, name: Optional[str] = None) -> ContractEnforcer:
        """Get enforcer for contract."""
        name = name or self._default_contract
        return self._enforcers.get(name)
    
    def switch_contract(self, name: str) -> bool:
        """Switch default contract."""
        if name in self._contracts:
            self._default_contract = name
            return True
        return False
    
    def register_violation_callback(self, callback: Callable) -> None:
        """Register callback for violation notifications."""
        self._violation_callbacks.append(callback)
    
    def _handle_violation(self, report: ContractViolationReport) -> None:
        """Handle violation notification."""
        for callback in self._violation_callbacks:
            try:
                callback(report)
            except Exception:
                pass  # Don't let callbacks break enforcement
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all contracts."""
        return {
            name: enforcer.get_stats()
            for name, enforcer in self._enforcers.items()
        }
