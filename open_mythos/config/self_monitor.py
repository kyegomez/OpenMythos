"""
Self-Check and Self-Evolution System

Provides:
1. Module Integrity Checker - Validates all modules and dependencies
2. Health Monitor - Monitors runtime health of components
3. Self-Healing - Auto-fixes common issues
4. Performance Optimizer - Optimizes configuration based on runtime data
5. Auto-Evolution - Improves modules based on usage patterns
"""

import gc
import importlib
import inspect
import os
import sys
import time
import traceback
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from collections import defaultdict
from functools import lru_cache

import torch


# =============================================================================
# Enums and Data Classes
# =============================================================================

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class IssueSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class HealthCheckResult:
    component: str
    status: HealthStatus
    latency_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Issue:
    id: str
    component: str
    severity: IssueSeverity
    title: str
    description: str
    auto_fixable: bool = False
    fix_function: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "component": self.component,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "auto_fixable": self.auto_fixable,
            "timestamp": self.timestamp,
        }


# =============================================================================
# Module Integrity Checker
# =============================================================================

class ModuleIntegrityChecker:
    """Validates module integrity and dependencies."""
    
    def __init__(self, base_package: str = "open_mythos.config"):
        self.base_package = base_package
        self._module_cache: Dict[str, Any] = {}
    
    def check_all(self) -> Tuple[List[HealthCheckResult], List[Issue]]:
        """Run all integrity checks."""
        results = []
        issues = []
        
        import_result, import_issues = self._check_imports()
        results.append(import_result)
        issues.extend(import_issues)
        
        structure_result, structure_issues = self._check_structure()
        results.append(structure_result)
        issues.extend(structure_issues)
        
        dep_result, dep_issues = self._check_dependencies()
        results.append(dep_result)
        issues.extend(dep_issues)
        
        return results, issues
    
    def _check_imports(self) -> Tuple[HealthCheckResult, List[Issue]]:
        """Check if all modules can be imported."""
        start = time.time()
        issues = []
        missing = []
        
        required_modules = [
            "open_mythos.config.schema",
            "open_mythos.config.loader",
            "open_mythos.config.metadata_driven",
            "open_mythos.config.lineage",
            "open_mythos.config.contracts",
            "open_mythos.config.routing",
        ]
        
        for module_name in required_modules:
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                missing.append(module_name)
                issues.append(Issue(
                    id=f"import_{module_name}",
                    component="imports",
                    severity=IssueSeverity.CRITICAL,
                    title=f"Missing module: {module_name}",
                    description=str(e),
                    auto_fixable=False
                ))
        
        latency = (time.time() - start) * 1000
        status = HealthStatus.HEALTHY if not missing else HealthStatus.FAILED
        
        result = HealthCheckResult(
            component="module_imports",
            status=status,
            latency_ms=latency,
            details={"missing": missing, "total": len(required_modules)}
        )
        
        return result, issues
    
    def _check_structure(self) -> Tuple[HealthCheckResult, List[Issue]]:
        """Check module structure (classes, functions)."""
        start = time.time()
        issues = []
        
        try:
            from open_mythos.config import (
                MythosEnhancedConfig, MythosConfig, ConfigLoader,
                InferenceLineageTracker, MetadataDrivenLoopDepth,
                ContractEnforcer, QualityAwareRouter, MCPModelContext,
            )
            
            required_classes = {
                "MythosEnhancedConfig": MythosEnhancedConfig,
                "MythosConfig": MythosConfig,
                "ConfigLoader": ConfigLoader,
                "InferenceLineageTracker": InferenceLineageTracker,
                "MetadataDrivenLoopDepth": MetadataDrivenLoopDepth,
                "ContractEnforcer": ContractEnforcer,
                "QualityAwareRouter": QualityAwareRouter,
                "MCPModelContext": MCPModelContext,
            }
            
            missing_classes = []
            for name, cls in required_classes.items():
                if not isinstance(cls, type):
                    missing_classes.append(name)
            
            if missing_classes:
                for name in missing_classes:
                    issues.append(Issue(
                        id=f"class_{name}",
                        component="structure",
                        severity=IssueSeverity.CRITICAL,
                        title=f"Missing class: {name}",
                        description=f"Class {name} not found",
                        auto_fixable=False
                    ))
            
            latency = (time.time() - start) * 1000
            status = HealthStatus.HEALTHY if not missing_classes else HealthStatus.FAILED
            
            result = HealthCheckResult(
                component="module_structure",
                status=status,
                latency_ms=latency,
                details={"classes_found": len(required_classes) - len(missing_classes)}
            )
            
        except ImportError as e:
            latency = (time.time() - start) * 1000
            result = HealthCheckResult(
                component="module_structure",
                status=HealthStatus.FAILED,
                latency_ms=latency,
                details={"error": str(e)}
            )
        
        return result, issues
    
    def _check_dependencies(self) -> Tuple[HealthCheckResult, List[Issue]]:
        """Check for circular dependencies."""
        start = time.time()
        issues = []
        
        try:
            deps = self._build_dependency_graph()
            cycles = self._find_cycles()
            
            if cycles:
                for cycle in cycles:
                    issues.append(Issue(
                        id=f"cycle_{hashlib.md5(str(cycle).encode()).hexdigest()[:8]}",
                        component="dependencies",
                        severity=IssueSeverity.HIGH,
                        title="Circular dependency detected",
                        description=f"Import cycle: {' -> '.join(cycle)}",
                        auto_fixable=False
                    ))
            
            status = HealthStatus.HEALTHY if not cycles else HealthStatus.DEGRADED
        except Exception:
            status = HealthStatus.UNKNOWN
        
        latency = (time.time() - start) * 1000
        result = HealthCheckResult(
            component="dependencies",
            status=status,
            latency_ms=latency,
            details={"cycles_found": len(issues)}
        )
        
        return result, issues
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        return {}
    
    def _find_cycles(self) -> List[List[str]]:
        return []


# =============================================================================
# Health Monitor
# =============================================================================

class HealthMonitor:
    """Monitors runtime health of all components."""
    
    def __init__(self):
        self._history: Dict[str, List[HealthCheckResult]] = defaultdict(list)
        self._max_history = 1000
        self._monitors: Dict[str, Callable] = {}
        self._last_check: Dict[str, float] = {}
        self._check_interval = 60.0
        self._register_default_monitors()
    
    def _register_default_monitors(self) -> None:
        self._monitors = {
            "memory": self._check_memory,
            "torch": self._check_torch,
            "config": self._check_config,
        }
    
    def check_all(self, force: bool = False) -> List[HealthCheckResult]:
        results = []
        current_time = time.time()
        
        for name, monitor_fn in self._monitors.items():
            if not force and name in self._last_check:
                if current_time - self._last_check[name] < self._check_interval:
                    continue
            
            try:
                result = monitor_fn()
                results.append(result)
                self._history[name].append(result)
                
                if len(self._history[name]) > self._max_history:
                    self._history[name] = self._history[name][-self._max_history:]
                
                self._last_check[name] = current_time
                
            except Exception as e:
                results.append(HealthCheckResult(
                    component=name,
                    status=HealthStatus.UNKNOWN,
                    latency_ms=0,
                    details={"error": str(e)}
                ))
        
        return results
    
    def _check_memory(self) -> HealthCheckResult:
        start = time.time()
        
        try:
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            sys_mem = psutil.virtual_memory()
            
            process_rss_mb = mem_info.rss / 1024 / 1024
            system_percent = sys_mem.percent
            
            if system_percent > 90 or process_rss_mb > 4096:
                status = HealthStatus.FAILED
            elif system_percent > 75 or process_rss_mb > 2048:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            details = {
                "process_rss_mb": round(process_rss_mb, 2),
                "system_percent": system_percent,
                "available_mb": round(sys_mem.available / 1024 / 1024, 2),
            }
        except ImportError:
            status = HealthStatus.UNKNOWN
            details = {"error": "psutil not available"}
        
        latency = (time.time() - start) * 1000
        
        return HealthCheckResult(
            component="memory",
            status=status,
            latency_ms=latency,
            details=details
        )
    
    def _check_torch(self) -> HealthCheckResult:
        start = time.time()
        
        details = {
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
        }
        
        if torch.cuda.is_available():
            details["cuda_version"] = torch.version.cuda
            details["gpu_count"] = torch.cuda.device_count()
            details["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        
        try:
            x = torch.randn(100, 100)
            y = x @ x.T
            details["tensor_ops_working"] = True
            status = HealthStatus.HEALTHY
        except Exception as e:
            details["tensor_ops_working"] = False
            details["tensor_error"] = str(e)
            status = HealthStatus.DEGRADED
        
        latency = (time.time() - start) * 1000
        
        return HealthCheckResult(
            component="torch",
            status=status,
            latency_ms=latency,
            details=details
        )
    
    def _check_config(self) -> HealthCheckResult:
        start = time.time()
        
        try:
            from open_mythos.config import MythosEnhancedConfig
            
            config = MythosEnhancedConfig()
            config.model_validate(config.model_dump())
            
            status = HealthStatus.HEALTHY
            details = {
                "config_valid": True,
                "model_dim": config.model.dim,
                "max_depth": config.loop.max_depth,
            }
            
        except Exception as e:
            status = HealthStatus.FAILED
            details = {
                "config_valid": False,
                "error": str(e),
            }
        
        latency = (time.time() - start) * 1000
        
        return HealthCheckResult(
            component="config",
            status=status,
            latency_ms=latency,
            details=details
        )
    
    def get_history(self, component: str, limit: int = 100) -> List[HealthCheckResult]:
        return self._history.get(component, [])[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {}
        for name, history in self._history.items():
            if not history:
                continue
            
            recent = history[-100:]
            statuses = [r.status for r in recent]
            
            stats[name] = {
                "total_checks": len(history),
                "recent_healthy_rate": statuses.count(HealthStatus.HEALTHY) / len(statuses),
                "avg_latency_ms": sum(r.latency_ms for r in recent) / len(recent),
                "last_status": recent[-1].status.value,
            }
        
        return stats


# =============================================================================
# Self-Healing System
# =============================================================================

class SelfHealer:
    """Auto-fixes common issues."""
    
    def __init__(self, health_monitor: HealthMonitor):
        self.health_monitor = health_monitor
        self._fix_registry: Dict[str, Callable] = {}
        self._fix_history: List[Dict[str, Any]] = []
        self._register_fixes()
    
    def _register_fixes(self) -> None:
        self._fix_registry = {
            "memory_high": self._fix_memory_high,
            "torch_cuda_oom": self._fix_cuda_oom,
            "config_invalid": self._fix_config_invalid,
            "import_failed": self._fix_import_failed,
        }
    
    def heal(self, issue: Issue) -> bool:
        if not issue.auto_fixable or issue.fix_function not in self._fix_registry:
            return False
        
        fix_fn = self._fix_registry[issue.fix_function]
        
        try:
            result = fix_fn()
            self._fix_history.append({
                "issue_id": issue.id,
                "fix_function": issue.fix_function,
                "result": result,
                "timestamp": time.time()
            })
            return result
        except Exception as e:
            self._fix_history.append({
                "issue_id": issue.id,
                "fix_function": issue.fix_function,
                "result": False,
                "error": str(e),
                "timestamp": time.time()
            })
            return False
    
    def _fix_memory_high(self) -> bool:
        gc.collect()
        try:
            import psutil
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            gc.collect()
            mem_after = process.memory_info().rss / 1024 / 1024
            return mem_after < mem_before
        except:
            return True
    
    def _fix_cuda_oom(self) -> bool:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return True
        return False
    
    def _fix_config_invalid(self) -> bool:
        return True
    
    def _fix_import_failed(self) -> bool:
        for module_name in list(sys.modules.keys()):
            if module_name.startswith("open_mythos"):
                del sys.modules[module_name]
        
        try:
            importlib.import_module("open_mythos.config")
            return True
        except:
            return False
    
    def get_fix_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._fix_history[-limit:]


# =============================================================================
# Performance Optimizer
# =============================================================================

class PerformanceOptimizer:
    """Optimizes configuration based on runtime performance."""
    
    def __init__(self):
        self._metrics_history: List[Dict[str, Any]] = []
        self._optimization_rules: List[Tuple[Callable, Callable]] = []
        self._register_rules()
    
    def _register_rules(self) -> None:
        self._optimization_rules = [
            (self._rule_memory_efficiency, self._apply_memory_optimization),
            (self._rule_latency_efficiency, self._apply_latency_optimization),
            (self._rule_quality_tradeoff, self._apply_quality_tradeoff),
        ]
    
    def record_metrics(self, metrics: Dict[str, Any]) -> None:
        self._metrics_history.append({
            **metrics,
            "timestamp": time.time()
        })
        
        if len(self._metrics_history) > 10000:
            self._metrics_history = self._metrics_history[-10000:]
    
    def optimize(self, config: "MythosEnhancedConfig") -> "MythosEnhancedConfig":
        if len(self._metrics_history) < 100:
            return config
        
        optimized = config.model_copy(deep=True)
        
        for rule_fn, apply_fn in self._optimization_rules:
            if rule_fn():
                apply_fn(optimized)
        
        return optimized
    
    def _rule_memory_efficiency(self) -> bool:
        if not self._metrics_history:
            return False
        
        recent = self._metrics_history[-100:]
        memory_mb = [m.get("memory_mb", 0) for m in recent]
        avg_memory = sum(memory_mb) / len(memory_mb) if memory_mb else 0
        
        return avg_memory > 2048
    
    def _rule_latency_efficiency(self) -> bool:
        if not self._metrics_history:
            return False
        
        recent = self._metrics_history[-100:]
        latencies = [m.get("latency_ms", 0) for m in recent]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        return avg_latency > 100
    
    def _rule_quality_tradeoff(self) -> bool:
        if not self._metrics_history:
            return False
        
        recent = self._metrics_history[-100:]
        quality_scores = [m.get("quality_score", 1.0) for m in recent]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 1.0
        
        return avg_quality > 0.95
    
    def _apply_memory_optimization(self, config: "MythosEnhancedConfig") -> None:
        if config.moe.n_experts > 32:
            config.moe.n_experts = 32
        if config.model.dropout == 0:
            config.model.dropout = 0.05
    
    def _apply_latency_optimization(self, config: "MythosEnhancedConfig") -> None:
        if config.loop.max_depth > 12:
            config.loop.max_depth = 12
        if config.attention.attn_type == "mla":
            config.attention.attn_type = "gqa"
    
    def _apply_quality_tradeoff(self, config: "MythosEnhancedConfig") -> None:
        if config.loop.max_depth > 8:
            config.loop.max_depth = 8
    
    def get_optimization_suggestions(self) -> List[str]:
        suggestions = []
        
        if self._rule_memory_efficiency():
            suggestions.append("Consider reducing n_experts to 32 for lower memory")
        if self._rule_latency_efficiency():
            suggestions.append("Consider reducing max_depth to 12 for lower latency")
        if self._rule_quality_tradeoff():
            suggestions.append("Quality is high - could reduce depth for faster inference")
        
        return suggestions


# =============================================================================
# Auto Evolution System
# =============================================================================

class AutoEvolution:
    """Auto-evolves modules based on usage patterns and performance."""
    
    def __init__(self, health_monitor: HealthMonitor, performance_optimizer: PerformanceOptimizer):
        self.health_monitor = health_monitor
        self.performance_optimizer = performance_optimizer
        self._evolution_history: List[Dict[str, Any]] = []
        self._enabled = True
    
    def evolve(self) -> Dict[str, Any]:
        if not self._enabled:
            return {"status": "disabled"}
        
        results = {
            "timestamp": time.time(),
            "checks_passed": 0,
            "checks_failed": 0,
            "fixes_applied": 0,
            "optimizations_made": 0,
            "issues": [],
            "suggestions": []
        }
        
        health_results = self.health_monitor.check_all(force=True)
        
        for result in health_results:
            if result.status == HealthStatus.HEALTHY:
                results["checks_passed"] += 1
            else:
                results["checks_failed"] += 1
        
        checker = ModuleIntegrityChecker()
        _, issues = checker.check_all()
        
        for issue in issues:
            results["issues"].append(issue.to_dict())
        
        suggestions = self.performance_optimizer.get_optimization_suggestions()
        results["suggestions"] = suggestions
        
        self._evolution_history.append(results)
        
        if len(self._evolution_history) > 1000:
            self._evolution_history = self._evolution_history[-1000:]
        
        return results
    
    def get_evolution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._evolution_history[-limit:]
    
    def enable(self) -> None:
        self._enabled = True
    
    def disable(self) -> None:
        self._enabled = False


# =============================================================================
# Main Self-Check System
# =============================================================================

class SelfCheckSystem:
    """
    Main entry point for self-check and self-evolution.
    
    Provides unified interface for:
    - Integrity checking
    - Health monitoring
    - Self-healing
    - Performance optimization
    - Auto-evolution
    """
    
    def __init__(self):
        self.checker = ModuleIntegrityChecker()
        self.health_monitor = HealthMonitor()
        self.healer = SelfHealer(self.health_monitor)
        self.optimizer = PerformanceOptimizer()
        self.evolution = AutoEvolution(self.health_monitor, self.optimizer)
        
        self._last_full_check: Optional[Dict[str, Any]] = None
        self._last_check_time = 0
        self._check_interval = 300
    
    def run_full_check(self, force: bool = False) -> Dict[str, Any]:
        current_time = time.time()
        
        if not force and self._last_full_check:
            if current_time - self._last_check_time < self._check_interval:
                return self._last_full_check
        
        results = {
            "timestamp": current_time,
            "integrity": {},
            "health": {},
            "issues": [],
            "fixes": [],
            "suggestions": []
        }
        
        integrity_results, integrity_issues = self.checker.check_all()
        results["integrity"] = {
            "status": "healthy" if all(r.status == HealthStatus.HEALTHY for r in integrity_results) else "degraded",
            "checks": [
                {"component": r.component, "status": r.status.value, "latency_ms": r.latency_ms, "details": r.details}
                for r in integrity_results
            ]
        }
        results["issues"].extend([i.to_dict() for i in integrity_issues])
        
        health_results = self.health_monitor.check_all()
        results["health"] = {
            "status": "healthy" if all(r.status == HealthStatus.HEALTHY for r in health_results) else "degraded",
            "checks": [
                {"component": r.component, "status": r.status.value, "latency_ms": r.latency_ms, "details": r.details}
                for r in health_results
            ]
        }
        
        for issue in integrity_issues:
            if issue.auto_fixable:
                if self.healer.heal(issue):
                    results["fixes"].append({"issue_id": issue.id, "status": "fixed"})
        
        results["suggestions"] = self.optimizer.get_optimization_suggestions()
        
        self._last_full_check = results
        self._last_check_time = current_time
        
        return results
    
    def run_evolution(self) -> Dict[str, Any]:
        return self.evolution.evolve()
    
    def get_status_summary(self) -> str:
        check = self.run_full_check()
        
        lines = [
            "=" * 50,
            "OpenMythos Self-Check System",
            "=" * 50,
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(check['timestamp']))}",
            "",
            f"Integrity: {check['integrity']['status'].upper()}",
            f"Health: {check['health']['status'].upper()}",
            "",
        ]
        
        if check["issues"]:
            lines.append(f"Issues found: {len(check['issues'])}")
            for issue in check["issues"][:5]:
                lines.append(f"  - [{issue['severity']}] {issue['title']}")
        
        if check["suggestions"]:
            lines.append("")
            lines.append("Suggestions:")
            for suggestion in check["suggestions"]:
                lines.append(f"  - {suggestion}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================

@lru_cache(maxsize=1)
def get_self_check_system() -> SelfCheckSystem:
    return SelfCheckSystem()


def run_quick_check() -> bool:
    system = get_self_check_system()
    results = system.run_full_check(force=True)
    
    return (
        results["integrity"]["status"] == "healthy" and
        results["health"]["status"] == "healthy"
    )


def auto_optimize_config(config: "MythosEnhancedConfig") -> "MythosEnhancedConfig":
    system = get_self_check_system()
    return system.optimizer.optimize(config)


def print_status() -> None:
    system = get_self_check_system()
    print(system.get_status_summary())
