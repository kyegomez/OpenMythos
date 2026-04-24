"""
Tool Execution Engine - Safe tool execution with timeout, truncation, error handling
"""

import asyncio
import concurrent.futures
import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict


class ExecutionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    APPROVAL_DENIED = "approval_denied"
    RATE_LIMITED = "rate_limited"


@dataclass
class ExecutionResult:
    tool_name: str
    status: ExecutionStatus
    result: str
    execution_time_ms: float
    truncated: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPolicy:
    max_timeout_seconds: float = 120.0
    max_result_size: int = 100000
    max_concurrent: int = 5
    require_approval_for_dangerous: bool = True
    rate_limit_per_minute: int = 60


class ApprovalCallback:
    def __init__(self):
        self._callbacks: List[Callable[[str, Dict], bool]] = []
    
    def register(self, callback: Callable[[str, Dict], bool]) -> None:
        self._callbacks.append(callback)
    
    def request_approval(self, tool_name: str, args: Dict) -> bool:
        for callback in self._callbacks:
            try:
                if callback(tool_name, args):
                    return True
            except Exception:
                pass
        return False


class RateLimiter:
    def __init__(self, max_per_minute: int = 60):
        self.max_per_minute = max_per_minute
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def is_allowed(self, tool_name: str) -> bool:
        with self._lock:
            now = time.time()
            minute_ago = now - 60
            self._requests[tool_name] = [t for t in self._requests[tool_name] if t > minute_ago]
            return len(self._requests[tool_name]) < self.max_per_minute
    
    def record(self, tool_name: str) -> None:
        with self._lock:
            self._requests[tool_name].append(time.time())
    
    def wait_time(self, tool_name: str) -> float:
        with self._lock:
            if tool_name not in self._requests or not self._requests[tool_name]:
                return 0
            oldest = min(self._requests[tool_name])
            cutoff = time.time() - 60
            return max(0, oldest - cutoff + 1)


class ToolExecutor:
    """
    Executes tools safely with timeout, truncation, and error handling.
    
    Usage:
        executor = ToolExecutor()
        result = executor.execute("read_file", {"path": "/tmp/test.txt"})
    """
    
    def __init__(self, policy: Optional[ExecutionPolicy] = None):
        self.policy = policy or ExecutionPolicy()
        self.rate_limiter = RateLimiter(self.policy.max_concurrent)
        self.approval_callback = ApprovalCallback()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.policy.max_concurrent)
    
    def execute(self, tool_name: str, args: Dict[str, Any]) -> ExecutionResult:
        from .registry import registry, tool_error
        
        start_time = time.time()
        
        if not self.rate_limiter.is_allowed(tool_name):
            wait = self.rate_limiter.wait_time(tool_name)
            return ExecutionResult(
                tool_name=tool_name, status=ExecutionStatus.RATE_LIMITED,
                result=tool_error(f"Rate limited. Wait {wait:.1f}s"),
                execution_time_ms=(time.time() - start_time)*1000, error=f"Rate limited"
            )
        
        entry = registry.get_entry(tool_name)
        if not entry:
            return ExecutionResult(
                tool_name=tool_name, status=ExecutionStatus.FAILED,
                result=tool_error(f"Tool '{tool_name}' not found"),
                execution_time_ms=(time.time() - start_time)*1000, error="Tool not found"
            )
        
        if entry.danger_level >= 2 and self.policy.require_approval_for_dangerous:
            if not self.approval_callback.request_approval(tool_name, args):
                return ExecutionResult(
                    tool_name=tool_name, status=ExecutionStatus.APPROVAL_DENIED,
                    result=tool_error("Approval denied for dangerous tool"),
                    execution_time_ms=(time.time() - start_time)*1000, error="Approval denied"
                )
        
        self.rate_limiter.record(tool_name)
        
        try:
            if entry.is_async:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(asyncio.wait_for(entry.handler(**args), timeout=self.policy.max_timeout_seconds))
                finally:
                    loop.close()
            else:
                future = self._executor.submit(entry.handler, **args)
                result = future.result(timeout=self.policy.max_timeout_seconds)
            
            execution_time_ms = (time.time() - start_time) * 1000
            result_str = result if isinstance(result, str) else json.dumps(result)
            
            truncated = len(result_str) > self.policy.max_result_size
            if truncated:
                result_str = result_str[:self.policy.max_result_size] + "... [truncated]"
            
            return ExecutionResult(
                tool_name=tool_name, status=ExecutionStatus.SUCCESS,
                result=result_str, execution_time_ms=execution_time_ms, truncated=truncated
            )
            
        except concurrent.futures.TimeoutError:
            return ExecutionResult(
                tool_name=tool_name, status=ExecutionStatus.TIMEOUT,
                result=tool_error(f"Tool execution timed out after {self.policy.max_timeout_seconds}s"),
                execution_time_ms=(time.time() - start_time)*1000, error="Timeout"
            )
        except Exception as e:
            return ExecutionResult(
                tool_name=tool_name, status=ExecutionStatus.FAILED,
                result=tool_error(str(e)), execution_time_ms=(time.time() - start_time)*1000, error=str(e)
            )
    
    async def execute_async(self, tool_name: str, args: Dict[str, Any]) -> ExecutionResult:
        from .registry import registry, tool_error
        start_time = time.time()
        
        if not self.rate_limiter.is_allowed(tool_name):
            return ExecutionResult(
                tool_name=tool_name, status=ExecutionStatus.RATE_LIMITED,
                result=tool_error("Rate limited"), execution_time_ms=(time.time()-start_time)*1000
            )
        
        entry = registry.get_entry(tool_name)
        if not entry:
            return ExecutionResult(
                tool_name=tool_name, status=ExecutionStatus.FAILED,
                result=tool_error("Tool not found"), execution_time_ms=(time.time()-start_time)*1000
            )
        
        self.rate_limiter.record(tool_name)
        
        try:
            if entry.is_async:
                result = await asyncio.wait_for(entry.handler(**args), timeout=self.policy.max_timeout_seconds)
            else:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(loop.run_in_executor(self._executor, lambda: entry.handler(**args)), timeout=self.policy.max_timeout_seconds)
            
            result_str = result if isinstance(result, str) else json.dumps(result)
            truncated = len(result_str) > self.policy.max_result_size
            if truncated:
                result_str = result_str[:self.policy.max_result_size] + "... [truncated]"
            
            return ExecutionResult(tool_name=tool_name, status=ExecutionStatus.SUCCESS, result=result_str, execution_time_ms=(time.time()-start_time)*1000, truncated=truncated)
        except asyncio.TimeoutError:
            return ExecutionResult(tool_name=tool_name, status=ExecutionStatus.TIMEOUT, result=tool_error("Timeout"), execution_time_ms=(time.time()-start_time)*1000, error="Timeout")
        except Exception as e:
            return ExecutionResult(tool_name=tool_name, status=ExecutionStatus.FAILED, result=tool_error(str(e)), execution_time_ms=(time.time()-start_time)*1000, error=str(e))
    
    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


class ToolExecutionMonitor:
    """Monitor tool execution metrics."""
    
    def __init__(self):
        self._metrics: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def record(self, result: ExecutionResult) -> None:
        with self._lock:
            self._metrics[result.tool_name].append({
                "status": result.status.value,
                "execution_time_ms": result.execution_time_ms,
                "truncated": result.truncated,
                "timestamp": time.time()
            })
            if len(self._metrics[result.tool_name]) > 1000:
                self._metrics[result.tool_name] = self._metrics[result.tool_name][-1000:]
    
    def get_stats(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if tool_name:
                return self._calc_stats(tool_name, self._metrics.get(tool_name, []))
            return {name: self._calc_stats(name, metrics) for name, metrics in self._metrics.items()}
    
    def _calc_stats(self, tool_name: str, metrics: List[Dict]) -> Dict[str, Any]:
        if not metrics:
            return {"count": 0}
        recent = metrics[-100:]
        success = sum(1 for m in recent if m["status"] == "success")
        return {
            "count": len(metrics),
            "recent_success_rate": success / len(recent) if recent else 0,
            "avg_time_ms": sum(m["execution_time_ms"] for m in recent) / len(recent) if recent else 0,
            "total_timeouts": sum(1 for m in metrics if m["status"] == "timeout")
        }
