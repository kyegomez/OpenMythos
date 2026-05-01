"""
Task router — decides whether to use local inference or escalate to Claude API.

Routing logic (rule-based, no external model needed):
  LOCAL  → fast, private, free, works offline
  API    → complex reasoning, returns compressed_context to reduce token cost
"""

from dataclasses import dataclass
from typing import Literal

# Token count heuristic (rough chars-per-token estimate)
_CHARS_PER_TOKEN = 4
_LOCAL_TOKEN_LIMIT = 1500   # tasks with more code context escalate
_LOCAL_KEYWORDS = {
    "explain", "docstring", "comment", "summarize", "summary",
    "lint", "format", "style", "rename", "boilerplate", "scaffold",
    "type hint", "typing", "simple", "quick", "what does",
}
_API_KEYWORDS = {
    "architect", "design", "system", "multi-file", "refactor across",
    "debug complex", "security audit", "performance", "algorithm",
    "why does", "how should", "trade-off", "compare",
}


@dataclass
class RouteDecision:
    use: Literal["local", "api"]
    reason: str
    confidence: float            # 0-1
    compress_first: bool         # always compress large contexts before API call


def route_task(task: str, code: str = "", offline: bool = False) -> RouteDecision:
    """Decide whether to handle locally or escalate to Claude API."""
    if offline:
        return RouteDecision(use="local", reason="offline mode", confidence=1.0, compress_first=False)

    task_lower = task.lower()
    code_tokens = len(code) // _CHARS_PER_TOKEN

    # Large context always benefits from local compression before API
    compress_first = code_tokens > _LOCAL_TOKEN_LIMIT

    # Explicit API signals
    for kw in _API_KEYWORDS:
        if kw in task_lower:
            return RouteDecision(
                use="api",
                reason=f"task keyword '{kw}' indicates need for deep reasoning",
                confidence=0.85,
                compress_first=compress_first,
            )

    # Explicit local signals
    for kw in _LOCAL_KEYWORDS:
        if kw in task_lower:
            return RouteDecision(
                use="local",
                reason=f"task keyword '{kw}' is well-suited for local inference",
                confidence=0.8,
                compress_first=False,
            )

    # Context size decides
    if code_tokens > _LOCAL_TOKEN_LIMIT:
        return RouteDecision(
            use="api",
            reason=f"code context ~{code_tokens} tokens exceeds local limit",
            confidence=0.7,
            compress_first=True,
        )

    # Default: try local for short tasks
    return RouteDecision(
        use="local",
        reason="short task, defaulting to local inference",
        confidence=0.6,
        compress_first=False,
    )
