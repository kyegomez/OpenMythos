"""
Tests for Tools System
"""

import pytest
import json
import tempfile
import os

from open_mythos.tools.registry import (
    ToolRegistry, ToolEntry, register_tool, tool_result, tool_error, get_registry
)
from open_mythos.tools.execution import (
    ToolExecutor, ExecutionPolicy, ExecutionStatus, RateLimiter, ApprovalCallback
)


class TestToolRegistry:
    """Tests for ToolRegistry."""
    
    def setup_method(self):
        """Reset registry before each test."""
        # Create fresh registry for testing
        self.registry = ToolRegistry.__new__(ToolRegistry)
        self.registry._init()
    
    def test_register_tool(self):
        """Test registering a basic tool."""
        def handler(x: int, y: int) -> int:
            return x + y
        
        self.registry.register(
            name="add",
            toolset="math",
            schema={
                "name": "add",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"}
                    }
                }
            },
            handler=handler,
            description="Add two numbers",
            emoji="➕"
        )
        
        assert "add" in self.registry.get_all_tool_names()
        assert self.registry.get_tool_to_toolset_map()["add"] == "math"
    
    def test_deregister_tool(self):
        """Test removing a tool."""
        def handler() -> str:
            return "test"
        
        self.registry.register(
            name="test_tool",
            toolset="test",
            schema={"name": "test_tool"},
            handler=handler
        )
        
        assert "test_tool" in self.registry.get_all_tool_names()
        
        self.registry.deregister("test_tool")
        
        assert "test_tool" not in self.registry.get_all_tool_names()
    
    def test_get_entry(self):
        """Test getting tool entry."""
        def handler() -> str:
            return "test"
        
        self.registry.register(
            name="get_test",
            toolset="test",
            schema={"name": "get_test"},
            handler=handler,
            danger_level=2
        )
        
        entry = self.registry.get_entry("get_test")
        assert entry is not None
        assert entry.danger_level == 2
    
    def test_tool_availability_check(self):
        """Test tool availability check."""
        available_check = lambda: True
        
        self.registry.register(
            name="available_tool",
            toolset="test",
            schema={"name": "available_tool"},
            handler=lambda: "test",
            check_fn=available_check
        )
        
        entry = self.registry.get_entry("available_tool")
        assert entry.is_available() == True
    
    def test_duplicate_registration_rejected(self):
        """Test that duplicate tool names are rejected."""
        def handler1() -> str:
            return "one"
        
        def handler2() -> str:
            return "two"
        
        self.registry.register(
            name="dup_test",
            toolset="test",
            schema={"name": "dup_test"},
            handler=handler1
        )
        
        # Second registration should be rejected
        self.registry.register(
            name="dup_test",
            toolset="test2",
            schema={"name": "dup_test"},
            handler=handler2
        )
        
        # First handler should still be there
        entry = self.registry.get_entry("dup_test")
        assert entry.handler() == "one"


class TestToolExecutor:
    """Tests for ToolExecutor."""
    
    def setup_method(self):
        """Reset registry before each test."""
        self.registry = ToolRegistry.__new__(ToolRegistry)
        self.registry._init()
        
        # Register a test tool
        def add(x: int, y: int) -> int:
            return x + y
        
        self.registry.register(
            name="add",
            toolset="math",
            schema={"name": "add"},
            handler=add
        )
        
        # Register a slow tool
        import time
        def slow() -> str:
            time.sleep(0.1)
            return "slow result"
        
        self.registry.register(
            name="slow",
            toolset="test",
            schema={"name": "slow"},
            handler=slow
        )
        
        self.executor = ToolExecutor()
    
    def test_execute_success(self):
        """Test successful tool execution."""
        result = self.executor.execute("add", {"x": 2, "y": 3})
        
        assert result.status == ExecutionStatus.SUCCESS
        assert "5" in result.result
        assert result.execution_time_ms > 0
    
    def test_execute_tool_not_found(self):
        """Test executing non-existent tool."""
        result = self.executor.execute("nonexistent", {})
        
        assert result.status == ExecutionStatus.FAILED
        assert "not found" in result.result
    
    def test_execute_timeout(self):
        """Test tool timeout."""
        # Create executor with very short timeout
        policy = ExecutionPolicy(max_timeout_seconds=0.01)
        executor = ToolExecutor(policy=policy)
        
        result = executor.execute("slow", {})
        
        assert result.status == ExecutionStatus.TIMEOUT
    
    def test_result_truncation(self):
        """Test result truncation for large outputs."""
        def big_output() -> str:
            return "x" * 200000
        
        self.registry.register(
            name="big_output",
            toolset="test",
            schema={"name": "big_output"},
            handler=big_output
        )
        
        policy = ExecutionPolicy(max_result_size=1000)
        executor = ToolExecutor(policy=policy)
        
        result = executor.execute("big_output", {})
        
        assert result.truncated == True
        assert len(result.result) < 200000


class TestRateLimiter:
    """Tests for RateLimiter."""
    
    def test_rate_limit(self):
        """Test rate limiting."""
        limiter = RateLimiter(max_per_minute=2)
        
        assert limiter.is_allowed("test") == True
        limiter.record("test")
        assert limiter.is_allowed("test") == True
        limiter.record("test")
        assert limiter.is_allowed("test") == False
    
    def test_wait_time(self):
        """Test wait time calculation."""
        limiter = RateLimiter(max_per_minute=1)
        
        limiter.record("test")
        wait = limiter.wait_time("test")
        
        assert wait > 0


class TestToolResult:
    """Tests for tool result helpers."""
    
    def test_tool_result(self):
        """Test tool_result helper."""
        result = tool_result({"key": "value"})
        data = json.loads(result)
        
        assert data["key"] == "value"
    
    def test_tool_error(self):
        """Test tool_error helper."""
        error = tool_error("Something went wrong")
        data = json.loads(error)
        
        assert data["error"] == "Something went wrong"


class TestApprovalCallback:
    """Tests for ApprovalCallback."""
    
    def test_approval_granted(self):
        """Test approval is granted."""
        callback = ApprovalCallback()
        
        def approve(tool_name: str, args: dict) -> bool:
            return True
        
        callback.register(approve)
        
        assert callback.request_approval("dangerous_tool", {}) == True
    
    def test_approval_denied(self):
        """Test approval is denied."""
        callback = ApprovalCallback()
        
        def deny(tool_name: str, args: dict) -> bool:
            return False
        
        callback.register(deny)
        
        assert callback.request_approval("dangerous_tool", {}) == False
    
    def test_no_callbacks(self):
        """Test no callbacks registered."""
        callback = ApprovalCallback()
        
        assert callback.request_approval("any_tool", {}) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
