"""
Tests for Context Compression Engine
"""

import pytest
import json

from open_mythos.context.context_engine import (
    ContextEngine, Message, CompressionStrategy,
    SummarizingCompressor, ReferenceCompressor, PriorityCompressor,
    ContextCache, ContextPressure
)


class TestMessage:
    """Tests for Message."""
    
    def test_create_message(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello", importance=0.8)
        
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.importance == 0.8
    
    def test_to_dict(self):
        """Test converting to dict."""
        msg = Message(role="assistant", content="Hi there", importance=0.5)
        
        d = msg.to_dict()
        
        assert d["role"] == "assistant"
        assert d["content"] == "Hi there"
        assert d["importance"] == 0.5


class TestContextEngine:
    """Tests for ContextEngine."""
    
    def setup_method(self):
        """Set up fresh context engine for each test."""
        self.engine = ContextEngine()
    
    def test_add_message(self):
        """Test adding messages."""
        self.engine.add_message("user", "Hello", importance=0.8)
        self.engine.add_message("assistant", "Hi there", importance=0.5)
        
        assert len(self.engine.messages) == 2
        assert self.engine.messages[0].content == "Hello"
    
    def test_add_system_message(self):
        """Test adding system messages."""
        self.engine.add_system_message("You are a helpful assistant.")
        
        assert len(self.engine.messages) == 1
        assert self.engine.messages[0].role == "system"
    
    def test_add_tool_message(self):
        """Test adding tool messages."""
        self.engine.add_message("user", "Use the tool", importance=0.5)
        self.engine.add_tool_result("tool_123", "tool result content")
        
        # Should have user message + tool result
        assert len(self.engine.messages) == 2
    
    def test_check_pressure(self):
        """Test pressure calculation."""
        # Add many messages
        for i in range(100):
            self.engine.add_message("user", f"Message {i}" * 100, importance=0.5)
        
        pressure = self.engine.check_pressure(max_tokens=1000)
        
        assert pressure.should_compress == True
        assert pressure.pressure_ratio > 0.5
    
    def test_get_context(self):
        """Test getting context."""
        self.engine.add_message("system", "You are helpful.", importance=1.0)
        self.engine.add_message("user", "Hello", importance=0.8)
        self.engine.add_message("assistant", "Hi", importance=0.5)
        
        context = self.engine.get_context(max_tokens=5000)
        
        assert len(context) == 3
    
    def test_compress_summarize(self):
        """Test summarization compression."""
        # Add many messages
        for i in range(50):
            self.engine.add_message("user", f"Message {i}", importance=0.5)
        
        original_count = len(self.engine.messages)
        
        result = self.engine.compress(strategy=CompressionStrategy.SUMMARIZE, max_tokens=500)
        
        assert result.compressed_count < original_count
        assert result.compression_ratio > 0
    
    def test_compress_reference(self):
        """Test reference compression."""
        for i in range(50):
            self.engine.add_message("user", f"Message {i}", importance=0.5)
        
        original_count = len(self.engine.messages)
        
        result = self.engine.compress(strategy=CompressionStrategy.REFERENCE, max_tokens=500)
        
        assert result.compressed_count <= original_count
    
    def test_compress_priority(self):
        """Test priority compression."""
        self.engine.add_message("user", "Low priority", importance=0.1)
        self.engine.add_message("user", "High priority", importance=0.9)
        self.engine.add_message("assistant", "Response", importance=0.7)
        
        result = self.engine.compress(strategy=CompressionStrategy.PRIORITY, max_tokens=500)
        
        # High priority message should be preserved
        context = self.engine.get_context()
        contents = [m.content for m in context]
        assert "High priority" in contents
    
    def test_cache(self):
        """Test context caching."""
        self.engine.add_message("user", "Test message", importance=0.5)
        
        # First call - cache miss
        cache1 = self.engine.get_context(max_tokens=1000)
        
        # Second call - cache hit
        cache2 = self.engine.get_context(max_tokens=1000)
        
        assert cache1 == cache2
    
    def test_cache_invalidation(self):
        """Test cache invalidation on new message."""
        self.engine.add_message("user", "First", importance=0.5)
        
        cache1 = self.engine.get_context(max_tokens=1000)
        
        # Add new message - should invalidate cache
        self.engine.add_message("user", "Second", importance=0.5)
        
        cache2 = self.engine.get_context(max_tokens=1000)
        
        assert cache1 != cache2


class TestContextCache:
    """Tests for ContextCache."""
    
    def test_cache_hit(self):
        """Test cache hit."""
        cache = ContextCache(ttl_seconds=60)
        
        cache.set("key1", ["value1", "value2"])
        
        result = cache.get("key1")
        assert result == ["value1", "value2"]
    
    def test_cache_miss(self):
        """Test cache miss."""
        cache = ContextCache(ttl_seconds=60)
        
        result = cache.get("nonexistent")
        assert result is None
    
    def test_cache_expiry(self):
        """Test cache expiry."""
        import time
        
        cache = ContextCache(ttl_seconds=1)
        
        cache.set("expiring", ["value"])
        
        time.sleep(1.5)
        
        result = cache.get("expiring")
        assert result is None


class TestCompressionStrategies:
    """Tests for compression strategies."""
    
    def test_summarizer_basic(self):
        """Test basic summarization."""
        compressor = SummarizingCompressor()
        
        messages = [
            Message(role="user", content="Message 1", importance=0.5),
            Message(role="user", content="Message 2", importance=0.5),
            Message(role="user", content="Message 3", importance=0.5),
        ]
        
        result = compressor.compress(messages, max_tokens=200)
        
        assert len(result) < len(messages)
    
    def test_reference_preservation(self):
        """Test that references are preserved."""
        compressor = ReferenceCompressor()
        
        messages = [
            Message(role="system", content="System prompt", importance=1.0),
            Message(role="user", content="Old message", importance=0.3),
        ]
        
        result = compressor.compress(messages, max_tokens=1000)
        
        # System should always be preserved
        roles = [m.role for m in result]
        assert "system" in roles


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
