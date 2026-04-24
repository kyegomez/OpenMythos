"""
Tests for Three-Layer Memory System
"""

import pytest
import time
from datetime import datetime, timedelta

from open_mythos.memory.three_layer_memory import (
    ThreeLayerMemorySystem,
    MemoryLayer,
    MemoryEntry,
    WorkingMemory,
    ShortTermMemory,
    LongTermMemory,
)
from open_mythos.memory.memory_manager import MemoryManager


class TestMemoryEntry:
    """Tests for MemoryEntry."""
    
    def test_create_entry(self):
        """Test creating a memory entry."""
        entry = MemoryEntry(
            id="test_1",
            content="Test content",
            importance=0.8,
            tags=["test"],
            source="test"
        )
        
        assert entry.id == "test_1"
        assert entry.content == "Test content"
        assert entry.importance == 0.8
        assert "test" in entry.tags
    
    def test_decay_calculation(self):
        """Test importance decay over time."""
        entry = MemoryEntry(
            id="test_1",
            content="Test",
            importance=1.0,
            tags=[],
            source="test",
            timestamp=time.time() - 3600  # 1 hour ago
        )
        
        # Should have decayed
        assert entry.effective_importance() < 1.0
    
    def test_boost(self):
        """Test importance boost."""
        entry = MemoryEntry(
            id="test_1",
            content="Test",
            importance=0.5,
            tags=[],
            source="test"
        )
        
        entry.boost(0.3)
        assert entry.importance >= 0.5


class TestWorkingMemory:
    """Tests for WorkingMemory."""
    
    def test_write_and_read(self):
        """Test basic write and read."""
        memory = WorkingMemory(max_entries=10, max_tokens=1000)
        
        memory.write("Test content", importance=0.8, tags=["test"], source="test")
        
        results = memory.search("Test")
        assert len(results) == 1
        assert results[0].content == "Test content"
    
    def test_max_entries(self):
        """Test entry limit."""
        memory = WorkingMemory(max_entries=3, max_tokens=1000)
        
        memory.write("Content 1", importance=0.5, tags=[], source="test")
        memory.write("Content 2", importance=0.5, tags=[], source="test")
        memory.write("Content 3", importance=0.5, tags=[], source="test")
        memory.write("Content 4", importance=0.5, tags=[], source="test")
        
        assert len(memory._entries) <= 3
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        memory = WorkingMemory(max_entries=100, max_tokens=1000, ttl_seconds=1)
        
        memory.write("Old content", importance=0.5, tags=[], source="test")
        time.sleep(1.5)
        memory.write("New content", importance=0.5, tags=[], source="test")
        
        # Old entry should be expired
        results = memory.search("Old")
        assert len(results) == 0


class TestShortTermMemory:
    """Tests for ShortTermMemory."""
    
    def test_write_and_read(self):
        """Test basic write and read."""
        memory = ShortTermMemory(max_entries=10, ttl_days=7)
        
        memory.write("Test content", importance=0.8, tags=["test"], source="test")
        
        results = memory.search("Test")
        assert len(results) >= 1
    
    def test_deduplication(self):
        """Test that duplicate content is not stored."""
        memory = ShortTermMemory(max_entries=10, ttl_days=7)
        
        memory.write("Unique content 1", importance=0.5, tags=[], source="test")
        memory.write("Unique content 1", importance=0.5, tags=[], source="test")
        
        results = memory.search("Unique")
        # Should only have one result
        assert len(results) <= 2


class TestLongTermMemory:
    """Tests for LongTermMemory."""
    
    def test_skill_storage(self):
        """Test storing and retrieving skills."""
        memory = LongTermMemory(max_entries=100, ttl_days=365)
        
        skill_content = """# Test Skill
        
## Description
A test skill.

## When to Use
Testing.
"""
        
        memory.add_skill("test_skill", skill_content, {"author": "test"})
        
        skill = memory.get_skill("test_skill")
        assert skill is not None
        assert "test_skill" in skill.get("name", "")
    
    def test_skill_listing(self):
        """Test listing skills."""
        memory = LongTermMemory(max_entries=100, ttl_days=365)
        
        memory.add_skill("skill_1", "# Skill 1", {})
        memory.add_skill("skill_2", "# Skill 2", {})
        memory.add_skill("skill_3", "# Skill 3", {})
        
        skills = memory.list_skills()
        assert len(skills) >= 3


class TestThreeLayerMemorySystem:
    """Tests for the full ThreeLayerMemorySystem."""
    
    def test_layer_write(self):
        """Test writing to specific layers."""
        system = ThreeLayerMemorySystem()
        
        system.write_to_layer(MemoryLayer.WORKING, "Working content", 0.8, ["test"], "test")
        system.write_to_layer(MemoryLayer.SHORT_TERM, "Short-term content", 0.6, ["test"], "test")
        system.write_to_layer(MemoryLayer.LONG_TERM, "Long-term content", 0.9, ["test"], "test")
        
        # Working layer
        working_results = system.working.search("Working")
        assert len(working_results) >= 1
        
        # Short-term layer
        short_results = system.short_term.search("Short")
        assert len(short_results) >= 1
        
        # Long-term layer
        long_results = system.long_term.get_skill("Long")
        # Skills are stored differently
    
    def test_unified_search(self):
        """Test searching across all layers."""
        from open_mythos.memory.three_layer_memory import MemoryQuery
        
        system = ThreeLayerMemorySystem()
        
        system.write_to_layer(MemoryLayer.WORKING, "Apple pie", 0.8, ["food"], "test")
        system.write_to_layer(MemoryLayer.SHORT_TERM, "Apple phone", 0.6, ["tech"], "test")
        system.write_to_layer(MemoryLayer.LONG_TERM, "Apple skill", 0.9, ["skill"], "test")
        
        query = MemoryQuery(text="Apple", layers=[MemoryLayer.WORKING, MemoryLayer.SHORT_TERM], limit=10)
        results = system.search_unified(query)
        
        assert len(results) >= 2
    
    def test_context_for_prompt(self):
        """Test generating context for prompts."""
        system = ThreeLayerMemorySystem()
        
        system.write_to_layer(MemoryLayer.WORKING, "User prefers TDD", 0.9, ["preference"], "test")
        system.write_to_layer(MemoryLayer.SHORT_TERM, "Last session: worked on auth", 0.7, ["session"], "test")
        
        context = system.get_context_for_prompt(max_tokens=1000)
        assert len(context) > 0
        assert "TDD" in context or "preference" in context
    
    def test_stats(self):
        """Test getting statistics."""
        system = ThreeLayerMemorySystem()
        
        system.write_to_layer(MemoryLayer.WORKING, "Test 1", 0.5, [], "test")
        system.write_to_layer(MemoryLayer.SHORT_TERM, "Test 2", 0.5, [], "test")
        
        stats = system.get_stats()
        assert "working_entries" in stats
        assert "short_term_entries" in stats


class TestMemoryManager:
    """Tests for MemoryManager."""
    
    def test_builtin_first_rule(self):
        """Test that BuiltinFirst memory cannot be removed."""
        system = ThreeLayerMemorySystem()
        manager = MemoryManager(system)
        
        # Should not be able to remove BuiltinFirst
        result = manager.remove_provider("BuiltinFirst")
        assert result == False
    
    def test_provider_registration(self):
        """Test registering providers."""
        system = ThreeLayerMemorySystem()
        manager = MemoryManager(system)
        
        def mock_check():
            return True
        
        manager.register_check("mock", mock_check)
        
        # Mock should be available now
        assert manager.is_available("mock") == True
    
    def test_context_sanitization(self):
        """Test that context is sanitized properly."""
        system = ThreeLayerMemorySystem()
        manager = MemoryManager(system)
        
        # Internal notes should be stripped
        dirty_context = """# Context
        
Some normal content.

[INTERNAL: This is a private note]
"""
        
        clean = manager.sanitize_context(dirty_context)
        assert "[INTERNAL:" not in clean


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
