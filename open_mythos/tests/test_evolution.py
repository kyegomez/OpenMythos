"""
Tests for Auto-Evolution System
"""

import pytest
import time

from open_mythos.evolution.auto_evolution import (
    TaskOutcome, TaskRecord, SkillTemplate, SkillImprovement,
    PeriodicReminder, PatternDetector, SkillImprover, PeriodicReminderSystem
)
from open_mythos.evolution.evolution_core import AutoEvolution


class TestTaskRecord:
    """Tests for TaskRecord."""
    
    def test_create_record(self):
        """Test creating a task record."""
        record = TaskRecord(
            id="task_1",
            task="Fix bug in auth",
            outcome=TaskOutcome.SUCCESS,
            approach="Used TDD approach",
            quality_score=0.9,
            duration_seconds=120.0
        )
        
        assert record.id == "task_1"
        assert record.task == "Fix bug in auth"
        assert record.outcome == TaskOutcome.SUCCESS
        assert record.quality_score == 0.9
    
    def test_outcome_enum(self):
        """Test TaskOutcome enum values."""
        assert TaskOutcome.SUCCESS.value == "success"
        assert TaskOutcome.FAILED.value == "failed"
        assert TaskOutcome.PARTIAL.value == "partial"


class TestPatternDetector:
    """Tests for PatternDetector."""
    
    def test_record_task(self):
        """Test recording a task."""
        detector = PatternDetector()
        
        record = TaskRecord(
            id="task_1",
            task="Fix bug",
            outcome=TaskOutcome.SUCCESS,
            approach="Used TDD",
            quality_score=0.9
        )
        
        detector.record_task(record)
        
        assert len(detector._task_records) == 1
    
    def test_detect_approach_patterns(self):
        """Test detecting approach patterns."""
        detector = PatternDetector()
        
        # Add multiple tasks with same approach
        for i in range(4):
            record = TaskRecord(
                id=f"task_{i}",
                task=f"Task {i}",
                outcome=TaskOutcome.SUCCESS,
                approach="TDD approach",
                quality_score=0.8
            )
            detector.record_task(record)
        
        patterns = detector.detect_approach_patterns()
        
        assert len(patterns) >= 1
        # Should find TDD approach
        approach_names = [p[0] for p in patterns]
        assert any("TDD approach" in name for name in approach_names)
    
    def test_detect_skill_patterns(self):
        """Test detecting skill patterns."""
        detector = PatternDetector()
        
        # Add successful tasks
        for i in range(5):
            record = TaskRecord(
                id=f"task_{i}",
                task="API development task",
                outcome=TaskOutcome.SUCCESS,
                approach="REST API pattern",
                quality_score=0.85,
                skills_invoked=["api_skill"]
            )
            detector.record_task(record)
        
        templates = detector.detect_skill_patterns()
        
        assert len(templates) >= 1


class TestSkillImprover:
    """Tests for SkillImprover."""
    
    def test_record_feedback(self):
        """Test recording skill feedback."""
        improver = SkillImprover()
        
        improver.record_feedback(
            skill_name="test_skill",
            task_id="task_1",
            success=True,
            quality_score=0.9
        )
        
        assert "test_skill" in improver._skill_feedback
        assert len(improver._skill_feedback["test_skill"]) == 1
    
    def test_record_failed_feedback(self):
        """Test recording failed feedback."""
        improver = SkillImprover()
        
        improver.record_feedback(
            skill_name="test_skill",
            task_id="task_1",
            success=False,
            quality_score=0.3
        )
        
        assert "test_skill" in improver._skill_feedback
        # Should be marked as failure
        feedback = improver._skill_feedback["test_skill"][0]
        assert feedback["success"] == False
    
    def test_suggest_improvements(self):
        """Test suggesting improvements."""
        improver = SkillImprover()
        
        # Add multiple failed uses of a skill
        for i in range(3):
            improver.record_feedback(
                skill_name="buggy_skill",
                task_id=f"task_{i}",
                success=False,
                quality_score=0.2
            )
        
        improvements = improver.suggest_improvements("buggy_skill")
        
        assert len(improvements) >= 1


class TestPeriodicReminderSystem:
    """Tests for PeriodicReminderSystem."""
    
    def test_add_reminder(self):
        """Test adding a reminder."""
        system = PeriodicReminderSystem()
        
        reminder = system.add_reminder(
            message="Review old skills",
            interval_hours=24,
            reminder_type="skill_review"
        )
        
        assert reminder is not None
        assert reminder.enabled == True
        assert reminder.message == "Review old skills"
    
    def test_check_reminders_due(self):
        """Test checking due reminders."""
        system = PeriodicReminderSystem()
        
        # Add a reminder that's always due (immediate)
        reminder = system.add_reminder(
            message="Test reminder",
            interval_hours=0,  # Due immediately
            reminder_type="test"
        )
        
        due = system.check_reminders()
        
        assert len(due) >= 1
        assert any(r.message == "Test reminder" for r in due)
    
    def test_disable_reminder(self):
        """Test disabling a reminder."""
        system = PeriodicReminderSystem()
        
        reminder = system.add_reminder(
            message="To be disabled",
            interval_hours=1,
            reminder_type="test"
        )
        
        system.disable_reminder(reminder.id)
        
        assert reminder.enabled == False


class TestAutoEvolution:
    """Tests for AutoEvolution main class."""
    
    def test_initialization(self):
        """Test creating AutoEvolution instance."""
        evolution = AutoEvolution()
        
        assert evolution._stats["total_tasks"] == 0
        assert evolution._stats["successful_tasks"] == 0
    
    def test_record_outcome(self):
        """Test recording task outcome."""
        evolution = AutoEvolution()
        
        record = evolution.record_outcome(
            task="Test task",
            outcome=TaskOutcome.SUCCESS,
            approach="Test approach",
            quality_score=0.8
        )
        
        assert record is not None
        assert record.task == "Test task"
        assert evolution._stats["total_tasks"] == 1
        assert evolution._stats["successful_tasks"] == 1
    
    def test_record_failed_outcome(self):
        """Test recording failed outcome."""
        evolution = AutoEvolution()
        
        evolution.record_outcome(
            task="Failed task",
            outcome=TaskOutcome.FAILED,
            approach="Bad approach",
            quality_score=0.2
        )
        
        assert evolution._stats["total_tasks"] == 1
        assert evolution._stats["successful_tasks"] == 0
    
    def test_check_for_new_skill(self):
        """Test checking for new skills."""
        evolution = AutoEvolution()
        
        # Add multiple successful tasks with same approach
        for i in range(4):
            evolution.record_outcome(
                task=f"Task {i}",
                outcome=TaskOutcome.SUCCESS,
                approach="Consistent approach",
                quality_score=0.85
            )
        
        new_skills = evolution.check_for_new_skill()
        
        # Should detect pattern and suggest skill
        assert isinstance(new_skills, list)
    
    def test_get_stats(self):
        """Test getting statistics."""
        evolution = AutoEvolution()
        
        evolution.record_outcome(
            task="Test",
            outcome=TaskOutcome.SUCCESS,
            approach="Test"
        )
        
        stats = evolution.get_stats()
        
        assert "total_tasks" in stats
        assert stats["total_tasks"] == 1
    
    def test_get_recent_tasks(self):
        """Test getting recent tasks."""
        evolution = AutoEvolution()
        
        for i in range(5):
            evolution.record_outcome(
                task=f"Task {i}",
                outcome=TaskOutcome.SUCCESS,
                approach="Test"
            )
        
        recent = evolution.get_recent_tasks(limit=3)
        
        assert len(recent) == 3
    
    def test_get_task_by_id(self):
        """Test getting specific task."""
        evolution = AutoEvolution()
        
        record = evolution.record_outcome(
            task="Specific task",
            outcome=TaskOutcome.SUCCESS,
            approach="Test"
        )
        
        found = evolution.get_task_by_id(record.id)
        
        assert found is not None
        assert found.task == "Specific task"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
