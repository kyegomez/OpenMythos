"""
AutoEvolution - Main orchestrator for the auto-evolution system.
"""

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..memory import MemoryManager

from .auto_evolution import (
    TaskOutcome, TaskRecord, SkillTemplate, SkillImprovement,
    PeriodicReminder, PatternDetector, SkillImprover, PeriodicReminderSystem
)


class AutoEvolution:
    """
    Main auto-evolution system.
    
    Orchestrates:
    - Task outcome tracking
    - Pattern detection
    - Skill generation
    - Skill improvement
    - Periodic reminders
    
    Usage:
        evolution = AutoEvolution(memory_manager)
        
        # After each task
        evolution.record_outcome(
            task="Fixed auth bug",
            success=True,
            approach="Used TDD approach",
            quality_score=0.9
        )
        
        # Check for new skills to create
        new_skills = evolution.check_for_new_skill()
        
        # Get periodic reminders
        reminders = evolution.check_reminders()
    """
    
    def __init__(self, memory_manager: Optional["MemoryManager"] = None):
        self.memory_manager = memory_manager
        
        # Components
        self.pattern_detector = PatternDetector()
        self.skill_improver = SkillImprover()
        self.reminder_system = PeriodicReminderSystem()
        
        # Task history
        self._task_history: List[TaskRecord] = []
        self._task_index: Dict[str, TaskRecord] = {}
        
        # Generated skills storage
        self._generated_skills: Dict[str, str] = {}
        
        # Stats
        self._stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "skills_generated": 0,
            "improvements_suggested": 0,
            "reminders_triggered": 0
        }
    
    def record_outcome(
        self,
        task: str,
        outcome: TaskOutcome,
        approach: str,
        quality_score: float = 0.5,
        duration_seconds: float = 0,
        tools_used: Optional[List[str]] = None,
        skills_invoked: Optional[List[str]] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TaskRecord:
        """
        Record a task outcome.
        
        Call this after each significant task.
        """
        task_id = f"task_{hashlib.md5(task.encode()).hexdigest()[:8]}_{int(time.time()*1000)}"
        
        record = TaskRecord(
            id=task_id,
            task=task,
            outcome=outcome,
            approach=approach,
            quality_score=quality_score,
            duration_seconds=duration_seconds,
            tools_used=tools_used or [],
            skills_invoked=skills_invoked or [],
            error=error,
            metadata=metadata or {}
        )
        
        self._task_history.append(record)
        self._task_index[task_id] = record
        
        self._stats["total_tasks"] += 1
        if outcome == TaskOutcome.SUCCESS:
            self._stats["successful_tasks"] += 1
        
        self.pattern_detector.record_task(record)
        
        if skills_invoked:
            for skill_name in skills_invoked:
                self.skill_improver.record_feedback(
                    skill_name=skill_name, task_id=task_id,
                    success=(outcome == TaskOutcome.SUCCESS),
                    quality_score=quality_score
                )
        
        self.reminder_system.on_task_complete(record)
        
        return record
    
    def check_for_new_skill(self) -> List[Tuple[SkillTemplate, str]]:
        """
        Check if any patterns suggest new skills.
        
        Returns list of (template, content) tuples for new skills.
        """
        templates = self.pattern_detector.detect_skill_patterns()
        new_skills = []
        
        for template in templates:
            if template.name not in self._generated_skills:
                content = self._generate_skill_content(template)
                new_skills.append((template, content))
                self._generated_skills[template.name] = content
                self._stats["skills_generated"] += 1
        
        return new_skills
    
    def _generate_skill_content(self, template: SkillTemplate) -> str:
        """Generate SKILL.md content from template."""
        examples_text = "\n".join(f"{i+1}. {ex}" for i, ex in enumerate(template.examples))
        
        return f"""# {template.name}

## Description
{template.description}

## Trigger Conditions
{" ".join(f'`{c}`' for c in template.trigger_conditions)}

## When to Use
This skill is triggered when task matches trigger conditions.

## Approach
{template.content_template}

## Examples
{examples_text}

## Confidence
{template.confidence:.0%}

## Metadata
- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
- Pattern occurrences: {len(template.examples)}
"""
    
    def check_skill_improvements(self) -> List[SkillImprovement]:
        """Check for skill improvements based on feedback."""
        improvements = []
        for skill_name in self.skill_improver._skill_feedback.keys():
            suggestions = self.skill_improver.suggest_improvements(skill_name)
            improvements.extend(suggestions)
        self._stats["improvements_suggested"] = len(improvements)
        return improvements
    
    def check_reminders(self) -> List[PeriodicReminder]:
        """Check for periodic reminders to trigger."""
        reminders = self.reminder_system.check_reminders()
        self._stats["reminders_triggered"] += len(reminders)
        return reminders
    
    def get_skill_suggestions(self) -> Dict[str, List[str]]:
        """Get skill suggestions based on recent tasks."""
        suggestions = {}
        recent = [t for t in self._task_history[-20:] if t.outcome == TaskOutcome.SUCCESS]
        
        approaches: Dict[str, List[str]] = {}
        for task in recent:
            approaches[task.approach].append(task.id)
        
        for approach, task_ids in approaches.items():
            if len(task_ids) >= 3:
                skill_name = f"skill_{hashlib.md5(approach.encode()).hexdigest()[:6]}"
                suggestions[skill_name] = [
                    f"Based on {len(task_ids)} successful tasks: {approach[:50]}...",
                    "Consider creating a skill to formalize this approach"
                ]
        
        return suggestions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        return {
            **self._stats,
            "task_history_size": len(self._task_history),
            "unique_skills_in_feedback": len(self.skill_improver._skill_feedback),
            "patterns_detected": len(self.pattern_detector._approach_patterns),
            "reminders_enabled": sum(1 for r in self.reminder_system._reminders if r.enabled)
        }
    
    def get_recent_tasks(self, limit: int = 10) -> List[TaskRecord]:
        """Get recent task records."""
        return self._task_history[-limit:]
    
    def get_task_by_id(self, task_id: str) -> Optional[TaskRecord]:
        """Get a specific task record."""
        return self._task_index.get(task_id)
    
    def get_generated_skills(self) -> Dict[str, str]:
        """Get all generated skill contents."""
        return self._generated_skills.copy()
