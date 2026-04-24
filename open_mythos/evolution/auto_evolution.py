"""
Auto-Evolution System - Hermes-Style automatic skill creation and improvement

Key Features:
1. Task Outcome Tracker - Records task success/failure patterns
2. Skill Generator - Creates new skills from successful task patterns
3. Skill Improver - Automatically improves existing skills based on usage
4. Periodic Reminders - Memory nudges to trigger reflection
"""

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict


class TaskOutcome(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    ABANDONED = "abandoned"


@dataclass
class TaskRecord:
    id: str
    task: str
    outcome: TaskOutcome
    approach: str
    quality_score: float
    duration_seconds: float
    tools_used: List[str] = field(default_factory=list)
    skills_invoked: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "task": self.task, "outcome": self.outcome.value,
            "approach": self.approach, "quality_score": self.quality_score,
            "duration_seconds": self.duration_seconds, "tools_used": self.tools_used,
            "skills_invoked": self.skills_invoked, "timestamp": self.timestamp,
            "error": self.error, "metadata": self.metadata,
        }


@dataclass
class SkillTemplate:
    name: str
    description: str
    trigger_conditions: List[str]
    content_template: str
    examples: List[str]
    confidence: float = 0.5


@dataclass
class SkillImprovement:
    skill_name: str
    improvement_type: str
    suggestion: str
    based_on_tasks: List[str]
    confidence: float = 0.5


@dataclass
class PeriodicReminder:
    id: str
    message: str
    trigger_type: str
    trigger_value: Any
    last_triggered: float = 0
    enabled: bool = True


class PatternDetector:
    """Detects patterns in task history that could become skills."""
    
    def __init__(self, min_occurrences: int = 3, confidence_threshold: float = 0.7):
        self.min_occurrences = min_occurrences
        self.confidence_threshold = confidence_threshold
        self._approach_patterns: Dict[str, List[str]] = defaultdict(list)
        self._tool_sequences: Dict[str, int] = defaultdict(int)
    
    def record_task(self, task: TaskRecord) -> None:
        if task.outcome == TaskOutcome.SUCCESS:
            self._approach_patterns[task.approach].append(task.id)
        if task.tools_used:
            sequence = " -> ".join(sorted(task.tools_used))
            self._tool_sequences[sequence] += 1
    
    def detect_skill_patterns(self) -> List[SkillTemplate]:
        templates = []
        for approach, task_ids in self._approach_patterns.items():
            if len(task_ids) >= self.min_occurrences:
                template = SkillTemplate(
                    name=self._generate_skill_name(approach),
                    description=f"Approach for: {approach[:50]}",
                    trigger_conditions=self._extract_keywords(approach)[:5],
                    content_template=self._generate_content(approach),
                    examples=task_ids[:3],
                    confidence=min(0.9, 0.5 + len(task_ids) * 0.1)
                )
                templates.append(template)
        return templates
    
    def _extract_keywords(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'for', 'of', 'and', 'or', 'in', 'on', 'at'}
        return [w for w in words if len(w) > 3 and w not in stopwords]
    
    def _generate_skill_name(self, approach: str) -> str:
        name = re.sub(r'[^\w]', '_', approach.lower())
        name = re.sub(r'_+', '_', name).strip('_')[:30]
        return f"auto_{name}"
    
    def _generate_content(self, approach: str) -> str:
        return f"""# Auto-Generated Skill

## When to Use
When encountering tasks related to: {approach}

## Approach
{approach}

## Steps
1. Analyze the task
2. Apply the approach
3. Verify outcome

## Tips
- Monitor quality score
- Adjust based on feedback
"""


class SkillImprover:
    """Automatically improves skills based on usage patterns."""
    
    def __init__(self):
        self._skill_feedback: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def record_feedback(self, skill_name: str, task_id: str, success: bool, quality_score: float, notes: Optional[str] = None) -> None:
        self._skill_feedback[skill_name].append({
            "task_id": task_id, "success": success, "quality_score": quality_score,
            "notes": notes, "timestamp": time.time()
        })
    
    def suggest_improvements(self, skill_name: str) -> List[SkillImprovement]:
        feedback = self._skill_feedback.get(skill_name, [])
        if len(feedback) < 2:
            return []
        
        improvements = []
        success_count = sum(1 for f in feedback if f["success"])
        success_rate = success_count / len(feedback)
        
        if success_rate < 0.7:
            improvements.append(SkillImprovement(
                skill_name=skill_name, improvement_type="clarify",
                suggestion="Low success rate. Consider clarifying approach steps.",
                based_on_tasks=[f["task_id"] for f in feedback[-5:]], confidence=0.8
            ))
        
        return improvements


class PeriodicReminderSystem:
    """Generates periodic reminders to encourage reflection."""
    
    def __init__(self, task_count_interval: int = 10, time_interval_seconds: float = 3600):
        self.task_count_interval = task_count_interval
        self.time_interval = time_interval_seconds
        self._reminders: List[PeriodicReminder] = []
        self._task_count = 0
        self._register_default_reminders()
    
    def _register_default_reminders(self) -> None:
        self._reminders = [
            PeriodicReminder(id="daily_reflection", message="Daily reflection: What did you learn today?",
                trigger_type="time", trigger_value=3600*24),
            PeriodicReminder(id="skill_review", message="Consider reviewing your skills for updates.",
                trigger_type="task_count", trigger_value=10),
            PeriodicReminder(id="memory_check", message="Time to consolidate important info to long-term memory?",
                trigger_type="time", trigger_value=3600*4),
        ]
    
    def on_task_complete(self, task: TaskRecord) -> None:
        self._task_count += 1
    
    def check_reminders(self) -> List[PeriodicReminder]:
        triggered = []
        current_time = time.time()
        for reminder in self._reminders:
            if not reminder.enabled:
                continue
            if reminder.trigger_type == "time":
                if current_time - reminder.last_triggered >= reminder.trigger_value:
                    triggered.append(reminder)
                    reminder.last_triggered = current_time
            elif reminder.trigger_type == "task_count":
                if self._task_count >= reminder.trigger_value:
                    triggered.append(reminder)
        return triggered
    
    def add_reminder(self, message: str, trigger_type: str, trigger_value: Any) -> PeriodicReminder:
        reminder = PeriodicReminder(id=f"custom_{len(self._reminders)}", message=message,
            trigger_type=trigger_type, trigger_value=trigger_value)
        self._reminders.append(reminder)
        return reminder
