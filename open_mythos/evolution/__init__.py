"""
Evolution System - Hermes-Style Auto-Evolution

Provides:
- Task outcome tracking
- Pattern detection for skill creation
- Skill generation from successful patterns
- Skill improvement based on feedback
- Periodic reminders
"""

from .auto_evolution import (
    TaskOutcome,
    TaskRecord,
    SkillTemplate,
    SkillImprovement,
    PeriodicReminder,
    PatternDetector,
    SkillImprover,
    PeriodicReminderSystem,
)
from .evolution_core import AutoEvolution

__all__ = [
    "TaskOutcome",
    "TaskRecord", 
    "SkillTemplate",
    "SkillImprovement",
    "PeriodicReminder",
    "PatternDetector",
    "SkillImprover",
    "PeriodicReminderSystem",
    "AutoEvolution",
]
