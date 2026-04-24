"""
Integration System - Connect Enhanced Hermes to OpenMythos

Provides:
- MythosIntegration: Bridge between enhanced systems and OpenMythos
- SkillMigration: Migrate skills to new format
- Integration callbacks and events
"""

from .mythos_integration import (
    MythosIntegration,
    IntegrationConfig,
    SkillMigration,
)

__all__ = [
    "MythosIntegration",
    "IntegrationConfig",
    "SkillMigration",
]
