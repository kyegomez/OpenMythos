"""
Persistence Layer for OpenMythos

Provides storage backends for memory and skills.
"""

from .json_store import JSONStore, SkillStore, StoredEntry
from .sqlite_store import SQLiteStore, SQLiteSkillStore

# Default stores
_default_memory_store = None
_default_skill_store = None


def get_default_memory_store() -> SQLiteStore:
    """Get or create the default memory store."""
    global _default_memory_store
    if _default_memory_store is None:
        _default_memory_store = SQLiteStore()
    return _default_memory_store


def get_default_skill_store() -> SQLiteSkillStore:
    """Get or create the default skill store."""
    global _default_skill_store
    if _default_skill_store is None:
        _default_skill_store = SQLiteSkillStore()
    return _default_skill_store


__all__ = [
    # JSON stores
    "JSONStore",
    "SkillStore",
    "StoredEntry",
    # SQLite stores
    "SQLiteStore",
    "SQLiteSkillStore",
    # Helpers
    "get_default_memory_store",
    "get_default_skill_store",
]
