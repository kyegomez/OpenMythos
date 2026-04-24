"""
SQLite-based Persistence Layer for OpenMythos Memory

Provides efficient SQLite storage for memory entries.
"""

import sqlite3
import json
import time
import threading
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import contextmanager


@dataclass
class StoredEntry:
    """A stored memory entry."""
    id: str
    content: str
    layer: str
    importance: float
    created_at: float
    last_accessed: float
    tags: List[str]
    source: str
    metadata: Dict[str, Any]
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredEntry":
        """Create from dictionary."""
        return cls(**data)


class SQLiteStore:
    """
    Thread-safe SQLite-based storage for memory entries.

    Provides:
    - ACID transactions
    - Efficient querying
    - Automatic schema migrations
    """

    def __init__(self, db_path: str = "~/.open_mythos/memory.db"):
        self.db_path = Path(os.path.expanduser(db_path))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @contextmanager
    def _transaction(self):
        """Context manager for transactions."""
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_db(self):
        """Initialize database schema."""
        with self._transaction() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    tags TEXT DEFAULT '[]',
                    source TEXT DEFAULT '',
                    metadata TEXT DEFAULT '{}',
                    access_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_layer ON entries(layer)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_importance ON entries(importance)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created ON entries(created_at)
            """)

    def put(self, entry: StoredEntry) -> bool:
        """Store an entry."""
        try:
            with self._transaction() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO entries
                    (id, content, layer, importance, created_at, last_accessed, tags, source, metadata, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.id,
                    entry.content,
                    entry.layer,
                    entry.importance,
                    entry.created_at,
                    time.time(),
                    json.dumps(entry.tags),
                    entry.source,
                    json.dumps(entry.metadata),
                    entry.access_count
                ))
            return True
        except Exception:
            return False

    def get(self, entry_id: str, layer: Optional[str] = None) -> Optional[StoredEntry]:
        """Get an entry by ID."""
        conn = self._get_conn()

        if layer:
            row = conn.execute(
                "SELECT * FROM entries WHERE id = ? AND layer = ?",
                (entry_id, layer)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM entries WHERE id = ?",
                (entry_id,)
            ).fetchone()

        if row:
            # Update access count
            conn.execute(
                "UPDATE entries SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (time.time(), entry_id)
            )
            return self._row_to_entry(row)

        return None

    def delete(self, entry_id: str, layer: Optional[str] = None) -> bool:
        """Delete an entry."""
        with self._transaction() as conn:
            if layer:
                cursor = conn.execute(
                    "DELETE FROM entries WHERE id = ? AND layer = ?",
                    (entry_id, layer)
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM entries WHERE id = ?",
                    (entry_id,)
                )
            return cursor.rowcount > 0

    def list_all(self, layer: str, limit: int = 100, offset: int = 0) -> List[StoredEntry]:
        """List all entries in a layer."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM entries WHERE layer = ? ORDER BY importance DESC LIMIT ? OFFSET ?",
            (layer, limit, offset)
        ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def search(self, layer: str, query: str, limit: int = 10) -> List[StoredEntry]:
        """Search entries by content."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM entries WHERE layer = ? AND content LIKE ? ORDER BY importance DESC LIMIT ?",
            (layer, f"%{query}%", limit)
        ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def count(self, layer: Optional[str] = None) -> int:
        """Count entries in a layer or total."""
        conn = self._get_conn()
        if layer:
            row = conn.execute(
                "SELECT COUNT(*) FROM entries WHERE layer = ?",
                (layer,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM entries").fetchone()
        return row[0] if row else 0

    def clear(self, layer: str) -> int:
        """Clear all entries in a layer."""
        with self._transaction() as conn:
            cursor = conn.execute("DELETE FROM entries WHERE layer = ?", (layer,))
            return cursor.rowcount

    def get_recent(self, layer: str, limit: int = 10) -> List[StoredEntry]:
        """Get recently accessed entries."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM entries WHERE layer = ? ORDER BY last_accessed DESC LIMIT ?",
            (layer, limit)
        ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        conn = self._get_conn()

        total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

        layer_counts = {}
        for layer in ["working", "short_term", "long_term"]:
            count = conn.execute(
                "SELECT COUNT(*) FROM entries WHERE layer = ?",
                (layer,)
            ).fetchone()[0]
            layer_counts[layer] = count

        return {
            "db_path": str(self.db_path),
            "total_entries": total,
            "layers": layer_counts
        }

    def _row_to_entry(self, row: sqlite3.Row) -> StoredEntry:
        """Convert a database row to StoredEntry."""
        return StoredEntry(
            id=row["id"],
            content=row["content"],
            layer=row["layer"],
            importance=row["importance"],
            created_at=row["created_at"],
            last_accessed=row["last_accessed"],
            tags=json.loads(row["tags"]),
            source=row["source"],
            metadata=json.loads(row["metadata"]),
            access_count=row["access_count"]
        )


class SQLiteSkillStore:
    """
    SQLite-based store for skills.
    """

    def __init__(self, db_path: str = "~/.open_mythos/skills.db"):
        self.db_path = Path(os.path.expanduser(db_path))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @contextmanager
    def _transaction(self):
        """Context manager for transactions."""
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_db(self):
        """Initialize database schema."""
        with self._transaction() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_updated ON skills(updated_at)
            """)

    def save(self, skill_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save a skill."""
        now = time.time()
        try:
            with self._transaction() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO skills (id, content, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, COALESCE((SELECT created_at FROM skills WHERE id = ?), ?), ?)
                """, (skill_id, content, json.dumps(metadata or {}), skill_id, now, now))
            return True
        except Exception:
            return False

    def load(self, skill_id: str) -> Optional[str]:
        """Load a skill by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT content FROM skills WHERE id = ?",
            (skill_id,)
        ).fetchone()
        return row["content"] if row else None

    def delete(self, skill_id: str) -> bool:
        """Delete a skill."""
        with self._transaction() as conn:
            cursor = conn.execute("DELETE FROM skills WHERE id = ?", (skill_id,))
            return cursor.rowcount > 0

    def list_all(self) -> List[str]:
        """List all skill IDs."""
        conn = self._get_conn()
        rows = conn.execute("SELECT id FROM skills ORDER BY updated_at DESC").fetchall()
        return [row["id"] for row in rows]

    def exists(self, skill_id: str) -> bool:
        """Check if a skill exists."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT 1 FROM skills WHERE id = ?",
            (skill_id,)
        ).fetchone()
        return row is not None


__all__ = [
    "SQLiteStore",
    "SQLiteSkillStore",
    "StoredEntry",
]
