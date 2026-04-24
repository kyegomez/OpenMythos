"""
JSON-based Persistence Layer for OpenMythos Memory

Provides simple file-based storage for memory entries.
"""

import json
import os
import time
import threading
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


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


class JSONStore:
    """
    Thread-safe JSON file-based storage.

    Stores memory entries in JSON files with automatic backup.
    """

    def __init__(self, data_dir: str = "~/.open_mythos/data"):
        self.data_dir = Path(os.path.expanduser(data_dir))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cache: Dict[str, StoredEntry] = {}
        self._cache_loaded = False

    def _get_file_path(self, layer: str) -> Path:
        """Get the file path for a layer."""
        return self.data_dir / f"{layer}.json"

    def _load_layer(self, layer: str) -> Dict[str, StoredEntry]:
        """Load entries from a layer file."""
        path = self._get_file_path(layer)
        if not path.exists():
            return {}

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return {k: StoredEntry.from_dict(v) for k, v in data.items()}
        except (json.JSONDecodeError, KeyError):
            return {}

    def _save_layer(self, layer: str, entries: Dict[str, StoredEntry]):
        """Save entries to a layer file."""
        path = self._get_file_path(layer)
        temp_path = path.with_suffix('.tmp')

        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump({k: v.to_dict() for k, v in entries.items()}, f, indent=2)

        temp_path.replace(path)

    def _ensure_loaded(self):
        """Ensure all layers are loaded."""
        if not self._cache_loaded:
            for layer in ["working", "short_term", "long_term"]:
                self._cache[layer] = self._load_layer(layer)
            self._cache_loaded = True

    def put(self, entry: StoredEntry) -> bool:
        """Store an entry."""
        with self._lock:
            self._ensure_loaded()
            layer = entry.layer

            if layer not in self._cache:
                self._cache[layer] = {}

            entry.last_accessed = time.time()
            self._cache[layer][entry.id] = entry
            self._save_layer(layer, self._cache[layer])

            return True

    def get(self, entry_id: str, layer: str) -> Optional[StoredEntry]:
        """Get an entry by ID."""
        with self._lock:
            self._ensure_loaded()

            if layer not in self._cache:
                return None

            entry = self._cache[layer].get(entry_id)

            if entry:
                entry.access_count += 1
                entry.last_accessed = time.time()
                self._save_layer(layer, self._cache[layer])

            return entry

    def delete(self, entry_id: str, layer: str) -> bool:
        """Delete an entry."""
        with self._lock:
            self._ensure_loaded()

            if layer not in self._cache:
                return False

            if entry_id in self._cache[layer]:
                del self._cache[layer][entry_id]
                self._save_layer(layer, self._cache[layer])
                return True

            return False

    def list_all(self, layer: str) -> List[StoredEntry]:
        """List all entries in a layer."""
        with self._lock:
            self._ensure_loaded()

            if layer not in self._cache:
                return []

            return list(self._cache[layer].values())

    def search(self, layer: str, query: str, limit: int = 10) -> List[StoredEntry]:
        """Search entries by content."""
        with self._lock:
            self._ensure_loaded()

            if layer not in self._cache:
                return []

            results = []
            query_lower = query.lower()

            for entry in self._cache[layer].values():
                if query_lower in entry.content.lower():
                    results.append(entry)

            results.sort(key=lambda e: e.importance, reverse=True)
            return results[:limit]

    def count(self, layer: str) -> int:
        """Count entries in a layer."""
        with self._lock:
            self._ensure_loaded()
            return len(self._cache.get(layer, {}))

    def clear(self, layer: str) -> int:
        """Clear all entries in a layer."""
        with self._lock:
            self._ensure_loaded()

            if layer not in self._cache:
                return 0

            count = len(self._cache[layer])
            self._cache[layer] = {}
            self._save_layer(layer, {})

            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._lock:
            self._ensure_loaded()

            total = sum(len(entries) for entries in self._cache.values())

            return {
                "data_dir": str(self.data_dir),
                "layers": {
                    layer: len(entries)
                    for layer, entries in self._cache.items()
                },
                "total_entries": total
            }


class SkillStore:
    """
    Store for skills (long-term memory).
    """

    def __init__(self, skills_dir: str = "~/.open_mythos/skills"):
        self.skills_dir = Path(os.path.expanduser(skills_dir))
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_loaded = False

    def _ensure_loaded(self):
        """Ensure skills are loaded."""
        if not self._cache_loaded:
            self._load_all_skills()
            self._cache_loaded = True

    def _load_all_skills(self):
        """Load all skills from files."""
        for path in self.skills_dir.glob("*.md"):
            skill_id = path.stem
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self._cache[skill_id] = {
                    "content": content,
                    "file": str(path),
                    "loaded_at": time.time()
                }
            except Exception:
                pass

    def save(self, skill_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save a skill."""
        with self._lock:
            path = self.skills_dir / f"{skill_id}.md"

            header = f"---\n"
            header += f"skill_id: {skill_id}\n"
            header += f"saved_at: {time.time()}\n"
            if metadata:
                for k, v in metadata.items():
                    header += f"{k}: {v}\n"
            header += f"---\n\n"

            full_content = header + content

            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(full_content)

                self._cache[skill_id] = {
                    "content": content,
                    "file": str(path),
                    "loaded_at": time.time()
                }
                return True
            except Exception:
                return False

    def load(self, skill_id: str) -> Optional[str]:
        """Load a skill by ID."""
        with self._lock:
            self._ensure_loaded()
            entry = self._cache.get(skill_id)
            return entry["content"] if entry else None

    def delete(self, skill_id: str) -> bool:
        """Delete a skill."""
        with self._lock:
            self._ensure_loaded()

            path = self.skills_dir / f"{skill_id}.md"
            if path.exists():
                path.unlink()

            if skill_id in self._cache:
                del self._cache[skill_id]
                return True

            return False

    def list_all(self) -> List[str]:
        """List all skill IDs."""
        with self._lock:
            self._ensure_loaded()
            return list(self._cache.keys())

    def exists(self, skill_id: str) -> bool:
        """Check if a skill exists."""
        with self._lock:
            self._ensure_loaded()
            return skill_id in self._cache


__all__ = [
    "JSONStore",
    "SkillStore",
    "StoredEntry",
]
