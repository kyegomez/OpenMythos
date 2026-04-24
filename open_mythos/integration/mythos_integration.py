"""
Mythos Integration - Connect Enhanced Hermes to OpenMythos

Provides:
- Integration between enhanced memory and existing mythos
- Skill creation from Hermes evolution
- Tool discovery and registration
- Session management
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import json


@dataclass
class IntegrationConfig:
    """Configuration for mythos integration."""
    mythos_dir: str = "~/.hermes"
    skills_dir: str = "~/.hermes/skills"
    memory_dir: str = "~/.hermes/memory"
    auto_create_skills: bool = True
    auto_register_tools: bool = True


class MythosIntegration:
    """
    Integration layer between Enhanced Hermes and OpenMythos.
    
    This class bridges the enhanced memory/evolution systems
    with the existing OpenMythos codebase.
    
    Usage:
        integration = MythosIntegration()
        
        # Load existing mythos skills
        skills = integration.load_mythos_skills()
        
        # Create skill from evolution
        integration.create_skill_from_evolution("new_skill", skill_content)
        
        # Get integrated context
        context = integration.get_integrated_context()
    """
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()
        
        # Expand paths
        self.mythos_dir = os.path.expanduser(self.config.mythos_dir)
        self.skills_dir = os.path.expanduser(self.config.skills_dir)
        self.memory_dir = os.path.expanduser(self.config.memory_dir)
        
        # Ensure directories exist
        os.makedirs(self.mythos_dir, exist_ok=True)
        os.makedirs(self.skills_dir, exist_ok=True)
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # State
        self._loaded_skills: Dict[str, str] = {}
        self._integration_callbacks: List[Callable] = []
    
    def load_mythos_skills(self) -> Dict[str, str]:
        """
        Load skills from the mythos skills directory.
        
        Returns:
            Dict mapping skill name to skill content
        """
        skills = {}
        
        if not os.path.exists(self.skills_dir):
            return skills
        
        for filename in os.listdir(self.skills_dir):
            if filename.endswith(".md") or filename.endswith(".skill"):
                skill_name = filename.rsplit(".", 1)[0]
                filepath = os.path.join(self.skills_dir, filename)
                
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        skills[skill_name] = f.read()
                    self._loaded_skills[skill_name] = filepath
                except Exception:
                    pass
        
        return skills
    
    def save_skill(self, skill_name: str, content: str) -> str:
        """
        Save a skill to the skills directory.
        
        Args:
            skill_name: Name of the skill
            content: SKILL.md content
            
        Returns:
            Path to saved skill file
        """
        safe_name = self._sanitize_filename(skill_name)
        filepath = os.path.join(self.skills_dir, f"{safe_name}.md")
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        self._loaded_skills[skill_name] = filepath
        return filepath
    
    def delete_skill(self, skill_name: str) -> bool:
        """
        Delete a skill.
        
        Args:
            skill_name: Name of skill to delete
            
        Returns:
            True if deleted, False if not found
        """
        filepath = self._loaded_skills.get(skill_name)
        if not filepath or not os.path.exists(filepath):
            return False
        
        try:
            os.remove(filepath)
            del self._loaded_skills[skill_name]
            return True
        except Exception:
            return False
    
    def create_skill_from_evolution(
        self,
        skill_name: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a skill from evolution system output.
        
        This is called when the evolution system detects a pattern
        and generates a new skill.
        
        Args:
            skill_name: Name for the new skill
            content: Generated skill content
            metadata: Optional metadata about the skill
            
        Returns:
            Path to created skill file
        """
        # Add metadata header
        header = f"""---
generated: true
created_from: evolution_system
"""
        if metadata:
            for key, value in metadata.items():
                header += f"{key}: {value}\n"
        header += "---\n\n"
        
        full_content = header + content
        
        return self.save_skill(skill_name, full_content)
    
    def get_integrated_context(
        self,
        enhanced_hermes: Any,
        max_tokens: int = 4000
    ) -> str:
        """
        Get integrated context from multiple sources.
        
        Combines:
        - Enhanced Hermes memory context
        - Loaded mythos skills
        - Conversation history
        
        Args:
            enhanced_hermes: EnhancedHermes instance
            max_tokens: Maximum tokens for context
            
        Returns:
            Formatted context string
        """
        parts = []
        
        # Memory context
        memory_context = enhanced_hermes.get_memory_context(max_tokens=int(max_tokens * 0.3))
        if memory_context:
            parts.append(f"# Memory Context\n{memory_context}")
        
        # Skills context
        skills_context = self._get_skills_context(max_tokens=int(max_tokens * 0.3))
        if skills_context:
            parts.append(f"# Active Skills\n{skills_context}")
        
        # Recent conversation
        recent = enhanced_hermes.evolution.get_recent_tasks(limit=5)
        if recent:
            task_summary = self._summarize_recent_tasks(recent)
            parts.append(f"# Recent Tasks\n{task_summary}")
        
        return "\n\n".join(parts)
    
    def _get_skills_context(self, max_tokens: int) -> str:
        """Get context from loaded skills."""
        if not self._loaded_skills:
            return ""
        
        # Get skill names
        skill_names = list(self._loaded_skills.keys())[:10]  # Limit to 10
        
        context_parts = []
        for name in skill_names:
            filepath = self._loaded_skills[name]
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Extract first 200 chars as preview
                preview = content[:200].replace("\n", " ")
                context_parts.append(f"- **{name}**: {preview}...")
            except Exception:
                pass
        
        if context_parts:
            return "\n".join(context_parts)
        return ""
    
    def _summarize_recent_tasks(self, tasks: List) -> str:
        """Summarize recent tasks for context."""
        summaries = []
        for task in tasks:
            outcome = "SUCCESS" if task.outcome.value == "success" else "FAILED"
            summaries.append(f"- [{outcome}] {task.task[:50]}...")
        return "\n".join(summaries)
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a skill name for use as filename."""
        # Remove invalid characters
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        return safe.lower()
    
    def register_integration_callback(self, callback: Callable) -> None:
        """Register a callback to be called on integration events."""
        self._integration_callbacks.append(callback)
    
    def trigger_event(self, event: str, data: Dict[str, Any]) -> None:
        """Trigger an integration event."""
        for callback in self._integration_callbacks:
            try:
                callback(event, data)
            except Exception:
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        return {
            "loaded_skills": len(self._loaded_skills),
            "skills_dir": self.skills_dir,
            "memory_dir": self.memory_dir,
            "callbacks_registered": len(self._integration_callbacks)
        }
    
    def export_memory_snapshot(self, filepath: Optional[str] = None) -> str:
        """
        Export a snapshot of all memory data.
        
        Args:
            filepath: Optional path to save snapshot
            
        Returns:
            Path to saved snapshot
        """
        if filepath is None:
            filepath = os.path.join(self.memory_dir, f"snapshot_{int(time.time())}.json")
        
        # Collect all data
        snapshot = {
            "skills": self._loaded_skills.copy(),
            "timestamp": time.time()
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        
        return filepath
    
    def import_memory_snapshot(self, filepath: str) -> bool:
        """
        Import a memory snapshot.
        
        Args:
            filepath: Path to snapshot file
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
            
            # Import skills
            if "skills" in snapshot:
                for skill_name, content in snapshot["skills"].items():
                    if isinstance(content, str):
                        self.save_skill(skill_name, content)
            
            return True
        except Exception:
            return False


# Import time for export
import time


class SkillMigration:
    """
    Helper to migrate skills from old format to new format.
    
    Old format: Simple text files
    New format: SKILL.md with YAML frontmatter
    """
    
    @staticmethod
    def migrate_file(old_path: str, new_path: str) -> bool:
        """
        Migrate a skill file to new format.
        
        Args:
            old_path: Path to old skill file
            new_path: Path to save new skill file
            
        Returns:
            True if successful
        """
        try:
            with open(old_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Create new format with frontmatter
            new_content = f"""---
migrated: true
original_path: {old_path}
---

{content}
"""
            
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            
            with open(new_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            return True
        except Exception:
            return False
    
    @staticmethod
    def migrate_directory(old_dir: str, new_dir: str) -> int:
        """
        Migrate all skill files in a directory.
        
        Args:
            old_dir: Directory containing old skill files
            new_dir: Directory to save new skill files
            
        Returns:
            Number of files migrated
        """
        import shutil
        
        count = 0
        os.makedirs(new_dir, exist_ok=True)
        
        for filename in os.listdir(old_dir):
            old_path = os.path.join(old_dir, filename)
            
            if os.path.isfile(old_path):
                name = filename.rsplit(".", 1)[0]
                new_path = os.path.join(new_dir, f"{name}.md")
                
                if SkillMigration.migrate_file(old_path, new_path):
                    count += 1
        
        return count
