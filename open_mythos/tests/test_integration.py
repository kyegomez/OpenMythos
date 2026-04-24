"""
Tests for Integration System
"""

import pytest
import os
import tempfile
import shutil

from open_mythos.integration.mythos_integration import (
    MythosIntegration, IntegrationConfig, SkillMigration
)


class TestMythosIntegration:
    """Tests for MythosIntegration."""
    
    def setup_method(self):
        """Set up temp directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = IntegrationConfig(
            mythos_dir=self.temp_dir,
            skills_dir=os.path.join(self.temp_dir, "skills"),
            memory_dir=os.path.join(self.temp_dir, "memory")
        )
        self.integration = MythosIntegration(self.config)
    
    def teardown_method(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test creating integration."""
        assert os.path.exists(self.temp_dir)
        assert os.path.exists(self.config.skills_dir)
        assert os.path.exists(self.config.memory_dir)
    
    def test_save_and_load_skill(self):
        """Test saving and loading skills."""
        skill_content = "# Test Skill\n\nDescription here."
        
        path = self.integration.save_skill("test_skill", skill_content)
        
        assert os.path.exists(path)
        
        skills = self.integration.load_mythos_skills()
        assert "test_skill" in skills
    
    def test_delete_skill(self):
        """Test deleting a skill."""
        skill_content = "# To Delete\n\nWill be deleted."
        
        self.integration.save_skill("to_delete", skill_content)
        result = self.integration.delete_skill("to_delete")
        
        assert result == True
        assert "to_delete" not in self.integration._loaded_skills
    
    def test_delete_nonexistent_skill(self):
        """Test deleting non-existent skill."""
        result = self.integration.delete_skill("nonexistent")
        assert result == False
    
    def test_create_skill_from_evolution(self):
        """Test creating skill from evolution output."""
        content = "# Generated Skill\n\nAuto-generated."
        metadata = {"pattern": "test", "confidence": 0.9}
        
        path = self.integration.create_skill_from_evolution(
            "evolved_skill", content, metadata
        )
        
        assert os.path.exists(path)
        
        with open(path, "r") as f:
            saved = f.read()
        
        assert "generated: true" in saved
        assert "evolution_system" in saved
        assert "pattern: test" in saved
    
    def test_get_stats(self):
        """Test getting integration stats."""
        self.integration.save_skill("skill1", "# Skill 1")
        self.integration.save_skill("skill2", "# Skill 2")
        
        stats = self.integration.get_stats()
        
        assert stats["loaded_skills"] == 2
        assert "skills_dir" in stats
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        unsafe = "My Skill! @#$"
        safe = self.integration._sanitize_filename(unsafe)
        
        assert safe.isalnum() or "_" in safe
        assert "@" not in safe
    
    def test_export_memory_snapshot(self):
        """Test exporting memory snapshot."""
        self.integration.save_skill("test", "# Test")
        
        snapshot_path = self.integration.export_memory_snapshot()
        
        assert os.path.exists(snapshot_path)
        
        import json
        with open(snapshot_path, "r") as f:
            data = json.load(f)
        
        assert "skills" in data
        assert "timestamp" in data
    
    def test_import_memory_snapshot(self):
        """Test importing memory snapshot."""
        # Create a snapshot file
        snapshot_dir = tempfile.mkdtemp()
        snapshot_path = os.path.join(snapshot_dir, "snapshot.json")
        
        import json
        with open(snapshot_path, "w") as f:
            json.dump({
                "skills": {
                    "imported_skill": "# Imported Skill\n\nFrom snapshot."
                },
                "timestamp": 123456
            }, f)
        
        # Import
        result = self.integration.import_memory_snapshot(snapshot_path)
        
        assert result == True
        assert "imported_skill" in self.integration._loaded_skills
        
        shutil.rmtree(snapshot_dir, ignore_errors=True)


class TestSkillMigration:
    """Tests for SkillMigration."""
    
    def setup_method(self):
        """Set up temp directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.old_dir = os.path.join(self.temp_dir, "old")
        self.new_dir = os.path.join(self.temp_dir, "new")
        os.makedirs(self.old_dir)
        os.makedirs(self.new_dir)
    
    def teardown_method(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_migrate_file(self):
        """Test migrating a single file."""
        old_path = os.path.join(self.old_dir, "myskill.txt")
        new_path = os.path.join(self.new_dir, "myskill.md")
        
        with open(old_path, "w") as f:
            f.write("Original skill content.")
        
        result = SkillMigration.migrate_file(old_path, new_path)
        
        assert result == True
        assert os.path.exists(new_path)
        
        with open(new_path, "r") as f:
            content = f.read()
        
        assert "migrated: true" in content
        assert "original_path" in content
        assert "Original skill content" in content
    
    def test_migrate_directory(self):
        """Test migrating entire directory."""
        # Create old-style skill files
        with open(os.path.join(self.old_dir, "skill1.txt"), "w") as f:
            f.write("Skill 1 content")
        with open(os.path.join(self.old_dir, "skill2.txt"), "w") as f:
            f.write("Skill 2 content")
        
        count = SkillMigration.migrate_directory(self.old_dir, self.new_dir)
        
        assert count == 2
        assert os.path.exists(os.path.join(self.new_dir, "skill1.md"))
        assert os.path.exists(os.path.join(self.new_dir, "skill2.md"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
