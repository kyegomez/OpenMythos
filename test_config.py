"""
Test Schema-First Configuration System

Run with: python test_config.py
"""

import json
import tempfile
from pathlib import Path

# Test imports
from open_mythos.config import (
    MythosEnhancedConfig,
    MythosConfig,
    ModelConfig,
    LoopConfig,
    MoEConfig,
    ConfigLoader,
)


def test_default_config():
    """Test creating default enhanced config."""
    print("Testing default config creation...")
    
    config = MythosEnhancedConfig()
    
    assert config.version == "1.0.0"
    assert config.model.dim == 2048
    assert config.loop.min_depth == 4
    assert config.loop.max_depth == 16
    assert config.moe.n_experts == 64
    
    print("  ✓ Default config created successfully")


def test_yaml_load_save():
    """Test YAML loading and saving."""
    print("Testing YAML load/save...")
    
    # Create config
    config = MythosEnhancedConfig(
        name="test-config",
        description="Test configuration",
        model=ModelConfig(dim=1024, n_heads=8),
        loop=LoopConfig(min_depth=2, max_depth=8),
        moe=MoEConfig(n_experts=32)
    )
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name
    
    config.to_yaml(temp_path)
    
    # Load back
    loaded = MythosEnhancedConfig.from_yaml(temp_path)
    
    assert loaded.name == "test-config"
    assert loaded.model.dim == 1024
    assert loaded.loop.min_depth == 2
    assert loaded.moe.n_experts == 32
    
    # Cleanup
    Path(temp_path).unlink()
    
    print("  ✓ YAML load/save working")


def test_json_load_save():
    """Test JSON loading and saving."""
    print("Testing JSON load/save...")
    
    config = MythosEnhancedConfig(
        model=ModelConfig(dim=512),
        enhancements__p0__multiscale_loop_enabled=False
    )
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    config.to_json(temp_path)
    
    # Load back
    loaded = MythosEnhancedConfig.from_json(temp_path)
    
    assert loaded.model.dim == 512
    assert loaded.enhancements.p0.multiscale_loop_enabled == False
    
    # Cleanup
    Path(temp_path).unlink()
    
    print("  ✓ JSON load/save working")


def test_backward_compatibility():
    """Test backward compatibility with legacy MythosConfig."""
    print("Testing backward compatibility...")
    
    # Create legacy config
    legacy = MythosConfig(
        dim=4096,
        n_heads=32,
        n_experts=128,
        enable_multiscale_loop=False
    )
    
    # Convert to enhanced
    enhanced = legacy.to_enhanced()
    
    assert enhanced.model.dim == 4096
    assert enhanced.model.n_heads == 32
    assert enhanced.moe.n_experts == 128
    assert enhanced.enhancements.p0.multiscale_loop_enabled == False
    
    # Convert back to legacy
    legacy2 = MythosConfig.from_enhanced(enhanced)
    
    assert legacy2.dim == 4096
    assert legacy2.n_heads == 32
    assert legacy2.n_experts == 128
    assert legacy2.enable_multiscale_loop == False
    
    print("  ✓ Backward compatibility maintained")


def test_config_loader():
    """Test ConfigLoader."""
    print("Testing ConfigLoader...")
    
    loader = ConfigLoader()
    
    # Create sample config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name
        f.write("""
version: "1.0.0"
model:
  dim: 1536
  n_heads: 12
loop:
  min_depth: 3
  max_depth: 12
""")
    
    # Load
    config = loader.load_from_file(temp_path)
    
    assert config.model.dim == 1536
    assert config.model.n_heads == 12
    assert config.loop.min_depth == 3
    
    # Cleanup
    Path(temp_path).unlink()
    
    print("  ✓ ConfigLoader working")


def test_schema_generation():
    """Test JSON Schema generation."""
    print("Testing JSON Schema generation...")
    
    config = MythosEnhancedConfig()
    schema = config.generate_json_schema()
    
    assert "$schema" in schema
    assert schema["title"] == "OpenMythos Configuration Schema"
    assert "properties" in schema
    
    # Save schema to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    config.generate_json_schema(temp_path)
    
    # Verify file was created
    assert Path(temp_path).exists()
    
    # Load and verify
    with open(temp_path) as f:
        loaded_schema = json.load(f)
    
    assert loaded_schema["title"] == "OpenMythos Configuration Schema"
    
    # Cleanup
    Path(temp_path).unlink()
    
    print("  ✓ JSON Schema generation working")


def test_validation():
    """Test configuration validation."""
    print("Testing validation...")
    
    # Valid config
    config = MythosEnhancedConfig(
        model=ModelConfig(dim=1024)
    )
    
    # Should not raise
    config.model_validate(config.model_dump())
    
    # Invalid dim (not divisible by 64)
    try:
        invalid_config = MythosEnhancedConfig(
            model=ModelConfig(dim=1000)  # 1000 % 64 != 0
        )
        raise AssertionError("Should have raised ValidationError")
    except ValueError as e:
        assert "divisible by 64" in str(e)
    
    # Invalid depth bounds
    try:
        invalid_config = MythosEnhancedConfig(
            loop=LoopConfig(min_depth=10, max_depth=5)  # min > max
        )
        raise AssertionError("Should have raised ValidationError")
    except ValueError as e:
        assert "cannot exceed" in str(e)
    
    print("  ✓ Validation working")


def test_enhancement_dependencies():
    """Test enhancement dependency validation."""
    print("Testing enhancement dependencies...")
    
    # P3 requires P0
    try:
        config = MythosEnhancedConfig(
            enhancements__p0__enabled=False,
            enhancements__p3__enabled=True
        )
        raise AssertionError("Should have raised ValidationError")
    except ValueError as e:
        assert "P3" in str(e) and "P0" in str(e)
    
    print("  ✓ Enhancement dependencies validated")


def main():
    """Run all tests."""
    print("=" * 60)
    print("OpenMythos Schema-First Configuration Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_default_config,
        test_yaml_load_save,
        test_json_load_save,
        test_backward_compatibility,
        test_config_loader,
        test_schema_generation,
        test_validation,
        test_enhancement_dependencies,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
