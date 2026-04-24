"""
Configuration Loader

Provides utilities for loading and validating OpenMythos configurations
from YAML/JSON files, environment variables, and command-line arguments.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .schema import MythosConfig, MythosEnhancedConfig


class ConfigLoader:
    """
    Configuration loader with support for multiple sources.
    
    Supports loading from:
    - YAML files (*.yaml, *.yml)
    - JSON files (*.json)
    - Environment variables (with OM_ prefix)
    - Command-line arguments
    
    Example:
        loader = ConfigLoader()
        
        # Load from file
        config = loader.load_from_file("config.yaml")
        
        # Load with env overrides
        config = loader.load("config.yaml", env=True)
        
        # Load with CLI overrides
        config = loader.load("config.yaml", cli_args={"dim": 1024})
    """
    
    ENV_PREFIX = "OM_"
    
    def __init__(self):
        self._cache: Dict[str, MythosEnhancedConfig] = {}
    
    def load_from_file(
        self,
        path: Union[str, Path],
        validate: bool = True
    ) -> MythosEnhancedConfig:
        """
        Load configuration from YAML or JSON file.
        
        Args:
            path: Path to configuration file
            validate: Whether to validate the configuration
            
        Returns:
            MythosEnhancedConfig instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        path = str(path)
        
        if path in self._cache:
            return self._cache[path]
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        # Determine format from extension
        ext = Path(path).suffix.lower()
        
        if ext in (".yaml", ".yml"):
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        elif ext == ".json":
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use .yaml, .yml, or .json")
        
        if data is None:
            raise ValueError(f"Empty configuration file: {path}")
        
        config = MythosEnhancedConfig(**data)
        
        if validate:
            config.model_validate(config.model_dump())
        
        self._cache[path] = config
        return config
    
    def load_from_env(self, base_config: Optional[MythosEnhancedConfig] = None) -> MythosEnhancedConfig:
        """
        Load configuration from environment variables.
        
        Environment variables use the OM_ prefix followed by the config path.
        
        Example:
            # Set environment variables
            export OM_MODEL_DIM=1024
            export OM_MOE_N_EXPERTS=32
            export OM_LOOP_MIN_DEPTH=2
            
            # Load with env
            config = loader.load_from_env(base_config)
        """
        if base_config is None:
            base_config = MythosEnhancedConfig()
        
        env_overrides = self._parse_env_vars()
        
        if not env_overrides:
            return base_config
        
        # Convert env overrides to nested dict
        nested = self._to_nested_dict(env_overrides)
        
        # Merge with base config
        config_dict = base_config.model_dump(exclude_none=True)
        config_dict = self._deep_merge(config_dict, nested)
        
        return MythosEnhancedConfig(**config_dict)
    
    def load_from_dict(
        self,
        data: Dict[str, Any],
        validate: bool = True
    ) -> MythosEnhancedConfig:
        """
        Load configuration from dictionary.
        
        Args:
            data: Configuration dictionary
            validate: Whether to validate
            
        Returns:
            MythosEnhancedConfig instance
        """
        config = MythosEnhancedConfig(**data)
        
        if validate:
            config.model_validate(config.model_dump())
        
        return config
    
    def load(
        self,
        path: Optional[Union[str, Path]] = None,
        env: bool = False,
        cli_args: Optional[Dict[str, Any]] = None,
        base_config: Optional[MythosEnhancedConfig] = None
    ) -> MythosEnhancedConfig:
        """
        Load configuration with multiple sources and precedence.
        
        Precedence (highest to lowest):
        1. CLI arguments
        2. Environment variables
        3. File configuration
        4. Base configuration (if provided)
        
        Args:
            path: Path to configuration file
            env: Whether to apply environment variable overrides
            cli_args: Command-line argument overrides
            base_config: Base configuration to start from
            
        Returns:
            MythosEnhancedConfig with all overrides applied
        """
        # Start with base or file
        if path:
            config = self.load_from_file(path, validate=False)
        elif base_config:
            config = base_config
        else:
            config = MythosEnhancedConfig()
        
        # Apply environment overrides
        if env:
            config = self.load_from_env(config)
        
        # Apply CLI overrides
        if cli_args:
            config_dict = config.model_dump(exclude_none=True)
            nested = self._to_nested_dict(cli_args)
            config_dict = self._deep_merge(config_dict, nested)
            config = MythosEnhancedConfig(**config_dict)
        
        return config
    
    def _parse_env_vars(self) -> Dict[str, str]:
        """Parse environment variables with OM_ prefix."""
        env_overrides = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.ENV_PREFIX):
                # Remove prefix and convert to config path
                config_key = key[len(self.ENV_PREFIX):].lower()
                
                # Convert SNAKE_CASE to dot.path
                # OM_MODEL_DIM -> model.dim
                parts = []
                for part in config_key.split('_'):
                    # Handle special cases
                    if part.isdigit():
                        parts.append(int(part))
                    else:
                        parts.append(part.lower())
                
                # Build path
                path = ".".join(str(p) for p in parts[:-1])
                if path:
                    final_key = f"{path}.{parts[-1]}"
                else:
                    final_key = str(parts[-1])
                
                # Parse value
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                elif value.lower() == "none":
                    value = None
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # Keep as string
                
                env_overrides[final_key] = value
        
        return env_overrides
    
    def _to_nested_dict(self, flat_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat dict with dot paths to nested dict."""
        result = {}
        
        for key, value in flat_dict.items():
            parts = key.split('.')
            current = result
            
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = value
        
        return result
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def create_sample_config(
        self,
        path: Union[str, Path],
        format: str = "yaml"
    ) -> None:
        """
        Create a sample configuration file.
        
        Args:
            path: Output path
            format: "yaml" or "json"
        """
        config = MythosEnhancedConfig(
            name="sample-config",
            description="Sample OpenMythos configuration with all enhancements"
        )
        
        if format == "yaml":
            config.to_yaml(str(path))
        else:
            config.to_json(str(path))
    
    def generate_schema(self, path: Union[str, Path]) -> None:
        """
        Generate JSON Schema for configuration.
        
        Args:
            path: Output path for schema file
        """
        config = MythosEnhancedConfig()
        config.generate_json_schema(str(path))


def load_config(
    path: Optional[str] = None,
    env: bool = False,
    **kwargs
) -> MythosConfig:
    """
    Convenience function to load MythosConfig.
    
    This function provides backward-compatible interface for loading
    the legacy MythosConfig from files or environment.
    
    Args:
        path: Path to configuration file
        env: Whether to load from environment
        **kwargs: Additional overrides
        
    Returns:
        MythosConfig (legacy dataclass) for backward compatibility
    """
    loader = ConfigLoader()
    
    if path:
        enhanced = loader.load_from_file(path)
    else:
        enhanced = MythosEnhancedConfig()
    
    if env:
        enhanced = loader.load_from_env(enhanced)
    
    # Apply any additional kwargs
    if kwargs:
        config_dict = enhanced.model_dump(exclude_none=True)
        for key, value in kwargs.items():
            # Handle nested keys like "model.dim"
            parts = key.split('.')
            current = config_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        enhanced = MythosEnhancedConfig(**config_dict)
    
    return MythosConfig.from_enhanced(enhanced)
