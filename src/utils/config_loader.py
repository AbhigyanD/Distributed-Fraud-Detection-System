"""Configuration loader utility."""
import yaml
import os
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    project_root = Path(__file__).parent.parent.parent
    config_file = project_root / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get nested configuration value using dot notation."""
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value

