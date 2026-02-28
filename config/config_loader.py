"""Configuration load and manage"""
import os
import yaml
from typing import Any, Dict
from pathlib import Path

class Config:
    """Configuration manager with dot notation access"""
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self._parse_nested(config_dict)

    def _parse_nested(self, d: Dict[str, Any]):
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                # env var overrides yaml value
                env_key = key.upper()
                setattr(self, key, os.getenv(env_key, value))
    
    def get(self, key: str, default: Any = None) -> Any:
            """Get value by dot-path e.g. get('bedrock.models.rag')"""
            keys = key.split(".")
            value = self.config
            for k in keys:
                if not isinstance(value, dict):
                    return default
                value = value.get(k)
                if value is None:
                    return default
            return value

    def to_dict(self) -> Dict[str, Any]:
        return self.config


def load_config(config_path: str = None) -> Config:
    if config_path is None:
        # default: <project_root>/config/config.yaml
        root = Path(__file__).resolve().parent.parent
        config_path = root / "config" / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)

config = load_config()