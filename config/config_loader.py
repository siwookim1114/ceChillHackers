"""Configuration loader with dot-notation access and env var overrides."""
import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# Load .env from project root before anything else so AWS credentials
# and all other env vars are available for boto3 and os.getenv()
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


class Config:
    """Configuration manager with dot-notation access.

    Nested yaml keys become attributes (config.bedrock.models.rag).
    Leaf values check os.getenv(KEY.upper()) first; yaml value is the fallback.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self._parse_nested(config_dict)

    def _parse_nested(self, d: Dict[str, Any]) -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                env_key = key.upper()
                setattr(self, key, os.getenv(env_key, value))

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by dot-path, e.g. get('bedrock.models.rag')."""
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
        config_path = _ROOT / "config" / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


config = load_config()
