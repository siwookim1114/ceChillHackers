"""Configuration load and manage"""
import os
import yaml
from typing import Any, Dict
from pathlib import Path


_MISSING = object()


class Config:
    """Configuration manager with dot notation access"""

    def __init__(self, config_dict: Dict[str, Any], prefix: str = ""):
        self.config = config_dict
        self._parse_nested(config_dict, prefix)

    @staticmethod
    def _coerce_env_value(raw: str, template: Any) -> Any:
        if isinstance(template, bool):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(template, int) and not isinstance(template, bool):
            try:
                return int(raw)
            except ValueError:
                return template
        if isinstance(template, float):
            try:
                return float(raw)
            except ValueError:
                return template
        return raw

    @staticmethod
    def _join_prefix(prefix: str, key: str) -> str:
        return f"{prefix}_{key}" if prefix else key

    def _parse_nested(self, d: Dict[str, Any], prefix: str):
        for key, value in d.items():
            full_key = self._join_prefix(prefix, key)
            if isinstance(value, dict):
                setattr(self, key, Config(value, prefix=full_key))
            else:
                # env var overrides yaml value, using full dot-path equivalent.
                env_key = full_key.upper()
                env_value = os.getenv(env_key)
                if env_value is None:
                    setattr(self, key, value)
                else:
                    setattr(self, key, self._coerce_env_value(env_value, value))

    def _resolve_config_value(self, key: str) -> Any:
        keys = key.split(".")
        value: Any = self.config
        for part in keys:
            if not isinstance(value, dict):
                return _MISSING
            if part not in value:
                return _MISSING
            value = value[part]
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get value by dot-path e.g. get('agents.professor.llm.model_id')."""
        env_key = key.replace(".", "_").upper()
        env_value = os.getenv(env_key)
        config_value = self._resolve_config_value(key)
        if env_value is not None:
            template = None if config_value is _MISSING else config_value
            return self._coerce_env_value(env_value, template)

        if config_value is _MISSING:
            return default
        return config_value

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
