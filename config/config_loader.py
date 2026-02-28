"""Configuration loader with dot-notation access and env var overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# Load .env from project root before anything else.
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

_MISSING = object()


class Config:
    """Configuration manager with dot-notation access.

    Nested YAML keys become attributes (e.g. ``config.bedrock.knowledge_base_id``).
    For leaf values, env var overrides are checked in this order:
    1) FULL_PATH style (e.g. ``AGENTS_PROFESSOR_MODE_DEFAULT``)
    2) LEAF_ONLY style (e.g. ``MODE_DEFAULT``)
    """

    def __init__(self, config_dict: Dict[str, Any], prefix: str = ""):
        self.config = config_dict
        self._prefix = prefix
        self._parse_nested(config_dict)

    @staticmethod
    def _join_prefix(prefix: str, key: str) -> str:
        if not prefix:
            return key
        return f"{prefix}.{key}"

    @staticmethod
    def _env_key_from_path(path: str) -> str:
        return path.replace(".", "_").upper()

    def _resolve_env_override(self, full_key: str, leaf_key: str, fallback: Any) -> Any:
        full_env_key = self._env_key_from_path(full_key)
        if full_env_key in os.environ:
            return os.environ[full_env_key]

        leaf_env_key = leaf_key.upper()
        if leaf_env_key in os.environ:
            return os.environ[leaf_env_key]

        return fallback

    def _parse_nested(self, d: Dict[str, Any]) -> None:
        for key, value in d.items():
            full_key = self._join_prefix(self._prefix, key)
            if isinstance(value, dict):
                setattr(self, key, Config(value, prefix=full_key))
            else:
                setattr(self, key, self._resolve_env_override(full_key, key, value))

    def get(self, key: str, default: Any = None) -> Any:
        """Get value by dot-path, e.g. ``get('bedrock.models.rag')``."""
        keys = key.split(".")
        value: Any = self.config
        for k in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(k, _MISSING)
            if value is _MISSING:
                return default

        if isinstance(value, dict):
            return value

        full_env_key = self._env_key_from_path(key)
        if full_env_key in os.environ:
            return os.environ[full_env_key]

        leaf_env_key = keys[-1].upper()
        if leaf_env_key in os.environ:
            return os.environ[leaf_env_key]

        return value

    def to_dict(self) -> Dict[str, Any]:
        return self.config


def load_config(config_path: str | None = None) -> Config:
    if config_path is None:
        config_path = _ROOT / "config" / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}

    if not isinstance(config_dict, dict):
        raise TypeError("Config YAML root must be an object")

    return Config(config_dict)


config = load_config()
