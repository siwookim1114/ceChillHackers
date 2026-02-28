"""Professor Agent runtime entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.config_loader import config

from agents.tools import ProfessorRespondTool
from db.models import ProfessorTurnResponse


PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "system_prompt.txt"
)


def load_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8").strip()
    return ""


def _resolve_mode_default() -> str:
    mode_default = str(config.get("agents.professor.mode_default", "strict")).strip().lower()
    if mode_default in {"strict", "convenience"}:
        return mode_default
    return "strict"


def _apply_mode_default(payload: dict[str, Any]) -> dict[str, Any]:
    mode_value = payload.get("mode")
    if mode_value is not None and str(mode_value).strip():
        return payload
    normalized_payload = dict(payload)
    normalized_payload["mode"] = _resolve_mode_default()
    return normalized_payload


def invoke_professor(payload: dict[str, Any]) -> dict[str, Any]:
    """Run professor BaseTool and return validated strict payload."""
    tool = ProfessorRespondTool(config=config, system_prompt=load_system_prompt())
    raw_result = tool._run(_apply_mode_default(payload))
    response = ProfessorTurnResponse.model_validate(json.loads(raw_result))
    return response.model_dump()
