"""Professor Agent runtime entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.config_loader import config

from backend.agents.schemas.professor import ProfessorTurnResponse
from backend.agents.tools import ProfessorRespondTool

try:
    from bedrock_agentcore import BedrockAgentCoreApp
except ImportError:  # pragma: no cover - optional in local test env
    BedrockAgentCoreApp = None


PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "system_prompt.txt"
)


def load_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8").strip()
    return ""


def invoke_professor(payload: dict[str, Any]) -> dict[str, Any]:
    """Run professor BaseTool and return validated strict payload."""
    tool = ProfessorRespondTool(config=config)
    raw_result = tool._run(payload)
    response = ProfessorTurnResponse.model_validate(json.loads(raw_result))
    return response.model_dump()


if BedrockAgentCoreApp is not None:
    app = BedrockAgentCoreApp()

    @app.entrypoint
    def professor_invocation(payload: dict[str, Any], context: Any) -> dict[str, Any]:
        _ = context
        _ = config
        _ = load_system_prompt()
        return {"result": invoke_professor(payload)}

    if __name__ == "__main__":
        app.run()
