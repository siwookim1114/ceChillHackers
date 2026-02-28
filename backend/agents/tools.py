"""Professor agent tools and schema helpers.

This module intentionally avoids persistence and raw-content logging.
"""

from __future__ import annotations

from typing import Any

from backend.agents.schemas.professor import (
    Citation,
    ProfessorMode,
    ProfessorNextAction,
    ProfessorTurnRequest,
    ProfessorTurnResponse,
    ProfessorTurnStrategy,
)
from config.config_loader import load_config


PROFESSOR_CONFIG_PATH = "agents.professor"


def get_professor_runtime_config() -> dict[str, Any]:
    """Load professor runtime config from config.yaml as single source of truth."""
    app_config = load_config()
    professor_config = app_config.get(PROFESSOR_CONFIG_PATH)

    if not isinstance(professor_config, dict):
        raise ValueError(f"Missing or invalid config mapping at '{PROFESSOR_CONFIG_PATH}'")

    llm_cfg = professor_config.get("llm")
    tutoring_cfg = professor_config.get("tutoring")
    if not isinstance(llm_cfg, dict):
        raise ValueError("Missing or invalid config mapping at 'agents.professor.llm'")
    if not isinstance(tutoring_cfg, dict):
        raise ValueError("Missing or invalid config mapping at 'agents.professor.tutoring'")
    if not llm_cfg.get("provider"):
        raise ValueError("Missing required config key: 'agents.professor.llm.provider'")
    if not llm_cfg.get("model_id"):
        raise ValueError("Missing required config key: 'agents.professor.llm.model_id'")

    return professor_config


def validate_professor_turn_request(payload: dict[str, Any]) -> ProfessorTurnRequest:
    """Validate inbound payload against the strict Professor request schema."""
    return ProfessorTurnRequest.model_validate(payload)


def sanitize_for_log(request: ProfessorTurnRequest) -> dict[str, Any]:
    """Return metadata-only logs. Never include raw student content."""
    return {
        "session_id": request.session_id,
        "mode": request.mode.value,
        "topic": request.topic,
        "message_length": len(request.student_message),
        "profile_level": request.profile.level,
    }


def retrieve_citations_for_professor(request: ProfessorTurnRequest) -> list[Citation]:
    """Temporary citation stub until RAG/KB is connected."""
    config = get_professor_runtime_config()
    citations_enabled = bool(config.get("tutoring", {}).get("citations_enabled", True))
    if not citations_enabled:
        return []

    if request.mode is ProfessorMode.STRICT:
        source_id = "local_stub_strict"
    else:
        source_id = "local_stub_convenience"

    return [
        Citation(
            source_id=source_id,
            title=f"Intro to {request.topic}",
            snippet=f"Core concept recap for {request.topic}.",
            url=None,
        )
    ]


def build_professor_response(request: ProfessorTurnRequest) -> ProfessorTurnResponse:
    """Generate deterministic mock response that always respects no-answer-reveal policy."""
    config = get_professor_runtime_config()
    citations = retrieve_citations_for_professor(request)
    socratic_default = bool(config.get("tutoring", {}).get("socratic_default", True))
    if socratic_default:
        strategy = ProfessorTurnStrategy.SOCRATIC_QUESTION
        question = (
            f"Before we solve it, can you explain in one sentence what the key idea in "
            f"{request.topic} is?"
        )
    else:
        strategy = ProfessorTurnStrategy.CONCEPT_EXPLAIN
        question = (
            f"Let's do a short recap: in {request.topic}, focus on the core principle "
            "first, then we can apply it to your problem."
        )

    return ProfessorTurnResponse(
        assistant_response=question,
        strategy=strategy,
        revealed_final_answer=False,
        next_action=ProfessorNextAction.CONTINUE,
        citations=citations,
    )


def get_professor_json_schemas() -> dict[str, dict[str, Any]]:
    """Expose JSON Schemas for transport contracts and UI/backend validation."""
    return {
        "ProfessorTurnRequest": ProfessorTurnRequest.model_json_schema(),
        "ProfessorTurnResponse": ProfessorTurnResponse.model_json_schema(),
    }
