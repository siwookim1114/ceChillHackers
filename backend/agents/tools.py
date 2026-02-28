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
    citations = retrieve_citations_for_professor(request)
    question = (
        f"Before we solve it, can you explain in one sentence what the key idea in "
        f"{request.topic} is?"
    )
    return ProfessorTurnResponse(
        assistant_response=question,
        strategy=ProfessorTurnStrategy.SOCRATIC_QUESTION,
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
