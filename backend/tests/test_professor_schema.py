from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.agents.schemas.professor import (
    ProfessorTurnRequest,
    ProfessorTurnResponse,
    ProfessorTurnStrategy,
)
from backend.agents.tools import (
    build_professor_response,
    get_professor_json_schemas,
    sanitize_for_log,
    validate_professor_turn_request,
)


FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "professor_turn_valid.json"
)


def load_valid_payload() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_request_schema_accepts_valid_fixture() -> None:
    payload = load_valid_payload()
    request = validate_professor_turn_request(payload)
    assert isinstance(request, ProfessorTurnRequest)
    assert request.mode.value == "strict"


def test_request_schema_rejects_extra_field() -> None:
    payload = load_valid_payload()
    payload["unexpected"] = "nope"

    with pytest.raises(ValidationError):
        validate_professor_turn_request(payload)


def test_response_schema_blocks_answer_reveal_flag() -> None:
    with pytest.raises(ValidationError):
        ProfessorTurnResponse(
            assistant_response="Here is the final answer.",
            strategy=ProfessorTurnStrategy.HINT,
            revealed_final_answer=True,
            next_action="continue",
            citations=[],
        )


def test_professor_tool_returns_valid_response() -> None:
    request = validate_professor_turn_request(load_valid_payload())
    response = build_professor_response(request)
    assert isinstance(response, ProfessorTurnResponse)
    assert response.revealed_final_answer is False


def test_sanitize_for_log_excludes_raw_message() -> None:
    request = validate_professor_turn_request(load_valid_payload())
    log_data = sanitize_for_log(request)
    assert "message" not in log_data
    assert "student_message" not in log_data
    assert log_data["message_length"] > 0


def test_json_schemas_exposed() -> None:
    schema_bundle = get_professor_json_schemas()
    assert "ProfessorTurnRequest" in schema_bundle
    assert "ProfessorTurnResponse" in schema_bundle
    assert schema_bundle["ProfessorTurnRequest"]["type"] == "object"

