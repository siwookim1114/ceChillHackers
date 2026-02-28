from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from db.models import (  # noqa: E402
    ProblemGenTARequest,
    ProblemGenTAResponse,
    ProblemSolvingTARequest,
    ProblemSolvingTAResponse,
)


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_json(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def test_problem_gen_ta_request_fixture_is_valid() -> None:
    payload = _load_json("ta_problem_gen_request.json")
    parsed = ProblemGenTARequest.model_validate(payload)

    assert parsed.request_id == "gen-001"
    assert parsed.profile.level == "beginner"
    assert parsed.mode.value == "mixed"
    assert parsed.num_problems == 3


def test_problem_gen_ta_response_fixture_is_valid() -> None:
    payload = _load_json("ta_problem_gen_response.json")
    parsed = ProblemGenTAResponse.model_validate(payload)

    assert parsed.request_id == "gen-001"
    assert parsed.problems
    assert parsed.problems[0].difficulty.value == "easy"


def test_problem_solving_ta_request_fixture_is_valid() -> None:
    payload = _load_json("ta_problem_solving_request.json")
    parsed = ProblemSolvingTARequest.model_validate(payload)

    assert parsed.request_id == "solve-001"
    assert parsed.scan_parse.steps
    assert len(parsed.rubric) == 3


def test_problem_solving_ta_response_fixture_is_valid() -> None:
    payload = _load_json("ta_problem_solving_response.json")
    parsed = ProblemSolvingTAResponse.model_validate(payload)

    assert parsed.attempt_id == "attempt-777"
    assert parsed.overall_verdict == "partial"
    assert parsed.partial_score.percent == 60


def test_problem_solving_request_rejects_duplicate_rubric_ids() -> None:
    payload = _load_json("ta_problem_solving_request.json")
    payload["rubric"].append(payload["rubric"][0])

    with pytest.raises(ValidationError, match="rubric criterion_id values must be unique"):
        ProblemSolvingTARequest.model_validate(payload)


def test_problem_solving_response_requires_justification_error_tag_when_flags_present() -> None:
    payload = _load_json("ta_problem_solving_response.json")
    payload["detected_error_tags"] = [
        tag for tag in payload["detected_error_tags"] if tag != "JUSTIFICATION_MISSING"
    ]

    with pytest.raises(
        ValidationError,
        match="missing_justification_flags require JUSTIFICATION_MISSING",
    ):
        ProblemSolvingTAResponse.model_validate(payload)


def test_problem_solving_response_rejects_correct_verdict_with_low_percent() -> None:
    payload = _load_json("ta_problem_solving_response.json")
    payload["overall_verdict"] = "correct"

    with pytest.raises(
        ValidationError,
        match="overall_verdict=correct requires partial_score.percent >= 90",
    ):
        ProblemSolvingTAResponse.model_validate(payload)


def test_problem_solving_response_requires_citations_for_internal_only_feedback() -> None:
    payload = _load_json("ta_problem_solving_response.json")
    payload["mode"] = "internal_only"
    payload["citations"] = []

    with pytest.raises(
        ValidationError,
        match="mode=internal_only requires citations for graded feedback",
    ):
        ProblemSolvingTAResponse.model_validate(payload)
