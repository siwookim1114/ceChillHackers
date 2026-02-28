from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agents.scan_parser import parse_scan_submission  # noqa: E402
from app.main import parse_scan  # noqa: E402
from db.models import ScanParserRequest, ScanParserResponse  # noqa: E402


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_json(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def test_scan_parser_extracts_problem_steps_and_final_answer() -> None:
    payload = ScanParserRequest.model_validate(_load_json("scan_parser_request.json"))
    response = parse_scan_submission(payload)

    assert isinstance(response, ScanParserResponse)
    assert response.scan_parse.problem_statement
    assert response.scan_parse.final_answer
    assert response.scan_parse.steps
    assert response.scan_parse.raw_parser_confidence is not None
    assert response.scan_parse.raw_parser_confidence >= 0.6
    assert response.diagnostics.equation_like_lines >= 2
    assert all(
        "final answer" not in step.content.lower()
        for step in response.scan_parse.steps
    )


def test_scan_parser_image_only_uses_hints_and_sets_warning() -> None:
    payload = ScanParserRequest.model_validate(
        _load_json("scan_parser_request_image_only.json")
    )
    response = parse_scan_submission(payload)

    assert response.scan_parse.problem_statement == "Solve 2x + 7 = 19."
    assert response.scan_parse.final_answer == "x = 6"
    assert response.scan_parse.raw_parser_confidence is not None
    assert response.scan_parse.raw_parser_confidence <= 0.35
    assert any("missing ocr text" in item.lower() for item in response.diagnostics.warnings)


def test_parse_scan_route_handler_returns_strict_schema() -> None:
    payload = ScanParserRequest.model_validate(_load_json("scan_parser_request.json"))
    response = parse_scan(payload)
    validated = ScanParserResponse.model_validate(response)

    assert validated.scan_parse.steps
    assert validated.diagnostics.focus_score >= 0
    assert validated.diagnostics.effort_score >= 0


def test_parse_scan_route_returns_422_when_no_extractable_content() -> None:
    payload = ScanParserRequest.model_validate(
        _load_json("scan_parser_request_image_no_anchors.json")
    )
    with pytest.raises(HTTPException) as exc_info:
        _ = parse_scan(payload)

    assert exc_info.value.status_code == 422
    assert "could not extract problem content" in str(exc_info.value.detail).lower()


def test_scan_parser_request_rejects_invalid_base64() -> None:
    payload = _load_json("scan_parser_request_image_no_anchors.json")
    payload["image_bytes_b64"] = "not-valid-base64***"

    with pytest.raises(Exception) as exc_info:
        _ = ScanParserRequest.model_validate(payload)

    assert "base64" in str(exc_info.value).lower()
