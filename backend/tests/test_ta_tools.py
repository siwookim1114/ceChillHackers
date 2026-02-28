from __future__ import annotations

import json
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agents.TA_tools import ProblemGenTATool, ProblemSolvingTATool  # noqa: E402
from db.models import ProblemGenTAResponse, ProblemSolvingTAResponse  # noqa: E402


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _load_json(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def test_problem_gen_ta_tool_returns_valid_response() -> None:
    payload = _load_json("ta_problem_gen_request.json")
    tool = ProblemGenTATool()
    raw = tool._run(payload)
    parsed = ProblemGenTAResponse.model_validate(json.loads(raw))

    assert parsed.request_id == payload["request_id"]
    assert len(parsed.problems) == payload["num_problems"]
    assert parsed.problems[0].problem_id.startswith(f"{payload['request_id']}-p")


def test_problem_gen_ta_tool_adapts_when_high_stuck_score() -> None:
    payload = _load_json("ta_problem_gen_request.json")
    payload["stuck_signals"]["stuck_score"] = 88
    tool = ProblemGenTATool()
    raw = tool._run(payload)
    parsed = ProblemGenTAResponse.model_validate(json.loads(raw))

    assert any("Difficulty was reduced" in note for note in parsed.adaptation_notes)


def test_problem_solving_ta_tool_returns_valid_response_and_citations() -> None:
    payload = _load_json("ta_problem_solving_request.json")
    tool = ProblemSolvingTATool()
    raw = tool._run(payload)
    parsed = ProblemSolvingTAResponse.model_validate(json.loads(raw))

    assert parsed.request_id == payload["request_id"]
    assert parsed.attempt_id == payload["attempt_id"]
    assert parsed.citations, "internal_only mode should include citations"


def test_problem_solving_ta_tool_uses_stuck_score_for_next_action() -> None:
    payload = _load_json("ta_problem_solving_request.json")
    payload["stuck_signals"]["stuck_score"] = 95
    payload["scan_parse"]["steps"] = [
        {"step_index": 1, "content": "idk"},
        {"step_index": 2, "content": "not sure"},
    ]
    payload["scan_parse"]["final_answer"] = ""

    tool = ProblemSolvingTATool()
    raw = tool._run(payload)
    parsed = ProblemSolvingTAResponse.model_validate(json.loads(raw))

    assert parsed.recommended_next_action in {"easier_problem", "escalate"}


def test_problem_solving_ta_tool_invalid_payload_returns_error() -> None:
    tool = ProblemSolvingTATool()
    raw = tool._run({"request_id": "x"})
    parsed = json.loads(raw)

    assert "error" in parsed

