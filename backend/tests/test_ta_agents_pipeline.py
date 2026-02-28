from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agents.problem_gen_ta_agent import invoke_problem_gen_ta  # noqa: E402
from agents.problem_solve_ta_agent import invoke_problem_solve_ta  # noqa: E402
from db.models import ProblemGenTAResponse, ProblemSolvingTAResponse  # noqa: E402


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class DummyRagRunner:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        return self.result


class FailingRagRunner:
    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        _ = payload
        raise RuntimeError("rag unavailable")


def _load_json(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def test_problem_gen_agent_pipeline_attaches_rag_context() -> None:
    payload = _load_json("ta_problem_gen_request.json")
    rag = DummyRagRunner(
        {
            "found": True,
            "mode": "external_ok",
            "citations": [
                {"index": 1, "doc": "lecture_week_03.pdf", "page": 12, "score": 0.82},
            ],
        }
    )

    output = invoke_problem_gen_ta(payload, rag_runner=rag)
    response = ProblemGenTAResponse.model_validate(output)

    assert response.request_id == payload["request_id"]
    assert response.citations, "RAG citations should be attached"
    assert any("Added RAG context" in note for note in response.adaptation_notes)
    assert rag.calls and rag.calls[0]["caller"] == "ta"
    assert rag.calls[0]["mode"] == "external_ok"


def test_problem_gen_agent_pipeline_falls_back_without_rag() -> None:
    payload = _load_json("ta_problem_gen_request.json")
    output = invoke_problem_gen_ta(payload, rag_runner=FailingRagRunner())
    response = ProblemGenTAResponse.model_validate(output)

    assert response.request_id == payload["request_id"]
    assert any("No RAG context found" in note for note in response.adaptation_notes)


def test_problem_solve_agent_pipeline_merges_rag_citations() -> None:
    payload = _load_json("ta_problem_solving_request.json")
    rag = DummyRagRunner(
        {
            "found": True,
            "mode": "internal_only",
            "citations": [
                {"index": 1, "doc": "derivatives_notes.pdf", "page": 12, "score": 0.9},
            ],
        }
    )

    output = invoke_problem_solve_ta(payload, rag_runner=rag)
    response = ProblemSolvingTAResponse.model_validate(output)

    assert response.request_id == payload["request_id"]
    assert response.citations, "Internal-only solve response must keep citations"
    source_ids = {item.source_id for item in response.citations}
    assert "internal-rubric" in source_ids
    assert "derivatives_notes.pdf" in source_ids
    assert rag.calls and rag.calls[0]["mode"] == "internal_only"


def test_problem_solve_agent_pipeline_respects_external_only_mode() -> None:
    payload = _load_json("ta_problem_solving_request.json")
    payload["mode"] = "external_only"
    rag = DummyRagRunner({"found": False, "mode": "external_only", "citations": []})

    output = invoke_problem_solve_ta(payload, rag_runner=rag)
    response = ProblemSolvingTAResponse.model_validate(output)

    assert response.mode.value == "external_only"
    assert rag.calls and rag.calls[0]["mode"] == "external_only"

