from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agents import problem_gen_ta_agent as gen_agent  # noqa: E402
from agents import problem_solve_ta_agent as solve_agent  # noqa: E402
from db.models import (  # noqa: E402
    CitationRef,
    KnowledgeMode,
    ProblemGenTARequest,
    ProblemGenTAResponse,
    ProblemSolvingTARequest,
    ProblemSolvingTAResponse,
    RagMode,
)


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


class NonDictRagRunner:
    def run(self, payload: dict[str, Any]) -> str:
        _ = payload
        return "not-a-dict"


def _load_json(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))


def _gen_request() -> ProblemGenTARequest:
    return ProblemGenTARequest.model_validate(_load_json("ta_problem_gen_request.json"))


def _solve_request() -> ProblemSolvingTARequest:
    return ProblemSolvingTARequest.model_validate(_load_json("ta_problem_solving_request.json"))


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (KnowledgeMode.INTERNAL_ONLY, RagMode.INTERNAL_ONLY),
        (KnowledgeMode.MIXED, RagMode.EXTERNAL_OK),
        (KnowledgeMode.EXTERNAL_ONLY, RagMode.EXTERNAL_ONLY),
    ],
)
def test_problem_gen_mode_mapping(mode: KnowledgeMode, expected: RagMode) -> None:
    assert gen_agent._map_knowledge_mode_to_rag_mode(mode) is expected


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (KnowledgeMode.INTERNAL_ONLY, RagMode.INTERNAL_ONLY),
        (KnowledgeMode.MIXED, RagMode.EXTERNAL_OK),
        (KnowledgeMode.EXTERNAL_ONLY, RagMode.EXTERNAL_ONLY),
    ],
)
def test_problem_solve_mode_mapping(mode: KnowledgeMode, expected: RagMode) -> None:
    assert solve_agent._map_knowledge_mode_to_rag_mode(mode) is expected


def test_problem_gen_build_rag_prompt_includes_profile_weak_concepts_and_errors() -> None:
    req = _gen_request()
    prompt = gen_agent._build_rag_prompt(req)

    assert "topic 'calculus'" in prompt
    assert "area 'chain-rule'" in prompt
    assert "level=beginner" in prompt
    assert "learning_style=example_first" in prompt
    assert "pace=slow" in prompt
    assert "derivative_definition" in prompt
    assert "chain_rule" in prompt
    assert "CONCEPT_GAP" in prompt
    assert "PROCEDURE_SLIP" in prompt


def test_problem_solve_build_rag_prompt_uses_fallback_text_when_steps_missing() -> None:
    req = _solve_request().model_copy(
        update={"scan_parse": _solve_request().scan_parse.model_copy(update={"steps": []})}
    )
    prompt = solve_agent._build_rag_prompt(req)

    assert "topic 'calculus'" in prompt
    assert "Problem statement: Differentiate y = (3x^2 + 1)^4." in prompt
    assert "- no explicit steps provided" in prompt
    assert "identify_outer_inner" in prompt
    assert "apply_chain_rule" in prompt


def test_problem_gen_normalize_rag_citations_handles_sparse_and_invalid_items() -> None:
    citations = [
        {"doc": "lecture.pdf", "title": "Week 3", "snippet": "Chain rule", "page": 12},
        {"url": "https://example.com/math", "index": 9},
        "skip-me",
        {"doc": "", "title": "", "snippet": ""},
    ]

    normalized = gen_agent._normalize_rag_citations(citations)

    assert len(normalized) == 3
    assert normalized[0].source_id == "lecture.pdf"
    assert normalized[0].location == "page 12"
    assert normalized[1].source_id == "https://example.com/math"
    assert normalized[1].location == "index 9"
    assert "RAG retrieval item" in normalized[1].snippet
    assert normalized[2].source_id == "rag-source-4"


def test_problem_solve_normalize_rag_citations_handles_sparse_and_invalid_items() -> None:
    citations = [
        {"doc": "notes.pdf", "title": "Notes", "snippet": "Derivative", "index": 1},
        {"url": "https://example.com", "page": 2},
        None,
    ]

    normalized = solve_agent._normalize_rag_citations(citations)

    assert len(normalized) == 2
    assert normalized[0].source_id == "notes.pdf"
    assert normalized[0].location == "index 1"
    assert normalized[1].source_id == "https://example.com"
    assert normalized[1].location == "page 2"


def test_problem_solve_merge_citations_dedupes_on_source_and_location() -> None:
    primary = [
        CitationRef(
            source_id="internal-rubric",
            title="Internal",
            snippet="Rubric source",
            location="calculus",
        )
    ]
    secondary = [
        CitationRef(
            source_id="internal-rubric",
            title="Internal duplicate",
            snippet="Duplicate",
            location="calculus",
        ),
        CitationRef(
            source_id="lecture.pdf",
            title="Lecture",
            snippet="Chain rule",
            location="page 12",
        ),
    ]

    merged = solve_agent._merge_citations(primary, secondary)
    assert len(merged) == 2
    assert merged[0].source_id == "internal-rubric"
    assert merged[1].source_id == "lecture.pdf"


def test_problem_gen_run_rag_context_without_runner_returns_graceful_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    req = _gen_request()
    monkeypatch.setattr(gen_agent, "_build_rag_agent", lambda: None)

    result = gen_agent._run_rag_context(req)
    assert result == {"found": False, "citations": [], "mode": "external_ok"}


def test_problem_solve_run_rag_context_without_runner_returns_graceful_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    req = _solve_request()
    monkeypatch.setattr(solve_agent, "_build_rag_agent", lambda: None)

    result = solve_agent._run_rag_context(req)
    assert result == {"found": False, "citations": [], "mode": "internal_only"}


def test_problem_gen_run_rag_context_handles_runner_exception() -> None:
    req = _gen_request()
    result = gen_agent._run_rag_context(req, rag_runner=FailingRagRunner())

    assert result["found"] is False
    assert result["citations"] == []
    assert result["mode"] == "external_ok"


def test_problem_solve_run_rag_context_handles_runner_exception() -> None:
    req = _solve_request()
    result = solve_agent._run_rag_context(req, rag_runner=FailingRagRunner())

    assert result["found"] is False
    assert result["citations"] == []
    assert result["mode"] == "internal_only"


def test_problem_gen_run_rag_context_handles_non_dict_output() -> None:
    req = _gen_request()
    result = gen_agent._run_rag_context(req, rag_runner=NonDictRagRunner())

    assert result["found"] is False
    assert result["citations"] == []


def test_problem_solve_run_rag_context_handles_non_dict_output() -> None:
    req = _solve_request()
    result = solve_agent._run_rag_context(req, rag_runner=NonDictRagRunner())

    assert result["found"] is False
    assert result["citations"] == []


def test_problem_gen_invoke_adds_rag_notes_and_normalized_citations() -> None:
    payload = _load_json("ta_problem_gen_request.json")
    rag = DummyRagRunner(
        {
            "found": True,
            "mode": "external_ok",
            "citations": [
                {"doc": "lecture_week_03.pdf", "title": "Week 3", "page": 12, "snippet": "Chain rule"},
            ],
        }
    )

    output = gen_agent.invoke_problem_gen_ta(payload, rag_runner=rag)
    parsed = ProblemGenTAResponse.model_validate(output)

    assert parsed.request_id == payload["request_id"]
    assert any("Added RAG context" in note for note in parsed.adaptation_notes)
    assert any("RAG mode used: external_ok" in note for note in parsed.adaptation_notes)
    assert parsed.citations
    assert parsed.citations[0].source_id == "lecture_week_03.pdf"
    assert rag.calls and rag.calls[0]["caller"] == "ta"


def test_problem_gen_invoke_fallback_note_when_rag_fails() -> None:
    payload = _load_json("ta_problem_gen_request.json")
    output = gen_agent.invoke_problem_gen_ta(payload, rag_runner=FailingRagRunner())
    parsed = ProblemGenTAResponse.model_validate(output)

    assert any("No RAG context found" in note for note in parsed.adaptation_notes)


def test_problem_solve_invoke_merges_and_dedupes_citations() -> None:
    payload = _load_json("ta_problem_solving_request.json")
    rag = DummyRagRunner(
        {
            "found": True,
            "mode": "internal_only",
            "citations": [
                {
                    "doc": "internal-rubric",
                    "title": "TA Internal Rubric",
                    "snippet": "Scoring reference",
                    "index": "calculus",
                },
                {"doc": "derivatives_notes.pdf", "title": "Week 3", "page": 12, "snippet": "Chain rule"},
            ],
        }
    )

    output = solve_agent.invoke_problem_solve_ta(payload, rag_runner=rag)
    parsed = ProblemSolvingTAResponse.model_validate(output)

    keys = {(item.source_id, item.location) for item in parsed.citations}
    assert ("internal-rubric", "calculus") in keys
    assert ("derivatives_notes.pdf", "page 12") in keys
    assert len(parsed.citations) == len(keys)
    assert rag.calls and rag.calls[0]["caller"] == "ta"


def test_problem_solve_invoke_keeps_internal_citation_when_rag_empty() -> None:
    payload = _load_json("ta_problem_solving_request.json")
    rag = DummyRagRunner({"found": False, "mode": "internal_only", "citations": []})

    output = solve_agent.invoke_problem_solve_ta(payload, rag_runner=rag)
    parsed = ProblemSolvingTAResponse.model_validate(output)

    assert parsed.mode.value == "internal_only"
    assert any(c.source_id == "internal-rubric" for c in parsed.citations)


def test_problem_gen_invoke_raises_on_invalid_payload() -> None:
    with pytest.raises(ValidationError):
        _ = gen_agent.invoke_problem_gen_ta({"request_id": "x"})


def test_problem_solve_invoke_raises_on_invalid_payload() -> None:
    with pytest.raises(ValidationError):
        _ = solve_agent.invoke_problem_solve_ta({"request_id": "x"})

