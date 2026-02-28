"""Problem Generation TA agent runtime.

Flow:
1) Validate request schema.
2) Retrieve TA context from RAG (programmatic call).
3) Generate strict TA output via ProblemGenTATool.
4) Attach normalized RAG citations and adaptation notes.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from config.config_loader import config

from agents.TA_tools import ProblemGenTATool
from db.models import CitationRef, KnowledgeMode, ProblemGenTARequest, ProblemGenTAResponse, RagMode

logger = logging.getLogger(__name__)


def _map_knowledge_mode_to_rag_mode(mode: KnowledgeMode) -> RagMode:
    if mode is KnowledgeMode.INTERNAL_ONLY:
        return RagMode.INTERNAL_ONLY
    if mode is KnowledgeMode.EXTERNAL_ONLY:
        return RagMode.EXTERNAL_ONLY
    return RagMode.EXTERNAL_OK


def _build_rag_prompt(request: ProblemGenTARequest) -> str:
    weak_concepts = [
        m.concept_id for m in request.mastery if m.mastery_score < 0.6
    ][:5]
    weak_concepts_str = ", ".join(weak_concepts) if weak_concepts else "none provided"
    error_tags = (
        ", ".join(tag.value for tag in request.recent_error_tags)
        if request.recent_error_tags
        else "none"
    )
    area = request.subtopic or request.unit or request.topic
    return (
        f"Generate TA practice context for topic '{request.topic}' and area '{area}'.\n"
        f"Student profile: level={request.profile.level}, learning_style={request.profile.learning_style}, "
        f"pace={request.profile.pace}.\n"
        f"Recent weak concepts: {weak_concepts_str}.\n"
        f"Recent error tags: {error_tags}.\n"
        "Return practical problem patterns, worked-solution structure, and common pitfalls "
        "that can guide generation of scaffolded practice problems."
    )


def _normalize_rag_citations(citations: list[Any]) -> list[CitationRef]:
    normalized: list[CitationRef] = []
    for idx, raw in enumerate(citations, start=1):
        if not isinstance(raw, dict):
            continue

        source_id = str(raw.get("doc") or raw.get("url") or f"rag-source-{idx}")
        title = str(raw.get("title") or raw.get("doc") or f"RAG Source {idx}")
        snippet = str(
            raw.get("snippet")
            or f"RAG retrieval item {raw.get('index', idx)}"
        ).strip()
        if not snippet:
            snippet = f"RAG retrieval item {idx}"
        location = None
        if raw.get("page") is not None:
            location = f"page {raw.get('page')}"
        elif raw.get("index") is not None:
            location = f"index {raw.get('index')}"

        try:
            normalized.append(
                CitationRef(
                    source_id=source_id,
                    title=title,
                    snippet=snippet,
                    location=location,
                    url=raw.get("url"),
                )
            )
        except Exception:
            continue
    return normalized


def _build_rag_agent() -> Any | None:
    try:
        from agents.rag_agent import RagAgent
    except Exception as exc:  # pragma: no cover - dependency/env dependent
        logger.warning("RAG import unavailable for problem_gen_ta_agent: %s", exc)
        return None

    try:
        return RagAgent(config=config)
    except Exception as exc:  # pragma: no cover - dependency/env dependent
        logger.warning("RAG initialization failed for problem_gen_ta_agent: %s", exc)
        return None


def _run_rag_context(
    request: ProblemGenTARequest, rag_runner: Any | None = None
) -> dict[str, Any]:
    runner = rag_runner or _build_rag_agent()
    rag_mode = _map_knowledge_mode_to_rag_mode(request.mode)
    if runner is None:
        return {"found": False, "citations": [], "mode": rag_mode.value}

    payload = {
        "prompt": _build_rag_prompt(request),
        "caller": "ta",
        "subject": request.topic,
        "mode": rag_mode.value,
        "level": request.profile.level,
    }
    try:
        output = runner.run(payload)
        return output if isinstance(output, dict) else {"found": False, "citations": []}
    except Exception as exc:  # pragma: no cover - dependency/env dependent
        logger.warning("RAG call failed for problem_gen_ta_agent: %s", exc)
        return {"found": False, "citations": [], "mode": rag_mode.value}


def invoke_problem_gen_ta(
    payload: dict[str, Any],
    *,
    rag_runner: Any | None = None,
    tool: ProblemGenTATool | None = None,
) -> dict[str, Any]:
    request = ProblemGenTARequest.model_validate(payload)

    effective_tool = tool or ProblemGenTATool()
    raw = effective_tool._run(request.model_dump(mode="json"))
    response = ProblemGenTAResponse.model_validate(json.loads(raw))

    rag_result = _run_rag_context(request, rag_runner=rag_runner)
    rag_citations = _normalize_rag_citations(rag_result.get("citations", []))

    notes = list(response.adaptation_notes)
    if rag_result.get("found"):
        notes.append("Added RAG context for TA problem generation.")
    else:
        notes.append("No RAG context found; generated from learner signals only.")
    if rag_result.get("mode"):
        notes.append(f"RAG mode used: {rag_result['mode']}.")

    merged = response.model_copy(
        update={
            "adaptation_notes": notes,
            "citations": rag_citations or response.citations,
        }
    )
    return merged.model_dump(mode="json")

