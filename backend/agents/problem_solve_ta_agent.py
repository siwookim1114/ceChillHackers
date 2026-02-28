"""Problem Solving TA agent runtime.

Flow:
1) Validate request schema.
2) Retrieve TA grading context from RAG (programmatic call).
3) Produce strict grading output via ProblemSolvingTATool.
4) Merge normalized RAG citations into response.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from config.config_loader import config

from agents.TA_tools import ProblemSolvingTATool
from db.models import (
    CitationRef,
    KnowledgeMode,
    ProblemSolvingTARequest,
    ProblemSolvingTAResponse,
    RagMode,
)


logger = logging.getLogger(__name__)


def _map_knowledge_mode_to_rag_mode(mode: KnowledgeMode) -> RagMode:
    if mode is KnowledgeMode.INTERNAL_ONLY:
        return RagMode.INTERNAL_ONLY
    if mode is KnowledgeMode.EXTERNAL_ONLY:
        return RagMode.EXTERNAL_ONLY
    return RagMode.EXTERNAL_OK


def _build_rag_prompt(request: ProblemSolvingTARequest) -> str:
    statement = (
        request.problem_ref.statement
        or request.scan_parse.problem_statement
        or "(no statement)"
    )
    attempted_steps = "\n".join(
        f"- Step {step.step_index}: {step.content}" for step in request.scan_parse.steps[:6]
    )
    rubric_focus = "\n".join(
        f"- {item.criterion_id}: {item.description}" for item in request.rubric
    )

    return (
        f"Provide TA grading context for topic '{request.problem_ref.topic}'.\n"
        f"Problem statement: {statement}\n"
        f"Student level={request.profile.level}, learning_style={request.profile.learning_style}, pace={request.profile.pace}\n"
        f"Student attempt:\n{attempted_steps or '- no explicit steps provided'}\n"
        f"Rubric criteria:\n{rubric_focus}\n"
        "Return high-signal guidance for evaluating common mistakes, missing justifications, "
        "and expected procedural structure for this problem type."
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


def _merge_citations(
    primary: list[CitationRef],
    secondary: list[CitationRef],
) -> list[CitationRef]:
    deduped: list[CitationRef] = []
    seen: set[tuple[str, str | None]] = set()

    for item in [*primary, *secondary]:
        key = (item.source_id, item.location)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _build_rag_agent() -> Any | None:
    try:
        from agents.rag_agent import RagAgent
    except Exception as exc:  # pragma: no cover - dependency/env dependent
        logger.warning("RAG import unavailable for problem_solve_ta_agent: %s", exc)
        return None

    try:
        return RagAgent(config=config)
    except Exception as exc:  # pragma: no cover - dependency/env dependent
        logger.warning("RAG initialization failed for problem_solve_ta_agent: %s", exc)
        return None


def _run_rag_context(
    request: ProblemSolvingTARequest, rag_runner: Any | None = None
) -> dict[str, Any]:
    runner = rag_runner or _build_rag_agent()
    rag_mode = _map_knowledge_mode_to_rag_mode(request.mode)
    if runner is None:
        return {"found": False, "citations": [], "mode": rag_mode.value}

    payload = {
        "prompt": _build_rag_prompt(request),
        "caller": "ta",
        "subject": request.problem_ref.topic,
        "mode": rag_mode.value,
        "level": request.profile.level,
    }

    try:
        output = runner.run(payload)
        if not isinstance(output, dict):
            output = {"found": False, "citations": []}
    except Exception as exc:  # pragma: no cover - dependency/env dependent
        logger.warning("RAG call failed for problem_solve_ta_agent: %s", exc)
        output = {"found": False, "citations": [], "mode": rag_mode.value}

    # Auto-upgrade: if internal KB returned nothing, retry with external_ok
    # so the RAG agent's built-in Exa web search fallback activates.
    if (
        not output.get("found")
        and rag_mode is RagMode.INTERNAL_ONLY
        and runner is not None
    ):
        logger.info(
            "Internal KB returned no results for problem_solve; "
            "retrying with external_ok to trigger web search fallback."
        )
        payload["mode"] = RagMode.EXTERNAL_OK.value
        try:
            output = runner.run(payload)
            if not isinstance(output, dict):
                output = {"found": False, "citations": []}
        except Exception as exc:  # pragma: no cover
            logger.warning("RAG external_ok retry failed: %s", exc)
            output = {
                "found": False, "citations": [],
                "mode": RagMode.EXTERNAL_OK.value,
            }

    return output


def invoke_problem_solve_ta(
    payload: dict[str, Any],
    *,
    rag_runner: Any | None = None,
    tool: ProblemSolvingTATool | None = None,
) -> dict[str, Any]:
    request = ProblemSolvingTARequest.model_validate(payload)

    effective_tool = tool or ProblemSolvingTATool()
    raw = effective_tool._run(request.model_dump(mode="json"))
    response = ProblemSolvingTAResponse.model_validate(json.loads(raw))

    rag_result = _run_rag_context(request, rag_runner=rag_runner)
    rag_citations = _normalize_rag_citations(rag_result.get("citations", []))

    merged = response.model_copy(
        update={
            "citations": _merge_citations(response.citations, rag_citations),
        }
    )
    return merged.model_dump(mode="json")

