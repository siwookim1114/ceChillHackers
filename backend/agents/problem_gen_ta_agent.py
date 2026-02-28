"""Problem Generation TA agent runtime.

Flow:
1) Validate request schema.
2) Retrieve TA context from RAG (practice problems, worked examples).
3) Generate problems via LLM grounded in RAG context.
4) Fall back to template-based generation if LLM fails.
5) Attach normalized RAG citations and adaptation notes.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from config.config_loader import config

from agents.TA_tools import (
    ProblemGenTATool,
    _build_difficulty_plan,
    _default_rubric_hooks,
)
from db.models import (
    CitationRef,
    CommonMistake,
    ErrorTag,
    GeneratedProblem,
    KnowledgeMode,
    ProblemGenTARequest,
    ProblemGenTAResponse,
    RagMode,
)

logger = logging.getLogger(__name__)


def _map_knowledge_mode_to_rag_mode(mode: KnowledgeMode) -> RagMode:
    if mode is KnowledgeMode.INTERNAL_ONLY:
        return RagMode.INTERNAL_ONLY
    if mode is KnowledgeMode.EXTERNAL_ONLY:
        return RagMode.EXTERNAL_ONLY
    return RagMode.EXTERNAL_OK


def _build_rag_prompt(request: ProblemGenTARequest, user_message: str = "") -> str:
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

    # Use the user's actual message for RAG search when available
    # (e.g., "I want to solve probability problems" is a better query than
    # the generic session topic "linear algebra")
    search_query = user_message.strip() if user_message.strip() else (
        f"practice problems exercises examples for {request.topic} {area}"
    )

    return (
        f"{search_query}\n\n"
        f"Context: Generate TA practice for topic '{request.topic}', area '{area}'.\n"
        f"Student profile: level={request.profile.level}, "
        f"learning_style={request.profile.learning_style}, "
        f"pace={request.profile.pace}.\n"
        f"Recent weak concepts: {weak_concepts_str}.\n"
        f"Recent error tags: {error_tags}.\n"
        "Return practical problem patterns, worked-solution structure, and common "
        "pitfalls that can guide generation of scaffolded practice problems."
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
    except Exception as exc:  # pragma: no cover
        logger.warning("RAG import unavailable for problem_gen_ta_agent: %s", exc)
        return None

    try:
        return RagAgent(config=config)
    except Exception as exc:  # pragma: no cover
        logger.warning("RAG initialization failed for problem_gen_ta_agent: %s", exc)
        return None


def _run_rag_context(
    request: ProblemGenTARequest,
    rag_runner: Any | None = None,
    user_message: str = "",
) -> dict[str, Any]:
    runner = rag_runner or _build_rag_agent()
    rag_mode = _map_knowledge_mode_to_rag_mode(request.mode)
    if runner is None:
        return {"found": False, "citations": [], "context": "", "mode": rag_mode.value}

    prompt = _build_rag_prompt(request, user_message)
    payload = {
        "prompt": prompt,
        "caller": "ta",
        "subject": request.topic,
        "mode": rag_mode.value,
        "level": request.profile.level,
    }
    try:
        output = runner.run(payload)
        if not isinstance(output, dict):
            output = {"found": False, "citations": [], "context": ""}
    except Exception as exc:  # pragma: no cover
        logger.warning("RAG call failed for problem_gen_ta_agent: %s", exc)
        output = {"found": False, "citations": [], "context": "", "mode": rag_mode.value}

    # Auto-upgrade: if internal KB returned nothing, retry with external_ok
    # so the RAG agent's built-in Exa web search fallback activates.
    if (
        not output.get("found")
        and rag_mode is RagMode.INTERNAL_ONLY
        and runner is not None
    ):
        logger.info(
            "Internal KB returned no results for problem_gen; "
            "retrying with external_ok to trigger web search fallback."
        )
        payload["mode"] = RagMode.EXTERNAL_OK.value
        try:
            output = runner.run(payload)
            if not isinstance(output, dict):
                output = {"found": False, "citations": [], "context": ""}
        except Exception as exc:  # pragma: no cover
            logger.warning("RAG external_ok retry failed: %s", exc)
            output = {
                "found": False, "citations": [], "context": "",
                "mode": RagMode.EXTERNAL_OK.value,
            }

    return output


# ---------------------------------------------------------------------------
# LLM-based problem generation
# ---------------------------------------------------------------------------

def _generate_problems_llm(
    request: ProblemGenTARequest,
    user_message: str,
    rag_context: str,
    rag_found: bool,
) -> ProblemGenTAResponse:
    """Generate actual problems using an LLM grounded in RAG context.

    Follows the same pattern as ProfessorAgent: RAG retrieval → prompt → LLM → parse.
    Falls through to the caller's template-based fallback on any error.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    from prompts.ta_problem_gen_prompt import (
        TA_PROBLEM_GEN_SYSTEM_PROMPT,
        build_ta_problem_gen_user_prompt,
    )

    api_key = os.environ.get("FEATHERLESSAI_API_KEY")
    if not api_key:
        raise ValueError("Missing FEATHERLESSAI_API_KEY")

    llm = ChatOpenAI(
        model=config.get("llm.model", "meta-llama/Llama-3.3-70B-Instruct"),
        base_url=config.get("llm.base_url", "https://api.featherless.ai/v1"),
        api_key=api_key,
        temperature=0.7,
    )

    # Build difficulty plan
    difficulty_plan = _build_difficulty_plan(
        request.desired_difficulty_curve.current,
        request.desired_difficulty_curve.target,
        request.num_problems,
    )
    difficulty_strs = [d.value for d in difficulty_plan]

    error_tag_strs = (
        [t.value for t in request.recent_error_tags]
        if request.recent_error_tags
        else None
    )

    user_prompt = build_ta_problem_gen_user_prompt(
        user_message=user_message or f"Generate practice problems on {request.topic}",
        topic=request.topic,
        level=request.profile.level,
        learning_style=request.profile.learning_style,
        pace=request.profile.pace,
        num_problems=request.num_problems,
        difficulty_plan=difficulty_strs,
        rag_context=rag_context,
        rag_found=rag_found,
        recent_error_tags=error_tag_strs,
    )

    # LLM call
    llm_response = llm.invoke([
        SystemMessage(content=TA_PROBLEM_GEN_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])

    # Parse response -- handle code fences, think tags, etc.
    raw_text = llm_response.content.strip()
    raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    if raw_text.startswith("```"):
        first_nl = raw_text.find("\n")
        if first_nl != -1:
            raw_text = raw_text[first_nl + 1:]
        if raw_text.rstrip().endswith("```"):
            raw_text = raw_text.rstrip()[:-3].rstrip()

    # Fix invalid backslash escapes that LLMs sometimes produce (e.g. \e, \s, \m).
    # Only fix backslashes NOT followed by valid JSON escape chars: " \ / b f n r t u
    raw_text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw_text)

    parsed = json.loads(raw_text)
    llm_problems = parsed.get("problems", [])

    if not llm_problems:
        raise ValueError("LLM returned no problems")

    # Build GeneratedProblem objects from LLM output
    generated: list[GeneratedProblem] = []
    for idx, difficulty in enumerate(difficulty_plan):
        if idx >= len(llm_problems):
            break

        p = llm_problems[idx]
        mistakes = []
        for m in p.get("common_mistakes", []):
            if isinstance(m, dict) and m.get("label") and m.get("reason") and m.get("fix"):
                mistakes.append(CommonMistake(
                    label=m["label"],
                    reason=m["reason"],
                    fix=m["fix"],
                ))
        if not mistakes:
            mistakes = [CommonMistake(
                label="Concept misapplication",
                reason="Applying the wrong formula or rule.",
                fix="Re-read the problem statement and identify the core concept.",
            )]

        solution_outline = p.get("solution_outline", [])
        if not solution_outline:
            solution_outline = ["Identify the approach.", "Apply it step by step.", "Verify the result."]

        hint_ladder = p.get("hint_ladder", [])
        if not hint_ladder:
            hint_ladder = ["Think about what concept applies here.", "Set up the relevant formula."]

        generated.append(GeneratedProblem(
            problem_id=f"{request.request_id}-p{idx + 1}",
            statement=p.get("statement", f"Problem {idx + 1}"),
            topic=p.get("topic", request.topic),
            difficulty=difficulty,
            estimated_minutes=15,
            solution_outline=solution_outline,
            hint_ladder=hint_ladder,
            common_mistakes=mistakes,
            rubric_hooks=_default_rubric_hooks() if request.include_rubric_hooks else [],
            targets_error_tags=request.recent_error_tags or [ErrorTag.CONCEPT_GAP],
        ))

    if not generated:
        raise ValueError("Could not parse any valid problems from LLM response")

    return ProblemGenTAResponse(
        request_id=request.request_id,
        generated_for_profile=request.profile,
        problems=generated,
        adaptation_notes=[],
        citations=[],
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def invoke_problem_gen_ta(
    payload: dict[str, Any],
    *,
    rag_runner: Any | None = None,
    tool: ProblemGenTATool | None = None,
    user_message: str = "",
) -> dict[str, Any]:
    """Generate practice problems with LLM + RAG, falling back to templates.

    Pipeline:
    1. Validate request
    2. Retrieve RAG context (course material, practice problems)
    3. Try LLM-based generation grounded in RAG context
    4. Fall back to template-based generation if LLM fails
    5. Attach RAG citations
    """
    request = ProblemGenTARequest.model_validate(payload)

    # Step 1: Get RAG context
    rag_result = _run_rag_context(request, rag_runner=rag_runner, user_message=user_message)
    rag_citations = _normalize_rag_citations(rag_result.get("citations", []))
    rag_context = rag_result.get("context", rag_result.get("answer", ""))
    rag_found = rag_result.get("found", False)

    # Step 2: Try LLM-based generation with RAG context
    try:
        response = _generate_problems_llm(
            request, user_message, rag_context, rag_found
        )
        notes = list(response.adaptation_notes)
        if rag_found:
            notes.append("Generated with LLM grounded in RAG course material.")
        else:
            notes.append("Generated with LLM (no RAG course material found).")

        merged = response.model_copy(
            update={
                "citations": rag_citations or response.citations,
                "adaptation_notes": notes,
            }
        )
        return merged.model_dump(mode="json")
    except Exception as exc:
        logger.warning("LLM-based problem gen failed: %s; falling back to templates", exc)

    # Step 3: Fallback to template-based generation
    effective_tool = tool or ProblemGenTATool()
    raw = effective_tool._run(request.model_dump(mode="json"))
    response = ProblemGenTAResponse.model_validate(json.loads(raw))

    notes = list(response.adaptation_notes)
    notes.append("Generated from templates (LLM unavailable).")
    if rag_result.get("mode"):
        notes.append(f"RAG mode: {rag_result['mode']}.")

    merged = response.model_copy(
        update={
            "adaptation_notes": notes,
            "citations": rag_citations or response.citations,
        }
    )
    return merged.model_dump(mode="json")
