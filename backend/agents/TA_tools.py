"""TA agent tools.

This module contains TA-specific tools only:
- ProblemGenTATool: generate practice problems from learner context.
- ProblemSolvingTATool: evaluate a student's worked solution.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Union

from langchain_core.tools import BaseTool
from pydantic import ValidationError

try:
    from db.models import (
        CitationRef,
        CommonMistake,
        DifficultyBand,
        ErrorTag,
        GeneratedProblem,
        PartialScore,
        ProblemGenTARequest,
        ProblemGenTAResponse,
        ProblemSolvingTARequest,
        ProblemSolvingTAResponse,
        RubricCriterion,
        StepVerdict,
        StuckBucket,
    )
    from utils.helpers import parse_tool_input
except ImportError:  # pragma: no cover - package-style imports
    from backend.db.models import (
        CitationRef,
        CommonMistake,
        DifficultyBand,
        ErrorTag,
        GeneratedProblem,
        PartialScore,
        ProblemGenTARequest,
        ProblemGenTAResponse,
        ProblemSolvingTARequest,
        ProblemSolvingTAResponse,
        RubricCriterion,
        StepVerdict,
        StuckBucket,
    )
    from backend.utils.helpers import parse_tool_input


logger = logging.getLogger(__name__)

_DIFFICULTY_ORDER: list[DifficultyBand] = [
    DifficultyBand.VERY_EASY,
    DifficultyBand.EASY,
    DifficultyBand.MEDIUM,
    DifficultyBand.HARD,
    DifficultyBand.CHALLENGE,
]

_COMMON_MISTAKE_BY_TAG: dict[ErrorTag, tuple[str, str, str]] = {
    ErrorTag.CONCEPT_GAP: (
        "Concept mismatch",
        "The method selected does not match the concept required by the prompt.",
        "Restate the governing concept before doing calculations.",
    ),
    ErrorTag.PROCEDURE_SLIP: (
        "Procedure skipped",
        "One or more intermediate transformations were skipped or reordered incorrectly.",
        "Write each transformation explicitly and check operation order.",
    ),
    ErrorTag.CALCULATION_ERROR: (
        "Arithmetic/algebra error",
        "A symbolic or arithmetic manipulation introduced a wrong term.",
        "Recompute the previous line and verify signs/coefficient arithmetic.",
    ),
    ErrorTag.MISREAD: (
        "Prompt misread",
        "Important problem constraints were ignored.",
        "Rewrite the givens/target before starting.",
    ),
    ErrorTag.JUSTIFICATION_MISSING: (
        "Missing justification",
        "A conclusion was written without explaining why it follows.",
        "Add one sentence that names the rule/theorem used.",
    ),
    ErrorTag.TIME_PRESSURE: (
        "Time pressure",
        "Response quality dropped due to rushed steps.",
        "Break the task into shorter checkpoints and verify after each checkpoint.",
    ),
}


def _difficulty_index(band: DifficultyBand) -> int:
    return _DIFFICULTY_ORDER.index(band)


def _clamp_difficulty(idx: int) -> DifficultyBand:
    return _DIFFICULTY_ORDER[max(0, min(len(_DIFFICULTY_ORDER) - 1, idx))]


def _build_difficulty_plan(
    current: DifficultyBand,
    target: DifficultyBand,
    num_problems: int,
) -> list[DifficultyBand]:
    if num_problems <= 1:
        return [target]

    start = _difficulty_index(current)
    end = _difficulty_index(target)
    plan: list[DifficultyBand] = []
    for i in range(num_problems):
        ratio = i / (num_problems - 1)
        idx = round(start + (end - start) * ratio)
        plan.append(_clamp_difficulty(idx))
    return plan


def _problem_statement(
    request: ProblemGenTARequest,
    difficulty: DifficultyBand,
    problem_number: int,
) -> str:
    area = request.subtopic or request.unit or request.topic
    style_suffix = {
        "visual": "Include a quick sketch/table to reason about structure.",
        "textual": "Explain each step in one short sentence.",
        "example_first": "Start by following the worked pattern, then generalize.",
        "mixed": "Use one equation step plus one sentence of explanation.",
    }[request.profile.learning_style]
    return (
        f"Problem {problem_number}: Solve one {difficulty.value.replace('_', ' ')} "
        f"{request.topic} exercise on '{area}'. {style_suffix}"
    )


def _solution_outline(topic: str, subtopic: str | None) -> list[str]:
    area = subtopic or topic
    return [
        f"Identify the governing rule for {area}.",
        "Substitute known quantities and simplify carefully.",
        "Check whether the final result satisfies the original condition.",
    ]


def _hint_ladder(topic: str) -> list[str]:
    return [
        f"Start by naming the core {topic} rule you need.",
        "Write one intermediate line before jumping to the final form.",
        "Verify sign/coefficient consistency by checking the previous line only.",
    ]


def _default_rubric_hooks() -> list[RubricCriterion]:
    return [
        RubricCriterion(
            criterion_id="method_selection",
            description="Chooses an appropriate method.",
            max_points=3,
            error_tags=[ErrorTag.CONCEPT_GAP, ErrorTag.MISREAD],
        ),
        RubricCriterion(
            criterion_id="procedure_execution",
            description="Executes steps in a valid order.",
            max_points=4,
            error_tags=[ErrorTag.PROCEDURE_SLIP, ErrorTag.CALCULATION_ERROR],
        ),
        RubricCriterion(
            criterion_id="justification",
            description="Justifies key transitions.",
            max_points=3,
            error_tags=[ErrorTag.JUSTIFICATION_MISSING],
        ),
    ]


class ProblemGenTATool(BaseTool):
    """Generate practice problems aligned to profile/mastery/stuck signals."""

    name: str = "ta_problem_gen"
    description: str = (
        "Generate practice problems aligned to learner profile, mastery signals, "
        "recent errors, and desired difficulty curve."
    )

    def _run(self, tool_input: Union[str, dict]) -> str:
        try:
            payload = parse_tool_input(tool_input)
            request = ProblemGenTARequest.model_validate(payload)
        except (TypeError, ValueError, ValidationError) as exc:
            return json.dumps({"error": f"Invalid ProblemGenTARequest: {exc}"})

        target = request.desired_difficulty_curve.target
        notes: list[str] = []
        if request.stuck_signals and request.stuck_signals.bucket in {
            StuckBucket.HIGH,
            StuckBucket.CRITICAL,
        }:
            target = _clamp_difficulty(_difficulty_index(target) - 1)
            notes.append(
                "Difficulty was reduced by one level due to high stuck score."
            )

        difficulty_plan = _build_difficulty_plan(
            current=request.desired_difficulty_curve.current,
            target=target,
            num_problems=request.num_problems,
        )

        problem_time = 15
        if request.time_budget_min:
            problem_time = max(5, int(request.time_budget_min / request.num_problems))

        tags = request.recent_error_tags or [ErrorTag.CONCEPT_GAP]
        mistakes = [
            CommonMistake(
                label=_COMMON_MISTAKE_BY_TAG[tag][0],
                reason=_COMMON_MISTAKE_BY_TAG[tag][1],
                fix=_COMMON_MISTAKE_BY_TAG[tag][2],
            )
            for tag in tags
        ]

        generated: list[GeneratedProblem] = []
        for idx, difficulty in enumerate(difficulty_plan, start=1):
            generated.append(
                GeneratedProblem(
                    problem_id=f"{request.request_id}-p{idx}",
                    statement=_problem_statement(request, difficulty, idx),
                    topic=request.topic,
                    unit=request.unit,
                    difficulty=difficulty,
                    estimated_minutes=problem_time,
                    solution_outline=(
                        _solution_outline(request.topic, request.subtopic)
                        if request.include_solution_outline
                        else []
                    ),
                    hint_ladder=_hint_ladder(request.topic),
                    easier_variant=(
                        f"Retry as a {DifficultyBand.EASY.value.replace('_', ' ')} "
                        f"version focusing only on first two steps."
                        if difficulty in {DifficultyBand.HARD, DifficultyBand.CHALLENGE}
                        else None
                    ),
                    common_mistakes=mistakes,
                    rubric_hooks=_default_rubric_hooks()
                    if request.include_rubric_hooks
                    else [],
                    targets_error_tags=tags,
                )
            )

        response = ProblemGenTAResponse(
            request_id=request.request_id,
            generated_for_profile=request.profile,
            problems=generated,
            adaptation_notes=notes,
            citations=[],
        )
        return json.dumps(response.model_dump(mode="json"))


def _contains_any(text: str, needles: list[str]) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in needles)


class ProblemSolvingTATool(BaseTool):
    """Evaluate student solution attempts and return structured rubric output."""

    name: str = "ta_problem_solve"
    description: str = (
        "Evaluate a student's parsed solution steps against rubric and produce "
        "step verdicts, partial score, and next action."
    )

    def _run(self, tool_input: Union[str, dict]) -> str:
        try:
            payload = parse_tool_input(tool_input)
            request = ProblemSolvingTARequest.model_validate(payload)
        except (TypeError, ValueError, ValidationError) as exc:
            return json.dumps({"error": f"Invalid ProblemSolvingTARequest: {exc}"})

        total_max = sum(item.max_points for item in request.rubric)
        steps = request.scan_parse.steps

        if not steps and not (request.scan_parse.final_answer or "").strip():
            response = ProblemSolvingTAResponse(
                request_id=request.request_id,
                attempt_id=request.attempt_id,
                mode=request.mode,
                overall_verdict="insufficient_work",
                partial_score=PartialScore(
                    earned_points=0.0,
                    max_points=total_max,
                    percent=0.0,
                ),
                step_verdicts=[],
                corrections=["Add at least one explicit solution step."],
                missing_justification_flags=[
                    "No derivation was provided, so reasoning cannot be graded."
                ],
                detected_error_tags=[ErrorTag.JUSTIFICATION_MISSING],
                recommended_next_action="request_hint",
                feedback_message=(
                    "There is not enough written work to evaluate yet. "
                    "Show one to two intermediate steps first."
                ),
                citations=(
                    [
                        CitationRef(
                            source_id="internal-rubric",
                            title="TA Internal Rubric",
                            snippet="Evaluation uses provided rubric and internal constraints.",
                            location=request.problem_ref.topic,
                        )
                    ]
                    if request.mode.value == "internal_only"
                    else []
                ),
            )
            return json.dumps(response.model_dump(mode="json"))

        step_weight = total_max / max(len(steps), 1)
        step_verdicts: list[StepVerdict] = []
        detected_tags: set[ErrorTag] = set()
        corrections: list[str] = []
        missing_flags: list[str] = []

        for idx, step in enumerate(steps, start=1):
            criterion = request.rubric[(idx - 1) % len(request.rubric)]
            content = step.content.strip()

            if len(content) < 12:
                verdict = "missing"
                factor = 0.0
                tags = [ErrorTag.JUSTIFICATION_MISSING]
                message = "Step is too short to verify reasoning."
                missing_flags.append(f"Step {step.step_index} lacks justification detail.")
            elif _contains_any(content, ["idk", "don't know", "not sure", "guess"]):
                verdict = "issue"
                factor = 0.25
                tags = [ErrorTag.CONCEPT_GAP]
                message = "Step indicates uncertainty in core method selection."
                corrections.append(
                    f"Step {step.step_index}: state the method first, then apply it."
                )
            elif _contains_any(content, ["maybe", "i think", "probably"]):
                verdict = "issue"
                factor = 0.6
                tags = [ErrorTag.JUSTIFICATION_MISSING]
                message = "Reasoning is tentative; add explicit rule-based justification."
                missing_flags.append(f"Step {step.step_index} needs explicit justification.")
            else:
                verdict = "ok"
                factor = 1.0
                tags = []
                message = "Step is clear and structurally valid."

            awarded = round(min(step_weight * factor, criterion.max_points), 2)
            detected_tags.update(tags)
            step_verdicts.append(
                StepVerdict(
                    step_index=step.step_index,
                    verdict=verdict,
                    message=message,
                    rubric_criterion_ids=[criterion.criterion_id],
                    error_tags=tags,
                    awarded_points=awarded,
                )
            )

        earned = round(sum(step.awarded_points for step in step_verdicts), 2)
        earned = min(earned, total_max)
        percent = round((earned / total_max) * 100, 1) if total_max > 0 else 0.0

        if missing_flags:
            detected_tags.add(ErrorTag.JUSTIFICATION_MISSING)

        if percent >= 90:
            overall = "correct"
            feedback = "Your method and execution are both strong. Keep this structure."
        elif percent >= 50:
            overall = "partial"
            feedback = "You are on the right track, but a few steps need tighter reasoning."
        else:
            overall = "incorrect"
            feedback = "Key method or justification gaps are blocking a correct result."

        bucket = request.stuck_signals.bucket if request.stuck_signals else None
        if bucket == StuckBucket.CRITICAL and percent < 35:
            next_action = "escalate"
        elif bucket in {StuckBucket.HIGH, StuckBucket.CRITICAL} and percent < 70:
            next_action = "easier_problem"
        elif percent < 70:
            next_action = "request_hint"
        else:
            next_action = "continue"

        citations: list[CitationRef] = []
        if request.mode.value == "internal_only":
            citations.append(
                CitationRef(
                    source_id="internal-rubric",
                    title="TA Internal Rubric",
                    snippet="Scoring is computed from the provided rubric and parsed steps.",
                    location=request.problem_ref.topic,
                )
            )

        response = ProblemSolvingTAResponse(
            request_id=request.request_id,
            attempt_id=request.attempt_id,
            mode=request.mode,
            overall_verdict=overall,
            partial_score=PartialScore(
                earned_points=earned,
                max_points=total_max,
                percent=percent,
            ),
            step_verdicts=step_verdicts,
            corrections=corrections,
            missing_justification_flags=missing_flags,
            detected_error_tags=sorted(detected_tags, key=lambda tag: tag.value),
            recommended_next_action=next_action,
            feedback_message=feedback,
            citations=citations,
        )
        return json.dumps(response.model_dump(mode="json"))

