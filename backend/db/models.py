from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class ProfessorMode(str, Enum):
    STRICT = "strict"
    CONVENIENCE = "convenience"


class ProfessorTurnStrategy(str, Enum):
    SOCRATIC_QUESTION = "socratic_question"
    CONCEPTUAL_EXPLANATION = "conceptual_explanation"
    PROCEDURAL_EXPLANATION = "procedural_explanation"
    BROKEN_DOWN_QUESTIONS = "broken_down_questions"


class ProfessorNextAction(str, Enum):
    CONTINUE = "continue"
    ROUTE_PROBLEM_TA = "route_problem_ta"
    ROUTE_PLANNER = "route_planner"


class ProfessorProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    level: Literal["beginner", "intermediate", "advanced"]
    learning_style: Literal["visual", "textual", "example_first", "mixed"]
    pace: Literal["slow", "medium", "fast"] = "medium"


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    snippet: str = Field(min_length=1)
    url: str | None = None


class ProfessorTurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    mode: ProfessorMode = ProfessorMode.STRICT
    profile: ProfessorProfile

    @property
    def student_message(self) -> str:
        return self.message


class ProfessorTurnResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant_response: str = Field(min_length=1)
    strategy: ProfessorTurnStrategy
    revealed_final_answer: Literal[False] = False
    next_action: ProfessorNextAction = ProfessorNextAction.CONTINUE
    citations: list[Citation] = Field(default_factory=list)


class KnowledgeMode(str, Enum):
    INTERNAL_ONLY = "internal_only"
    MIXED = "mixed"
    EXTERNAL_ONLY = "external_only"


class DifficultyBand(str, Enum):
    VERY_EASY = "very_easy"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    CHALLENGE = "challenge"


class ErrorTag(str, Enum):
    CONCEPT_GAP = "CONCEPT_GAP"
    PROCEDURE_SLIP = "PROCEDURE_SLIP"
    CALCULATION_ERROR = "CALCULATION_ERROR"
    MISREAD = "MISREAD"
    JUSTIFICATION_MISSING = "JUSTIFICATION_MISSING"
    TIME_PRESSURE = "TIME_PRESSURE"


class StuckBucket(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LearnerProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    level: str = Field(pattern="^(beginner|intermediate|advanced)$")
    learning_style: str = Field(pattern="^(visual|textual|example_first|mixed)$")
    pace: str = Field(pattern="^(slow|medium|fast)$")


class StuckSignals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idle_ms: int = Field(ge=0)
    erase_count_delta: int = Field(ge=0)
    repeated_error_count: int = Field(ge=0)
    stuck_score: int = Field(ge=0, le=100)
    bucket: StuckBucket | None = None

    @model_validator(mode="after")
    def _fill_bucket_if_missing(self) -> "StuckSignals":
        if self.bucket is not None:
            return self

        if self.stuck_score >= 80:
            self.bucket = StuckBucket.CRITICAL
        elif self.stuck_score >= 55:
            self.bucket = StuckBucket.HIGH
        elif self.stuck_score >= 25:
            self.bucket = StuckBucket.MEDIUM
        else:
            self.bucket = StuckBucket.LOW
        return self


class CitationRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    snippet: str = Field(min_length=1)
    location: str | None = None
    url: str | None = None


class RubricCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    max_points: float = Field(gt=0)
    error_tags: list[ErrorTag] = Field(default_factory=list)


class MasterySignal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    concept_id: str = Field(min_length=1)
    mastery_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    last_seen_at: datetime | None = None


class DifficultyCurve(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    current: DifficultyBand
    target: DifficultyBand
    shape: Literal["linear", "adaptive", "staircase"] = "adaptive"


class CommonMistake(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    fix: str = Field(min_length=1)


class GeneratedProblem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    problem_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    unit: str | None = None
    difficulty: DifficultyBand
    estimated_minutes: int = Field(ge=1, le=180)
    solution_outline: list[str] = Field(default_factory=list)
    hint_ladder: list[str] = Field(default_factory=list)
    easier_variant: str | None = None
    common_mistakes: list[CommonMistake] = Field(default_factory=list)
    rubric_hooks: list[RubricCriterion] = Field(default_factory=list)
    targets_error_tags: list[ErrorTag] = Field(default_factory=list)


class ProblemGenTARequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    unit: str | None = None
    subtopic: str | None = None
    profile: LearnerProfile
    mode: KnowledgeMode
    mastery: list[MasterySignal] = Field(default_factory=list)
    recent_error_tags: list[ErrorTag] = Field(default_factory=list)
    stuck_signals: StuckSignals | None = None
    desired_difficulty_curve: DifficultyCurve
    time_budget_min: int | None = Field(default=None, ge=1, le=24 * 60)
    num_problems: int = Field(ge=1, le=10)
    include_solution_outline: bool = True
    include_rubric_hooks: bool = True
    language: str = "en"


class ProblemGenTAResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    generated_for_profile: LearnerProfile
    problems: list[GeneratedProblem] = Field(min_length=1)
    adaptation_notes: list[str] = Field(default_factory=list)
    citations: list[CitationRef] = Field(default_factory=list)


class ProblemReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    problem_id: str | None = None
    statement: str | None = None
    topic: str = Field(min_length=1)
    unit: str | None = None

    @model_validator(mode="after")
    def _require_identifier_or_statement(self) -> "ProblemReference":
        if self.problem_id or self.statement:
            return self
        raise ValueError("problem_ref must include problem_id or statement")


class StudentStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    step_index: int = Field(ge=1)
    content: str = Field(min_length=1)
    extracted_math: str | None = None
    units: str | None = None


class ScanParseInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    problem_statement: str | None = None
    steps: list[StudentStep] = Field(default_factory=list)
    final_answer: str | None = None
    units: str | None = None
    raw_parser_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _require_some_content(self) -> "ScanParseInput":
        has_problem = bool(self.problem_statement and self.problem_statement.strip())
        has_steps = bool(self.steps)
        has_final = bool(self.final_answer and self.final_answer.strip())
        if has_problem or has_steps or has_final:
            return self
        raise ValueError("scan_parse must include problem_statement, steps, or final_answer")


class ProblemSolvingTARequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    profile: LearnerProfile
    mode: KnowledgeMode
    problem_ref: ProblemReference
    scan_parse: ScanParseInput
    reference_solution_outline: list[str] = Field(default_factory=list)
    rubric: list[RubricCriterion] = Field(min_length=1)
    stuck_signals: StuckSignals | None = None
    language: str = "en"

    @model_validator(mode="after")
    def _ensure_unique_rubric_ids(self) -> "ProblemSolvingTARequest":
        ids = [criterion.criterion_id for criterion in self.rubric]
        if len(ids) != len(set(ids)):
            raise ValueError("rubric criterion_id values must be unique")
        return self


class PartialScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    earned_points: float = Field(ge=0)
    max_points: float = Field(gt=0)
    percent: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _percent_consistency(self) -> "PartialScore":
        expected = (self.earned_points / self.max_points) * 100.0
        if abs(expected - self.percent) > 1.5:
            raise ValueError("partial_score.percent is inconsistent with earned/max")
        return self


class StepVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    step_index: int = Field(ge=1)
    verdict: Literal["ok", "issue", "missing"]
    message: str = Field(min_length=1)
    rubric_criterion_ids: list[str] = Field(default_factory=list)
    error_tags: list[ErrorTag] = Field(default_factory=list)
    awarded_points: float = Field(ge=0)


class ProblemSolvingTAResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    mode: KnowledgeMode
    overall_verdict: Literal["correct", "partial", "incorrect", "insufficient_work"]
    partial_score: PartialScore
    step_verdicts: list[StepVerdict] = Field(default_factory=list)
    corrections: list[str] = Field(default_factory=list)
    missing_justification_flags: list[str] = Field(default_factory=list)
    detected_error_tags: list[ErrorTag] = Field(default_factory=list)
    recommended_next_action: Literal[
        "continue",
        "request_hint",
        "easier_problem",
        "escalate",
    ]
    feedback_message: str = Field(min_length=1)
    citations: list[CitationRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_consistency(self) -> "ProblemSolvingTAResponse":
        step_ids = [step.step_index for step in self.step_verdicts]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("step_verdicts.step_index must be unique")

        if self.overall_verdict == "correct" and self.partial_score.percent < 90:
            raise ValueError("overall_verdict=correct requires partial_score.percent >= 90")

        if self.missing_justification_flags and ErrorTag.JUSTIFICATION_MISSING not in self.detected_error_tags:
            raise ValueError(
                "missing_justification_flags require JUSTIFICATION_MISSING in detected_error_tags"
            )

        if self.mode is KnowledgeMode.INTERNAL_ONLY and self.feedback_message.strip() and not self.citations:
            raise ValueError("mode=internal_only requires citations for graded feedback")

        return self
