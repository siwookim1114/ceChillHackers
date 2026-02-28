from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class Problem(BaseModel):
    id: str
    title: str
    prompt: str
    answer_key: str
    unit: str


class AttemptCreateRequest(BaseModel):
    guest_id: str | None = None
    problem_id: str | None = None
    problem_text: str | None = None
    answer_key: str | None = None
    unit: str | None = None


class AttemptCreateResponse(BaseModel):
    attempt_id: str
    started_at: datetime
    problem: Problem


EventType = Literal[
    "stroke_add",
    "stroke_erase",
    "idle_ping",
    "hint_request",
    "answer_submit",
]


class ClientEvent(BaseModel):
    type: EventType
    ts: datetime | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class EventBatchRequest(BaseModel):
    events: list[ClientEvent]


class StuckSignals(BaseModel):
    idle_ms: int
    erase_count_delta: int
    repeated_error_count: int
    stuck_score: int


class Intervention(BaseModel):
    level: Literal[1, 2, 3]
    reason: str
    tutor_message: str
    citations: list[dict[str, str]] | None = None
    created_at: datetime


class EventBatchResponse(BaseModel):
    accepted: int
    stuck_signals: StuckSignals
    intervention: Intervention | None = None
    solved: bool


class AttemptDetailResponse(BaseModel):
    attempt_id: str
    started_at: datetime
    solved_at: datetime | None
    problem: Problem


class TimelineEntry(BaseModel):
    at: datetime
    type: str
    label: str


class SummaryMetrics(BaseModel):
    time_to_solve_sec: int | None
    max_stuck: int
    hint_max_level: int
    erase_count: int


class AttemptSummaryResponse(BaseModel):
    attempt_id: str
    metrics: SummaryMetrics
    timeline: list[TimelineEntry]


@dataclass
class AttemptState:
    id: str
    started_at: datetime
    problem: Problem
    guest_id: str | None = None
    solved_at: datetime | None = None
    events: list[ClientEvent] = field(default_factory=list)
    interventions: list[Intervention] = field(default_factory=list)
    stuck_scores: list[int] = field(default_factory=list)
    last_intervention_at: datetime | None = None


DEFAULT_PROBLEMS: list[Problem] = [
    Problem(
        id="quadratic-01",
        title="Quadratic Basics",
        prompt="Solve x^2 - 5x + 6 = 0",
        answer_key="2,3",
        unit="Quadratic Equations",
    ),
    Problem(
        id="derivative-01",
        title="Derivative Basics",
        prompt="Find d/dx (3x^2 + 2x - 1)",
        answer_key="6x+2",
        unit="Differentiation",
    ),
    Problem(
        id="linear-01",
        title="Linear Equation",
        prompt="Solve 2x + 7 = 19",
        answer_key="6",
        unit="Linear Equations",
    ),
]

PROBLEM_BY_ID = {p.id: p for p in DEFAULT_PROBLEMS}
ATTEMPTS: dict[str, AttemptState] = {}


def normalize_text(value: str) -> str:
    return value.replace(" ", "").lower().strip()


def check_answer(problem: Problem, submitted: str) -> bool:
    if not submitted:
        return False
    correct = normalize_text(problem.answer_key)
    answer = normalize_text(submitted)
    if "," in correct:
        return sorted(correct.split(",")) == sorted(answer.split(","))
    return answer == correct


def compute_signals(events: list[ClientEvent]) -> StuckSignals:
    now = utcnow()
    recent_window = now - timedelta(minutes=2)

    idle_ms = 0
    erase_count_delta = 0
    for event in events:
        event_ts = ensure_utc(event.ts or now)
        if event_ts >= recent_window and event.type == "stroke_erase":
            erase_count_delta += 1
        if event.type == "idle_ping":
            idle_ms = max(idle_ms, int(event.payload.get("idle_ms", 0)))

    repeated_error_count = 0
    for event in reversed(events):
        if event.type != "answer_submit":
            continue
        if event.payload.get("correct", False):
            break
        repeated_error_count += 1

    stuck_score = min(
        100,
        int(idle_ms / 1000 * 0.25)
        + min(erase_count_delta * 9, 27)
        + min(repeated_error_count * 14, 42),
    )
    return StuckSignals(
        idle_ms=idle_ms,
        erase_count_delta=erase_count_delta,
        repeated_error_count=repeated_error_count,
        stuck_score=stuck_score,
    )


def pick_level(stuck_score: int) -> int:
    if stuck_score >= 70:
        return 3
    if stuck_score >= 45:
        return 2
    if stuck_score >= 25:
        return 1
    return 0


def build_hint(problem: Problem, signals: StuckSignals, level: int) -> Intervention:
    reason = (
        f"Intervened due to {signals.idle_ms // 1000}s idle, "
        f"{signals.erase_count_delta} erases, and {signals.repeated_error_count} repeated errors."
    )

    if level == 1:
        tutor_message = (
            f"Before solving `{problem.prompt}`, what is the first structure you notice? "
            "Try naming one strategy and test only that first step."
        )
    elif level == 2:
        tutor_message = (
            "Core idea:\n"
            "1) Identify the equation form.\n"
            "2) Apply one standard method carefully.\n"
            "3) Verify by substitution before finalizing."
        )
    else:
        tutor_message = (
            "Let's step down once. Mini-task: isolate one smaller target from the original expression, "
            "then compute only that part. First step: rewrite the equation in a standard form."
        )

    return Intervention(
        level=level, reason=reason, tutor_message=tutor_message, created_at=utcnow()
    )


def should_emit_intervention(attempt: AttemptState, level: int, force: bool = False) -> bool:
    if level == 0:
        return False
    if force:
        return True
    if attempt.last_intervention_at is None:
        return True
    return utcnow() - attempt.last_intervention_at > timedelta(seconds=15)


app = FastAPI(title="AI Coach MVP API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/problems", response_model=list[Problem])
def list_problems() -> list[Problem]:
    return DEFAULT_PROBLEMS


@app.post("/api/attempts", response_model=AttemptCreateResponse)
def create_attempt(payload: AttemptCreateRequest) -> AttemptCreateResponse:
    if payload.problem_id and payload.problem_id in PROBLEM_BY_ID:
        problem = PROBLEM_BY_ID[payload.problem_id]
    elif payload.problem_text and payload.answer_key:
        problem = Problem(
            id=f"custom-{uuid4().hex[:8]}",
            title="Custom Problem",
            prompt=payload.problem_text,
            answer_key=payload.answer_key,
            unit=payload.unit or "Custom",
        )
    else:
        problem = DEFAULT_PROBLEMS[0]

    attempt_id = uuid4().hex
    attempt = AttemptState(
        id=attempt_id,
        started_at=utcnow(),
        problem=problem,
        guest_id=payload.guest_id,
    )
    ATTEMPTS[attempt_id] = attempt
    return AttemptCreateResponse(
        attempt_id=attempt.id, started_at=attempt.started_at, problem=attempt.problem
    )


@app.get("/api/attempts/{attempt_id}", response_model=AttemptDetailResponse)
def get_attempt(attempt_id: str) -> AttemptDetailResponse:
    attempt = ATTEMPTS.get(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")
    return AttemptDetailResponse(
        attempt_id=attempt.id,
        started_at=attempt.started_at,
        solved_at=attempt.solved_at,
        problem=attempt.problem,
    )


@app.post("/api/attempts/{attempt_id}/events", response_model=EventBatchResponse)
def ingest_events(attempt_id: str, payload: EventBatchRequest) -> EventBatchResponse:
    attempt = ATTEMPTS.get(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")

    for event in payload.events:
        if event.ts is None:
            event.ts = utcnow()
        else:
            event.ts = ensure_utc(event.ts)

        if event.type == "answer_submit":
            answer = str(event.payload.get("answer", ""))
            correct = check_answer(attempt.problem, answer)
            event.payload["correct"] = correct
            if correct and attempt.solved_at is None:
                attempt.solved_at = utcnow()

        attempt.events.append(event)

    signals = compute_signals(attempt.events)
    attempt.stuck_scores.append(signals.stuck_score)

    forced_hint = any(evt.type == "hint_request" for evt in payload.events)
    hint_level = max(1, pick_level(signals.stuck_score)) if forced_hint else pick_level(signals.stuck_score)

    intervention = None
    if should_emit_intervention(attempt, hint_level, force=forced_hint):
        intervention = build_hint(attempt.problem, signals, hint_level)
        attempt.interventions.append(intervention)
        attempt.last_intervention_at = intervention.created_at

    return EventBatchResponse(
        accepted=len(payload.events),
        stuck_signals=signals,
        intervention=intervention,
        solved=attempt.solved_at is not None,
    )


@app.get("/api/attempts/{attempt_id}/intervention", response_model=Intervention | None)
def get_latest_intervention(attempt_id: str) -> Intervention | None:
    attempt = ATTEMPTS.get(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")
    if not attempt.interventions:
        return None
    return attempt.interventions[-1]


@app.get("/api/attempts/{attempt_id}/summary", response_model=AttemptSummaryResponse)
def get_summary(attempt_id: str) -> AttemptSummaryResponse:
    attempt = ATTEMPTS.get(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")

    erase_count = sum(1 for event in attempt.events if event.type == "stroke_erase")
    hint_max_level = max((item.level for item in attempt.interventions), default=0)
    max_stuck = max(attempt.stuck_scores, default=0)

    time_to_solve_sec = None
    if attempt.solved_at is not None:
        delta = attempt.solved_at - attempt.started_at
        time_to_solve_sec = int(delta.total_seconds())

    timeline: list[TimelineEntry] = [
        TimelineEntry(at=attempt.started_at, type="attempt_start", label="Practice started")
    ]
    for intervention in attempt.interventions:
        timeline.append(
            TimelineEntry(
                at=intervention.created_at,
                type="intervention",
                label=f"Hint Level {intervention.level}",
            )
        )
    if attempt.solved_at:
        timeline.append(
            TimelineEntry(
                at=attempt.solved_at,
                type="solved",
                label="Solved",
            )
        )

    timeline.sort(key=lambda item: item.at)

    return AttemptSummaryResponse(
        attempt_id=attempt.id,
        metrics=SummaryMetrics(
            time_to_solve_sec=time_to_solve_sec,
            max_stuck=max_stuck,
            hint_max_level=hint_max_level,
            erase_count=erase_count,
        ),
        timeline=timeline,
    )
