from __future__ import annotations

import json
import os
import base64
import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import psycopg
    from psycopg import errors as psycopg_errors
    from psycopg.rows import dict_row
    from psycopg.types.json import Json
except ImportError:
    psycopg = None
    psycopg_errors = None
    dict_row = None
    Json = None


# Load root-level .env as default runtime config.
ROOT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ROOT_ENV_PATH)
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()


def normalize_database_url(url: str) -> str:
    if not url:
        return url
    parts = urlsplit(url)
    if "rds.amazonaws.com" not in (parts.hostname or ""):
        return url

    query_items = dict(parse_qsl(parts.query))
    if "sslmode" not in query_items:
        query_items["sslmode"] = "require"
        return urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urlencode(query_items), parts.fragment)
        )
    return url


DATABASE_URL = normalize_database_url(DATABASE_URL)
DB_ENABLED = bool(DATABASE_URL) and "<DB_PASSWORD>" not in DATABASE_URL
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "dev-only-change-me-please")
ACCESS_TOKEN_TTL_SEC = int(os.getenv("ACCESS_TOKEN_TTL_SEC", "604800"))


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_json_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    if isinstance(value, dict):
        return value
    return {}


def parse_json_list(value: Any) -> list[dict[str, str]] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        return None
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return None


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def b64url_decode(value: str) -> bytes:
    padding = "=" * ((4 - len(value) % 4) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("utf-8"))


def hash_password(password: str) -> str:
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long")
    salt = secrets.token_bytes(16)
    n, r, p = 2**14, 8, 1
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=32
    )
    return f"scrypt${n}${r}${p}${b64url_encode(salt)}${b64url_encode(digest)}"


def verify_password(password: str, encoded_hash: str) -> bool:
    try:
        algo, n_str, r_str, p_str, salt_b64, digest_b64 = encoded_hash.split("$", 5)
        if algo != "scrypt":
            return False
        n, r, p = int(n_str), int(r_str), int(p_str)
        salt = b64url_decode(salt_b64)
        expected = b64url_decode(digest_b64)
        computed = hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=len(expected)
        )
        return hmac.compare_digest(expected, computed)
    except Exception:
        return False


def create_access_token(user_id: str, email: str, role: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "exp": int(time.time()) + ACCESS_TOKEN_TTL_SEC,
    }
    header_b64 = b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = b64url_encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    )
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(
        APP_SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256
    ).digest()
    return f"{header_b64}.{payload_b64}.{b64url_encode(signature)}"


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        header_b64, payload_b64, signature_b64 = token.split(".", 2)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid token format") from exc

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    expected_signature = hmac.new(
        APP_SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256
    ).digest()
    provided_signature = b64url_decode(signature_b64)
    if not hmac.compare_digest(expected_signature, provided_signature):
        raise HTTPException(status_code=401, detail="Invalid token signature")

    payload = json.loads(b64url_decode(payload_b64))
    if int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(status_code=401, detail="Token expired")
    return payload


def parse_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    return authorization.removeprefix("Bearer ").strip()


class Problem(BaseModel):
    id: str
    title: str
    prompt: str
    answer_key: str
    unit: str


class AttemptCreateRequest(BaseModel):
    guest_id: Optional[str] = None
    problem_id: Optional[str] = None
    problem_text: Optional[str] = None
    answer_key: Optional[str] = None
    unit: Optional[str] = None


class AttemptCreateResponse(BaseModel):
    attempt_id: str
    started_at: datetime
    problem: Problem


UserRole = Literal["student", "teacher", "parent"]
LearningStyle = Literal["explanation", "question", "problem_solving"]
LearningPace = Literal["fast", "normal", "slow"]


class SignupRequest(BaseModel):
    email: str
    password: str
    display_name: Optional[str] = None
    role: UserRole = "student"
    learning_style: LearningStyle = "explanation"
    learning_pace: LearningPace = "normal"
    target_goal: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthUser(BaseModel):
    id: str
    email: str
    display_name: str
    role: UserRole
    learning_style: Optional[LearningStyle] = None
    learning_pace: Optional[LearningPace] = None
    target_goal: Optional[str] = None


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: AuthUser


EventType = Literal[
    "stroke_add",
    "stroke_erase",
    "idle_ping",
    "hint_request",
    "answer_submit",
]


class ClientEvent(BaseModel):
    type: EventType
    ts: Optional[datetime] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class EventBatchRequest(BaseModel):
    events: List[ClientEvent]


class StuckSignals(BaseModel):
    idle_ms: int
    erase_count_delta: int
    repeated_error_count: int
    stuck_score: int


class Intervention(BaseModel):
    level: Literal[1, 2, 3]
    reason: str
    tutor_message: str
    citations: Optional[List[Dict[str, str]]] = None
    created_at: datetime


class EventBatchResponse(BaseModel):
    accepted: int
    stuck_signals: StuckSignals
    intervention: Optional[Intervention] = None
    solved: bool


class AttemptDetailResponse(BaseModel):
    attempt_id: str
    started_at: datetime
    solved_at: Optional[datetime]
    problem: Problem


class TimelineEntry(BaseModel):
    at: datetime
    type: str
    label: str


class SummaryMetrics(BaseModel):
    time_to_solve_sec: Optional[int]
    max_stuck: int
    hint_max_level: int
    erase_count: int


class AttemptSummaryResponse(BaseModel):
    attempt_id: str
    metrics: SummaryMetrics
    timeline: List[TimelineEntry]


@dataclass
class AttemptState:
    id: str
    started_at: datetime
    problem: Problem
    guest_id: Optional[str] = None
    solved_at: Optional[datetime] = None
    events: List[ClientEvent] = field(default_factory=list)
    interventions: List[Intervention] = field(default_factory=list)
    stuck_scores: List[int] = field(default_factory=list)
    last_intervention_at: Optional[datetime] = None


DEFAULT_PROBLEMS: List[Problem] = [
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
ATTEMPTS: Dict[str, AttemptState] = {}


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


def resolve_problem(payload: AttemptCreateRequest) -> Problem:
    if payload.problem_id and payload.problem_id in PROBLEM_BY_ID:
        return PROBLEM_BY_ID[payload.problem_id]
    if payload.problem_text and payload.answer_key:
        return Problem(
            id=f"custom-{uuid4().hex[:8]}",
            title="Custom Problem",
            prompt=payload.problem_text,
            answer_key=payload.answer_key,
            unit=payload.unit or "Custom",
        )
    return DEFAULT_PROBLEMS[0]


DB_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS attempts (
  id TEXT PRIMARY KEY,
  guest_id TEXT,
  started_at TIMESTAMPTZ NOT NULL,
  solved_at TIMESTAMPTZ,
  problem JSONB NOT NULL,
  last_intervention_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS attempt_events (
  id BIGSERIAL PRIMARY KEY,
  attempt_id TEXT NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
  type TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  payload JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_attempt_events_attempt_id ON attempt_events(attempt_id);

CREATE TABLE IF NOT EXISTS attempt_interventions (
  id BIGSERIAL PRIMARY KEY,
  attempt_id TEXT NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
  level INT NOT NULL,
  reason TEXT NOT NULL,
  tutor_message TEXT NOT NULL,
  citations JSONB,
  created_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_attempt_interventions_attempt_id
  ON attempt_interventions(attempt_id);

CREATE TABLE IF NOT EXISTS attempt_stuck_scores (
  id BIGSERIAL PRIMARY KEY,
  attempt_id TEXT NOT NULL REFERENCES attempts(id) ON DELETE CASCADE,
  score INT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_attempt_stuck_scores_attempt_id
  ON attempt_stuck_scores(attempt_id);
"""

AUTH_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role') THEN
    CREATE TYPE user_role AS ENUM ('student', 'teacher', 'parent');
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'learning_style') THEN
    CREATE TYPE learning_style AS ENUM ('explanation', 'question', 'problem_solving');
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'learning_pace') THEN
    CREATE TYPE learning_pace AS ENUM ('fast', 'normal', 'slow');
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT NOT NULL UNIQUE,
  password_hash TEXT NOT NULL,
  role user_role NOT NULL,
  display_name TEXT NOT NULL DEFAULT '',
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_users_email_lower ON users ((lower(email)));

CREATE TABLE IF NOT EXISTS student_profiles (
  user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  learning_style learning_style NOT NULL DEFAULT 'explanation',
  learning_pace learning_pace NOT NULL DEFAULT 'normal',
  target_goal TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def db_connect():
    if not DB_ENABLED:
        raise RuntimeError("DATABASE_URL is not configured")
    if psycopg is None:
        raise RuntimeError("psycopg is required when DATABASE_URL is configured")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def init_db_schema() -> None:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(DB_SCHEMA_SQL)
        conn.commit()


def init_auth_schema() -> None:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(AUTH_SCHEMA_SQL)
        conn.commit()


def require_db_enabled() -> None:
    if not DB_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Auth requires DATABASE_URL. Configure shared PostgreSQL first.",
        )


def normalize_email(email: str) -> str:
    return email.strip().lower()


def auth_user_from_row(row: dict[str, Any]) -> AuthUser:
    return AuthUser(
        id=str(row["id"]),
        email=row["email"],
        display_name=row["display_name"],
        role=row["role"],
        learning_style=row.get("learning_style"),
        learning_pace=row.get("learning_pace"),
        target_goal=row.get("target_goal"),
    )


def get_user_by_id_db(user_id: str) -> Optional[AuthUser]:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  u.id::text AS id,
                  u.email,
                  u.display_name,
                  u.role::text AS role,
                  sp.learning_style::text AS learning_style,
                  sp.learning_pace::text AS learning_pace,
                  sp.target_goal
                FROM users u
                LEFT JOIN student_profiles sp ON sp.user_id = u.id
                WHERE u.id = %s::uuid AND u.is_active = TRUE
                """,
                (user_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return auth_user_from_row(row)


def signup_db(payload: SignupRequest) -> AuthResponse:
    email = normalize_email(payload.email)
    display_name = (payload.display_name or email.split("@")[0]).strip() or "Student"
    password_hash = hash_password(payload.password)
    user_id: Optional[str] = None

    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (email, password_hash, role, display_name)
                    VALUES (%s, %s, %s::user_role, %s)
                    RETURNING id::text AS id, email, display_name, role::text AS role
                    """,
                    (email, password_hash, payload.role, display_name),
                )
                user_row = cur.fetchone()
                user_id = user_row["id"]

                if payload.role == "student":
                    cur.execute(
                        """
                        INSERT INTO student_profiles (
                          user_id, learning_style, learning_pace, target_goal
                        ) VALUES (%s::uuid, %s::learning_style, %s::learning_pace, %s)
                        ON CONFLICT (user_id) DO UPDATE SET
                          learning_style = EXCLUDED.learning_style,
                          learning_pace = EXCLUDED.learning_pace,
                          target_goal = EXCLUDED.target_goal,
                          updated_at = now()
                        """,
                        (
                            user_id,
                            payload.learning_style,
                            payload.learning_pace,
                            payload.target_goal,
                        ),
                    )
            conn.commit()
    except Exception as exc:
        if psycopg_errors and isinstance(exc, psycopg_errors.UniqueViolation):
            raise HTTPException(status_code=409, detail="Email already exists") from exc
        raise

    if not user_id:
        raise HTTPException(status_code=500, detail="Failed to create user")

    user = get_user_by_id_db(user_id)
    if not user:
        raise HTTPException(status_code=500, detail="Failed to create user profile")

    token = create_access_token(user.id, user.email, user.role)
    return AuthResponse(access_token=token, user=user)


def login_db(payload: LoginRequest) -> AuthResponse:
    email = normalize_email(payload.email)

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  u.id::text AS id,
                  u.email,
                  u.display_name,
                  u.role::text AS role,
                  u.password_hash,
                  sp.learning_style::text AS learning_style,
                  sp.learning_pace::text AS learning_pace,
                  sp.target_goal
                FROM users u
                LEFT JOIN student_profiles sp ON sp.user_id = u.id
                WHERE lower(u.email) = %s AND u.is_active = TRUE
                LIMIT 1
                """,
                (email,),
            )
            row = cur.fetchone()

    if not row or not verify_password(payload.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user = auth_user_from_row(row)
    token = create_access_token(user.id, user.email, user.role)
    return AuthResponse(access_token=token, user=user)


def load_attempt_state_db(attempt_id: str) -> AttemptState | None:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, guest_id, started_at, solved_at, problem, last_intervention_at
                FROM attempts
                WHERE id = %s
                """,
                (attempt_id,),
            )
            attempt_row = cur.fetchone()
            if not attempt_row:
                return None

            problem_raw = parse_json_dict(attempt_row["problem"])
            problem = Problem(**problem_raw)
            attempt = AttemptState(
                id=attempt_row["id"],
                guest_id=attempt_row["guest_id"],
                started_at=ensure_utc(attempt_row["started_at"]),
                solved_at=ensure_utc(attempt_row["solved_at"]) if attempt_row["solved_at"] else None,
                problem=problem,
                last_intervention_at=(
                    ensure_utc(attempt_row["last_intervention_at"])
                    if attempt_row["last_intervention_at"]
                    else None
                ),
            )

            cur.execute(
                """
                SELECT type, ts, payload
                FROM attempt_events
                WHERE attempt_id = %s
                ORDER BY id ASC
                """,
                (attempt_id,),
            )
            event_rows = cur.fetchall()
            attempt.events = [
                ClientEvent(
                    type=row["type"],
                    ts=ensure_utc(row["ts"]),
                    payload=parse_json_dict(row["payload"]),
                )
                for row in event_rows
            ]

            cur.execute(
                """
                SELECT level, reason, tutor_message, citations, created_at
                FROM attempt_interventions
                WHERE attempt_id = %s
                ORDER BY id ASC
                """,
                (attempt_id,),
            )
            intervention_rows = cur.fetchall()
            attempt.interventions = [
                Intervention(
                    level=int(row["level"]),
                    reason=row["reason"],
                    tutor_message=row["tutor_message"],
                    citations=parse_json_list(row["citations"]),
                    created_at=ensure_utc(row["created_at"]),
                )
                for row in intervention_rows
            ]

            cur.execute(
                """
                SELECT score
                FROM attempt_stuck_scores
                WHERE attempt_id = %s
                ORDER BY id ASC
                """,
                (attempt_id,),
            )
            stuck_rows = cur.fetchall()
            attempt.stuck_scores = [int(row["score"]) for row in stuck_rows]
            return attempt


def create_attempt_db(payload: AttemptCreateRequest) -> AttemptCreateResponse:
    problem = resolve_problem(payload)
    attempt_id = uuid4().hex
    started_at = utcnow()

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO attempts (id, guest_id, started_at, problem)
                VALUES (%s, %s, %s, %s)
                """,
                (attempt_id, payload.guest_id, started_at, Json(problem.model_dump())),
            )
        conn.commit()

    return AttemptCreateResponse(
        attempt_id=attempt_id,
        started_at=started_at,
        problem=problem,
    )


def ingest_events_db(attempt_id: str, payload: EventBatchRequest) -> EventBatchResponse:
    attempt = load_attempt_state_db(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")

    new_events: list[ClientEvent] = []
    for event in payload.events:
        event_ts = ensure_utc(event.ts or utcnow())
        event_payload = dict(event.payload)

        if event.type == "answer_submit":
            answer = str(event_payload.get("answer", ""))
            correct = check_answer(attempt.problem, answer)
            event_payload["correct"] = correct
            if correct and attempt.solved_at is None:
                attempt.solved_at = utcnow()

        new_event = ClientEvent(type=event.type, ts=event_ts, payload=event_payload)
        attempt.events.append(new_event)
        new_events.append(new_event)

    signals = compute_signals(attempt.events)
    attempt.stuck_scores.append(signals.stuck_score)

    forced_hint = any(evt.type == "hint_request" for evt in new_events)
    hint_level = (
        max(1, pick_level(signals.stuck_score))
        if forced_hint
        else pick_level(signals.stuck_score)
    )

    intervention = None
    if should_emit_intervention(attempt, hint_level, force=forced_hint):
        intervention = build_hint(attempt.problem, signals, hint_level)
        attempt.interventions.append(intervention)
        attempt.last_intervention_at = intervention.created_at

    with db_connect() as conn:
        with conn.cursor() as cur:
            if new_events:
                cur.executemany(
                    """
                    INSERT INTO attempt_events (attempt_id, type, ts, payload)
                    VALUES (%s, %s, %s, %s)
                    """,
                    [
                        (attempt_id, evt.type, ensure_utc(evt.ts or utcnow()), Json(evt.payload))
                        for evt in new_events
                    ],
                )

            cur.execute(
                """
                INSERT INTO attempt_stuck_scores (attempt_id, score)
                VALUES (%s, %s)
                """,
                (attempt_id, signals.stuck_score),
            )

            if attempt.solved_at is not None:
                cur.execute(
                    """
                    UPDATE attempts
                    SET solved_at = COALESCE(solved_at, %s)
                    WHERE id = %s
                    """,
                    (attempt.solved_at, attempt_id),
                )

            if intervention:
                cur.execute(
                    """
                    INSERT INTO attempt_interventions (
                      attempt_id, level, reason, tutor_message, citations, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        attempt_id,
                        intervention.level,
                        intervention.reason,
                        intervention.tutor_message,
                        Json(intervention.citations),
                        intervention.created_at,
                    ),
                )
                cur.execute(
                    """
                    UPDATE attempts
                    SET last_intervention_at = %s
                    WHERE id = %s
                    """,
                    (intervention.created_at, attempt_id),
                )

        conn.commit()

    return EventBatchResponse(
        accepted=len(new_events),
        stuck_signals=signals,
        intervention=intervention,
        solved=attempt.solved_at is not None,
    )


def get_latest_intervention_db(attempt_id: str) -> Intervention | None:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT level, reason, tutor_message, citations, created_at
                FROM attempt_interventions
                WHERE attempt_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (attempt_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return Intervention(
                level=int(row["level"]),
                reason=row["reason"],
                tutor_message=row["tutor_message"],
                citations=parse_json_list(row["citations"]),
                created_at=ensure_utc(row["created_at"]),
            )


def summary_from_attempt(attempt: AttemptState) -> AttemptSummaryResponse:
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


app = FastAPI(title="AI Coach MVP API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    if DB_ENABLED:
        if psycopg is None:
            raise RuntimeError(
                "DATABASE_URL is configured but psycopg is not installed. "
                "Install backend requirements first."
            )
        try:
            init_db_schema()
            init_auth_schema()
        except Exception as exc:
            raise RuntimeError(
                "Failed to connect to PostgreSQL. Check DATABASE_URL, RDS public access, "
                "security group inbound 5432, and SSL settings."
            ) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "storage": "postgres" if DB_ENABLED else "memory"}


@app.post("/api/auth/signup", response_model=AuthResponse)
def signup(payload: SignupRequest) -> AuthResponse:
    require_db_enabled()
    return signup_db(payload)


@app.post("/api/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest) -> AuthResponse:
    require_db_enabled()
    return login_db(payload)


@app.get("/api/auth/me", response_model=AuthUser)
def get_me(authorization: Optional[str] = Header(None, alias="Authorization")) -> AuthUser:
    require_db_enabled()
    token = parse_bearer_token(authorization)
    payload = decode_access_token(token)
    user = get_user_by_id_db(str(payload["sub"]))
    if not user:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


@app.get("/api/problems", response_model=list[Problem])
def list_problems() -> list[Problem]:
    return DEFAULT_PROBLEMS


@app.post("/api/attempts", response_model=AttemptCreateResponse)
def create_attempt(payload: AttemptCreateRequest) -> AttemptCreateResponse:
    if DB_ENABLED:
        return create_attempt_db(payload)

    problem = resolve_problem(payload)
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
    if DB_ENABLED:
        attempt = load_attempt_state_db(attempt_id)
        if not attempt:
            raise HTTPException(status_code=404, detail="Attempt not found")
        return AttemptDetailResponse(
            attempt_id=attempt.id,
            started_at=attempt.started_at,
            solved_at=attempt.solved_at,
            problem=attempt.problem,
        )

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
    if DB_ENABLED:
        return ingest_events_db(attempt_id, payload)

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
    hint_level = (
        max(1, pick_level(signals.stuck_score))
        if forced_hint
        else pick_level(signals.stuck_score)
    )

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


@app.get("/api/attempts/{attempt_id}/intervention", response_model=Optional[Intervention])
def get_latest_intervention(attempt_id: str) -> Intervention | None:
    if DB_ENABLED:
        attempt = load_attempt_state_db(attempt_id)
        if not attempt:
            raise HTTPException(status_code=404, detail="Attempt not found")
        return get_latest_intervention_db(attempt_id)

    attempt = ATTEMPTS.get(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")
    if not attempt.interventions:
        return None
    return attempt.interventions[-1]


@app.get("/api/attempts/{attempt_id}/summary", response_model=AttemptSummaryResponse)
def get_summary(attempt_id: str) -> AttemptSummaryResponse:
    if DB_ENABLED:
        attempt = load_attempt_state_db(attempt_id)
        if not attempt:
            raise HTTPException(status_code=404, detail="Attempt not found")
        return summary_from_attempt(attempt)

    attempt = ATTEMPTS.get(attempt_id)
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")
    return summary_from_attempt(attempt)
