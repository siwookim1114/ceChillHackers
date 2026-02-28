from __future__ import annotations

import json
import logging
import os
import base64
import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from urllib import error as urllib_error, request as urllib_request
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
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

try:
    import multipart  # type: ignore # noqa: F401
    MULTIPART_AVAILABLE = True
except ImportError:
    MULTIPART_AVAILABLE = False


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
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173").strip()
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
GOOGLE_REDIRECT_URI = os.getenv(
    "GOOGLE_REDIRECT_URI", "http://localhost:8000/api/auth/google/callback"
).strip()
OAUTH_STATE_TTL_SEC = int(os.getenv("OAUTH_STATE_TTL_SEC", "600"))
GOOGLE_OAUTH_ENABLED = bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET and GOOGLE_REDIRECT_URI)
DB_READY = False
_DB_INIT_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


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


def frontend_login_url() -> str:
    return f"{FRONTEND_URL.rstrip('/')}/login"


def normalize_return_to(return_to: Optional[str]) -> str:
    default_url = frontend_login_url()
    if not return_to:
        return default_url

    cleaned = return_to.strip()
    if not cleaned:
        return default_url
    if cleaned.startswith("/"):
        return f"{FRONTEND_URL.rstrip('/')}{cleaned}"

    front_parts = urlsplit(FRONTEND_URL)
    target_parts = urlsplit(cleaned)
    if (
        target_parts.scheme in {"http", "https"}
        and target_parts.netloc
        and target_parts.netloc == front_parts.netloc
    ):
        return cleaned
    return default_url


def append_fragment_params(url: str, params: dict[str, str]) -> str:
    parts = urlsplit(url)
    fragment_map = dict(parse_qsl(parts.fragment))
    fragment_map.update({k: v for k, v in params.items() if v})
    fragment = urlencode(fragment_map)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, fragment))


def create_oauth_state(return_to: str) -> str:
    payload = {
        "return_to": return_to,
        "exp": int(time.time()) + OAUTH_STATE_TTL_SEC,
        "nonce": secrets.token_urlsafe(12),
    }
    payload_b64 = b64url_encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signature = hmac.new(
        APP_SECRET_KEY.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256
    ).digest()
    return f"{payload_b64}.{b64url_encode(signature)}"


def decode_oauth_state(state: str) -> dict[str, Any]:
    try:
        payload_b64, signature_b64 = state.split(".", 1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid OAuth state") from exc

    expected_signature = hmac.new(
        APP_SECRET_KEY.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256
    ).digest()
    provided_signature = b64url_decode(signature_b64)
    if not hmac.compare_digest(expected_signature, provided_signature):
        raise HTTPException(status_code=400, detail="Invalid OAuth state signature")

    payload = json.loads(b64url_decode(payload_b64))
    if int(payload.get("exp", 0)) < int(time.time()):
        raise HTTPException(status_code=400, detail="OAuth state expired")
    return payload


def require_google_oauth_enabled() -> None:
    if not GOOGLE_OAUTH_ENABLED:
        raise HTTPException(
            status_code=503,
            detail=(
                "Google login is not configured. Set GOOGLE_CLIENT_ID, "
                "GOOGLE_CLIENT_SECRET, GOOGLE_REDIRECT_URI."
            ),
        )


def build_google_authorize_url(state: str) -> str:
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "prompt": "select_account",
        "access_type": "offline",
        "state": state,
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"


def exchange_google_code(code: str) -> dict[str, Any]:
    payload = urlencode(
        {
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        }
    ).encode("utf-8")
    request = urllib_request.Request(
        "https://oauth2.googleapis.com/token",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(status_code=400, detail=f"Google token exchange failed: {body}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Google token exchange failed") from exc


def fetch_google_userinfo(access_token: str) -> dict[str, Any]:
    request = urllib_request.Request(
        "https://openidconnect.googleapis.com/v1/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        method="GET",
    )
    try:
        with urllib_request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(status_code=400, detail=f"Google userinfo failed: {body}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Google userinfo request failed") from exc


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


DailyProgressEventType = Literal[
    "session_solved",
    "course_created",
    "coached_session",
    "set_current_topic",
]


class DailyProgress(BaseModel):
    date: str
    solved_sessions: int
    created_courses: int
    coached_sessions: int
    daily_target_sessions: int
    current_course_topic: Optional[str] = None


class DailyProgressEventRequest(BaseModel):
    event_type: DailyProgressEventType
    attempt_id: Optional[str] = None
    topic: Optional[str] = None


class CourseCreateRequest(BaseModel):
    title: str
    syllabus: Optional[str] = None


class LectureCreateRequest(BaseModel):
    title: str
    description: Optional[str] = None
    problem_prompt: str
    answer_key: str
    sort_order: Optional[int] = None


class LectureFileInfo(BaseModel):
    id: str
    file_name: str
    content_type: Optional[str] = None
    size_bytes: int
    created_at: datetime


class LectureItem(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    problem_prompt: str
    answer_key: str
    sort_order: int
    file_count: int
    created_at: datetime
    files: List[LectureFileInfo] = Field(default_factory=list)


class CourseFolder(BaseModel):
    id: str
    title: str
    syllabus: Optional[str] = None
    lecture_count: int
    file_count: int
    created_at: datetime


class CourseDetailResponse(BaseModel):
    id: str
    title: str
    syllabus: Optional[str] = None
    created_at: datetime
    lectures: List[LectureItem] = Field(default_factory=list)


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

CREATE TABLE IF NOT EXISTS user_daily_progress (
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  progress_date DATE NOT NULL,
  solved_sessions INT NOT NULL DEFAULT 0,
  created_courses INT NOT NULL DEFAULT 0,
  coached_sessions INT NOT NULL DEFAULT 0,
  daily_target_sessions INT NOT NULL DEFAULT 2,
  current_course_topic TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (user_id, progress_date)
);

CREATE TABLE IF NOT EXISTS user_daily_progress_events (
  id BIGSERIAL PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  progress_date DATE NOT NULL,
  event_type TEXT NOT NULL,
  attempt_id TEXT NOT NULL DEFAULT '',
  topic TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_user_daily_progress_events_dedup
  ON user_daily_progress_events(user_id, progress_date, event_type, attempt_id);

CREATE TABLE IF NOT EXISTS courses (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  syllabus TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_courses_user_created
  ON courses(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS lectures (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  course_id UUID NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  description TEXT,
  problem_prompt TEXT NOT NULL,
  answer_key TEXT NOT NULL,
  sort_order INT NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_lectures_course_sort
  ON lectures(course_id, sort_order, created_at);

CREATE TABLE IF NOT EXISTS lecture_files (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  lecture_id UUID NOT NULL REFERENCES lectures(id) ON DELETE CASCADE,
  file_name TEXT NOT NULL,
  content_type TEXT,
  size_bytes INT NOT NULL,
  file_data BYTEA NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_lecture_files_lecture_created
  ON lecture_files(lecture_id, created_at DESC);
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
    global DB_READY
    if not DB_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Auth requires DATABASE_URL. Configure shared PostgreSQL first.",
        )
    if DB_READY:
        return
    with _DB_INIT_LOCK:
        # Double-checked locking: another thread may have initialized while waiting.
        if DB_READY:
            return
        try:
            init_db_schema()
            init_auth_schema()
            DB_READY = True
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Database is temporarily unavailable. Check DATABASE_URL, RDS public access, "
                    "security group inbound 5432, and SSL settings."
                ),
            ) from exc


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


def get_user_by_email_db(email: str) -> Optional[AuthUser]:
    normalized = normalize_email(email)
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
                WHERE lower(u.email) = %s AND u.is_active = TRUE
                LIMIT 1
                """,
                (normalized,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return auth_user_from_row(row)


def find_or_create_google_user_db(email: str, display_name: str) -> AuthUser:
    normalized = normalize_email(email)
    existing = get_user_by_email_db(normalized)
    if existing:
        return existing

    user_id: Optional[str] = None
    random_password_hash = hash_password(secrets.token_urlsafe(32))
    safe_name = display_name.strip() or normalized.split("@")[0] or "Student"

    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (email, password_hash, role, display_name)
                    VALUES (%s, %s, 'student'::user_role, %s)
                    RETURNING id::text AS id
                    """,
                    (normalized, random_password_hash, safe_name),
                )
                inserted = cur.fetchone()
                user_id = inserted["id"] if inserted else None

                if user_id:
                    cur.execute(
                        """
                        INSERT INTO student_profiles (user_id, learning_style, learning_pace)
                        VALUES (%s::uuid, 'question'::learning_style, 'normal'::learning_pace)
                        ON CONFLICT (user_id) DO NOTHING
                        """,
                        (user_id,),
                    )
            conn.commit()
    except Exception as exc:
        if psycopg_errors and isinstance(exc, psycopg_errors.UniqueViolation):
            winner = get_user_by_email_db(normalized)
            if winner:
                return winner
        raise

    if user_id:
        created = get_user_by_id_db(user_id)
        if created:
            return created
    raise HTTPException(status_code=500, detail="Failed to create Google user")


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


def today_utc_date() -> str:
    return utcnow().date().isoformat()


def daily_progress_from_row(row: dict[str, Any]) -> DailyProgress:
    progress_date = row["progress_date"]
    date_text = (
        progress_date.isoformat()
        if hasattr(progress_date, "isoformat")
        else str(progress_date)
    )
    return DailyProgress(
        date=date_text,
        solved_sessions=int(row["solved_sessions"] or 0),
        created_courses=int(row["created_courses"] or 0),
        coached_sessions=int(row["coached_sessions"] or 0),
        daily_target_sessions=int(row["daily_target_sessions"] or 2),
        current_course_topic=row.get("current_course_topic"),
    )


def get_daily_progress_db(user_id: str) -> DailyProgress:
    progress_date = today_utc_date()
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_daily_progress (user_id, progress_date)
                VALUES (%s::uuid, %s::date)
                ON CONFLICT (user_id, progress_date) DO NOTHING
                """,
                (user_id, progress_date),
            )
            cur.execute(
                """
                SELECT
                  progress_date,
                  solved_sessions,
                  created_courses,
                  coached_sessions,
                  daily_target_sessions,
                  current_course_topic
                FROM user_daily_progress
                WHERE user_id = %s::uuid AND progress_date = %s::date
                """,
                (user_id, progress_date),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=500, detail="Failed to load daily progress")
    return daily_progress_from_row(row)


def record_daily_progress_event_db(
    user_id: str, payload: DailyProgressEventRequest
) -> DailyProgress:
    progress_date = today_utc_date()
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_daily_progress (user_id, progress_date)
                VALUES (%s::uuid, %s::date)
                ON CONFLICT (user_id, progress_date) DO NOTHING
                """,
                (user_id, progress_date),
            )

            cleaned_topic = (payload.topic or "").strip() or None
            if payload.event_type == "set_current_topic":
                if cleaned_topic:
                    cur.execute(
                        """
                        UPDATE user_daily_progress
                        SET current_course_topic = %s, updated_at = now()
                        WHERE user_id = %s::uuid AND progress_date = %s::date
                        """,
                        (cleaned_topic, user_id, progress_date),
                    )
            else:
                attempt_id = (payload.attempt_id or "").strip()
                if not attempt_id:
                    raise HTTPException(
                        status_code=400,
                        detail="attempt_id is required for this event type",
                    )

                cur.execute(
                    """
                    INSERT INTO user_daily_progress_events (
                      user_id, progress_date, event_type, attempt_id, topic
                    ) VALUES (%s::uuid, %s::date, %s, %s, %s)
                    ON CONFLICT (user_id, progress_date, event_type, attempt_id)
                    DO NOTHING
                    RETURNING id
                    """,
                    (user_id, progress_date, payload.event_type, attempt_id, cleaned_topic),
                )
                inserted = cur.fetchone()
                if inserted:
                    if payload.event_type == "session_solved":
                        cur.execute(
                            """
                            UPDATE user_daily_progress
                            SET solved_sessions = solved_sessions + 1, updated_at = now()
                            WHERE user_id = %s::uuid AND progress_date = %s::date
                            """,
                            (user_id, progress_date),
                        )
                    elif payload.event_type == "course_created":
                        cur.execute(
                            """
                            UPDATE user_daily_progress
                            SET
                              created_courses = created_courses + 1,
                              current_course_topic = COALESCE(%s, current_course_topic),
                              updated_at = now()
                            WHERE user_id = %s::uuid AND progress_date = %s::date
                            """,
                            (cleaned_topic, user_id, progress_date),
                        )
                    elif payload.event_type == "coached_session":
                        cur.execute(
                            """
                            UPDATE user_daily_progress
                            SET coached_sessions = coached_sessions + 1, updated_at = now()
                            WHERE user_id = %s::uuid AND progress_date = %s::date
                            """,
                            (user_id, progress_date),
                        )

            cur.execute(
                """
                SELECT
                  progress_date,
                  solved_sessions,
                  created_courses,
                  coached_sessions,
                  daily_target_sessions,
                  current_course_topic
                FROM user_daily_progress
                WHERE user_id = %s::uuid AND progress_date = %s::date
                """,
                (user_id, progress_date),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=500, detail="Failed to update daily progress")
    return daily_progress_from_row(row)


def course_folder_from_row(row: dict[str, Any]) -> CourseFolder:
    return CourseFolder(
        id=str(row["id"]),
        title=row["title"],
        syllabus=row.get("syllabus"),
        lecture_count=int(row.get("lecture_count") or 0),
        file_count=int(row.get("file_count") or 0),
        created_at=ensure_utc(row["created_at"]),
    )


def lecture_file_from_row(row: dict[str, Any]) -> LectureFileInfo:
    return LectureFileInfo(
        id=str(row["id"]),
        file_name=row["file_name"],
        content_type=row.get("content_type"),
        size_bytes=int(row["size_bytes"] or 0),
        created_at=ensure_utc(row["created_at"]),
    )


def lecture_item_from_row(
    row: dict[str, Any], files: Optional[List[LectureFileInfo]] = None
) -> LectureItem:
    return LectureItem(
        id=str(row["id"]),
        title=row["title"],
        description=row.get("description"),
        problem_prompt=row["problem_prompt"],
        answer_key=row["answer_key"],
        sort_order=int(row.get("sort_order") or 0),
        file_count=int(row.get("file_count") or 0),
        created_at=ensure_utc(row["created_at"]),
        files=files or [],
    )


def get_course_row_for_user_db(user_id: str, course_id: str) -> dict[str, Any]:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, syllabus, created_at
                FROM courses
                WHERE id = %s::uuid AND user_id = %s::uuid
                """,
                (course_id, user_id),
            )
            row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Course not found")
    return row


def list_courses_db(user_id: str) -> List[CourseFolder]:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  c.id,
                  c.title,
                  c.syllabus,
                  c.created_at,
                  (
                    SELECT COUNT(*)
                    FROM lectures l
                    WHERE l.course_id = c.id
                  ) AS lecture_count,
                  (
                    SELECT COUNT(*)
                    FROM lecture_files lf
                    JOIN lectures l2 ON l2.id = lf.lecture_id
                    WHERE l2.course_id = c.id
                  ) AS file_count
                FROM courses c
                WHERE c.user_id = %s::uuid
                ORDER BY c.created_at DESC
                """,
                (user_id,),
            )
            rows = cur.fetchall()
    return [course_folder_from_row(row) for row in rows]


def create_course_db(user_id: str, payload: CourseCreateRequest) -> CourseFolder:
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Course title is required")
    syllabus = (payload.syllabus or "").strip() or None

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO courses (user_id, title, syllabus)
                VALUES (%s::uuid, %s, %s)
                RETURNING id, title, syllabus, created_at
                """,
                (user_id, title, syllabus),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=500, detail="Failed to create course")
    return course_folder_from_row(
        {
            "id": row["id"],
            "title": row["title"],
            "syllabus": row.get("syllabus"),
            "created_at": row["created_at"],
            "lecture_count": 0,
            "file_count": 0,
        }
    )


def list_lecture_files_db(user_id: str, course_id: str, lecture_id: str) -> List[LectureFileInfo]:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT l.id
                FROM lectures l
                JOIN courses c ON c.id = l.course_id
                WHERE c.user_id = %s::uuid
                  AND c.id = %s::uuid
                  AND l.id = %s::uuid
                """,
                (user_id, course_id, lecture_id),
            )
            lecture_row = cur.fetchone()
            if not lecture_row:
                raise HTTPException(status_code=404, detail="Lecture not found")

            cur.execute(
                """
                SELECT id, file_name, content_type, size_bytes, created_at
                FROM lecture_files
                WHERE lecture_id = %s::uuid
                ORDER BY created_at DESC
                """,
                (lecture_id,),
            )
            file_rows = cur.fetchall()

    return [lecture_file_from_row(row) for row in file_rows]


def list_course_lectures_db(
    user_id: str, course_id: str, include_files: bool = False
) -> List[LectureItem]:
    get_course_row_for_user_db(user_id, course_id)

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  l.id,
                  l.title,
                  l.description,
                  l.problem_prompt,
                  l.answer_key,
                  l.sort_order,
                  l.created_at,
                  (
                    SELECT COUNT(*)
                    FROM lecture_files lf
                    WHERE lf.lecture_id = l.id
                  ) AS file_count
                FROM lectures l
                WHERE l.course_id = %s::uuid
                ORDER BY l.sort_order ASC, l.created_at ASC
                """,
                (course_id,),
            )
            lecture_rows = cur.fetchall()

            lecture_items: List[LectureItem] = []
            for row in lecture_rows:
                files: List[LectureFileInfo] = []
                if include_files:
                    cur.execute(
                        """
                        SELECT id, file_name, content_type, size_bytes, created_at
                        FROM lecture_files
                        WHERE lecture_id = %s::uuid
                        ORDER BY created_at DESC
                        """,
                        (row["id"],),
                    )
                    file_rows = cur.fetchall()
                    files = [lecture_file_from_row(file_row) for file_row in file_rows]
                lecture_items.append(lecture_item_from_row(row, files=files))

    return lecture_items


def get_course_detail_db(user_id: str, course_id: str) -> CourseDetailResponse:
    course_row = get_course_row_for_user_db(user_id, course_id)
    lectures = list_course_lectures_db(user_id, course_id, include_files=True)
    return CourseDetailResponse(
        id=str(course_row["id"]),
        title=course_row["title"],
        syllabus=course_row.get("syllabus"),
        created_at=ensure_utc(course_row["created_at"]),
        lectures=lectures,
    )


def create_lecture_db(user_id: str, course_id: str, payload: LectureCreateRequest) -> LectureItem:
    get_course_row_for_user_db(user_id, course_id)

    title = payload.title.strip()
    problem_prompt = payload.problem_prompt.strip()
    answer_key = payload.answer_key.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Lecture title is required")
    if not problem_prompt:
        raise HTTPException(status_code=400, detail="Problem prompt is required")
    if not answer_key:
        raise HTTPException(status_code=400, detail="Answer key is required")

    description = (payload.description or "").strip() or None

    with db_connect() as conn:
        with conn.cursor() as cur:
            sort_order = payload.sort_order
            if sort_order is None:
                cur.execute(
                    """
                    SELECT COALESCE(MAX(sort_order), -1) + 1 AS next_sort
                    FROM lectures
                    WHERE course_id = %s::uuid
                    """,
                    (course_id,),
                )
                next_sort_row = cur.fetchone()
                sort_order = int(next_sort_row["next_sort"] or 0) if next_sort_row else 0

            cur.execute(
                """
                INSERT INTO lectures (
                  course_id, title, description, problem_prompt, answer_key, sort_order
                ) VALUES (%s::uuid, %s, %s, %s, %s, %s)
                RETURNING
                  id,
                  title,
                  description,
                  problem_prompt,
                  answer_key,
                  sort_order,
                  created_at
                """,
                (
                    course_id,
                    title,
                    description,
                    problem_prompt,
                    answer_key,
                    sort_order,
                ),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=500, detail="Failed to create lecture")
    row_with_count = dict(row)
    row_with_count["file_count"] = 0
    return lecture_item_from_row(row_with_count)


def upload_lecture_file_db(
    user_id: str,
    course_id: str,
    lecture_id: str,
    file_name: str,
    content_type: Optional[str],
    file_data: bytes,
) -> LectureFileInfo:
    if not file_name.strip():
        raise HTTPException(status_code=400, detail="File name is required")
    if not file_data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(file_data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File is too large (max 20MB)")

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT l.id
                FROM lectures l
                JOIN courses c ON c.id = l.course_id
                WHERE c.user_id = %s::uuid
                  AND c.id = %s::uuid
                  AND l.id = %s::uuid
                """,
                (user_id, course_id, lecture_id),
            )
            lecture_row = cur.fetchone()
            if not lecture_row:
                raise HTTPException(status_code=404, detail="Lecture not found")

            cur.execute(
                """
                INSERT INTO lecture_files (lecture_id, file_name, content_type, size_bytes, file_data)
                VALUES (%s::uuid, %s, %s, %s, %s)
                RETURNING id, file_name, content_type, size_bytes, created_at
                """,
                (lecture_id, file_name.strip(), content_type, len(file_data), file_data),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=500, detail="Failed to upload lecture file")
    return lecture_file_from_row(row)


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
    global DB_READY
    if DB_ENABLED:
        if psycopg is None:
            raise RuntimeError(
                "DATABASE_URL is configured but psycopg is not installed. "
                "Install backend requirements first."
            )
        try:
            init_db_schema()
            init_auth_schema()
            DB_READY = True
        except Exception as exc:
            DB_READY = False
            logger.warning(
                "Startup DB initialization failed; server will stay up and retry on request. "
                "Check DATABASE_URL/RDS network. cause=%s",
                exc,
            )


@app.get("/health")
def health() -> dict[str, str]:
    storage = "postgres" if DB_ENABLED else "memory"
    db_status = "ready" if DB_READY else ("disabled" if not DB_ENABLED else "initializing")
    return {"status": "ok", "storage": storage, "db": db_status}


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


def get_current_auth_user(authorization: Optional[str]) -> AuthUser:
    token = parse_bearer_token(authorization)
    payload = decode_access_token(token)
    user = get_user_by_id_db(str(payload["sub"]))
    if not user:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


@app.get("/api/progress/daily", response_model=DailyProgress)
def get_daily_progress(
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> DailyProgress:
    require_db_enabled()
    user = get_current_auth_user(authorization)
    return get_daily_progress_db(user.id)


@app.post("/api/progress/daily/events", response_model=DailyProgress)
def post_daily_progress_event(
    payload: DailyProgressEventRequest,
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> DailyProgress:
    require_db_enabled()
    user = get_current_auth_user(authorization)
    return record_daily_progress_event_db(user.id, payload)


@app.get("/api/auth/google/start")
def google_auth_start(return_to: Optional[str] = Query(None)) -> RedirectResponse:
    require_db_enabled()
    require_google_oauth_enabled()
    target = normalize_return_to(return_to)
    state = create_oauth_state(target)
    auth_url = build_google_authorize_url(state)
    return RedirectResponse(url=auth_url, status_code=302)


@app.get("/api/auth/google/callback")
def google_auth_callback(
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
) -> RedirectResponse:
    require_db_enabled()
    require_google_oauth_enabled()

    fallback_return_to = frontend_login_url()
    if state:
        try:
            decoded_state = decode_oauth_state(state)
            fallback_return_to = normalize_return_to(decoded_state.get("return_to"))
        except HTTPException:
            pass

    if error:
        return RedirectResponse(
            url=append_fragment_params(
                fallback_return_to, {"oauth_error": f"google_error_{error}"}
            ),
            status_code=302,
        )

    if not code or not state:
        return RedirectResponse(
            url=append_fragment_params(
                fallback_return_to, {"oauth_error": "google_missing_code_or_state"}
            ),
            status_code=302,
        )

    try:
        decoded_state = decode_oauth_state(state)
        return_to = normalize_return_to(decoded_state.get("return_to"))
        token_payload = exchange_google_code(code)
        provider_access_token = str(token_payload.get("access_token") or "").strip()
        if not provider_access_token:
            raise HTTPException(status_code=400, detail="Missing Google access token")

        profile = fetch_google_userinfo(provider_access_token)
        email = normalize_email(str(profile.get("email") or ""))
        is_verified = bool(profile.get("email_verified"))
        if not email or not is_verified:
            raise HTTPException(status_code=400, detail="Google email is missing or not verified")

        display_name = str(profile.get("name") or email.split("@")[0]).strip() or "Student"
        user = find_or_create_google_user_db(email, display_name)
        token = create_access_token(user.id, user.email, user.role)
        success_url = append_fragment_params(
            return_to, {"access_token": token, "oauth_provider": "google"}
        )
        return RedirectResponse(url=success_url, status_code=302)
    except Exception:
        return RedirectResponse(
            url=append_fragment_params(fallback_return_to, {"oauth_error": "google_login_failed"}),
            status_code=302,
        )


@app.get("/api/courses", response_model=List[CourseFolder])
def list_courses(
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> List[CourseFolder]:
    require_db_enabled()
    user = get_current_auth_user(authorization)
    return list_courses_db(user.id)


@app.post("/api/courses", response_model=CourseFolder, status_code=201)
def create_course(
    payload: CourseCreateRequest,
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> CourseFolder:
    require_db_enabled()
    user = get_current_auth_user(authorization)
    return create_course_db(user.id, payload)


@app.get("/api/courses/{course_id}", response_model=CourseDetailResponse)
def get_course_detail(
    course_id: str,
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> CourseDetailResponse:
    require_db_enabled()
    user = get_current_auth_user(authorization)
    return get_course_detail_db(user.id, course_id)


@app.get("/api/courses/{course_id}/lectures", response_model=List[LectureItem])
def list_course_lectures(
    course_id: str,
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> List[LectureItem]:
    require_db_enabled()
    user = get_current_auth_user(authorization)
    return list_course_lectures_db(user.id, course_id, include_files=False)


@app.post("/api/courses/{course_id}/lectures", response_model=LectureItem, status_code=201)
def create_lecture(
    course_id: str,
    payload: LectureCreateRequest,
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> LectureItem:
    require_db_enabled()
    user = get_current_auth_user(authorization)
    return create_lecture_db(user.id, course_id, payload)


@app.get("/api/courses/{course_id}/lectures/{lecture_id}/files", response_model=List[LectureFileInfo])
def list_lecture_files(
    course_id: str,
    lecture_id: str,
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> List[LectureFileInfo]:
    require_db_enabled()
    user = get_current_auth_user(authorization)
    return list_lecture_files_db(user.id, course_id, lecture_id)


if MULTIPART_AVAILABLE:
    @app.post(
        "/api/courses/{course_id}/lectures/{lecture_id}/files",
        response_model=LectureFileInfo,
        status_code=201,
    )
    async def upload_lecture_file(
        course_id: str,
        lecture_id: str,
        file: UploadFile = File(...),
        authorization: Optional[str] = Header(None, alias="Authorization"),
    ) -> LectureFileInfo:
        require_db_enabled()
        user = get_current_auth_user(authorization)
        file_bytes = await file.read()
        return upload_lecture_file_db(
            user.id,
            course_id,
            lecture_id,
            file.filename or "uploaded_file",
            file.content_type,
            file_bytes,
        )
else:
    @app.post(
        "/api/courses/{course_id}/lectures/{lecture_id}/files",
        response_model=LectureFileInfo,
        status_code=201,
    )
    def upload_lecture_file_unavailable(
        course_id: str,
        lecture_id: str,
        authorization: Optional[str] = Header(None, alias="Authorization"),
    ) -> LectureFileInfo:
        _ = course_id, lecture_id, authorization
        raise HTTPException(
            status_code=503,
            detail="File upload requires python-multipart. Install backend requirements.",
        )


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
