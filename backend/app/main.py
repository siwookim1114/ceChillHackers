from __future__ import annotations

import json
import logging
import os
import sys
import base64
import hashlib
import hmac
import re
import secrets
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from urllib import error as urllib_error, request as urllib_request
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

# Ensure the project root (parent of backend/) is on sys.path so that
# `from config.config_loader import ...` resolves correctly when uvicorn
# is started from the backend/ directory.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from db.models import ScanParserRequest, ScanParserResponse
from agents.scan_parser import parse_scan_submission

try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    boto3 = None
    BotoConfig = None
    BotoCoreError = Exception
    ClientError = Exception

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
S3_LECTURE_UPLOAD_URI = os.getenv(
    "S3_LECTURE_UPLOAD_URI", "s3://cechillhacker-filebucket/docs/lecture_slides/"
).strip()
FEATHERLESS_API_BASE_URL = os.getenv(
    "FEATHERLESS_API_BASE_URL", "https://api.featherless.ai/v1"
).strip()
FEATHERLESSAI_API_KEY = os.getenv("FEATHERLESSAI_API_KEY", "").strip()
FEATHERLESS_MODEL = os.getenv("FEATHERLESS_MODEL", "Qwen/Qwen2.5-3B-Instruct").strip()
FEATHERLESS_TIMEOUT_SEC = float(os.getenv("FEATHERLESS_TIMEOUT_SEC", "30"))
FEATHERLESS_HTTP_REFERER = os.getenv("FEATHERLESS_HTTP_REFERER", FRONTEND_URL).strip()
FEATHERLESS_X_TITLE = os.getenv("FEATHERLESS_X_TITLE", "TutorCoach").strip()
FEATHERLESS_USER_AGENT = os.getenv(
    "FEATHERLESS_USER_AGENT", "TutorCoach/1.0 (Codex backend)"
).strip()
FEATHERLESS_FORCE_CURL = os.getenv("FEATHERLESS_FORCE_CURL", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
FEATHERLESS_LOCAL_FALLBACK = os.getenv("FEATHERLESS_LOCAL_FALLBACK", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
WHISPER_API_BASE_URL = os.getenv("WHISPER_API_BASE_URL", "https://api.openai.com/v1").strip()
WHISPER_API_KEY = os.getenv("WHISPER_API_KEY", "").strip()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1").strip()
WHISPER_TRANSCRIBE_PATH = os.getenv("WHISPER_TRANSCRIBE_PATH", "/audio/transcriptions").strip()
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en").strip().lower()
WHISPER_TIMEOUT_SEC = float(os.getenv("WHISPER_TIMEOUT_SEC", "45"))
MINIMAX_API_BASE_URL = os.getenv("MINIMAX_API_BASE_URL", "https://api.minimax.io").strip()
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "").strip()
MINIMAX_GROUP_ID = os.getenv("MINIMAX_GROUP_ID", "").strip()
MINIMAX_TTS_MODEL = os.getenv("MINIMAX_TTS_MODEL", "speech-02-hd").strip()
MINIMAX_TTS_VOICE_ID = os.getenv("MINIMAX_TTS_VOICE_ID", "English_Insightful_Speaker").strip()
MINIMAX_TTS_OUTPUT_FORMAT = os.getenv("MINIMAX_TTS_OUTPUT_FORMAT", "hex").strip().lower()
MINIMAX_TTS_TIMEOUT_SEC = float(os.getenv("MINIMAX_TTS_TIMEOUT_SEC", "30"))
VOICE_SESSION_TTL_SEC = int(os.getenv("VOICE_SESSION_TTL_SEC", "3600"))
VOICE_SESSION_MAX_TURNS = int(os.getenv("VOICE_SESSION_MAX_TURNS", "14"))
AWS_REGION = os.getenv("AWS_REGION", "").strip()
RAG_DOCS_PREFIX = os.getenv("RAG_DOCS_PREFIX", "docs/").strip()
AWS_S3_FORCE_PATH_STYLE = os.getenv("AWS_S3_FORCE_PATH_STYLE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DB_READY = False
_DB_INIT_LOCK = threading.Lock()
_S3_CLIENT_LOCK = threading.Lock()
_S3_CLIENT = None
_VOICE_SESSION_LOCK = threading.Lock()
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


@dataclass(frozen=True)
class S3Location:
    bucket: str
    prefix: str


def parse_s3_uri(value: str) -> S3Location:
    raw = value.strip()
    if not raw:
        raise ValueError("S3 URI is empty")
    parts = urlsplit(raw)
    if parts.scheme != "s3" or not parts.netloc:
        raise ValueError("S3 URI must use the s3://bucket/prefix format")
    prefix = parts.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix = f"{prefix}/"
    return S3Location(bucket=parts.netloc, prefix=prefix)


try:
    LECTURE_S3_LOCATION = parse_s3_uri(S3_LECTURE_UPLOAD_URI)
except ValueError:
    LECTURE_S3_LOCATION = None


def sanitized_file_name(file_name: str) -> str:
    base_name = Path(file_name).name.strip() or "uploaded_file"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base_name).strip("._")
    return cleaned[:180] or "uploaded_file"


def normalized_s3_prefix(prefix: str) -> str:
    cleaned = prefix.strip().lstrip("/")
    if cleaned and not cleaned.endswith("/"):
        cleaned = f"{cleaned}/"
    return cleaned


def sanitized_rag_subject(value: str) -> str:
    lowered = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return normalized[:80] or "general"


def get_s3_client() -> Any:
    global _S3_CLIENT
    if _S3_CLIENT is not None:
        return _S3_CLIENT
    if boto3 is None:
        raise HTTPException(
            status_code=503,
            detail="S3 upload requires boto3. Install backend requirements.",
        )

    with _S3_CLIENT_LOCK:
        if _S3_CLIENT is not None:
            return _S3_CLIENT
        client_kwargs: dict[str, Any] = {}
        if AWS_REGION:
            client_kwargs["region_name"] = AWS_REGION
        if AWS_S3_FORCE_PATH_STYLE and BotoConfig is not None:
            client_kwargs["config"] = BotoConfig(s3={"addressing_style": "path"})
        _S3_CLIENT = boto3.client("s3", **client_kwargs)
    return _S3_CLIENT


def upload_lecture_file_to_s3(
    user_id: str,
    course_id: str,
    lecture_id: str,
    file_name: str,
    content_type: Optional[str],
    file_data: bytes,
) -> tuple[str, str]:
    if LECTURE_S3_LOCATION is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "S3 lecture storage is not configured. "
                "Set S3_LECTURE_UPLOAD_URI to s3://bucket/prefix/."
            ),
        )

    safe_name = sanitized_file_name(file_name)
    timestamp = utcnow().strftime("%Y%m%dT%H%M%SZ")
    random_suffix = uuid4().hex[:10]
    key = (
        f"{LECTURE_S3_LOCATION.prefix}"
        f"{user_id}/{course_id}/{lecture_id}/{timestamp}_{random_suffix}_{safe_name}"
    )

    put_object_args: dict[str, Any] = {
        "Bucket": LECTURE_S3_LOCATION.bucket,
        "Key": key,
        "Body": file_data,
    }
    if content_type:
        put_object_args["ContentType"] = content_type

    s3_client = get_s3_client()
    try:
        s3_client.put_object(**put_object_args)
    except (BotoCoreError, ClientError, Exception) as exc:
        logger.exception("Failed to upload lecture file to S3: key=%s", key)
        raise HTTPException(
            status_code=502,
            detail="Failed to upload lecture file to S3.",
        ) from exc

    return key, f"s3://{LECTURE_S3_LOCATION.bucket}/{key}"


def sync_pdf_to_rag_docs(
    *,
    user_id: str,
    course_id: str,
    lecture_id: str,
    course_title: str,
    file_name: str,
    content_type: Optional[str],
    file_data: bytes,
) -> Optional[str]:
    if LECTURE_S3_LOCATION is None:
        return None

    ext = Path(file_name).suffix.lower()
    ctype = (content_type or "").lower()
    if ext != ".pdf" and "pdf" not in ctype:
        return None

    docs_prefix = normalized_s3_prefix(RAG_DOCS_PREFIX or "docs/")
    subject = sanitized_rag_subject(course_title)
    safe_name = sanitized_file_name(file_name)
    timestamp = utcnow().strftime("%Y%m%dT%H%M%SZ")
    random_suffix = uuid4().hex[:10]
    rag_key = (
        f"{docs_prefix}{subject}/"
        f"{user_id}/{course_id}/{lecture_id}/{timestamp}_{random_suffix}_{safe_name}"
    )

    put_object_args: dict[str, Any] = {
        "Bucket": LECTURE_S3_LOCATION.bucket,
        "Key": rag_key,
        "Body": file_data,
    }
    if content_type:
        put_object_args["ContentType"] = content_type

    s3_client = get_s3_client()
    s3_client.put_object(**put_object_args)
    return rag_key


def warmup_rag_index_async(*, course_title: str, lecture_title: str, file_name: str) -> None:
    def _worker() -> None:
        try:
            from agents.rag_agent import RagAgent

            rag_agent = RagAgent()
            rag_agent.run(
                {
                    "prompt": (
                        f"Warm up retrieval index for course '{course_title}', "
                        f"lecture '{lecture_title}', file '{file_name}'."
                    ),
                    "caller": "professor",
                    "subject": sanitized_rag_subject(course_title),
                    "mode": "internal_only",
                    "retrieve_only": True,
                }
            )
        except Exception as exc:
            logger.warning("RAG warm-up skipped for uploaded lecture file: %s", exc)

    threading.Thread(target=_worker, daemon=True).start()


@dataclass
class VoiceSessionState:
    session_id: str
    created_at: datetime
    updated_at: datetime
    user_id: Optional[str]
    attempt_id: Optional[str]
    course_id: Optional[str]
    lecture_id: Optional[str]
    lecture_context: str
    history: list[dict[str, str]] = field(default_factory=list)


VOICE_SESSIONS: Dict[str, VoiceSessionState] = {}


def join_api_url(base_url: str, path: str) -> str:
    cleaned_base = base_url.rstrip("/")
    cleaned_path = path.strip()
    if not cleaned_path:
        return cleaned_base
    if cleaned_path.startswith("http://") or cleaned_path.startswith("https://"):
        return cleaned_path
    if not cleaned_path.startswith("/"):
        cleaned_path = f"/{cleaned_path}"
    return f"{cleaned_base}{cleaned_path}"


def parse_json_object_from_text(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def build_multipart_form_data(
    fields: dict[str, str],
    file_field_name: str,
    file_name: str,
    file_bytes: bytes,
    file_content_type: str,
) -> tuple[bytes, str]:
    boundary = f"----codex-{uuid4().hex}"
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )

    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field_name}"; '
                f'filename="{sanitized_file_name(file_name)}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {file_content_type}\r\n\r\n".encode("utf-8"),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(chunks), boundary


def call_featherless_chat_via_curl(payload: dict[str, Any], request_url: str) -> dict[str, Any]:
    command = [
        "curl",
        "-sS",
        "--http1.1",
        "--connect-timeout",
        "10",
        "--max-time",
        str(int(max(15, FEATHERLESS_TIMEOUT_SEC))),
        "-X",
        "POST",
        request_url,
        "-H",
        f"Authorization: Bearer {FEATHERLESSAI_API_KEY}",
        "-H",
        "Content-Type: application/json",
        "-H",
        "Accept: application/json",
        "-H",
        f"User-Agent: {FEATHERLESS_USER_AGENT or 'TutorCoach/1.0'}",
        "-d",
        json.dumps(payload),
        "-w",
        "\n%{http_code}",
    ]
    if FEATHERLESS_HTTP_REFERER:
        command.extend(["-H", f"HTTP-Referer: {FEATHERLESS_HTTP_REFERER}"])
    if FEATHERLESS_X_TITLE:
        command.extend(["-H", f"X-Title: {FEATHERLESS_X_TITLE}"])

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Featherless curl fallback failed to execute.",
        ) from exc

    output = (result.stdout or "").strip()
    if not output:
        stderr_text = (result.stderr or "").strip()
        raise HTTPException(
            status_code=502,
            detail=f"Featherless curl fallback returned empty output. {stderr_text}",
        )

    lines = output.splitlines()
    status_candidate = lines[-1].strip() if lines else ""
    response_body = "\n".join(lines[:-1]).strip() if len(lines) > 1 else output
    try:
        status_code = int(status_candidate)
    except ValueError:
        status_code = 0
        response_body = output

    if not (200 <= status_code < 300):
        lowered_body = response_body.lower()
        if "error code: 1010" in lowered_body or "error code:1010" in lowered_body:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Featherless blocked this request (error code 1010) even with curl fallback. "
                    "This is usually account/network firewall policy."
                ),
            )
        detail = response_body or (result.stderr or "unknown error")
        raise HTTPException(
            status_code=502,
            detail=f"Featherless curl fallback failed: {detail}",
        )

    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail="Featherless curl fallback returned non-JSON response.",
        ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=502,
            detail="Featherless curl fallback returned invalid JSON.",
        )
    return parsed


def call_featherless_chat(messages: list[dict[str, str]]) -> str:
    if not FEATHERLESSAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="FEATHERLESSAI_API_KEY is not configured.",
        )

    payload = {
        "model": FEATHERLESS_MODEL,
        "messages": messages,
        "temperature": 0.35,
        "max_tokens": 480,
    }
    request_url = join_api_url(FEATHERLESS_API_BASE_URL, "/chat/completions")
    if FEATHERLESS_FORCE_CURL:
        response_json = call_featherless_chat_via_curl(payload, request_url)
        choices = response_json.get("choices")
        if not isinstance(choices, list) or not choices:
            raise HTTPException(status_code=502, detail="Featherless returned no choices.")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise HTTPException(status_code=502, detail="Featherless returned empty content.")
        return content.strip()

    headers = {
        "Authorization": f"Bearer {FEATHERLESSAI_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": FEATHERLESS_USER_AGENT or "TutorCoach/1.0",
    }
    if FEATHERLESS_HTTP_REFERER:
        headers["HTTP-Referer"] = FEATHERLESS_HTTP_REFERER
    if FEATHERLESS_X_TITLE:
        headers["X-Title"] = FEATHERLESS_X_TITLE

    request = urllib_request.Request(
        request_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=FEATHERLESS_TIMEOUT_SEC) as response:
            response_json = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        if "error code: 1010" in body.lower() or "error code:1010" in body.lower():
            response_json = call_featherless_chat_via_curl(payload, request_url)
        else:
            raise HTTPException(
                status_code=502,
                detail=f"Featherless request failed: {body or exc.reason}",
            ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Featherless request failed.") from exc

    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        raise HTTPException(status_code=502, detail="Featherless returned no choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise HTTPException(status_code=502, detail="Featherless returned empty content.")
    return content.strip()


def transcribe_audio_with_whisper(file_name: str, content_type: Optional[str], file_bytes: bytes) -> str:
    if not WHISPER_API_KEY:
        raise HTTPException(status_code=503, detail="WHISPER_API_KEY is not configured.")
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    mime_type = content_type or "audio/webm"
    fields = {"model": WHISPER_MODEL}
    if WHISPER_LANGUAGE and WHISPER_LANGUAGE != "auto":
        fields["language"] = WHISPER_LANGUAGE
    form_body, boundary = build_multipart_form_data(
        fields=fields,
        file_field_name="file",
        file_name=file_name or "voice.webm",
        file_bytes=file_bytes,
        file_content_type=mime_type,
    )
    request_url = join_api_url(WHISPER_API_BASE_URL, WHISPER_TRANSCRIBE_PATH)
    request = urllib_request.Request(
        request_url,
        data=form_body,
        headers={
            "Authorization": f"Bearer {WHISPER_API_KEY}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=WHISPER_TIMEOUT_SEC) as response:
            response_json = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(
            status_code=502,
            detail=f"Whisper transcription failed: {body or exc.reason}",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Whisper transcription failed.") from exc

    transcript = response_json.get("text")
    if not isinstance(transcript, str) or not transcript.strip():
        raise HTTPException(status_code=502, detail="Whisper returned empty transcript.")
    return transcript.strip()


def synthesize_tts_with_minimax(text: str) -> tuple[str, str]:
    if not MINIMAX_API_KEY:
        raise HTTPException(status_code=503, detail="MINIMAX_API_KEY is not configured.")
    if not text.strip():
        raise HTTPException(status_code=400, detail="TTS text is empty.")

    request_url = f"{MINIMAX_API_BASE_URL.rstrip('/')}/v1/t2a_v2"
    if MINIMAX_GROUP_ID:
        request_url = f"{request_url}?{urlencode({'GroupId': MINIMAX_GROUP_ID})}"

    output_format = MINIMAX_TTS_OUTPUT_FORMAT if MINIMAX_TTS_OUTPUT_FORMAT else "hex"
    payload = {
        "model": MINIMAX_TTS_MODEL,
        "text": text.strip(),
        "output_format": output_format,
        "stream": False,
        "voice_setting": {
            "voice_id": MINIMAX_TTS_VOICE_ID,
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0,
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3",
            "channel": 1,
        },
    }
    request = urllib_request.Request(
        request_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {MINIMAX_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(request, timeout=MINIMAX_TTS_TIMEOUT_SEC) as response:
            response_json = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(
            status_code=502,
            detail=f"MiniMax TTS failed: {body or exc.reason} (url={request_url})",
        ) from exc
    except urllib_error.URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"MiniMax TTS failed: network error ({exc.reason}) (url={request_url})",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"MiniMax TTS failed: unexpected error ({exc.__class__.__name__})",
        ) from exc

    base_resp = response_json.get("base_resp")
    if isinstance(base_resp, dict):
        status_code = int(base_resp.get("status_code", 0) or 0)
        if status_code != 0:
            status_msg = str(base_resp.get("status_msg") or "unknown error")
            raise HTTPException(status_code=502, detail=f"MiniMax TTS failed: {status_msg}")

    data = response_json.get("data")
    audio_raw = data.get("audio") if isinstance(data, dict) else None
    if not isinstance(audio_raw, str) or not audio_raw.strip():
        raise HTTPException(status_code=502, detail="MiniMax TTS returned empty audio.")

    audio_raw = audio_raw.strip()
    audio_format = str(
        (data.get("audio_format") if isinstance(data, dict) else "") or output_format
    ).lower()
    mime_type = "audio/mpeg"

    if audio_format == "hex":
        try:
            audio_bytes = bytes.fromhex(audio_raw)
        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail="MiniMax TTS returned invalid hex audio.",
            ) from exc
        return base64.b64encode(audio_bytes).decode("utf-8"), mime_type

    if audio_format == "base64":
        return audio_raw, mime_type

    # Some deployments may still return base64 without an explicit format.
    if re.fullmatch(r"[A-Fa-f0-9]+", audio_raw) and len(audio_raw) % 2 == 0:
        try:
            audio_bytes = bytes.fromhex(audio_raw)
            return base64.b64encode(audio_bytes).decode("utf-8"), mime_type
        except ValueError:
            pass
    return audio_raw, mime_type


def get_optional_auth_user(authorization: Optional[str]) -> Optional["AuthUser"]:
    if not authorization:
        return None
    return get_current_auth_user(authorization)


def get_attempt_context(attempt_id: Optional[str]) -> str:
    if not attempt_id:
        return ""
    if DB_ENABLED:
        attempt = load_attempt_state_db(attempt_id)
    else:
        attempt = ATTEMPTS.get(attempt_id)
    if not attempt:
        return ""
    return (
        f"Problem title: {attempt.problem.title}\n"
        f"Unit: {attempt.problem.unit}\n"
        f"Problem prompt: {attempt.problem.prompt}\n"
        f"Answer key reference: {attempt.problem.answer_key}"
    )


def get_lecture_context_db(user_id: str, course_id: str, lecture_id: str) -> str:
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  c.title AS course_title,
                  c.syllabus AS course_syllabus,
                  l.title AS lecture_title,
                  l.description AS lecture_description,
                  l.problem_prompt,
                  l.answer_key
                FROM lectures l
                JOIN courses c ON c.id = l.course_id
                WHERE c.user_id = %s::uuid
                  AND c.id = %s::uuid
                  AND l.id = %s::uuid
                """,
                (user_id, course_id, lecture_id),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Lecture not found for voice session.")

            cur.execute(
                """
                SELECT file_name
                FROM lecture_files
                WHERE lecture_id = %s::uuid
                ORDER BY created_at DESC
                LIMIT 8
                """,
                (lecture_id,),
            )
            file_rows = cur.fetchall()

    file_names = [str(item.get("file_name") or "").strip() for item in file_rows if item.get("file_name")]
    file_hint = ", ".join(file_names) if file_names else "No file names available."
    return (
        f"Course: {row['course_title']}\n"
        f"Course syllabus: {row.get('course_syllabus') or 'N/A'}\n"
        f"Lecture title: {row['lecture_title']}\n"
        f"Lecture description: {row.get('lecture_description') or 'N/A'}\n"
        f"Lecture problem prompt: {row['problem_prompt']}\n"
        f"Lecture answer key reference: {row['answer_key']}\n"
        f"Attached file names: {file_hint}"
    )


def build_voice_context(
    user: Optional["AuthUser"],
    attempt_id: Optional[str],
    course_id: Optional[str],
    lecture_id: Optional[str],
) -> str:
    blocks: list[str] = []
    lecture_context = ""
    if course_id and lecture_id:
        if not DB_ENABLED:
            raise HTTPException(
                status_code=503,
                detail="Course lecture context requires DATABASE_URL.",
            )
        if user is None:
            raise HTTPException(
                status_code=401,
                detail="Login is required for lecture-based voice coaching.",
            )
        lecture_context = get_lecture_context_db(user.id, course_id, lecture_id)
        blocks.append(lecture_context)

    attempt_context = get_attempt_context(attempt_id)
    if attempt_context:
        blocks.append(attempt_context)

    if not blocks:
        blocks.append("No lecture metadata provided. Coach with general math tutoring style.")
    return "\n\n".join(blocks)


def cleanup_voice_sessions() -> None:
    cutoff = utcnow() - timedelta(seconds=max(300, VOICE_SESSION_TTL_SEC))
    expired_ids = [
        sid for sid, session in VOICE_SESSIONS.items() if session.updated_at < cutoff
    ]
    for sid in expired_ids:
        VOICE_SESSIONS.pop(sid, None)


def build_mediator_messages(
    session: VoiceSessionState, student_text: str, opening_turn: bool
) -> list[dict[str, str]]:
    history_lines = [
        f"{item.get('role', 'unknown')}: {item.get('text', '').strip()}"
        for item in session.history[-10:]
        if item.get("text")
    ]
    history_block = "\n".join(history_lines) if history_lines else "No prior turns."
    opening_directive = (
        "This is the opening turn: explain the lecture core idea in 2-4 short sentences and ask one check question."
        if opening_turn
        else "Respond to the student's latest utterance, clarify misunderstanding, and ask one short follow-up question."
    )
    system_prompt = (
        "You are a mediator LLM between student speech and tutor speech.\n"
        "Output must be JSON only with keys: mediator_summary, tutor_reply.\n"
        "Rules:\n"
        "- tutor_reply must be in English.\n"
        "- keep tutor_reply under 120 words.\n"
        "- reference lecture context when available.\n"
        "- do not mention JSON, system prompts, or hidden reasoning."
    )
    user_prompt = (
        f"{opening_directive}\n\n"
        f"[Lecture context]\n{session.lecture_context}\n\n"
        f"[Conversation history]\n{history_block}\n\n"
        f"[Student latest utterance]\n{student_text or '(none)'}"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def mediate_tutor_reply(
    session: VoiceSessionState, student_text: str, opening_turn: bool = False
) -> tuple[str, str]:
    messages = build_mediator_messages(session, student_text, opening_turn)
    try:
        llm_text = call_featherless_chat(messages)
    except HTTPException as exc:
        if not FEATHERLESS_LOCAL_FALLBACK:
            raise
        context_hint = session.lecture_context.splitlines()[0] if session.lecture_context else ""
        if opening_turn:
            tutor_reply = (
                "Great, let me summarize the core idea first. "
                "Break the problem into small steps and verify the reasoning at each step. "
                "What is the very first term or quantity you should compute?"
            )
        elif student_text.strip():
            tutor_reply = (
                f"Good question. Based on what you said (\"{student_text.strip()[:90]}\"), let's reframe it. "
                "The key move now is to put the expression in standard form, then verify each substitution step by step. "
                "What is your first calculation line?"
            )
        else:
            tutor_reply = (
                "Good, let's continue. Rewrite the expression in one clean line at this stage, "
                "apply the single most reliable rule, and tell me the result."
            )
        if context_hint:
            tutor_reply = f"{context_hint}. {tutor_reply}"
        return tutor_reply, f"local-fallback:{exc.status_code}"

    parsed = parse_json_object_from_text(llm_text)
    tutor_reply = str(parsed.get("tutor_reply") or "").strip()
    mediator_summary = str(parsed.get("mediator_summary") or "").strip()
    if not tutor_reply:
        tutor_reply = llm_text.strip()
    if not mediator_summary:
        mediator_summary = "direct-pass"
    return tutor_reply, mediator_summary


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
    storage_provider: str = "s3"
    storage_key: Optional[str] = None
    file_url: Optional[str] = None
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


class AttemptGradeResponse(BaseModel):
    attempt_id: str
    solved: bool
    answer_checked_correct: bool
    accepted_events: int
    stuck_signals: StuckSignals
    parser_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    scan_parse: Dict[str, Any] = Field(default_factory=dict)
    ta_result: Dict[str, Any] = Field(default_factory=dict)
    rag_citations_count: int = 0


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


class VoiceSessionStartRequest(BaseModel):
    attempt_id: Optional[str] = None
    course_id: Optional[str] = None
    lecture_id: Optional[str] = None


class VoiceSessionStartResponse(BaseModel):
    session_id: str
    tutor_text: str
    mediator_summary: str
    audio_base64: str
    audio_mime_type: str = "audio/mpeg"


class VoiceSessionTurnResponse(BaseModel):
    session_id: str
    transcript: str
    tutor_text: str
    mediator_summary: str
    audio_base64: str
    audio_mime_type: str = "audio/mpeg"


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


def to_agent_learning_style(style: Optional[LearningStyle]) -> str:
    style_map = {
        "explanation": "mixed",
        "question": "textual",
        "problem_solving": "example_first",
    }
    if style in style_map:
        return style_map[style]
    return "mixed"


def to_agent_pace(pace: Optional[LearningPace]) -> str:
    pace_map = {
        "slow": "slow",
        "normal": "medium",
        "fast": "fast",
    }
    if pace in pace_map:
        return pace_map[pace]
    return "medium"


def default_ta_rubric() -> list[dict[str, Any]]:
    return [
        {
            "criterion_id": "setup",
            "description": "Sets up the equation or representation from the problem statement.",
            "max_points": 3.0,
            "error_tags": ["CONCEPT_GAP", "MISREAD"],
        },
        {
            "criterion_id": "method",
            "description": "Applies a valid method and keeps transformations logically consistent.",
            "max_points": 4.0,
            "error_tags": ["PROCEDURE_SLIP", "JUSTIFICATION_MISSING"],
        },
        {
            "criterion_id": "accuracy",
            "description": "Carries arithmetic/algebraic steps with minimal calculation mistakes.",
            "max_points": 2.0,
            "error_tags": ["CALCULATION_ERROR"],
        },
        {
            "criterion_id": "conclusion",
            "description": "States a final answer that matches the solved expression.",
            "max_points": 1.0,
            "error_tags": ["JUSTIFICATION_MISSING"],
        },
    ]


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
  file_data BYTEA,
  storage_provider TEXT NOT NULL DEFAULT 'db',
  storage_key TEXT,
  file_url TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_lecture_files_lecture_created
  ON lecture_files(lecture_id, created_at DESC);

ALTER TABLE lecture_files
  ALTER COLUMN file_data DROP NOT NULL;

ALTER TABLE lecture_files
  ADD COLUMN IF NOT EXISTS storage_provider TEXT NOT NULL DEFAULT 'db';

ALTER TABLE lecture_files
  ADD COLUMN IF NOT EXISTS storage_key TEXT;

ALTER TABLE lecture_files
  ADD COLUMN IF NOT EXISTS file_url TEXT;
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
        storage_provider=row.get("storage_provider") or "db",
        storage_key=row.get("storage_key"),
        file_url=row.get("file_url"),
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
                SELECT
                  id,
                  file_name,
                  content_type,
                  size_bytes,
                  storage_provider,
                  storage_key,
                  file_url,
                  created_at
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
                        SELECT
                          id,
                          file_name,
                          content_type,
                          size_bytes,
                          storage_provider,
                          storage_key,
                          file_url,
                          created_at
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

    s3_key: Optional[str] = None
    file_url: Optional[str] = None
    rag_doc_key: Optional[str] = None
    course_title = ""
    lecture_title = ""

    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT l.id, l.title AS lecture_title, c.title AS course_title
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
            course_title = str(lecture_row.get("course_title") or "").strip()
            lecture_title = str(lecture_row.get("lecture_title") or "").strip()

            s3_key, file_url = upload_lecture_file_to_s3(
                user_id=user_id,
                course_id=course_id,
                lecture_id=lecture_id,
                file_name=file_name,
                content_type=content_type,
                file_data=file_data,
            )
            try:
                rag_doc_key = sync_pdf_to_rag_docs(
                    user_id=user_id,
                    course_id=course_id,
                    lecture_id=lecture_id,
                    course_title=course_title,
                    file_name=file_name,
                    content_type=content_type,
                    file_data=file_data,
                )
            except Exception as exc:
                logger.warning("RAG sync skipped for lecture file '%s': %s", file_name, exc)

            cur.execute(
                """
                INSERT INTO lecture_files (
                  lecture_id,
                  file_name,
                  content_type,
                  size_bytes,
                  file_data,
                  storage_provider,
                  storage_key,
                  file_url
                )
                VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s)
                RETURNING
                  id,
                  file_name,
                  content_type,
                  size_bytes,
                  storage_provider,
                  storage_key,
                  file_url,
                  created_at
                """,
                (
                    lecture_id,
                    file_name.strip(),
                    content_type,
                    len(file_data),
                    None,
                    "s3",
                    s3_key,
                    file_url,
                ),
            )
            row = cur.fetchone()
        conn.commit()

    if not row:
        raise HTTPException(status_code=500, detail="Failed to upload lecture file")
    if rag_doc_key:
        warmup_rag_index_async(
            course_title=course_title or "general",
            lecture_title=lecture_title or "lecture",
            file_name=file_name,
        )
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


def create_voice_session_state(
    user: Optional[AuthUser],
    payload: VoiceSessionStartRequest,
) -> VoiceSessionState:
    lecture_context = build_voice_context(
        user=user,
        attempt_id=payload.attempt_id,
        course_id=payload.course_id,
        lecture_id=payload.lecture_id,
    )
    now = utcnow()
    return VoiceSessionState(
        session_id=uuid4().hex,
        created_at=now,
        updated_at=now,
        user_id=user.id if user else None,
        attempt_id=payload.attempt_id,
        course_id=payload.course_id,
        lecture_id=payload.lecture_id,
        lecture_context=lecture_context,
        history=[],
    )


def push_voice_turn(session: VoiceSessionState, role: str, text: str) -> None:
    clean_text = text.strip()
    if not clean_text:
        return
    session.history.append({"role": role, "text": clean_text})
    max_messages = max(8, VOICE_SESSION_MAX_TURNS * 2)
    if len(session.history) > max_messages:
        session.history = session.history[-max_messages:]
    session.updated_at = utcnow()


def get_voice_session_or_404(session_id: str) -> VoiceSessionState:
    with _VOICE_SESSION_LOCK:
        cleanup_voice_sessions()
        session = VOICE_SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Voice session not found or expired.")
    return session


app = FastAPI(title="AI Coach MVP API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Professor Agent routes (text + voice) — optional dependency.
try:
    from app.routes.professor import router as professor_router
except ModuleNotFoundError as exc:
    logger.warning(
        "Professor routes disabled due to missing dependency: %s",
        exc,
    )
else:
    app.include_router(professor_router)


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


@app.post("/api/scan/parse", response_model=ScanParserResponse)
def parse_scan(payload: ScanParserRequest) -> ScanParserResponse:
    """Parse handwritten/typed student work into structured TA scan payload.

    Raw image bytes are processed in-memory only and not persisted.
    """
    try:
        return parse_scan_submission(payload)
    except (ValueError, PydanticValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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


@app.post("/api/voice/session/start", response_model=VoiceSessionStartResponse)
def start_voice_session(
    payload: VoiceSessionStartRequest,
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> VoiceSessionStartResponse:
    user = get_optional_auth_user(authorization)

    session = create_voice_session_state(user, payload)
    tutor_text, mediator_summary = mediate_tutor_reply(
        session=session,
        student_text="",
        opening_turn=True,
    )
    audio_base64, audio_mime_type = synthesize_tts_with_minimax(tutor_text)
    push_voice_turn(session, "tutor", tutor_text)

    with _VOICE_SESSION_LOCK:
        cleanup_voice_sessions()
        VOICE_SESSIONS[session.session_id] = session

    return VoiceSessionStartResponse(
        session_id=session.session_id,
        tutor_text=tutor_text,
        mediator_summary=mediator_summary,
        audio_base64=audio_base64,
        audio_mime_type=audio_mime_type,
    )


if MULTIPART_AVAILABLE:
    @app.post("/api/voice/session/turn", response_model=VoiceSessionTurnResponse)
    async def voice_session_turn(
        session_id: str = Form(...),
        audio: UploadFile = File(...),
        authorization: Optional[str] = Header(None, alias="Authorization"),
    ) -> VoiceSessionTurnResponse:
        user = get_optional_auth_user(authorization)
        session = get_voice_session_or_404(session_id.strip())
        if session.user_id and user and session.user_id != user.id:
            raise HTTPException(status_code=403, detail="Voice session owner mismatch.")
        if session.user_id and user is None:
            raise HTTPException(status_code=401, detail="Login is required for this voice session.")

        audio_bytes = await audio.read()
        transcript = transcribe_audio_with_whisper(
            file_name=audio.filename or "voice.webm",
            content_type=audio.content_type,
            file_bytes=audio_bytes,
        )
        push_voice_turn(session, "student", transcript)
        tutor_text, mediator_summary = mediate_tutor_reply(
            session=session,
            student_text=transcript,
            opening_turn=False,
        )
        audio_base64, audio_mime_type = synthesize_tts_with_minimax(tutor_text)
        push_voice_turn(session, "tutor", tutor_text)

        with _VOICE_SESSION_LOCK:
            VOICE_SESSIONS[session.session_id] = session

        return VoiceSessionTurnResponse(
            session_id=session.session_id,
            transcript=transcript,
            tutor_text=tutor_text,
            mediator_summary=mediator_summary,
            audio_base64=audio_base64,
            audio_mime_type=audio_mime_type,
        )
else:
    @app.post("/api/voice/session/turn", response_model=VoiceSessionTurnResponse)
    def voice_session_turn_unavailable(
        session_id: str,
        authorization: Optional[str] = Header(None, alias="Authorization"),
    ) -> VoiceSessionTurnResponse:
        _ = session_id, authorization
        raise HTTPException(
            status_code=503,
            detail="Voice turn upload requires python-multipart. Install backend requirements.",
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


if MULTIPART_AVAILABLE:
    @app.post("/api/attempts/{attempt_id}/grade", response_model=AttemptGradeResponse)
    async def grade_attempt_submission(
        attempt_id: str,
        file: UploadFile = File(...),
        work_notes: Optional[str] = Form(None),
        final_answer: Optional[str] = Form(None),
        authorization: Optional[str] = Header(None, alias="Authorization"),
    ) -> AttemptGradeResponse:
        if DB_ENABLED:
            attempt = load_attempt_state_db(attempt_id)
        else:
            attempt = ATTEMPTS.get(attempt_id)
        if not attempt:
            raise HTTPException(status_code=404, detail="Attempt not found")

        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Submitted file is empty")
        if len(file_bytes) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Submitted file must be <= 5MB")

        notes_text = (work_notes or "").strip()
        answer_text = (final_answer or "").strip()
        scan_request = ScanParserRequest(
            image_bytes_b64=base64.b64encode(file_bytes).decode("utf-8"),
            image_mime_type=file.content_type,
            attempt_id=attempt.id,
            topic=attempt.problem.unit,
            problem_statement_hint=attempt.problem.prompt,
            ocr_text=notes_text or None,
            answer_hint=answer_text or None,
        )
        try:
            parsed_scan = parse_scan_submission(scan_request)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to parse submitted work: {exc}",
            ) from exc

        try:
            from agents.problem_solve_ta_agent import invoke_problem_solve_ta
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"TA grading agent is unavailable: {exc}",
            ) from exc

        auth_user = get_optional_auth_user(authorization)
        profile = {
            "level": "intermediate",
            "learning_style": to_agent_learning_style(
                auth_user.learning_style if auth_user else None
            ),
            "pace": to_agent_pace(auth_user.learning_pace if auth_user else None),
        }
        signals = compute_signals(attempt.events)
        ta_payload = {
            "request_id": uuid4().hex,
            "user_id": auth_user.id if auth_user else (attempt.guest_id or "guest"),
            "attempt_id": attempt.id,
            "session_id": attempt.id,
            "profile": profile,
            "mode": "internal_only",
            "problem_ref": {
                "problem_id": attempt.problem.id,
                "statement": attempt.problem.prompt,
                "topic": sanitized_rag_subject(attempt.problem.unit or attempt.problem.title),
                "unit": attempt.problem.unit or None,
            },
            "scan_parse": parsed_scan.scan_parse.model_dump(mode="json"),
            "reference_solution_outline": [
                "Match the final expression to the expected answer format.",
                f"Expected answer key: {attempt.problem.answer_key}",
            ],
            "rubric": default_ta_rubric(),
            "stuck_signals": {
                "idle_ms": signals.idle_ms,
                "erase_count_delta": signals.erase_count_delta,
                "repeated_error_count": signals.repeated_error_count,
                "stuck_score": signals.stuck_score,
            },
            "language": "en",
        }
        try:
            ta_result = invoke_problem_solve_ta(ta_payload)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"TA grading failed: {exc}",
            ) from exc

        submission_answer = answer_text or (parsed_scan.scan_parse.final_answer or "")
        event_payload: dict[str, Any] = {
            "answer": submission_answer,
            "agent_overall_verdict": ta_result.get("overall_verdict"),
            "agent_feedback_message": ta_result.get("feedback_message", ""),
        }
        partial_score = ta_result.get("partial_score")
        if isinstance(partial_score, dict):
            event_payload["agent_partial_score_percent"] = partial_score.get("percent")

        batch = ingest_events(
            attempt_id,
            EventBatchRequest(
                events=[ClientEvent(type="answer_submit", payload=event_payload)]
            ),
        )
        answer_checked_correct = check_answer(attempt.problem, submission_answer)

        return AttemptGradeResponse(
            attempt_id=attempt_id,
            solved=batch.solved,
            answer_checked_correct=answer_checked_correct,
            accepted_events=batch.accepted,
            stuck_signals=batch.stuck_signals,
            parser_diagnostics=parsed_scan.diagnostics.model_dump(mode="json"),
            scan_parse=parsed_scan.scan_parse.model_dump(mode="json"),
            ta_result=ta_result,
            rag_citations_count=len(ta_result.get("citations", [])),
        )
else:
    @app.post("/api/attempts/{attempt_id}/grade", response_model=AttemptGradeResponse)
    def grade_attempt_submission_unavailable(
        attempt_id: str,
        authorization: Optional[str] = Header(None, alias="Authorization"),
    ) -> AttemptGradeResponse:
        _ = attempt_id, authorization
        raise HTTPException(
            status_code=503,
            detail="Grading upload requires python-multipart. Install backend requirements.",
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


# ---------------------------------------------------------------------------
# Orchestration endpoints (LangGraph multi-agent)
# ---------------------------------------------------------------------------

class OrchestratorChatRequest(BaseModel):
    """Incoming chat message routed through the LangGraph orchestration graph."""
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    topic: str = Field(default="general", min_length=1)
    user_id: str | None = None
    level: str = "intermediate"
    learning_style: str = "mixed"
    pace: str = "medium"
    mode: str = "strict"
    knowledge_mode: str = "internal_only"
    require_human_review: bool = True   # HITL always on


class OrchestratorChatResponse(BaseModel):
    """Response from the orchestration graph."""
    session_id: str
    agent_name: str
    response_text: str
    structured_data: dict = Field(default_factory=dict)
    citations: list = Field(default_factory=list)
    next_action: str = "continue"
    route_used: str = ""
    intent: str = ""
    awaiting_feedback: bool = False
    turn_count: int = 0
    rag_found: bool = False
    rag_citations_count: int = 0


class OrchestratorFeedbackRequest(BaseModel):
    """Human-in-the-loop feedback submission to resume the graph."""
    session_id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    action: str = Field(min_length=1)       # approve | revise | reroute | cancel
    feedback_text: str = ""
    reroute_to: str | None = None


@app.post("/api/chat", response_model=OrchestratorChatResponse)
def orchestrator_chat(req: OrchestratorChatRequest):
    """Main chat endpoint -- invokes the LangGraph orchestration graph.

    Routes the user message through Manager → specialist agent → response.
    HITL is always enabled: every agent response pauses for voice/text feedback.
    Auto-detects pending HITL interrupts: if the graph is paused, the user's
    message is automatically classified as feedback (approve/revise/reroute)
    by the Manager and the graph is resumed.
    """
    from agents.graph import get_graph
    from agents.nodes import classify_hitl_feedback

    graph = get_graph()
    thread_id = req.session_id
    config = {"configurable": {"thread_id": thread_id}}

    # Check graph state: is there a pending HITL interrupt?
    snapshot = None
    try:
        snapshot = graph.get_state(config)
    except Exception:
        pass

    is_hitl_pending = bool(
        snapshot and snapshot.next and "human_feedback" in snapshot.next
    )

    if is_hitl_pending:
        # ── HITL feedback path ──
        # The user's message is feedback on the previous agent response.
        # The Manager classifies it automatically (no manual buttons needed).
        from langgraph.types import Command

        current_agent = ""
        if snapshot.values:
            current_agent = (
                snapshot.values.get("agent_output", {}).get("agent_name", "")
            )

        feedback = classify_hitl_feedback(req.message, current_agent)

        try:
            result = graph.invoke(Command(resume=feedback), config)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Feedback resume error: {exc}"
            )

        turn_count = result.get("session", {}).get("turn_count", 0)
    else:
        # ── New query path ──
        existing_state = (
            snapshot.values
            if snapshot and snapshot.values
            and snapshot.values.get("agent_outputs_history")
            else None
        )

        turn_count = 0
        if existing_state:
            turn_count = (
                existing_state.get("session", {}).get("turn_count", 0) + 1
            )

        input_state = {
            "user_message": req.message,
            "user_profile": {
                "user_id": req.user_id or "anonymous",
                "level": req.level,
                "learning_style": req.learning_style,
                "pace": req.pace,
            },
            "session": {
                "session_id": req.session_id,
                "topic": req.topic,
                "subject": req.topic,
                "mode": req.mode,
                "knowledge_mode": req.knowledge_mode,
                "require_human_review": True,   # HITL always on
                "turn_count": turn_count,
            },
            "route_history": [],
            "routing": {},
            "agent_output": {},
            "human_feedback": None,
            "error": None,
            "final_response": {},
            "awaiting_feedback": False,
            "rag_context": "",
            "rag_citations": [],
            "rag_found": False,
        }

        if not existing_state:
            input_state["agent_outputs_history"] = []

        try:
            result = graph.invoke(input_state, config)
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Orchestration error: {exc}"
            )

    # ── Build response (shared by both paths) ──
    final = result.get("final_response") or {}
    routing = result.get("routing") or {}
    agent_output = result.get("agent_output") or {}
    rag_found = result.get("rag_found", False)
    rag_citations = result.get("rag_citations", [])

    is_interrupted = (
        not final.get("response_text")
        and agent_output.get("response_text")
    )

    if is_interrupted:
        return OrchestratorChatResponse(
            session_id=req.session_id,
            agent_name=agent_output.get("agent_name", "unknown"),
            response_text=agent_output.get("response_text", ""),
            structured_data=agent_output.get("structured_data", {}),
            citations=agent_output.get("citations", []),
            next_action=agent_output.get("next_action", "continue"),
            route_used=routing.get("route", ""),
            intent=routing.get("intent", ""),
            awaiting_feedback=True,
            turn_count=turn_count,
            rag_found=rag_found,
            rag_citations_count=len(rag_citations),
        )

    return OrchestratorChatResponse(
        session_id=req.session_id,
        agent_name=final.get("agent_name", "unknown"),
        response_text=final.get("response_text", "No response generated."),
        structured_data=final.get("structured_data", {}),
        citations=final.get("citations", []),
        next_action=final.get("next_action", "continue"),
        route_used=routing.get("route", ""),
        intent=routing.get("intent", ""),
        awaiting_feedback=False,
        turn_count=turn_count,
        rag_found=rag_found,
        rag_citations_count=len(rag_citations),
    )


# ---------------------------------------------------------------------------
# Voice Orchestration endpoint (STT → LangGraph → TTS)
# ---------------------------------------------------------------------------

class OrchestratorVoiceResponse(BaseModel):
    """Response from the voice orchestration pipeline."""
    session_id: str
    transcript: str
    agent_name: str
    response_text: str
    structured_data: dict = Field(default_factory=dict)
    citations: list = Field(default_factory=list)
    next_action: str = "continue"
    route_used: str = ""
    intent: str = ""
    awaiting_feedback: bool = False
    turn_count: int = 0
    rag_found: bool = False
    rag_citations_count: int = 0
    audio_base64: str = ""


@app.post("/api/chat/voice", response_model=OrchestratorVoiceResponse)
async def orchestrator_voice(
    audio: UploadFile = File(...),
    session_id: str = Form("voice-session"),
    topic: str = Form("general"),
    level: str = Form("intermediate"),
    learning_style: str = Form("mixed"),
    pace: str = Form("medium"),
    mode: str = Form("strict"),
    knowledge_mode: str = Form("internal_only"),
):
    """Voice-in, voice-out orchestration endpoint.

    Pipeline: STT (Whisper) → LangGraph orchestration → TTS (MiniMax)

    HITL is always enabled. Auto-detects pending HITL interrupts: if the
    graph is paused awaiting feedback, the user's speech is automatically
    classified as feedback (approve/revise/reroute) by the Manager and
    the graph is resumed. No manual buttons needed.
    """
    import asyncio
    import base64 as b64mod

    from services.voice_service import VoiceService

    voice = VoiceService()

    # Step 1: Read audio
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    content_type = audio.content_type or "audio/webm"

    # Step 2: STT
    try:
        transcript = await voice.speech_to_text(audio_bytes, content_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Speech-to-text error: {exc}")

    # Step 3: Route through LangGraph (auto-detect HITL)
    from agents.graph import get_graph
    from agents.nodes import classify_hitl_feedback

    graph = get_graph()
    graph_config = {"configurable": {"thread_id": session_id}}

    # Check for pending HITL interrupt
    snapshot = None
    try:
        snapshot = graph.get_state(graph_config)
    except Exception:
        pass

    is_hitl_pending = bool(
        snapshot and snapshot.next and "human_feedback" in snapshot.next
    )

    if is_hitl_pending:
        # ── HITL feedback path ──
        # User's speech is feedback on the previous agent response.
        from langgraph.types import Command

        current_agent = ""
        if snapshot.values:
            current_agent = (
                snapshot.values.get("agent_output", {}).get("agent_name", "")
            )

        feedback = classify_hitl_feedback(transcript, current_agent)

        try:
            result = await asyncio.to_thread(
                graph.invoke, Command(resume=feedback), graph_config
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Feedback resume error: {exc}"
            )

        turn_count = result.get("session", {}).get("turn_count", 0)
    else:
        # ── New query path ──
        existing_state = (
            snapshot.values
            if snapshot and snapshot.values
            and snapshot.values.get("agent_outputs_history")
            else None
        )

        turn_count = 0
        if existing_state:
            turn_count = (
                existing_state.get("session", {}).get("turn_count", 0) + 1
            )

        input_state = {
            "user_message": transcript,
            "user_profile": {
                "user_id": "voice-user",
                "level": level,
                "learning_style": learning_style,
                "pace": pace,
            },
            "session": {
                "session_id": session_id,
                "topic": topic,
                "subject": topic,
                "mode": mode,
                "knowledge_mode": knowledge_mode,
                "require_human_review": True,   # HITL always on
                "turn_count": turn_count,
            },
            "route_history": [],
            "routing": {},
            "agent_output": {},
            "human_feedback": None,
            "error": None,
            "final_response": {},
            "awaiting_feedback": False,
            "rag_context": "",
            "rag_citations": [],
            "rag_found": False,
        }

        if not existing_state:
            input_state["agent_outputs_history"] = []

        try:
            result = await asyncio.to_thread(
                graph.invoke, input_state, graph_config
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"Orchestration error: {exc}"
            )

    # ── Build response (shared by both paths) ──
    final = result.get("final_response") or {}
    routing = result.get("routing") or {}
    agent_output = result.get("agent_output") or {}
    rag_found = result.get("rag_found", False)
    rag_citations = result.get("rag_citations", [])

    is_interrupted = (
        not final.get("response_text")
        and agent_output.get("response_text")
    )

    if is_interrupted:
        response_text = agent_output.get("response_text", "")
        agent_name_val = agent_output.get("agent_name", "unknown")
        citations_val = agent_output.get("citations", [])
        structured_val = agent_output.get("structured_data", {})
        next_action_val = agent_output.get("next_action", "continue")
        awaiting = True
    else:
        response_text = final.get("response_text", "No response generated.")
        agent_name_val = final.get("agent_name", "unknown")
        citations_val = final.get("citations", [])
        structured_val = final.get("structured_data", {})
        next_action_val = final.get("next_action", "continue")
        awaiting = False

    # Step 4: TTS
    audio_b64 = ""
    if response_text:
        try:
            tts_bytes = await voice.text_to_speech(response_text)
            audio_b64 = b64mod.b64encode(tts_bytes).decode("utf-8")
        except Exception:
            pass  # TTS failure is non-fatal

    return OrchestratorVoiceResponse(
        session_id=session_id,
        transcript=transcript,
        agent_name=agent_name_val,
        response_text=response_text,
        structured_data=structured_val,
        citations=citations_val,
        next_action=next_action_val,
        route_used=routing.get("route", ""),
        intent=routing.get("intent", ""),
        awaiting_feedback=awaiting,
        turn_count=turn_count,
        rag_found=rag_found,
        rag_citations_count=len(rag_citations),
        audio_base64=audio_b64,
    )


@app.post("/api/chat/feedback", response_model=OrchestratorChatResponse)
def orchestrator_feedback(req: OrchestratorFeedbackRequest):
    """Resume graph execution after human-in-the-loop feedback.

    Called when the user reviews an agent response and chooses to
    approve, revise, reroute, or cancel.
    """
    from langgraph.types import Command

    from agents.graph import get_graph

    graph = get_graph()
    config = {"configurable": {"thread_id": req.thread_id}}

    feedback_value = {
        "feedback_text": req.feedback_text,
        "action": req.action,
        "reroute_to": req.reroute_to,
    }

    try:
        result = graph.invoke(Command(resume=feedback_value), config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feedback resume error: {exc}")

    final = result.get("final_response") or {}
    routing = result.get("routing") or {}
    agent_output = result.get("agent_output") or {}
    turn_count = result.get("session", {}).get("turn_count", 0)

    rag_found = result.get("rag_found", False)
    rag_citations = result.get("rag_citations", [])

    # Check if the graph paused again (e.g., revise → agent → HITL again)
    is_interrupted = (
        not final.get("response_text")
        and agent_output.get("response_text")
    )

    if is_interrupted:
        return OrchestratorChatResponse(
            session_id=req.session_id,
            agent_name=agent_output.get("agent_name", "unknown"),
            response_text=agent_output.get("response_text", ""),
            structured_data=agent_output.get("structured_data", {}),
            citations=agent_output.get("citations", []),
            next_action=agent_output.get("next_action", "continue"),
            route_used=routing.get("route", ""),
            intent=routing.get("intent", ""),
            awaiting_feedback=True,
            turn_count=turn_count,
            rag_found=rag_found,
            rag_citations_count=len(rag_citations),
        )

    return OrchestratorChatResponse(
        session_id=req.session_id,
        agent_name=final.get("agent_name", "unknown"),
        response_text=final.get("response_text", "No response generated."),
        structured_data=final.get("structured_data", {}),
        citations=final.get("citations", []),
        next_action=final.get("next_action", "continue"),
        route_used=routing.get("route", ""),
        intent=routing.get("intent", ""),
        awaiting_feedback=False,
        turn_count=turn_count,
        rag_found=rag_found,
        rag_citations_count=len(rag_citations),
    )
