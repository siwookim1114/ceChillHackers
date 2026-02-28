"""Professor Agent API routes -- text and voice endpoints."""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from agents.professor_agent import ProfessorAgent
from db.models import (
    Citation,
    ProfessorChatResponse,
    ProfessorMode,
    ProfessorNextAction,
    ProfessorProfile,
    ProfessorTurnRequest,
    ProfessorTurnStrategy,
    ProfessorVoiceResponse,
)
from services.voice_service import VoiceService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/professor", tags=["professor"])

# Lazy-init singletons (avoid heavy init at import time)
_professor_agent: ProfessorAgent | None = None
_voice_service: VoiceService | None = None


def _get_professor_agent() -> ProfessorAgent:
    global _professor_agent
    if _professor_agent is None:
        _professor_agent = ProfessorAgent()
    return _professor_agent


def _get_voice_service() -> VoiceService:
    global _voice_service
    if _voice_service is None:
        _voice_service = VoiceService()
    return _voice_service


# -- Request model for the chat endpoint ------------------------------------

class ProfessorChatRequest(BaseModel):
    """API-layer request for text-based professor chat."""
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    mode: str = "strict"
    profile: ProfessorProfile


# -- Text endpoint -----------------------------------------------------------

@router.post("/chat", response_model=ProfessorChatResponse)
async def professor_chat(
    payload: ProfessorChatRequest,
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> ProfessorChatResponse:
    """Text-in, text-out professor tutoring turn.

    The professor uses RAG retrieval for grounding, then generates a
    Socratic tutoring response adapted to the student's profile.
    """
    # Build domain request
    try:
        mode = ProfessorMode(payload.mode)
    except ValueError:
        mode = ProfessorMode.STRICT

    request = ProfessorTurnRequest(
        session_id=payload.session_id,
        message=payload.message,
        topic=payload.topic,
        mode=mode,
        profile=payload.profile,
    )

    # Run professor agent (sync call wrapped for async)
    agent = _get_professor_agent()
    try:
        result = await asyncio.to_thread(agent.run, request)
    except Exception as exc:
        logger.error("Professor chat failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Professor agent error: {exc}")

    return ProfessorChatResponse(
        assistant_response=result.assistant_response,
        strategy=result.strategy,
        revealed_final_answer=result.revealed_final_answer,
        next_action=result.next_action,
        citations=result.citations,
        rag_found=True,
        rag_mode=mode.value,
    )


# -- Voice endpoint ----------------------------------------------------------

@router.post("/voice", response_model=ProfessorVoiceResponse)
async def professor_voice(
    audio: UploadFile = File(...),
    topic: str = Form(...),
    session_id: str = Form("voice-session"),
    mode: str = Form("strict"),
    level: str = Form("intermediate"),
    learning_style: str = Form("mixed"),
    pace: str = Form("medium"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> ProfessorVoiceResponse:
    """Audio-in, audio-out professor tutoring turn.

    Pipeline:
    1. Read audio blob from multipart upload
    2. STT via OpenAI Whisper -> transcript
    3. Build ProfessorTurnRequest from transcript + form fields
    4. ProfessorAgent.run() -> tutoring response
    5. TTS via MiniMax -> audio bytes
    6. Return JSON with text response + base64-encoded audio
    """
    voice = _get_voice_service()
    agent = _get_professor_agent()

    # Step 1: Read audio
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    content_type = audio.content_type or "audio/webm"

    # Step 2: STT
    try:
        transcript = await voice.speech_to_text(audio_bytes, content_type)
    except Exception as exc:
        logger.error("STT failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Speech-to-text error: {exc}")

    # Step 3: Build request
    try:
        prof_mode = ProfessorMode(mode)
    except ValueError:
        prof_mode = ProfessorMode.STRICT

    # Validate profile fields
    valid_levels = {"beginner", "intermediate", "advanced"}
    valid_styles = {"visual", "textual", "example_first", "mixed"}
    valid_paces = {"slow", "medium", "fast"}

    request = ProfessorTurnRequest(
        session_id=session_id,
        message=transcript,
        topic=topic,
        mode=prof_mode,
        profile=ProfessorProfile(
            level=level if level in valid_levels else "intermediate",
            learning_style=learning_style if learning_style in valid_styles else "mixed",
            pace=pace if pace in valid_paces else "medium",
        ),
    )

    # Step 4: Professor agent
    try:
        result = await asyncio.to_thread(agent.run, request)
    except Exception as exc:
        logger.error("Professor voice agent failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Professor agent error: {exc}")

    # Step 5: TTS
    try:
        audio_response = await voice.text_to_speech(result.assistant_response)
        audio_b64 = base64.b64encode(audio_response).decode("utf-8")
    except Exception as exc:
        logger.error("TTS failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Text-to-speech error: {exc}")

    # Step 6: Return
    return ProfessorVoiceResponse(
        transcript=transcript,
        assistant_response=result.assistant_response,
        strategy=result.strategy,
        next_action=result.next_action,
        citations=result.citations,
        audio_base64=audio_b64,
    )


# -- Chat + Voice endpoint (text-in, text+audio-out) --------------------------
# No STT needed — useful when you have MiniMax but not OpenAI Whisper.

class ProfessorChatVoiceResponse(BaseModel):
    """Text response + TTS audio from professor."""
    assistant_response: str
    strategy: str
    next_action: str
    citations: list = Field(default_factory=list)
    audio_base64: str = ""


@router.post("/chat-voice", response_model=ProfessorChatVoiceResponse)
async def professor_chat_voice(
    payload: ProfessorChatRequest,
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> ProfessorChatVoiceResponse:
    """Text-in, text+audio-out. Runs the professor then converts response to speech.

    Same as /chat but also returns audio_base64 (MP3) of the professor's response.
    No microphone or OpenAI key needed — only MiniMax TTS.
    """
    try:
        mode = ProfessorMode(payload.mode)
    except ValueError:
        mode = ProfessorMode.STRICT

    request = ProfessorTurnRequest(
        session_id=payload.session_id,
        message=payload.message,
        topic=payload.topic,
        mode=mode,
        profile=payload.profile,
    )

    agent = _get_professor_agent()
    try:
        result = await asyncio.to_thread(agent.run, request)
    except Exception as exc:
        logger.error("Professor chat-voice failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Professor agent error: {exc}")

    # TTS
    voice = _get_voice_service()
    audio_b64 = ""
    try:
        audio_bytes = await voice.text_to_speech(result.assistant_response)
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as exc:
        logger.warning("TTS failed, returning text only: %s", exc)

    return ProfessorChatVoiceResponse(
        assistant_response=result.assistant_response,
        strategy=result.strategy.value,
        next_action=result.next_action.value,
        citations=[c.model_dump() for c in result.citations],
        audio_base64=audio_b64,
    )
