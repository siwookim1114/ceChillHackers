"""Voice service -- STT (local Whisper or API Whisper) and TTS (MiniMax T2A v2).

Stateless service layer, completely decoupled from agent logic.

STT: Uses faster-whisper locally when available. If unavailable, falls back
     to Whisper-compatible API when WHISPER_API_KEY is configured.
     Local model download may occur on first use (~150 MB).
TTS: Uses MiniMax T2A v2 API (requires MINIMAX_API_KEY + MINIMAX_GROUP_ID).
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Any

import httpx

from config.config_loader import config as default_config

logger = logging.getLogger(__name__)


class VoiceService:
    """Handles Speech-to-Text and Text-to-Speech conversions."""

    def __init__(self, config: Any = None) -> None:
        self.config = config or default_config

        # STT config
        self.stt_provider = self.config.get("voice.stt.provider", "local_whisper")
        self.stt_model_size = self.config.get("voice.stt.model_size", "base")
        self.stt_language = str(
            os.environ.get(
                "WHISPER_LANGUAGE",
                self.config.get("voice.stt.language", "auto"),
            )
        ).strip().lower()
        self._whisper_model = None  # lazy-loaded on first STT call
        self.whisper_api_base_url = os.environ.get(
            "WHISPER_API_BASE_URL", "https://api.openai.com/v1"
        ).strip()
        self.whisper_api_key = os.environ.get("WHISPER_API_KEY", "").strip()
        self.whisper_model = os.environ.get("WHISPER_MODEL", "whisper-1").strip()
        self.whisper_transcribe_path = os.environ.get(
            "WHISPER_TRANSCRIBE_PATH", "/audio/transcriptions"
        ).strip()
        self.whisper_timeout = float(os.environ.get("WHISPER_TIMEOUT_SEC", "45"))

        # TTS config (MiniMax)
        self.minimax_api_key = os.environ.get("MINIMAX_API_KEY", "")
        self.minimax_group_id = os.environ.get("MINIMAX_GROUP_ID", "")
        self.tts_model = self.config.get("voice.tts.model", "speech-2.8-hd")
        self.tts_voice_id = self.config.get(
            "voice.tts.voice_id", "English_Insightful_Speaker"
        )
        self.tts_speed = float(self.config.get("voice.tts.speed", 1.0))
        self.tts_emotion = self.config.get("voice.tts.emotion", "calm")
        self.tts_format = self.config.get("voice.tts.format", "mp3")
        self.tts_sample_rate = int(self.config.get("voice.tts.sample_rate", 24000))
        self.tts_timeout = int(self.config.get("voice.tts.timeout_sec", 15))

    # -- STT: Local Whisper (faster-whisper) ---------------------------------

    def _get_whisper_model(self):
        """Lazy-load the Whisper model on first use."""
        if self._whisper_model is None:
            try:
                from faster_whisper import WhisperModel
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "No module named 'faster_whisper'. "
                    "Install it (`pip install faster-whisper`) or set WHISPER_API_KEY "
                    "to use API-based STT fallback."
                ) from exc

            logger.info(
                "Loading Whisper model '%s' (first call may download ~150 MB)...",
                self.stt_model_size,
            )
            self._whisper_model = WhisperModel(
                self.stt_model_size,
                device="cpu",
                compute_type="int8",
            )
            logger.info("Whisper model loaded.")
        return self._whisper_model

    async def _speech_to_text_api(
        self,
        audio_bytes: bytes,
        content_type: str,
    ) -> str:
        if not self.whisper_api_key:
            raise ValueError(
                "WHISPER_API_KEY is not configured for API STT fallback."
            )

        transcribe_path = self.whisper_transcribe_path.strip()
        if not transcribe_path.startswith("/"):
            transcribe_path = f"/{transcribe_path}"
        request_url = f"{self.whisper_api_base_url.rstrip('/')}{transcribe_path}"
        file_content_type = content_type or "audio/webm"

        files = {
            "file": ("voice.webm", audio_bytes, file_content_type),
        }
        data = {"model": self.whisper_model}
        if self.stt_language and self.stt_language != "auto":
            data["language"] = self.stt_language

        async with httpx.AsyncClient(timeout=self.whisper_timeout) as client:
            response = await client.post(
                request_url,
                headers={"Authorization": f"Bearer {self.whisper_api_key}"},
                data=data,
                files=files,
            )

        if response.status_code != 200:
            detail = response.text[:300]
            raise RuntimeError(
                f"Whisper API STT failed (HTTP {response.status_code}): {detail}"
            )

        payload = response.json()
        transcript = str(payload.get("text") or "").strip()
        if not transcript:
            raise ValueError("Whisper API returned empty transcript.")
        return transcript

    async def speech_to_text(
        self,
        audio_bytes: bytes,
        content_type: str = "audio/webm",
    ) -> str:
        """Convert audio bytes to text using local Whisper model.

        Parameters
        ----------
        audio_bytes : bytes
            Raw audio data (webm, wav, mp3, m4a, etc.)
        content_type : str
            MIME type of the audio (used to determine file extension).

        Returns
        -------
        str
            The transcribed text.
        """
        ext_map = {
            "audio/webm": ".webm",
            "audio/wav": ".wav",
            "audio/wave": ".wav",
            "audio/x-wav": ".wav",
            "audio/mpeg": ".mp3",
            "audio/mp3": ".mp3",
            "audio/mp4": ".m4a",
            "audio/m4a": ".m4a",
            "audio/ogg": ".ogg",
            "audio/flac": ".flac",
        }
        ext = ext_map.get(content_type, ".webm")

        # Explicitly use API path when configured.
        if str(self.stt_provider).strip().lower() in {
            "openai_api",
            "whisper_api",
            "api_whisper",
        }:
            transcript = await self._speech_to_text_api(audio_bytes, content_type)
        else:
            # faster-whisper needs a file path, so write to a temp file.
            def _transcribe_local() -> str:
                with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
                    tmp.write(audio_bytes)
                    tmp.flush()

                    model = self._get_whisper_model()
                    transcribe_kwargs: dict[str, Any] = {
                        "beam_size": 5,
                        "task": "transcribe",
                    }
                    if self.stt_language and self.stt_language != "auto":
                        transcribe_kwargs["language"] = self.stt_language
                    segments, _ = model.transcribe(tmp.name, **transcribe_kwargs)
                    return " ".join(seg.text.strip() for seg in segments).strip()

            try:
                transcript = await asyncio.to_thread(_transcribe_local)
            except ModuleNotFoundError as exc:
                logger.warning(
                    "Local faster-whisper unavailable; falling back to API STT: %s",
                    exc,
                )
                transcript = await self._speech_to_text_api(audio_bytes, content_type)

        if not transcript:
            raise ValueError("Whisper returned an empty transcript.")

        logger.info("STT transcript (%d chars): %.80s...", len(transcript), transcript)
        return transcript

    # -- TTS: MiniMax T2A v2 -------------------------------------------------

    async def text_to_speech(self, text: str) -> bytes:
        """Convert text to audio using MiniMax T2A v2 API.

        Parameters
        ----------
        text : str
            The text to synthesize into speech.

        Returns
        -------
        bytes
            Raw audio bytes (MP3 format by default).
        """
        if not self.minimax_api_key:
            raise ValueError(
                "Missing MINIMAX_API_KEY -- required for text-to-speech."
            )
        if not self.minimax_group_id:
            raise ValueError(
                "Missing MINIMAX_GROUP_ID -- required for text-to-speech."
            )

        url = f"https://api.minimax.io/v1/t2a_v2?GroupId={self.minimax_group_id}"
        payload = {
            "model": self.tts_model,
            "text": text,
            "stream": False,
            "voice_setting": {
                "voice_id": self.tts_voice_id,
                "speed": self.tts_speed,
                "vol": 1.0,
                "pitch": 0,
                "emotion": self.tts_emotion,
            },
            "audio_setting": {
                "sample_rate": self.tts_sample_rate,
                "bitrate": 128000,
                "format": self.tts_format,
                "channel": 1,
            },
            "output_format": "hex",
        }

        async with httpx.AsyncClient(timeout=self.tts_timeout) as client:
            response = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.minimax_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

        if response.status_code != 200:
            error_detail = response.text[:300]
            logger.error("MiniMax TTS failed (HTTP %d): %s", response.status_code, error_detail)
            raise RuntimeError(
                f"MiniMax TTS failed (HTTP {response.status_code}): {error_detail}"
            )

        result = response.json()
        base_resp = result.get("base_resp", {})
        if base_resp.get("status_code", -1) != 0:
            error_msg = base_resp.get("status_msg", "Unknown MiniMax error")
            raise RuntimeError(f"MiniMax TTS error: {error_msg}")

        audio_hex = result.get("data", {}).get("audio", "")
        if not audio_hex:
            raise ValueError("MiniMax TTS returned empty audio data.")

        audio_bytes = bytes.fromhex(audio_hex)
        logger.info(
            "TTS generated %d bytes (%s) for %d chars of text",
            len(audio_bytes),
            self.tts_format,
            len(text),
        )
        return audio_bytes
