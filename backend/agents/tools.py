"""Agent tools for the AI tutoring platform.

Provides the three BaseTool classes used by RagAgent for knowledge-base
retrieval and document management.  These tools are shared across all
calling agents (Professor, TA, Manager) -- no agent-specific logic here.

Document uploads are handled directly by the FastAPI layer (not the agent).

Tools
-----
RetrieveContextTool      - Semantic search via local FAISS (S3 -> PDF -> embeddings)
CheckIngestionStatusTool - Poll ingestion job progress
ListDocumentsTool        - List available documents (metadata only)
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Union

import boto3
import pdfplumber
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from db.models import (
    Citation,
    ProfessorMode,
    ProfessorNextAction,
    ProfessorTurnRequest,
    ProfessorTurnResponse,
    ProfessorTurnStrategy,
)
from utils.helpers import parse_tool_input

logger = logging.getLogger(__name__)

PROFESSOR_CONFIG_PATH = "agents.professor"


# ---------------------------------------------------------------------------
# 0. ProfessorRespondTool
# ---------------------------------------------------------------------------

class ProfessorRespondTool(BaseTool):
    """Professor tutoring tool with strict request/response schema handling."""

    name: str = "professor_respond"
    description: str = (
        "Generate one tutoring response turn for the Professor agent.\n"
        "\n"
        "INPUT (JSON object):\n"
        "  session_id (str, REQUIRED)\n"
        "  message    (str, REQUIRED)\n"
        "  topic      (str, REQUIRED)\n"
        "  mode       (str, optional) - strict | convenience (default: strict)\n"
        "  profile    (object, REQUIRED) - level/learning_style/pace\n"
        "\n"
        "OUTPUT (JSON object):\n"
        "  assistant_response (str)\n"
        "  strategy           (str)\n"
        "  revealed_final_answer (false)\n"
        "  next_action        (str)\n"
        "  citations          (list)\n"
        "\n"
        "Always returns strict JSON matching ProfessorTurnResponse schema."
    )

    professor_config: Any = None
    citations_enabled: bool = True
    socratic_default: bool = True
    llm_provider: str = ""
    llm_model_id: str = ""
    llm_temperature: float = 0.3
    llm_max_tokens: int = 800
    featherless_base_url: str = "https://api.featherless.ai/v1"
    featherless_api_key_env: str = "FEATHERLESS_API_KEY"
    featherless_user_agent: str = "ceChillHackers-professor/1.0"
    runtime_timeout_sec: int = 30
    runtime_retries: int = 0
    fallback_on_live_error: bool = False
    use_live_featherless: bool = False
    featherless_api_key: str = ""
    system_prompt_override: str = ""

    def __init__(self, config: Any, system_prompt: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.professor_config = self._load_professor_runtime_config(config)
        llm_cfg = self.professor_config.get("llm", {})
        runtime_cfg = self.professor_config.get("runtime", {})
        tutoring_cfg = self.professor_config.get("tutoring", {})
        provider_cfg = config.get("providers.featherless", {})
        if not isinstance(provider_cfg, dict):
            provider_cfg = {}

        self.llm_provider = str(llm_cfg.get("provider", ""))
        self.llm_model_id = str(llm_cfg.get("model_id", ""))
        self.llm_temperature = float(llm_cfg.get("temperature", 0.3))
        self.llm_max_tokens = int(llm_cfg.get("max_tokens", 800))
        self.featherless_base_url = str(
            provider_cfg.get("base_url", "https://api.featherless.ai/v1")
        ).rstrip("/")
        self.featherless_api_key_env = str(
            provider_cfg.get("api_key_env", "FEATHERLESS_API_KEY")
        )
        self.featherless_user_agent = str(
            provider_cfg.get("user_agent", "ceChillHackers-professor/1.0")
        ).strip() or "ceChillHackers-professor/1.0"
        self.runtime_timeout_sec = int(
            runtime_cfg.get("timeout_sec", provider_cfg.get("timeout_sec", 30))
        )
        self.runtime_retries = max(0, int(runtime_cfg.get("retries", 0)))
        self.fallback_on_live_error = self._as_bool(
            runtime_cfg.get("fallback_on_live_error", False)
        )

        self.citations_enabled = bool(tutoring_cfg.get("citations_enabled", True))
        self.socratic_default = bool(tutoring_cfg.get("socratic_default", True))
        live_from_cfg = self._as_bool(runtime_cfg.get("use_live_featherless", False))
        live_from_env = self._as_bool(
            os.getenv("PROFESSOR_USE_LIVE_FEATHERLESS", "false")
        )
        self.use_live_featherless = (
            self.llm_provider in {"featherless_openai", "featherless"}
            and (live_from_cfg or live_from_env)
        )
        self.featherless_api_key = os.getenv(self.featherless_api_key_env, "")
        self.system_prompt_override = system_prompt.strip()

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _load_professor_runtime_config(config: Any) -> dict[str, Any]:
        professor_config = config.get(PROFESSOR_CONFIG_PATH)
        if not isinstance(professor_config, dict):
            raise ValueError(
                f"Missing or invalid config mapping at '{PROFESSOR_CONFIG_PATH}'"
            )

        llm_cfg = professor_config.get("llm")
        tutoring_cfg = professor_config.get("tutoring")
        if not isinstance(llm_cfg, dict):
            raise ValueError("Missing or invalid config mapping at 'agents.professor.llm'")
        if not isinstance(tutoring_cfg, dict):
            raise ValueError("Missing or invalid config mapping at 'agents.professor.tutoring'")
        if not llm_cfg.get("provider"):
            raise ValueError("Missing required config key: 'agents.professor.llm.provider'")
        if not llm_cfg.get("model_id"):
            raise ValueError("Missing required config key: 'agents.professor.llm.model_id'")
        return professor_config

    @staticmethod
    def map_professor_mode_to_rag_mode(mode: ProfessorMode) -> str:
        """Normalize Professor mode to the equivalent RAG mode."""
        if mode is ProfessorMode.STRICT:
            return "internal_only"
        return "external_ok"

    @staticmethod
    def sanitize_for_log(request: ProfessorTurnRequest) -> dict[str, Any]:
        """Return metadata-only logs (no raw student message)."""
        return {
            "session_id": request.session_id,
            "mode": request.mode.value,
            "rag_mode": ProfessorRespondTool.map_professor_mode_to_rag_mode(request.mode),
            "topic": request.topic,
            "message_length": len(request.student_message),
            "profile_level": request.profile.level,
        }

    @staticmethod
    def get_professor_json_schemas() -> dict[str, dict[str, Any]]:
        """Expose strict transport schemas for validation in caller layers."""
        return {
            "ProfessorTurnRequest": ProfessorTurnRequest.model_json_schema(),
            "ProfessorTurnResponse": ProfessorTurnResponse.model_json_schema(),
        }

    def _retrieve_citations(self, request: ProfessorTurnRequest) -> list[Citation]:
        """Return citations from RAG when wired; empty list for now."""
        if not self.citations_enabled:
            return []
        # Do not fabricate citations. Until RAG retrieval is connected,
        # return an empty list instead of synthetic sources.
        _ = self.map_professor_mode_to_rag_mode(request.mode)
        return []

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        content = text.strip()
        if not content.startswith("```"):
            return content
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline + 1 :]
        if content.rstrip().endswith("```"):
            content = content.rstrip()[:-3].rstrip()
        return content.strip()

    @staticmethod
    def _extract_chat_completion_text(response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return ""

        message = first_choice.get("message")
        if not isinstance(message, dict):
            return ""

        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            return "\n".join(parts).strip()
        return ""

    @staticmethod
    def _parse_json_object_from_text(text: str) -> dict[str, Any]:
        cleaned = ProfessorRespondTool._strip_code_fences(text)

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
            raise TypeError("Featherless response JSON must be an object")
        except json.JSONDecodeError:
            pass

        text_len = len(cleaned)
        for start_idx, char in enumerate(cleaned):
            if char != "{":
                continue

            depth = 0
            in_string = False
            escaped = False
            for end_idx in range(start_idx, text_len):
                current = cleaned[end_idx]
                if in_string:
                    if escaped:
                        escaped = False
                    elif current == "\\":
                        escaped = True
                    elif current == '"':
                        in_string = False
                    continue

                if current == '"':
                    in_string = True
                    continue
                if current == "{":
                    depth += 1
                    continue
                if current == "}":
                    depth -= 1
                    if depth != 0:
                        continue
                    candidate = cleaned[start_idx : end_idx + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    raise TypeError("Featherless response JSON must be an object")

        raise ValueError("Featherless response did not contain a valid JSON object")

    @staticmethod
    def _normalize_live_response_payload(
        parsed: dict[str, Any],
        request: ProfessorTurnRequest,
        selected_strategy: ProfessorTurnStrategy,
        *,
        enforce_selected_strategy: bool,
    ) -> dict[str, Any]:
        strategy_aliases = {
            "socratic question": "socratic_question",
            "socratic-question": "socratic_question",
            "question": "socratic_question",
            "encourage": "socratic_question",
            "encouragement": "socratic_question",
            "explain": "conceptual_explanation",
            "concept explain": "conceptual_explanation",
            "concept_explain": "conceptual_explanation",
            "conceptual explanation": "conceptual_explanation",
            "step by step": "procedural_explanation",
            "step-by-step": "procedural_explanation",
            "walkthrough": "procedural_explanation",
            "procedural explanation": "procedural_explanation",
            "procedural_explanation": "procedural_explanation",
            "hint": "broken_down_questions",
            "breakdown": "broken_down_questions",
            "break down": "broken_down_questions",
            "scaffold": "broken_down_questions",
            "scaffolded_subproblem": "broken_down_questions",
            "broken down questions": "broken_down_questions",
            "misconception": "conceptual_explanation",
            "misconception_repair": "conceptual_explanation",
        }
        next_action_aliases = {
            "route_ta": "route_problem_ta",
            "route problem ta": "route_problem_ta",
            "route planner": "route_planner",
            "keep_going": "continue",
            "continue_tutoring": "continue",
        }

        strategy_raw = str(parsed.get("strategy", "")).strip().lower()
        strategy_raw = strategy_aliases.get(strategy_raw, strategy_raw)
        if strategy_raw not in {
            "socratic_question",
            "conceptual_explanation",
            "procedural_explanation",
            "broken_down_questions",
        }:
            strategy_raw = selected_strategy.value
        if enforce_selected_strategy:
            strategy_raw = selected_strategy.value

        next_action_raw = str(parsed.get("next_action", "")).strip().lower()
        next_action_raw = next_action_aliases.get(next_action_raw, next_action_raw)
        if next_action_raw not in {
            "continue",
            "route_problem_ta",
            "route_planner",
        }:
            next_action_raw = "continue"

        assistant_response = str(parsed.get("assistant_response", "")).strip()
        if not assistant_response:
            assistant_response = (
                f"In {request.topic}, start from the core concept first. "
                "Then apply that concept in one small step to your specific question."
            )

        return {
            "assistant_response": assistant_response,
            "strategy": strategy_raw,
            "revealed_final_answer": False,
            "next_action": next_action_raw,
            # RAG citations are intentionally disabled until RAG is fully wired.
            "citations": [],
        }

    @staticmethod
    def _normalize_text_for_compare(text: str) -> str:
        return " ".join(text.strip().lower().split())

    @staticmethod
    def _assistant_response_style_issue(
        assistant_response: str,
        request: ProfessorTurnRequest,
        strategy: ProfessorTurnStrategy,
    ) -> str | None:
        normalized_response = ProfessorRespondTool._normalize_text_for_compare(
            assistant_response
        )
        if not normalized_response:
            return "assistant_response_empty"

        normalized_student = ProfessorRespondTool._normalize_text_for_compare(
            request.student_message
        )
        if normalized_response == normalized_student:
            return "assistant_response_echoes_student_message"

        strategy_labels = {
            "socratic_question",
            "conceptual_explanation",
            "procedural_explanation",
            "broken_down_questions",
            "socratic question",
            "conceptual explanation",
            "procedural explanation",
            "broken down questions",
        }
        if normalized_response in strategy_labels:
            return "assistant_response_is_strategy_label"

        if len(normalized_response.split()) < 6:
            return "assistant_response_too_short"

        question_count = assistant_response.count("?")
        has_sequence_marker = any(
            marker in normalized_response
            for marker in (
                "step 1",
                "step one",
                "first",
                "next",
                "then",
                "finally",
            )
        )
        has_example_signal = any(
            marker in normalized_response
            for marker in (
                "for example",
                "example:",
                "imagine",
                "think of",
            )
        )
        has_explanation_signal = any(
            marker in normalized_response
            for marker in (
                "means",
                "because",
                "is when",
                "works by",
                "the idea",
                "core concept",
            )
        )
        all_question_sentences = (
            "?" in assistant_response
            and assistant_response.replace("?", "").strip() != ""
            and assistant_response.strip().endswith("?")
            and question_count >= 1
            and "." not in assistant_response
        )

        if strategy is ProfessorTurnStrategy.CONCEPTUAL_EXPLANATION:
            if all_question_sentences:
                return "conceptual_explanation_is_question_only"
            if not (has_explanation_signal or has_example_signal or "." in assistant_response):
                return "conceptual_explanation_missing_explanatory_anchor"
        elif strategy is ProfessorTurnStrategy.PROCEDURAL_EXPLANATION:
            if not has_sequence_marker:
                return "procedural_explanation_missing_step_sequence"
        elif strategy is ProfessorTurnStrategy.BROKEN_DOWN_QUESTIONS:
            if question_count < 2:
                return "broken_down_questions_needs_multiple_guiding_questions"
        elif strategy is ProfessorTurnStrategy.SOCRATIC_QUESTION:
            if question_count < 1:
                return "socratic_question_missing_question"
            if request.profile.level != "advanced" and all_question_sentences:
                return "socratic_question_overused_for_non_advanced_profile"

        return None

    def _request_live_chat_completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        response_data: dict[str, Any] | None = None
        for attempt in range(self.runtime_retries + 1):
            http_request = urllib.request.Request(
                url=f"{self.featherless_base_url}/chat/completions",
                data=json.dumps(request_payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.featherless_api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": self.featherless_user_agent,
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    http_request, timeout=self.runtime_timeout_sec
                ) as response:
                    response_data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="ignore")
                is_retryable = exc.code in {429, 500, 502, 503, 504}
                if is_retryable and attempt < self.runtime_retries:
                    time.sleep(min(0.2 * (2 ** attempt), 1.0))
                    continue
                raise RuntimeError(
                    f"Featherless HTTP {exc.code}: {error_body[:200]}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < self.runtime_retries:
                    time.sleep(min(0.2 * (2 ** attempt), 1.0))
                    continue
                raise RuntimeError(f"Featherless connection error: {exc.reason}") from exc

        if response_data is None:
            raise RuntimeError("Featherless request failed before receiving a response")
        return response_data

    def _choose_strategy(self, request: ProfessorTurnRequest) -> ProfessorTurnStrategy:
        text = request.student_message.strip().lower()
        asks_for_procedure_signals = (
            "step by step",
            "step-by-step",
            "procedure",
            "algorithm",
            "how do i solve",
            "process",
            "walk me through",
        )
        stuck_signals = (
            "break down",
            "simpler",
            "easier",
            "small example",
            "stuck",
            "don't know where to start",
            "dont know where to start",
            "mixing up",
            "confuse",
            "confused",
        )
        concept_signals = (
            "why",
            "intuition",
            "concept",
            "meaning",
            "understand",
            "what is",
        )
        asks_for_procedure = any(signal in text for signal in asks_for_procedure_signals)
        is_stuck = any(signal in text for signal in stuck_signals)
        asks_for_concept = any(signal in text for signal in concept_signals)

        profile = request.profile
        level = profile.level
        learning_style = profile.learning_style
        pace = profile.pace

        # Advanced textual/mixed learners can default to Socratic probing.
        if level == "advanced" and learning_style in {"textual", "mixed"}:
            if asks_for_procedure:
                return ProfessorTurnStrategy.PROCEDURAL_EXPLANATION
            if asks_for_concept:
                return ProfessorTurnStrategy.CONCEPTUAL_EXPLANATION
            return ProfessorTurnStrategy.SOCRATIC_QUESTION

        # Beginners or slower pace benefit from guided procedure and breakdowns.
        if level == "beginner" or pace == "slow":
            if is_stuck:
                return ProfessorTurnStrategy.BROKEN_DOWN_QUESTIONS
            if learning_style == "example_first" or asks_for_procedure:
                return ProfessorTurnStrategy.PROCEDURAL_EXPLANATION
            return ProfessorTurnStrategy.CONCEPTUAL_EXPLANATION

        # Intermediate defaults to concept-first, with scaffolding if student is stuck.
        if level == "intermediate":
            if is_stuck:
                return ProfessorTurnStrategy.BROKEN_DOWN_QUESTIONS
            if asks_for_procedure and learning_style == "example_first":
                return ProfessorTurnStrategy.PROCEDURAL_EXPLANATION
            return ProfessorTurnStrategy.CONCEPTUAL_EXPLANATION

        # Remaining advanced styles (e.g. visual/example_first) get concept/procedure focus.
        if asks_for_procedure:
            return ProfessorTurnStrategy.PROCEDURAL_EXPLANATION
        if is_stuck:
            return ProfessorTurnStrategy.BROKEN_DOWN_QUESTIONS
        if asks_for_concept:
            return ProfessorTurnStrategy.CONCEPTUAL_EXPLANATION
        if self.socratic_default and level == "advanced":
            return ProfessorTurnStrategy.SOCRATIC_QUESTION
        return ProfessorTurnStrategy.CONCEPTUAL_EXPLANATION

    def _build_live_response(self, request: ProfessorTurnRequest) -> ProfessorTurnResponse:
        if not self.featherless_api_key:
            raise RuntimeError(
                f"Missing API key env var: {self.featherless_api_key_env}"
            )

        selected_strategy = self._choose_strategy(request)
        rag_mode = self.map_professor_mode_to_rag_mode(request.mode)
        default_system_prompt = (
            "You are a concept-first tutoring Professor agent. "
            "Adapt explanations to learner profile (level, learning_style, pace). "
            "Return ONLY a strict JSON object matching the requested schema. "
            "Never reveal the final answer directly."
        )
        system_prompt = self.system_prompt_override or default_system_prompt
        user_prompt = (
            "Return JSON with keys: assistant_response, strategy, "
            "revealed_final_answer, next_action, citations.\n"
            "Allowed strategy values: conceptual_explanation, procedural_explanation, "
            "broken_down_questions, socratic_question.\n"
            "Allowed next_action values: continue, route_problem_ta, route_planner.\n"
            "Rules:\n"
            "- Start with a brief conceptual explanation before guidance.\n"
            "- revealed_final_answer must be false.\n"
            "- citations must be an array (can be empty).\n"
            "- Do not include markdown, code fences, or extra keys.\n"
            "- Do not return question-only output unless strategy is socratic_question.\n"
            f"- selected_strategy={selected_strategy.value}\n"
            f"- mode={request.mode.value}, rag_mode={rag_mode}\n"
            f"- student_level={request.profile.level}, learning_style={request.profile.learning_style}, pace={request.profile.pace}\n"
            f"- topic={request.topic}\n"
            f"- student_message={request.student_message}\n"
        )
        retry_user_prompt = (
            user_prompt
            + "Validation failed on your last response. Fix all issues now:\n"
            "- assistant_response must not echo the student's message.\n"
            "- assistant_response must not be a strategy label.\n"
            "- assistant_response must match the selected strategy style.\n"
            "- assistant_response must stay concept-first and profile-adaptive.\n"
            "- Return ONLY JSON.\n"
        )

        request_payload = {
            "model": self.llm_model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
            "response_format": {"type": "json_object"},
        }

        semantic_errors: list[str] = []
        for semantic_attempt in range(2):
            if semantic_attempt == 1:
                request_payload["messages"][1]["content"] = retry_user_prompt

            try:
                response_data = self._request_live_chat_completion(request_payload)
                raw_text = self._extract_chat_completion_text(response_data)
                if not raw_text:
                    raise ValueError("Featherless returned empty response text")

                parsed = self._parse_json_object_from_text(raw_text)
                normalized = self._normalize_live_response_payload(
                    parsed,
                    request,
                    selected_strategy=selected_strategy,
                    enforce_selected_strategy=True,
                )
                style_issue = self._assistant_response_style_issue(
                    normalized["assistant_response"],
                    request,
                    ProfessorTurnStrategy(normalized["strategy"]),
                )
                if style_issue:
                    raise ValueError(style_issue)

                return ProfessorTurnResponse.model_validate(normalized)
            except (ValidationError, TypeError, ValueError) as exc:
                semantic_errors.append(str(exc))
                if semantic_attempt == 0:
                    logger.warning(
                        "Professor live Featherless semantic validation failed once; retrying: %s",
                        str(exc)[:200],
                    )
                    continue
                raise ValueError(
                    "Featherless response failed semantic validation after retry: "
                    + "; ".join(semantic_errors[-2:])
                ) from exc

        raise RuntimeError("Featherless semantic validation loop exhausted unexpectedly")

    def _build_response(self, request: ProfessorTurnRequest) -> ProfessorTurnResponse:
        if self.use_live_featherless:
            try:
                return self._build_live_response(request)
            except Exception as exc:
                if not self.fallback_on_live_error:
                    raise
                logger.warning(
                    "Professor live Featherless call failed; using deterministic fallback: %s: %s",
                    type(exc).__name__,
                    str(exc)[:200],
                )

        strategy = self._choose_strategy(request)
        citations = self._retrieve_citations(request)
        if strategy is ProfessorTurnStrategy.SOCRATIC_QUESTION:
            response_text = (
                f"In {request.topic}, the key idea is connecting principle to method. "
                "Given your current setup, what assumption are you making first and why?"
            )
        elif strategy is ProfessorTurnStrategy.CONCEPTUAL_EXPLANATION:
            response_text = (
                f"In {request.topic}, start from the core concept and what it means in plain language. "
                "Once that idea is clear, apply it to your exact question one step at a time."
            )
        elif strategy is ProfessorTurnStrategy.PROCEDURAL_EXPLANATION:
            response_text = (
                f"In {request.topic}, Step 1 is to identify the knowns and unknowns. "
                "Step 2 is choose the governing rule. Step 3 is apply it on a small example before your full problem."
            )
        else:
            response_text = (
                f"The core concept in {request.topic} stays the same, so let's break your task into smaller checks. "
                "What is the first quantity you can define? What relation connects it to the target quantity?"
            )

        return ProfessorTurnResponse(
            assistant_response=response_text,
            strategy=strategy,
            revealed_final_answer=False,
            next_action=ProfessorNextAction.CONTINUE,
            citations=citations,
        )

    def _run(self, tool_input: Union[str, dict]) -> str:
        params = parse_tool_input(tool_input)
        request = ProfessorTurnRequest.model_validate(params)
        response = self._build_response(request)
        return response.model_dump_json()


# ---------------------------------------------------------------------------
# LocalVectorStore — S3 download → PDF parse → chunk → embed → FAISS
# ---------------------------------------------------------------------------

class LocalVectorStore:
    """Downloads PDFs from S3, parses, chunks, embeds, and provides FAISS search.

    Built once at tool init time. Supports subject-filtered or full-corpus indexing.
    """

    def __init__(
        self,
        s3_client: Any,
        bucket: str,
        docs_prefix: str,
        embedding_model_name: str,
        chunk_size: int,
        chunk_overlap: int,
        cache_dir: str,
    ) -> None:
        self.s3 = s3_client
        self.bucket = bucket
        self.docs_prefix = docs_prefix
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
        )

        # Cache: {subject_or_"__all__": (FAISS, set_of_s3_keys)}
        self._indexes: dict[str, tuple[FAISS, set[str]]] = {}

    def _list_s3_pdfs(self, prefix: str) -> list[str]:
        """List all PDF keys under a given S3 prefix."""
        keys: list[str] = []
        continuation_token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token
            resp = self.s3.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                if obj["Key"].lower().endswith(".pdf"):
                    keys.append(obj["Key"])
            if not resp.get("IsTruncated"):
                break
            continuation_token = resp.get("NextContinuationToken")
        return keys

    def _download_pdf(self, s3_key: str) -> Path:
        """Download PDF to local cache, skip if already cached."""
        safe_name = s3_key.replace("/", "_")
        local_path = self.cache_dir / safe_name
        if not local_path.exists():
            logger.info("Downloading %s from S3...", s3_key)
            self.s3.download_file(self.bucket, s3_key, str(local_path))
        return local_path

    def _parse_and_chunk(self, pdf_path: Path, s3_key: str) -> list[Document]:
        """Extract text page-by-page, then chunk with page number tracking.

        Each chunk is prefixed with ``[doc_name | Page N]`` so that the
        embedding model can distinguish document types (e.g. "Tutorial"
        vs "Lecture") — metadata alone is invisible to FAISS similarity
        search.
        """
        documents: list[Document] = []
        doc_name = s3_key.split("/")[-1]

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    continue

                metadata_prefix = f"[{doc_name} | Page {page_num}]\n"

                for chunk in self._chunk_text(text):
                    documents.append(Document(
                        page_content=f"{metadata_prefix}{chunk}",
                        metadata={
                            "source": s3_key,
                            "doc_name": doc_name,
                            "page_number": page_num,
                        },
                    ))
        return documents

    def _chunk_text(self, text: str) -> list[str]:
        """Paragraph/sentence-aware chunking via RecursiveCharacterTextSplitter.

        Splits on paragraph breaks first, then sentences, then words —
        keeps math problems and solution pairs intact instead of cutting
        mid-sentence at a fixed character offset.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]

    def get_or_build_index(self, subject: str = "") -> FAISS | None:
        """Return a FAISS index for the given subject (or all docs).

        Builds on first call. Auto-rebuilds when S3 contents change
        (new uploads or deletions detected).
        """
        cache_key = subject or "__all__"

        prefix = f"{self.docs_prefix}{subject}/" if subject else self.docs_prefix
        pdf_keys = self._list_s3_pdfs(prefix)

        # Fall back to all docs if subject-specific search finds nothing
        if not pdf_keys and subject:
            logger.info("No PDFs under '%s', falling back to all docs", prefix)
            pdf_keys = self._list_s3_pdfs(self.docs_prefix)

        if not pdf_keys:
            return None

        current_keys = set(pdf_keys)

        # Use cache if S3 contents haven't changed
        if cache_key in self._indexes:
            cached_index, cached_keys = self._indexes[cache_key]
            if cached_keys == current_keys:
                return cached_index
            logger.info("S3 contents changed for '%s', rebuilding FAISS index", cache_key)

        all_docs: list[Document] = []
        for s3_key in pdf_keys:
            local_path = self._download_pdf(s3_key)
            chunks = self._parse_and_chunk(local_path, s3_key)
            all_docs.extend(chunks)
            logger.info("Parsed %s: %d chunks", s3_key, len(chunks))

        if not all_docs:
            return None

        index = FAISS.from_documents(all_docs, self.embeddings)
        self._indexes[cache_key] = (index, current_keys)
        logger.info("FAISS index built: %d chunks for '%s'", len(all_docs), cache_key)
        return index


# ---------------------------------------------------------------------------
# RetrieveContextTool — semantic search via local FAISS
# ---------------------------------------------------------------------------

class RetrieveContextTool(BaseTool):
    """Semantic search via local FAISS vector store (S3 -> PDF -> embeddings).

    Returns ranked, citation-annotated text chunks -- never raw full
    documents.  Used by every calling agent for different purposes:
    - Professor Agent: concept explanations, lecture material, examples
    - TA Agent: practice problems, worked solutions, difficulty context
    - Manager Agent: content verification before document operations
    """

    name: str = "retrieve_context"
    description: str = (
        "Semantic search over the Knowledge Base. "
        "Call this for ANY content question — concepts, problems, examples, lecture material. "
        "Always call before answering; never rely on memory alone. "
        "Args: query (str, required), subject (str, optional — course filter), top_k (int, optional). "
        "Returns: {found, context (numbered [1]...[n] chunks), citations [{index, doc, page}]}."
    )
    top_k: Any = None
    vector_store: Any = None

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.top_k = config.rag.top_k

        s3 = boto3.client("s3", region_name=config.aws.region)
        self.vector_store = LocalVectorStore(
            s3_client=s3,
            bucket=config.aws.s3_bucket,
            docs_prefix=config.get("rag.s3_docs_prefix", "docs/"),
            embedding_model_name=config.get("rag.embedding_model", "all-MiniLM-L6-v2"),
            chunk_size=int(config.get("rag.chunk_size", 500)),
            chunk_overlap=int(config.get("rag.chunk_overlap", 100)),
            cache_dir=config.get("rag.cache_dir", "/tmp/cechillhackers_cache"),
        )

    def _run(self, tool_input: Union[str, dict]) -> str:
        """Execute retrieval against local FAISS vector store."""
        params = parse_tool_input(tool_input)
        query: str = params.get("query", "")
        subject: str = params.get("subject", "")
        top_k: int = int(params.get("top_k", self.top_k))

        if not query.strip():
            return json.dumps({"context": "", "citations": [], "found": False,
                               "error": "Empty query -- provide a search query."})

        try:
            index = self.vector_store.get_or_build_index(subject)
        except Exception as exc:
            logger.error("Failed to build vector index: %s", exc)
            return json.dumps({"context": "", "citations": [], "found": False,
                               "error": f"Index build error: {exc}"})

        if index is None:
            return json.dumps({"context": "", "citations": [], "found": False})

        try:
            results = index.similarity_search_with_score(query, k=top_k)
        except Exception as exc:
            logger.error("FAISS search failed: %s", exc)
            return json.dumps({"context": "", "citations": [], "found": False,
                               "error": f"Search error: {exc}"})

        if not results:
            return json.dumps({"context": "", "citations": [], "found": False})

        context_parts: list[str] = []
        citations: list[dict[str, Any]] = []
        for idx, (doc, l2_dist) in enumerate(results, start=1):
            metadata = doc.metadata or {}
            doc_name = metadata.get("doc_name", "unknown")
            page = metadata.get("page_number", "?")

            # Convert squared L2 distance to cosine similarity.
            # FAISS returns d_sq = ||a - b||^2 (already squared).
            # For normalized vectors: d_sq = 2(1 - cos_sim)
            # So cos_sim = 1 - d_sq / 2
            cos_sim = max(0.0, 1.0 - float(l2_dist) / 2.0)

            context_parts.append(f"[{idx}] {doc.page_content.strip()}")
            citation: dict[str, Any] = {
                "index": idx,
                "doc": doc_name,
                "page": page,
                "score": round(cos_sim, 4),
            }
            citations.append(citation)

        return json.dumps({
            "context": "\n\n".join(context_parts),
            "citations": citations,
            "found": True,
        })


# ---------------------------------------------------------------------------
# CheckIngestionStatusTool
# ---------------------------------------------------------------------------

class CheckIngestionStatusTool(BaseTool):
    """Poll the ingestion status of a previously uploaded document."""

    name: str = "check_ingestion_status"
    description: str = (
        "Poll the sync status of a document after it has been uploaded to S3. "
        "Call this with an ingestion_job_id; repeat until status=COMPLETE or FAILED. "
        "Args: ingestion_job_id (str, required). "
        "Returns: {status (STARTING|IN_PROGRESS|COMPLETE|FAILED), indexed_count, failed_count} or {error}."
    )

    kb_id: Any = None
    ds_id: Any = None
    bedrock: Any = None

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.kb_id = config.bedrock.knowledge_base_id
        self.ds_id = config.bedrock.data_source_id
        self.bedrock = boto3.client("bedrock-agent", region_name=config.aws.region)

    def _run(self, tool_input: Union[str, dict]) -> str:
        """Query Bedrock for ingestion job status."""
        params = parse_tool_input(tool_input)
        ingestion_job_id: str = params.get("ingestion_job_id", "")

        if not ingestion_job_id:
            return json.dumps({"error": "Missing required field: ingestion_job_id"})

        try:
            res = self.bedrock.get_ingestion_job(
                knowledgeBaseId=self.kb_id,
                dataSourceId=self.ds_id,
                ingestionJobId=ingestion_job_id,
            )
            job = res["ingestionJob"]
            stats = job.get("statistics", {})

            return json.dumps({
                "status": job["status"],
                "indexed_count": stats.get("numberOfDocumentsIndexed", 0),
                "failed_count": stats.get("numberOfDocumentsFailed", 0),
            })

        except Exception as exc:
            logger.error("Ingestion status check failed for job '%s': %s",
                         ingestion_job_id, exc)
            return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# ListDocumentsTool
# ---------------------------------------------------------------------------

class ListDocumentsTool(BaseTool):
    """List documents available in the Knowledge Base -- metadata only."""

    name: str = "list_available_documents"
    description: str = (
        "List documents stored in the Knowledge Base (metadata only, no content). "
        "Call this to see what course material exists or to verify a recent upload. "
        "Args: subject (str, optional — omit to list all subjects). "
        "Returns: {documents [{filename, s3_key, last_modified, size_kb}], total} or {error}."
    )

    bucket: Any = None
    s3: Any = None

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.bucket = config.aws.s3_bucket
        self.s3 = boto3.client("s3", region_name=config.aws.region)

    def _run(self, tool_input: Union[str, dict]) -> str:
        """List S3 objects under the docs/ prefix, returning metadata only."""
        params = parse_tool_input(tool_input)
        subject: str = params.get("subject", "")

        prefix = f"docs/{subject}/" if subject else "docs/"

        try:
            documents: list[dict[str, Any]] = []
            continuation_token: str | None = None

            while True:
                list_kwargs: dict[str, Any] = {
                    "Bucket": self.bucket,
                    "Prefix": prefix,
                }
                if continuation_token:
                    list_kwargs["ContinuationToken"] = continuation_token

                response = self.s3.list_objects_v2(**list_kwargs)

                for obj in response.get("Contents", []):
                    key: str = obj["Key"]
                    if key.endswith("/"):
                        continue
                    documents.append({
                        "filename": key.split("/")[-1],
                        "s3_key": key,
                        "last_modified": obj["LastModified"].isoformat(),
                        "size_kb": round(obj["Size"] / 1024, 1),
                    })

                if not response.get("IsTruncated"):
                    break
                continuation_token = response.get("NextContinuationToken")

            return json.dumps({"documents": documents, "total": len(documents)})

        except Exception as exc:
            logger.error("Failed to list documents (prefix='%s'): %s", prefix, exc)
            return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# ExaWebSearchTool — web search via Exa API for EXTERNAL_OK mode
# ---------------------------------------------------------------------------

class ExaWebSearchTool(BaseTool):
    """Web search via Exa API for supplementing KB content in EXTERNAL_OK mode.

    Caller-aware: professor queries target concepts/theory, TA queries
    target problems/exercises. Level-aware: query complexity adjusts
    to beginner/intermediate/advanced.
    """

    name: str = "exa_web_search"
    description: str = (
        "Search the web via Exa for educational content when the Knowledge Base "
        "is insufficient. Returns web results with text highlights and URLs. "
        "Args: query (str, required), caller (str), level (str), subject (str). "
        "Returns: {found, context, web_citations [{index, title, url, snippet}]}."
    )

    exa_client: Any = None
    num_results: int = 5

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from exa_py import Exa

        api_key = os.environ.get("EXA_API_KEY", "")
        if not api_key:
            api_key = config.get("exa.api_key", "")
        if not api_key:
            raise ValueError(
                "Missing EXA_API_KEY in environment or config -- "
                "add it to your .env file or config.yaml."
            )
        self.exa_client = Exa(api_key=api_key)
        self.num_results = int(config.get("exa.num_results", 5))

    def _run(self, tool_input: Union[str, dict]) -> str:
        """Execute web search via Exa, returning formatted context + citations."""
        params = parse_tool_input(tool_input)
        query: str = params.get("query", "")
        caller: str = params.get("caller", "")
        level: str = params.get("level", "intermediate")
        subject: str = params.get("subject", "")

        if not query.strip():
            return json.dumps({"context": "", "web_citations": [], "found": False,
                               "error": "Empty query."})

        search_query = self._build_search_query(query, caller, level, subject)

        try:
            response = self.exa_client.search(
                search_query,
                type="auto",
                num_results=self.num_results,
                contents={
                    "text": {"max_characters": 2000},
                    "highlights": {
                        "query": query,
                        "num_sentences": 5,
                    },
                },
            )
        except Exception as exc:
            logger.error("Exa search failed: %s", exc)
            return json.dumps({"context": "", "web_citations": [], "found": False,
                               "error": f"Web search error: {exc}"})

        if not response.results:
            return json.dumps({"context": "", "web_citations": [], "found": False})

        context_parts: list[str] = []
        web_citations: list[dict[str, Any]] = []

        for idx, result in enumerate(response.results, start=1):
            content = ""
            if hasattr(result, "highlights") and result.highlights:
                content = " ... ".join(result.highlights)
            elif hasattr(result, "text") and result.text:
                content = result.text[:1500]

            if not content.strip():
                continue

            context_parts.append(f"[W{idx}] {content.strip()}")
            web_citations.append({
                "index": f"W{idx}",
                "title": getattr(result, "title", "Untitled") or "Untitled",
                "url": getattr(result, "url", ""),
                "snippet": content[:300],
            })

        return json.dumps({
            "context": "\n\n".join(context_parts),
            "web_citations": web_citations,
            "found": len(context_parts) > 0,
        })

    @staticmethod
    def _build_search_query(
        query: str, caller: str, level: str, subject: str
    ) -> str:
        """Construct a search query based on caller role and student level."""
        parts: list[str] = []

        if subject:
            parts.append(subject)

        if caller == "professor":
            parts.append("concept explanation definition theory")
        elif caller == "ta":
            parts.append("practice problems worked examples exercises solutions")

        _LEVEL_MODIFIERS = {
            "beginner": "introductory basic simple explanation",
            "intermediate": "detailed comprehensive with examples",
            "advanced": "rigorous in-depth proof derivation graduate-level",
        }
        parts.append(_LEVEL_MODIFIERS.get(level, _LEVEL_MODIFIERS["intermediate"]))
        parts.append(query)

        return " ".join(parts)
