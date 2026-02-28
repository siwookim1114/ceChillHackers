"""Professor Agent -- RAG-grounded Socratic tutoring.

Uses the RAG Agent as its knowledge retrieval brain, then generates
Socratic tutoring responses grounded in retrieved course material.

Architecture:
  1. Select tutoring strategy based on student profile + message
  2. Call RagAgent.run(retrieve_only=True) for KB chunks + citations
  3. Build professor prompt injecting RAG context + strategy instructions
  4. Single LLM call generates the tutoring response
  5. Validate + return ProfessorTurnResponse

Designed for easy integration as a LangGraph node:
  professor_node(state) -> ProfessorAgent().run(state["request"])
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.rag_agent import RagAgent
from config.config_loader import config as default_config
from db.models import (
    Citation,
    ProfessorMode,
    ProfessorNextAction,
    ProfessorTurnRequest,
    ProfessorTurnResponse,
    ProfessorTurnStrategy,
)
from prompts.professor_prompt import (
    PROFESSOR_SYSTEM_PROMPT,
    build_professor_user_prompt,
)

logger = logging.getLogger(__name__)


class ProfessorAgent:
    """Finalized Professor Agent -- RAG-grounded Socratic tutoring.

    Usage
    -----
    agent = ProfessorAgent()
    response = agent.run(ProfessorTurnRequest(
        session_id="s1",
        message="I keep mixing up derivatives and slopes",
        topic="calculus",
        mode=ProfessorMode.STRICT,
        profile=ProfessorProfile(level="beginner", learning_style="mixed"),
    ))
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config or default_config
        self.rag_agent = RagAgent(self.config)
        self._init_llm()

        prof_cfg = self.config.get("agents.professor", {})
        if not isinstance(prof_cfg, dict):
            prof_cfg = {}
        tutoring = prof_cfg.get("tutoring", {})
        if not isinstance(tutoring, dict):
            tutoring = {}

        self.socratic_default: bool = bool(tutoring.get("socratic_default", True))
        self.citations_enabled: bool = bool(tutoring.get("citations_enabled", True))

    def _init_llm(self) -> None:
        """Initialize the LLM for tutoring generation (same pattern as RagAgent)."""
        provider = self.config.get("llm.provider", "featherless")

        if provider == "featherless":
            model = self.config.get("llm.model", "meta-llama/Llama-3.3-70B-Instruct")
            base_url = self.config.get("llm.base_url", "https://api.featherless.ai/v1")
            api_key = os.environ.get("FEATHERLESSAI_API_KEY")
            if not api_key:
                raise ValueError("Missing FEATHERLESSAI_API_KEY in environment.")
            self.llm = ChatOpenAI(
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=0.3,
            )
        else:
            from langchain.chat_models import init_chat_model

            model_id = self.config.get("bedrock.models.professor")
            if not model_id:
                model_id = self.config.get("bedrock.models.rag")
            if not model_id:
                raise ValueError("Missing bedrock model config for professor.")
            self.llm = init_chat_model(
                model_id,
                model_provider="bedrock_converse",
                region_name=self.config.get("aws.region", "us-east-1"),
                temperature=0.3,
            )

    # -- Main pipeline -------------------------------------------------------

    def run(self, request: ProfessorTurnRequest) -> ProfessorTurnResponse:
        """Execute one tutoring turn.

        Pipeline:
        1. Select strategy based on student profile + message
        2. Retrieve grounding context from RAG Agent (no LLM synthesis)
        3. Build professor prompt with RAG context + strategy instructions
        4. Generate Socratic tutoring response via LLM
        5. Parse, validate, and return ProfessorTurnResponse
        """
        # Step 1: Strategy selection
        strategy = self._choose_strategy(request)
        logger.info(
            "Professor strategy=%s for session=%s topic=%s level=%s",
            strategy.value,
            request.session_id,
            request.topic,
            request.profile.level,
        )

        # Step 2: RAG retrieval (retrieve_only — no LLM synthesis)
        rag_mode = self._map_mode_to_rag(request.mode)
        rag_result = self.rag_agent.run({
            "prompt": request.message,
            "caller": "professor",
            "subject": request.topic,
            "mode": rag_mode,
            "level": request.profile.level,
            "retrieve_only": True,
        })
        rag_found = rag_result.get("found", False)
        rag_context = rag_result.get("context", "")
        rag_citations = rag_result.get("citations", [])

        logger.info(
            "RAG retrieve_only: found=%s citations=%d mode=%s",
            rag_found,
            len(rag_citations),
            rag_result.get("mode", ""),
        )

        # Step 3: Build professor prompt
        user_prompt = build_professor_user_prompt(
            message=request.message,
            topic=request.topic,
            level=request.profile.level,
            learning_style=request.profile.learning_style,
            pace=request.profile.pace,
            strategy=strategy.value,
            rag_context=rag_context,
            rag_citations=rag_citations,
            rag_found=rag_found,
        )

        # Step 4: LLM generation
        try:
            llm_response = self.llm.invoke([
                SystemMessage(content=PROFESSOR_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])
            raw_text = self._strip_think_tags(llm_response.content)
        except Exception as exc:
            logger.error("Professor LLM call failed: %s", exc)
            return self._fallback_response(request, strategy)

        # Step 5: Parse and validate
        try:
            parsed = self._parse_json_from_text(raw_text)
            normalized = self._normalize_response(parsed, request, strategy)
            return ProfessorTurnResponse.model_validate(normalized)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Professor response parse/validation failed: %s", exc)
            return self._fallback_response(request, strategy)

    # -- Strategy selection (ported from ProfessorRespondTool) ----------------

    def _choose_strategy(self, request: ProfessorTurnRequest) -> ProfessorTurnStrategy:
        """Select tutoring strategy based on student profile and message content."""
        text = request.student_message.strip().lower()

        procedure_signals = (
            "step by step", "step-by-step", "procedure", "algorithm",
            "how do i solve", "process", "walk me through",
        )
        stuck_signals = (
            "break down", "simpler", "easier", "small example", "stuck",
            "don't know where to start", "dont know where to start",
            "mixing up", "confuse", "confused",
        )
        concept_signals = (
            "why", "intuition", "concept", "meaning", "understand", "what is",
        )

        asks_for_procedure = any(s in text for s in procedure_signals)
        is_stuck = any(s in text for s in stuck_signals)
        asks_for_concept = any(s in text for s in concept_signals)

        level = request.profile.level
        learning_style = request.profile.learning_style
        pace = request.profile.pace

        # Advanced textual/mixed learners → Socratic probing
        if level == "advanced" and learning_style in {"textual", "mixed"}:
            if asks_for_procedure:
                return ProfessorTurnStrategy.PROCEDURAL_EXPLANATION
            if asks_for_concept:
                return ProfessorTurnStrategy.CONCEPTUAL_EXPLANATION
            return ProfessorTurnStrategy.SOCRATIC_QUESTION

        # Beginners or slow pace → guided procedure and breakdowns
        if level == "beginner" or pace == "slow":
            if is_stuck:
                return ProfessorTurnStrategy.BROKEN_DOWN_QUESTIONS
            if learning_style == "example_first" or asks_for_procedure:
                return ProfessorTurnStrategy.PROCEDURAL_EXPLANATION
            return ProfessorTurnStrategy.CONCEPTUAL_EXPLANATION

        # Intermediate → concept-first, scaffolding if stuck
        if level == "intermediate":
            if is_stuck:
                return ProfessorTurnStrategy.BROKEN_DOWN_QUESTIONS
            if asks_for_procedure and learning_style == "example_first":
                return ProfessorTurnStrategy.PROCEDURAL_EXPLANATION
            return ProfessorTurnStrategy.CONCEPTUAL_EXPLANATION

        # Remaining advanced styles
        if asks_for_procedure:
            return ProfessorTurnStrategy.PROCEDURAL_EXPLANATION
        if is_stuck:
            return ProfessorTurnStrategy.BROKEN_DOWN_QUESTIONS
        if asks_for_concept:
            return ProfessorTurnStrategy.CONCEPTUAL_EXPLANATION
        if self.socratic_default and level == "advanced":
            return ProfessorTurnStrategy.SOCRATIC_QUESTION
        return ProfessorTurnStrategy.CONCEPTUAL_EXPLANATION

    # -- Response parsing and validation -------------------------------------

    @staticmethod
    def _normalize_response(
        parsed: dict[str, Any],
        request: ProfessorTurnRequest,
        selected_strategy: ProfessorTurnStrategy,
    ) -> dict[str, Any]:
        """Normalize LLM output into valid ProfessorTurnResponse fields."""
        strategy_aliases = {
            "socratic question": "socratic_question",
            "socratic-question": "socratic_question",
            "question": "socratic_question",
            "explain": "conceptual_explanation",
            "conceptual explanation": "conceptual_explanation",
            "step by step": "procedural_explanation",
            "step-by-step": "procedural_explanation",
            "procedural explanation": "procedural_explanation",
            "hint": "broken_down_questions",
            "breakdown": "broken_down_questions",
            "broken down questions": "broken_down_questions",
            "scaffold": "broken_down_questions",
        }
        next_action_aliases = {
            "route_ta": "route_problem_ta",
            "route problem ta": "route_problem_ta",
            "route planner": "route_planner",
            "keep_going": "continue",
            "continue_tutoring": "continue",
        }

        # Normalize strategy
        strategy_raw = str(parsed.get("strategy", "")).strip().lower()
        strategy_raw = strategy_aliases.get(strategy_raw, strategy_raw)
        valid_strategies = {
            "socratic_question", "conceptual_explanation",
            "procedural_explanation", "broken_down_questions",
        }
        if strategy_raw not in valid_strategies:
            strategy_raw = selected_strategy.value

        # Normalize next_action
        next_action_raw = str(parsed.get("next_action", "")).strip().lower()
        next_action_raw = next_action_aliases.get(next_action_raw, next_action_raw)
        if next_action_raw not in {"continue", "route_problem_ta", "route_planner"}:
            next_action_raw = "continue"

        # Extract assistant_response
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
            "citations": [],
        }

    @staticmethod
    def _parse_json_from_text(text: str) -> dict[str, Any]:
        """Extract a JSON object from LLM output, handling code fences."""
        cleaned = text.strip()
        # Strip markdown code fences
        if cleaned.startswith("```"):
            first_nl = cleaned.find("\n")
            if first_nl != -1:
                cleaned = cleaned[first_nl + 1:]
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3].rstrip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: find first balanced { ... } block
        for start_idx, char in enumerate(cleaned):
            if char != "{":
                continue
            depth = 0
            in_string = False
            escaped = False
            for end_idx in range(start_idx, len(cleaned)):
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
                elif current == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = cleaned[start_idx:end_idx + 1]
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                        break

        raise ValueError("LLM response did not contain a valid JSON object")

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Remove <think>...</think> blocks emitted by some models."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _map_mode_to_rag(mode: ProfessorMode) -> str:
        """Map professor mode to RAG mode string."""
        if mode is ProfessorMode.STRICT:
            return "internal_only"
        return "external_ok"

    @staticmethod
    def _fallback_response(
        request: ProfessorTurnRequest,
        strategy: ProfessorTurnStrategy,
    ) -> ProfessorTurnResponse:
        """Deterministic fallback when LLM call or parsing fails."""
        fallback_texts = {
            ProfessorTurnStrategy.SOCRATIC_QUESTION: (
                f"In {request.topic}, the key idea connects principle to method. "
                "Given your current setup, what assumption are you making first and why?"
            ),
            ProfessorTurnStrategy.CONCEPTUAL_EXPLANATION: (
                f"In {request.topic}, start from the core concept and what it means in plain language. "
                "Once that idea is clear, apply it to your exact question one step at a time."
            ),
            ProfessorTurnStrategy.PROCEDURAL_EXPLANATION: (
                f"In {request.topic}, Step 1 is to identify the knowns and unknowns. "
                "Step 2 is choose the governing rule. "
                "Step 3 is apply it on a small example before your full problem."
            ),
            ProfessorTurnStrategy.BROKEN_DOWN_QUESTIONS: (
                f"The core concept in {request.topic} stays the same, so let's break it into smaller checks. "
                "What is the first quantity you can define? What relation connects it to the target?"
            ),
        }
        return ProfessorTurnResponse(
            assistant_response=fallback_texts.get(strategy, fallback_texts[ProfessorTurnStrategy.CONCEPTUAL_EXPLANATION]),
            strategy=strategy,
            revealed_final_answer=False,
            next_action=ProfessorNextAction.CONTINUE,
            citations=[],
        )
