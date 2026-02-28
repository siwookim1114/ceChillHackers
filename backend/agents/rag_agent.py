"""
RAG Agent -- Core content engine for the AI tutoring platform.

Handles: semantic retrieval (local FAISS from S3) + document management
Called by: Professor Agent, TA Agent, Manager Agent
Architecture: Programmatic retrieve-then-generate — calls retrieval tools
directly in code, then passes context to the LLM for answer generation.
No LLM tool calling needed.

Modes
1. INTERNAL_ONLY - ALL content sourced exclusively from the Knowledge Base.
2. EXTERNAL_OK - Knowledge Base first; web search / LLM may supplement when KB is insufficient.
3. EXTERNAL_ONLY - Skip KB entirely; use only Exa web search for all content.

Dynamic mode detection: the user's query is analyzed for phrases that signal
a need for web content (e.g. "more examples", "not on the slides"), and the
mode is automatically upgraded (never downgraded).

Callers:
1. Professor Agent - Concept explanations, lecture flow, examples for teaching.
2. TA Agent - Practice problems, worked examples, difficulty-level context.
3. Manager Agent - Document management (list, check sync status).
"""
import json
import logging
import os
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.tools import (
    CheckIngestionStatusTool,
    ExaWebSearchTool,
    ListDocumentsTool,
    RetrieveContextTool,
)
from config.config_loader import config as default_config
from db.models import RagMode
from prompts.rag_prompt import (
    _CALLER_DEFAULT,
    _CALLER_MANAGER,
    _CALLER_PROFESSOR,
    _CALLER_TA,
    _EXTERNAL_ADDENDUM,
    _EXTERNAL_ONLY_ADDENDUM,
    _EXTERNAL_WEB_ADDENDUM,
    _INTERNAL_ADDENDUM,
)

from langchain.chat_models import init_chat_model

logger = logging.getLogger(__name__)


class RagAgent:
    """
    RAG Agent -- retrieves and manages educational content.

    Uses programmatic retrieve-then-generate: always calls retrieval tools
    first in code, then passes context to the LLM for answer generation.

    Usage
    -----
    agent = RagAgent()
    result = agent.run({
        "prompt": "Explain eigenvalues and their geometric interpretation",
        "caller": "professor",
        "subject": "linear_algebra",
        "mode": "internal_only",
    })
    """
    # Dynamic mode detection — compiled regex patterns

    # Patterns that upgrade INTERNAL_ONLY → EXTERNAL_OK (web supplement)
    _WEB_SUPPLEMENT_PATTERNS = [
        re.compile(r"\bmore\s+examples?\b", re.IGNORECASE),
        re.compile(r"\bbetter\s+explanation\b", re.IGNORECASE),
        re.compile(r"\bexplain\b.*\b(more\s+simply|in\s+simpler\s+terms|in\s+simple\s+terms)\b", re.IGNORECASE),
        re.compile(r"\breal[\s-]?world\s+examples?\b", re.IGNORECASE),
        re.compile(r"\bother\s+sources?\b", re.IGNORECASE),
        re.compile(r"\bdifferent\s+(perspective|explanation|approach)\b", re.IGNORECASE),
        re.compile(r"\bsupplement(ary)?\s+(material|content|information|resources)\b", re.IGNORECASE),
        re.compile(r"\badditional\s+(examples?|resources?|material|explanations?|information)\b", re.IGNORECASE),
        re.compile(r"\boutside\s+(the\s+)?(lecture|slides?|course|class|textbook|material)\b", re.IGNORECASE),
        re.compile(r"\bbeyond\s+(the\s+)?(lecture|slides?|course|class|textbook|material)\b", re.IGNORECASE),
        re.compile(r"\bfrom\s+(the\s+)?(web|internet|online)\b", re.IGNORECASE),
        re.compile(r"\bsearch\s+(the\s+)?(web|internet|online)\b", re.IGNORECASE),
    ]

    # Patterns that upgrade any mode → EXTERNAL_ONLY (skip KB entirely)
    _EXTERNAL_ONLY_PATTERNS = [
        re.compile(r"\bnot\s+(on|in|from)\s+(the\s+)?(lecture|slides?|course|class|textbook|material|kb|knowledge\s+base)\b", re.IGNORECASE),
        re.compile(r"\bdon'?t\s+use\s+(the\s+)?(lecture|slides?|course|class|textbook|material|kb|knowledge\s+base)\b", re.IGNORECASE),
        re.compile(r"\bskip\s+(the\s+)?(lecture|slides?|course|class|textbook|material|kb|knowledge\s+base)\b", re.IGNORECASE),
        re.compile(r"\bignore\s+(the\s+)?(lecture|slides?|course|class|textbook|material|kb|knowledge\s+base)\b", re.IGNORECASE),
        re.compile(r"\bonly\s+(from\s+)?(the\s+)?(web|internet|online|external)\b", re.IGNORECASE),
        re.compile(r"\bweb\s+only\b", re.IGNORECASE),
        re.compile(r"\bexclude\s+(the\s+)?(lecture|slides?|course|class|textbook|material|kb|knowledge\s+base)\b", re.IGNORECASE),
    ]

    def __init__(self, config: Any = None) -> None:
        self.config = config or default_config
        self._init_llm()
        self._init_tools()
        self._CALLER_PROMPTS: Dict[str, str] = {
            "professor": _CALLER_PROFESSOR,
            "ta": _CALLER_TA,
            "manager": _CALLER_MANAGER,
        }

    def _init_tools(self) -> None:
        """Instantiate the KB tools for direct programmatic use."""
        self.retrieve_tool = RetrieveContextTool(self.config)
        self.ingestion_tool = CheckIngestionStatusTool(self.config)
        self.list_tool = ListDocumentsTool(self.config)

        # Exa web search — optional, graceful None if no API key
        try:
            self.exa_tool = ExaWebSearchTool(self.config)
        except (ValueError, ImportError) as exc:
            logger.warning("Exa web search unavailable: %s", exc)
            self.exa_tool = None

    def _get_tools(self) -> List:
        """Return tool list (for compatibility with tests)."""
        tools = [self.retrieve_tool, self.ingestion_tool, self.list_tool]
        if self.exa_tool is not None:
            tools.append(self.exa_tool)
        return tools

    def _init_llm(self) -> None:
        """Initialize the LLM — Featherless (default) or Bedrock Converse."""
        provider = self.config.get("llm.provider", "featherless")

        if provider == "featherless":
            model = self.config.get("llm.model", "meta-llama/Llama-3.3-70B-Instruct")
            base_url = self.config.get("llm.base_url", "https://api.featherless.ai/v1")
            api_key = os.environ.get("FEATHERLESSAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Missing FEATHERLESSAI_API_KEY in environment -- "
                    "add it to your .env file."
                )
            self.llm = ChatOpenAI(
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=0,
            )
        else:
            model_id = self.config.get("bedrock.models.rag")
            if not model_id:
                raise ValueError(
                    "Missing required config key: bedrock.models.rag -- "
                    "set this to a valid Bedrock model ID in config.yaml"
                )
            self.llm = init_chat_model(
                model_id,
                model_provider="bedrock_converse",
                region_name=self.config.aws.region,
                temperature=0,
            )

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the RAG Agent with programmatic retrieve-then-generate.

        Parameters
        ----------
        payload : dict
            prompt         (str, REQUIRED) - The query or instruction.
            caller         (str, optional) - "professor" | "ta" | "manager"
            subject        (str, optional) - Course/subject filter.
            mode           (str, optional) - "internal_only" (default) | "external_ok" | "external_only"
            level          (str, optional) - "beginner" | "intermediate" | "advanced"
            retrieve_only  (bool, optional) - If True, return raw chunks without LLM synthesis.

        Returns
        -------
        dict with keys: answer, citations, found, mode, caller, tool_calls_count
             (retrieve_only adds: context — raw chunk text)
        """
        raw_prompt = payload.get("prompt", "")
        if not raw_prompt.strip():
            return self._error_response("Empty prompt provided.", payload)

        mode = RagMode(payload.get("mode", RagMode.INTERNAL_ONLY))
        caller: str = payload.get("caller", "unknown")
        subject: str = payload.get("subject", "")
        level: str = payload.get("level", "intermediate")
        retrieve_only: bool = payload.get("retrieve_only", False)

        # Dynamic mode detection — upgrade based on query phrasing (content callers only)
        if caller != "manager":
            mode, upgraded = self._detect_mode_upgrade(raw_prompt, mode)
            if upgraded:
                logger.info(
                    "Dynamic mode upgrade: %s -> %s (query: %.80s...)",
                    payload.get("mode", "internal_only"),
                    mode.value,
                    raw_prompt,
                )

        # Manager caller → route to management tools
        if caller == "manager":
            return self._handle_manager(raw_prompt, mode, subject, payload)

        # Retrieve-only mode — return raw chunks without LLM synthesis.
        # Used by Professor/TA agents to get grounding context for their own
        # LLM calls, avoiding double-LLM invocations.
        if retrieve_only:
            return self._handle_retrieve_only(
                raw_prompt, mode, caller, subject, level, payload
            )

        # Content callers (professor, ta, unknown) → retrieve then generate
        return self._handle_content(raw_prompt, mode, caller, subject, level, payload)

    # -- Core flows --------------------------------------------------------

    def _handle_content(
        self,
        prompt: str,
        mode: RagMode,
        caller: str,
        subject: str,
        level: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Retrieve from KB, then generate answer with LLM.

        Flow:
        0. EXTERNAL_ONLY → skip KB, go straight to Exa web search
        1. Always search KB first via FAISS (INTERNAL_ONLY / EXTERNAL_OK)
        2. If KB has results → use them (all modes)
        3. If KB empty AND mode=EXTERNAL_OK → call Exa web search
        4. If web results found → pass web context to LLM with [W1] citation style
        5. If web also empty → fall back to LLM supplement (existing behavior)
        """
        tool_calls_count = 0
        used_web = False

        # Step 0: EXTERNAL_ONLY — skip KB entirely, go straight to Exa
        if mode == RagMode.EXTERNAL_ONLY:
            return self._handle_external_only(prompt, caller, subject, level, payload)

        # Step 1: Always retrieve from KB first (INTERNAL_ONLY / EXTERNAL_OK)
        # Append lightweight caller-specific keywords to help FAISS match
        # the right chunk type (concepts vs. problems) without verbose noise.
        search_query = self._enrich_query_for_search(prompt, caller)
        top_k = int(self.config.get("rag.top_k", 5))
        retrieval_input = {"query": search_query, "subject": subject, "top_k": top_k}

        try:
            raw_retrieval = self.retrieve_tool._run(retrieval_input)
            tool_calls_count += 1
            retrieval = json.loads(raw_retrieval)
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            return self._error_response(f"Retrieval error: {exc}", payload)

        found = retrieval.get("found", False)
        context = retrieval.get("context", "")
        citations = retrieval.get("citations", [])

        # Relevance check (EXTERNAL_OK only): FAISS always returns nearest
        # neighbors even for irrelevant queries. If the best score is below
        # threshold, treat as not found so we fall through to web search.
        # In INTERNAL_ONLY mode, always trust KB results regardless of score.
        if mode == RagMode.EXTERNAL_OK and found and citations:
            relevance_threshold = float(self.config.get("rag.relevance_threshold", 0.25))
            best_score = max(c.get("score", 0) for c in citations)
            if best_score < relevance_threshold:
                logger.info(
                    "KB results below relevance threshold (%.4f < %.2f), treating as not found",
                    best_score, relevance_threshold,
                )
                found = False

        # Label KB citations with source_type
        for c in citations:
            c["source_type"] = "kb"

        # Step 2: Mode-dependent handling when KB has no results
        if not found and mode == RagMode.INTERNAL_ONLY:
            return {
                "answer": "The requested content was not found in the Knowledge Base.",
                "citations": [],
                "found": False,
                "mode": mode.value,
                "caller": caller,
                "tool_calls_count": tool_calls_count,
            }

        # Step 3: EXTERNAL_OK with no KB content → try Exa web search
        web_context = ""
        web_citations: list[dict[str, Any]] = []

        if not found and mode == RagMode.EXTERNAL_OK and self.exa_tool is not None:
            # KB results were irrelevant — drop them before web search
            citations = []
            try:
                raw_web = self.exa_tool._run({
                    "query": prompt,
                    "caller": caller,
                    "level": level,
                    "subject": subject,
                })
                tool_calls_count += 1
                web_result = json.loads(raw_web)

                if web_result.get("found", False):
                    used_web = True
                    web_context = web_result.get("context", "")
                    web_citations = web_result.get("web_citations", [])
                    for wc in web_citations:
                        wc["source_type"] = "web"
            except Exception as exc:
                logger.warning("Exa web search failed, falling back to LLM: %s", exc)

        # Step 4: Build system + user messages for LLM
        system_prompt = self._build_system_prompt(mode, caller, subject, used_web)
        focus = self._get_caller_focus(caller)

        if found:
            # KB content available — use only KB citations
            user_content = (
                f"## Retrieved Knowledge Base Content\n{context}\n\n"
                f"## User Query\n{prompt}\n\n"
                f"{focus}\n"
                "Use the retrieved content above to answer the query. "
                "Cite facts using [index] notation matching the chunk indices. "
                "Return ONLY the answer text — no JSON wrapping."
            )
        elif used_web:
            # Web search results available (KB was empty)
            user_content = (
                f"## Web Search Results\n{web_context}\n\n"
                f"## User Query\n{prompt}\n\n"
                f"{focus}\n"
                "Use the web search results above to answer the query. "
                "Cite facts using [W1], [W2], etc. matching the web result indices. "
                "Return ONLY the answer text — no JSON wrapping."
            )
        else:
            # EXTERNAL_OK with no KB and no web content → LLM supplement
            user_content = (
                f"## User Query\n{prompt}\n\n"
                "No relevant content was found in the Knowledge Base or web search. "
                "You may supplement with your own knowledge. "
                "Mark all such content as [LLM supplemented]. "
                "Return ONLY the answer text — no JSON wrapping."
            )

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ])
            raw_answer = response.content
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return self._error_response(f"LLM error: {exc}", payload)

        # Strip <think> tags (Qwen/deepseek models emit these)
        raw_answer = self._strip_think_tags(raw_answer)

        # If the LLM determined content is irrelevant, override found/citations
        answer_lower = raw_answer.lower()
        if any(phrase in answer_lower for phrase in [
            "not found in the knowledge base",
            "not found in the kb",
            "no relevant content",
        ]):
            found = False
            citations = []

        # Merge KB + web citations
        merged_citations = citations + web_citations
        effective_found = found or used_web

        return self._build_output(
            raw_answer, merged_citations, effective_found, mode, caller, tool_calls_count
        )

    def _handle_retrieve_only(
        self,
        prompt: str,
        mode: RagMode,
        caller: str,
        subject: str,
        level: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Return raw retrieved chunks + citations without LLM synthesis.

        Used by calling agents (Professor, TA) that run their own LLM step
        and only need the RAG Agent as a retrieval engine.
        """
        tool_calls_count = 0

        # EXTERNAL_ONLY → web search only
        if mode == RagMode.EXTERNAL_ONLY:
            if self.exa_tool is None:
                return {
                    "context": "",
                    "citations": [],
                    "found": False,
                    "mode": mode.value,
                    "caller": caller,
                    "tool_calls_count": 0,
                }
            try:
                raw_web = self.exa_tool._run({
                    "query": prompt,
                    "caller": caller,
                    "level": level,
                    "subject": subject,
                })
                tool_calls_count += 1
                web_result = json.loads(raw_web)
                web_citations = web_result.get("web_citations", [])
                for wc in web_citations:
                    wc["source_type"] = "web"
                return {
                    "context": web_result.get("context", ""),
                    "citations": web_citations,
                    "found": web_result.get("found", False),
                    "mode": mode.value,
                    "caller": caller,
                    "tool_calls_count": tool_calls_count,
                }
            except Exception as exc:
                logger.error("Retrieve-only web search failed: %s", exc)
                return self._error_response(f"Web search error: {exc}", payload)

        # KB retrieval (INTERNAL_ONLY or EXTERNAL_OK)
        search_query = self._enrich_query_for_search(prompt, caller)
        top_k = int(self.config.get("rag.top_k", 5))
        retrieval_input = {"query": search_query, "subject": subject, "top_k": top_k}

        try:
            raw_retrieval = self.retrieve_tool._run(retrieval_input)
            tool_calls_count += 1
            retrieval = json.loads(raw_retrieval)
        except Exception as exc:
            logger.error("Retrieve-only KB retrieval failed: %s", exc)
            return self._error_response(f"Retrieval error: {exc}", payload)

        found = retrieval.get("found", False)
        context = retrieval.get("context", "")
        citations = retrieval.get("citations", [])

        # Relevance check for EXTERNAL_OK
        if mode == RagMode.EXTERNAL_OK and found and citations:
            threshold = float(self.config.get("rag.relevance_threshold", 0.25))
            best_score = max(c.get("score", 0) for c in citations)
            if best_score < threshold:
                found = False

        for c in citations:
            c["source_type"] = "kb"

        # If KB empty and EXTERNAL_OK → try web search
        web_context = ""
        web_citations: list[dict[str, Any]] = []
        if not found and mode == RagMode.EXTERNAL_OK and self.exa_tool is not None:
            citations = []
            try:
                raw_web = self.exa_tool._run({
                    "query": prompt,
                    "caller": caller,
                    "level": level,
                    "subject": subject,
                })
                tool_calls_count += 1
                web_result = json.loads(raw_web)
                if web_result.get("found", False):
                    web_context = web_result.get("context", "")
                    web_citations = web_result.get("web_citations", [])
                    for wc in web_citations:
                        wc["source_type"] = "web"
            except Exception as exc:
                logger.warning("Retrieve-only web fallback failed: %s", exc)

        merged_context = context or web_context
        merged_citations = citations + web_citations
        effective_found = found or bool(web_citations)

        return {
            "context": merged_context,
            "citations": merged_citations,
            "found": effective_found,
            "mode": mode.value,
            "caller": caller,
            "tool_calls_count": tool_calls_count,
        }

    def _handle_manager(
        self,
        prompt: str,
        mode: RagMode,
        subject: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Route manager requests to the appropriate management tool."""
        tool_calls_count = 0
        prompt_lower = prompt.lower()

        try:
            if "status" in prompt_lower or "ingestion" in prompt_lower:
                raw_result = self.ingestion_tool._run(
                    {"ingestion_job_id": prompt.strip()}
                )
                tool_calls_count += 1
            else:
                # Default: list documents
                raw_result = self.list_tool._run({"subject": subject})
                tool_calls_count += 1

            tool_result = json.loads(raw_result)
        except Exception as exc:
            logger.error("Manager tool call failed: %s", exc)
            return self._error_response(f"Tool error: {exc}", payload)
        return {
            "answer": json.dumps(tool_result, indent=2),
            "citations": [],
            "found": True,
            "mode": mode.value,
            "caller": "manager",
            "tool_calls_count": tool_calls_count,
        }

    def _handle_external_only(
        self,
        prompt: str,
        caller: str,
        subject: str,
        level: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """EXTERNAL_ONLY mode — skip KB, use only Exa web search."""
        mode = RagMode.EXTERNAL_ONLY

        if self.exa_tool is None:
            return {
                "answer": (
                    "Web search is unavailable. Cannot fulfill EXTERNAL_ONLY request "
                    "without the Exa web search tool. Please check your EXA_API_KEY configuration."
                ),
                "citations": [],
                "found": False,
                "mode": mode.value,
                "caller": caller,
                "tool_calls_count": 0,
                "error": "Exa web search tool is not configured.",
            }

        try:
            raw_web = self.exa_tool._run({
                "query": prompt,
                "caller": caller,
                "level": level,
                "subject": subject,
            })
            web_result = json.loads(raw_web)
        except Exception as exc:
            logger.error("Exa web search failed in EXTERNAL_ONLY mode: %s", exc)
            return self._error_response(
                f"Web search failed in EXTERNAL_ONLY mode: {exc}", payload
            )

        if not web_result.get("found", False):
            return {
                "answer": "No relevant web results were found for your query.",
                "citations": [],
                "found": False,
                "mode": mode.value,
                "caller": caller,
                "tool_calls_count": 1,
            }

        web_context = web_result.get("context", "")
        web_citations = web_result.get("web_citations", [])
        for wc in web_citations:
            wc["source_type"] = "web"

        system_prompt = self._build_system_prompt(mode, caller, subject, used_web=True)
        focus = self._get_caller_focus(caller)
        user_content = (
            f"## Web Search Results\n{web_context}\n\n"
            f"## User Query\n{prompt}\n\n"
            f"{focus}\n"
            "Use the web search results above to answer the query. "
            "Cite facts using [W1], [W2], etc. matching the web result indices. "
            "Return ONLY the answer text — no JSON wrapping."
        )

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_content),
            ])
            raw_answer = self._strip_think_tags(response.content)
        except Exception as exc:
            logger.error("LLM generation failed in EXTERNAL_ONLY mode: %s", exc)
            return self._error_response(f"LLM error: {exc}", payload)

        return self._build_output(
            raw_answer, web_citations, True, mode, caller, 1
        )

    # Private helpers

    @classmethod
    def _detect_mode_upgrade(cls, query: str, current_mode: RagMode) -> tuple[RagMode, bool]:
        """Analyze the user's query and upgrade the mode if warranted.

        Rules:
        - EXTERNAL_ONLY patterns always win (upgrade to EXTERNAL_ONLY).
        - WEB_SUPPLEMENT patterns upgrade INTERNAL_ONLY → EXTERNAL_OK.
        - Never downgrades.

        Returns (effective_mode, was_upgraded).
        """
        # Check EXTERNAL_ONLY patterns first (strongest signal)
        for pattern in cls._EXTERNAL_ONLY_PATTERNS:
            if pattern.search(query):
                if current_mode != RagMode.EXTERNAL_ONLY:
                    return RagMode.EXTERNAL_ONLY, True
                return current_mode, False

        # Check web supplement patterns (upgrade INTERNAL_ONLY → EXTERNAL_OK)
        if current_mode == RagMode.INTERNAL_ONLY:
            for pattern in cls._WEB_SUPPLEMENT_PATTERNS:
                if pattern.search(query):
                    return RagMode.EXTERNAL_OK, True

        return current_mode, False

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Remove <think>...</think> blocks emitted by some models."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _build_system_prompt(
        self, mode: RagMode, caller: str, subject: str, used_web: bool = False
    ) -> str:
        """Assemble the full system prompt from mode, caller, and subject."""
        _BASE_PROMPT = """\
        You are the RAG Agent -- the core content engine of an AI tutoring platform.

        Your purpose is to organize and present educational content from the
        Knowledge Base (KB) so that other agents (Professor, TA, Manager)
        can serve students effectively.

        ## Core Rules
        1. Cite every KB-sourced fact with [index] notation matching the retrieved chunk indices.
        2. NEVER fabricate citations or invent content not found in the KB.
        3. Do NOT generate citation objects — citations are handled automatically.
        4. Return ONLY the answer text as a plain string. Do NOT wrap it in JSON.
        """
        parts = [_BASE_PROMPT]

        if mode == RagMode.EXTERNAL_ONLY:
            parts.append(_EXTERNAL_ONLY_ADDENDUM)
        elif mode == RagMode.INTERNAL_ONLY:
            parts.append(_INTERNAL_ADDENDUM)
        elif used_web:
            parts.append(_EXTERNAL_WEB_ADDENDUM)
        else:
            parts.append(_EXTERNAL_ADDENDUM)

        parts.append(self._CALLER_PROMPTS.get(caller, _CALLER_DEFAULT))

        if subject:
            parts.append(
                f"\n## SUBJECT CONTEXT\n"
                f"The current subject/course is: {subject}\n"
            )

        return "\n".join(parts)

    @staticmethod
    def _get_caller_focus(caller: str) -> str:
        """Return caller-specific INCLUDE/EXCLUDE focus instructions."""
        if caller == "professor":
            return (
                "## STRICT ROLE: Professor Agent\n"
                "Extract and present ONLY:\n"
                "- Concept definitions and theoretical foundations\n"
                "- Key formulas, theorems, and their intuition\n"
                "- Analogies and conceptual explanations\n"
                "- Prerequisites and how concepts connect\n\n"
                "EXCLUDE completely:\n"
                "- Problem statements (e.g. 'Q3. In a class...')\n"
                "- Numerical solutions or step-by-step calculations\n"
                "- Worked examples with specific numbers\n"
                "These belong to the TA Agent, not you.\n"
            )
        elif caller == "ta":
            return (
                "## STRICT ROLE: TA Agent\n"
                "Extract and present ONLY:\n"
                "- Practice problems and their exact statements\n"
                "- Step-by-step worked solutions with calculations\n"
                "- Difficulty level and sub-topic grouping\n"
                "- Hints and common mistakes for each problem\n\n"
                "EXCLUDE completely:\n"
                "- Concept definitions or theoretical explanations\n"
                "- Teaching narratives or lecture-style content\n"
                "- Prerequisite discussions\n"
                "These belong to the Professor Agent, not you.\n"
            )
        return ""

    @staticmethod
    def _enrich_query_for_search(prompt: str, caller: str) -> str:
        """Append lightweight caller-specific keywords for FAISS matching."""
        if caller == "professor":
            return f"{prompt} concept explanation definition theory"
        elif caller == "ta":
            return f"{prompt} practice problems exercises solutions examples"
        return prompt

    def _build_output(
        self,
        raw_content: str,
        citations: list,
        found: bool,
        mode: RagMode,
        caller: str,
        tool_calls_count: int,
    ) -> dict[str, Any]:
        """Normalize the LLM's response into a structured dict."""
        parsed: dict[str, Any] | None = None
        if isinstance(raw_content, str):
            content = raw_content.strip()
            # Strip markdown code fences
            if content.startswith("```"):
                first_newline = content.find("\n")
                if first_newline != -1:
                    content = content[first_newline + 1:]
                if content.rstrip().endswith("```"):
                    content = content.rstrip()[:-3].rstrip()
            try:
                parsed = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                pass

        if parsed and isinstance(parsed, dict):
            parsed.setdefault("mode", mode.value)
            parsed.setdefault("caller", caller)
            parsed.setdefault("found", found)
            # Always use retrieval citations (doc, page, score) —
            # never trust LLM-generated citation objects
            parsed["citations"] = citations
            parsed["tool_calls_count"] = tool_calls_count
            return parsed

        return {
            "answer": raw_content,
            "citations": citations,
            "found": found,
            "mode": mode.value,
            "caller": caller,
            "tool_calls_count": tool_calls_count,
        }

    @staticmethod
    def _error_response(message: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Build a structured error response."""
        return {
            "answer": message,
            "citations": [],
            "found": False,
            "mode": payload.get("mode", RagMode.INTERNAL_ONLY.value),
            "caller": payload.get("caller", "unknown"),
            "error": message,
            "tool_calls_count": 0,
        }
