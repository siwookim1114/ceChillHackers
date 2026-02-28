"""
RAG Agent -- Core content engine for the AI tutoring platform.

Handles: semantic retrieval (Bedrock Knowledge Base) + document management
Called by: Professor Agent, TA Agent, Manager Agent
Architecture: Pure Python class with a dynamic ReAct tool-calling loop.

Modes
1. INTERNAL_ONLY  - ALL content sourced exclusively from the Knowledge Base. If nothing is found, return a structured "not found" response. Never hallucinate or supplement with general knowledge.
2. EXTERNAL_OK  - Knowledge Base first; the LLM may supplement with its own knowledge when KB results are insufficient.

Callers:
1. Professor Agent - Concept explanations, lecture flow, examples for teaching.
2. TA - Practice problems, worked examples, difficulty-level context.
3. Manager Agent - Document management (upload, list, check sync status).
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agents.tools import (
    CheckIngestionStatusTool,
    ListDocumentsTool,
    RetrieveContextTool,
    UploadDocumentTool,
)
from config.config_loader import config as default_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_REACT_ITERATIONS = 10  # Safety cap to prevent infinite tool-calling loops


# ---------------------------------------------------------------------------
# Mode enum
# ---------------------------------------------------------------------------

class RagMode(str, Enum):
    """Controls whether the RAG Agent may supplement KB content with LLM knowledge."""
    INTERNAL_ONLY = "internal_only"
    EXTERNAL_OK = "external_ok"


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------

_BASE_PROMPT = """\
You are the RAG Agent -- the core content engine of an AI tutoring platform.

Your sole purpose is to retrieve, organize, and present educational content
from the Knowledge Base (KB) so that other agents (Professor, TA, Manager)
can serve students effectively.

## Core Rules
1. ALWAYS use the retrieve_context tool before answering any content question.
   Never rely on your own knowledge without first checking the KB.
2. Cite every fact with its source using [index] notation matching the
   citation indices returned by retrieve_context.
3. NEVER reveal raw, unprocessed document dumps. Summarize, structure, and
   annotate content for the calling agent's specific need.
4. NEVER fabricate citations. If the KB does not contain relevant content,
   say so explicitly.
5. Return your final answer as a valid JSON object (schema below) so the
   calling agent can parse it programmatically.

## Final Answer JSON Schema
{{
  "answer":    "<structured content tailored for the caller>",
  "citations": [<citation objects from retrieve_context>],
  "found":     <true if KB had relevant content, false otherwise>,
  "mode":      "<internal_only | external_ok>"
}}
"""

_INTERNAL_ADDENDUM = """
## MODE: INTERNAL_ONLY (strict)
- You MUST source ALL content exclusively from the Knowledge Base.
- If retrieve_context returns found=false or the retrieved content does not
  answer the query, you MUST return:
  {{
    "answer":    "The requested content was not found in the Knowledge Base.",
    "citations": [],
    "found":     false,
    "mode":      "internal_only",
    "suggestion": "Consider uploading relevant course material covering this topic."
  }}
- Do NOT supplement with your own training knowledge. Do NOT guess.
- Do NOT paraphrase beyond what the KB chunks actually say.
"""

_EXTERNAL_ADDENDUM = """
## MODE: EXTERNAL_OK (flexible)
- Search the Knowledge Base FIRST using retrieve_context.
- If KB content is sufficient, use it exclusively and cite it.
- If KB content is partial or missing, you MAY supplement with your own
  knowledge, but you MUST clearly distinguish KB-sourced content (cited)
  from LLM-supplemented content (marked as "[LLM supplemented]").
- Always prefer KB content over your own knowledge when both are available.
"""

_CALLER_PROFESSOR = """
## CALLER: Professor Agent
You are serving the Professor Agent, which teaches students through Socratic
dialogue, concept explanations, and guided discovery.

Tailor your retrieval and response for TEACHING purposes:
- Retrieve full concept explanations, definitions, and theoretical foundations.
- Include illustrative examples and analogies found in course materials.
- Organize content in a logical lecture flow: definition -> intuition ->
  examples -> common pitfalls.
- Highlight prerequisite concepts the student should already understand.
- If multiple explanations exist across documents, synthesize them into a
  coherent narrative rather than listing them separately.
- Include relevant formulas, theorems, or key equations exactly as they
  appear in the source material.
"""

_CALLER_TA = """
## CALLER: TA Agent
You are serving the TA Agent, which helps students practice through problem
solving, worked examples, and targeted exercises.

Tailor your retrieval and response for PRACTICE purposes:
- Retrieve practice problems, exercises, and worked examples.
- Include step-by-step solution procedures when available.
- Note the difficulty level if indicated in the source material
  (e.g., easy/medium/hard, or chapter/section context).
- Group problems by sub-topic or technique when multiple are found.
- Include any hints, common mistakes, or solution strategies mentioned
  in the source material.
- Preserve the exact problem statements -- do not paraphrase math problems
  as this can change their meaning.
"""

_CALLER_MANAGER = """
## CALLER: Manager Agent
You are serving the Manager Agent, which handles administrative tasks
like document upload, listing, and ingestion status checks.

Tailor your behavior for DOCUMENT MANAGEMENT:
- For upload requests, use the upload_document tool.
- For status checks, use the check_ingestion_status tool.
- For listing documents, use the list_available_documents tool.
- Provide clear, structured responses about document states and actions.
- Do NOT retrieve or summarize document content for management tasks --
  only handle the administrative operation requested.
"""

_CALLER_DEFAULT = """
## CALLER: Unknown
No specific caller context was provided.  Default to general-purpose
retrieval: return well-cited, clearly structured content from the KB.
"""

_CALLER_PROMPTS: dict[str, str] = {
    "professor": _CALLER_PROFESSOR,
    "ta": _CALLER_TA,
    "manager": _CALLER_MANAGER,
}


def _build_system_prompt(mode: RagMode, caller: str, subject: str) -> str:
    """Assemble the full system prompt from mode, caller, and subject."""
    parts = [_BASE_PROMPT]

    # Mode-specific rules
    if mode == RagMode.INTERNAL_ONLY:
        parts.append(_INTERNAL_ADDENDUM)
    else:
        parts.append(_EXTERNAL_ADDENDUM)

    # Caller-specific instructions
    parts.append(_CALLER_PROMPTS.get(caller, _CALLER_DEFAULT))

    # Subject context
    if subject:
        parts.append(
            f"\n## SUBJECT CONTEXT\n"
            f"The current subject/course is: {subject}\n"
            f"When using retrieve_context, pass subject='{subject}' to "
            f"filter results to this course's materials.\n"
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Query enrichment by caller
# ---------------------------------------------------------------------------

def _enrich_query(prompt: str, caller: str, subject: str) -> str:
    """Add caller-specific context to the raw user query.

    This helps the LLM formulate better retrieve_context tool calls by
    priming it with the kind of content the caller actually needs.
    """
    enrichment_prefix = ""

    if caller == "professor":
        enrichment_prefix = (
            "[Teaching context] I need to explain the following concept to a "
            "student. Retrieve comprehensive explanations, definitions, "
            "examples, and any relevant lecture material.\n\n"
        )
    elif caller == "ta":
        enrichment_prefix = (
            "[Practice context] I need practice material for a student. "
            "Retrieve relevant problems, exercises, worked examples, "
            "solution steps, and difficulty information.\n\n"
        )
    elif caller == "manager":
        # Manager queries are typically administrative -- no enrichment needed
        return prompt

    subject_hint = f"[Subject: {subject}] " if subject else ""
    return f"{enrichment_prefix}{subject_hint}{prompt}"


# ---------------------------------------------------------------------------
# RagAgent
# ---------------------------------------------------------------------------

class RagAgent:
    """RAG Agent -- retrieves and manages educational content from the
    Bedrock Knowledge Base.

    This is a pure Python class (no LangGraph) with a ReAct tool-calling
    loop.  It initializes four tools, binds them to a Bedrock Converse
    LLM, and iterates until the LLM produces a final text response or
    the max iteration guard is hit.

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

    def __init__(self, config: Any = None) -> None:
        self.config = config or default_config
        self._init_tools()
        self._init_llm()
        logger.info("RAG Agent initialized (model=%s)",
                     self.config.get("bedrock.models.rag"))

    # -- Initialization helpers -------------------------------------------

    def _init_tools(self) -> None:
        """Instantiate the four KB tools and build a name -> tool lookup."""
        self.retrieve_tool = RetrieveContextTool(self.config)
        self.upload_tool = UploadDocumentTool(self.config)
        self.check_tool = CheckIngestionStatusTool(self.config)
        self.list_tool = ListDocumentsTool(self.config)

        self.tools: list = [
            self.retrieve_tool,
            self.upload_tool,
            self.check_tool,
            self.list_tool,
        ]
        self.tools_map: dict[str, Any] = {t.name: t for t in self.tools}

    def _init_llm(self) -> None:
        """Initialize the Bedrock Converse LLM with tools bound."""
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
        ).bind_tools(self.tools)

    # -- Public interface --------------------------------------------------

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the RAG Agent ReAct loop.

        Parameters
        ----------
        payload : dict
            prompt  (str, REQUIRED) - The query or instruction.
            caller  (str, optional) - "professor" | "ta" | "manager"
            subject (str, optional) - Course/subject filter.
            mode    (str, optional) - "internal_only" (default) | "external_ok"

        Returns
        -------
        dict with keys: answer, citations, found, mode, caller, tool_calls_count
        """
        # -- Parse and validate payload -----------------------------------
        raw_prompt = payload.get("prompt", "")
        if not raw_prompt.strip():
            return self._error_response("Empty prompt provided.", payload)

        mode = RagMode(payload.get("mode", RagMode.INTERNAL_ONLY))
        caller: str = payload.get("caller", "unknown")
        subject: str = payload.get("subject", "")

        # -- Build messages ------------------------------------------------
        system_prompt = _build_system_prompt(mode, caller, subject)
        enriched_prompt = _enrich_query(raw_prompt, caller, subject)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=enriched_prompt),
        ]

        # -- ReAct tool-calling loop ---------------------------------------
        tool_calls_count = 0

        for iteration in range(MAX_REACT_ITERATIONS):
            try:
                response = self.llm.invoke(messages)
            except Exception as exc:
                logger.error("LLM invocation failed at iteration %d: %s",
                             iteration, exc)
                return self._error_response(
                    f"LLM invocation error: {exc}", payload
                )

            messages.append(response)

            # If the LLM produced no tool calls, we have the final answer
            if not response.tool_calls:
                break

            # Execute each tool call and append the results
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]

                logger.debug("Tool call [iter=%d]: %s(%s)",
                             iteration, tool_name, json.dumps(tool_args)[:200])

                result = self._execute_tool(tool_name, tool_args)
                messages.append(
                    ToolMessage(content=result, tool_call_id=tool_call_id)
                )
                tool_calls_count += 1
        else:
            # Loop exhausted without a final text response
            logger.warning(
                "RAG Agent hit max iterations (%d) without final answer. "
                "caller=%s, subject=%s", MAX_REACT_ITERATIONS, caller, subject
            )

        # -- Build structured output ---------------------------------------
        raw_content = messages[-1].content if messages else ""
        return self._build_output(raw_content, mode, caller, tool_calls_count)

    # -- Private helpers ---------------------------------------------------

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Dispatch a tool call by name. Returns JSON string."""
        tool = self.tools_map.get(tool_name)
        if tool is None:
            logger.warning("Unknown tool requested: %s", tool_name)
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            return tool._run(tool_input)
        except Exception as exc:
            logger.error("Tool '%s' raised an exception: %s", tool_name, exc)
            return json.dumps({"error": f"Tool execution error: {exc}"})

    def _build_output(
        self,
        raw_content: str,
        mode: RagMode,
        caller: str,
        tool_calls_count: int,
    ) -> dict[str, Any]:
        """Normalize the LLM's final response into a structured dict.

        The system prompt instructs the LLM to return JSON, so we attempt
        to parse it.  If the LLM returned plain text instead, we wrap it
        in the expected structure.
        """
        # Try to parse the LLM's response as JSON
        parsed: dict[str, Any] | None = None
        if isinstance(raw_content, str):
            # Strip markdown code fences if the LLM wrapped JSON in ```
            content = raw_content.strip()
            if content.startswith("```"):
                first_newline = content.find("\n")
                if first_newline != -1:
                    content = content[first_newline + 1:]
                if content.rstrip().endswith("```"):
                    content = content.rstrip()[:-3].rstrip()

            try:
                parsed = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                pass  # LLM returned plain text -- wrap it below

        if parsed and isinstance(parsed, dict):
            # Ensure required fields are present
            parsed.setdefault("mode", mode.value)
            parsed.setdefault("caller", caller)
            parsed.setdefault("found", True)
            parsed["tool_calls_count"] = tool_calls_count
            return parsed

        # Fallback: wrap plain text in the expected structure
        return {
            "answer": raw_content,
            "citations": [],
            "found": bool(raw_content and raw_content.strip()),
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
