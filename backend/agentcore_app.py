"""Bedrock AgentCore entry point — wraps the LangGraph orchestration graph.

Deploys the multi-agent tutoring platform as a managed Bedrock AgentCore
service. The compiled LangGraph graph (Manager → Professor/TA/RAG → Respond)
is exposed via the @app.entrypoint decorator.

Deployment:
    agentcore configure
    agentcore launch -e backend/agentcore_app.py

Local testing:
    agentcore dev
    agentcore invoke --dev '{"prompt": "Explain Bayes theorem"}'
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from uuid import uuid4

# Ensure project root and backend/ are on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
_BACKEND_DIR = str(Path(__file__).resolve().parent)
for _p in (_PROJECT_ROOT, _BACKEND_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv

load_dotenv(Path(_PROJECT_ROOT) / ".env")

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from agents.graph import get_graph
from agents.nodes import classify_hitl_feedback

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Compile the LangGraph graph once at import time
graph = get_graph()

app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload: dict, context: dict) -> dict:
    """AgentCore entry point — routes user queries through the LangGraph graph.

    Payload fields:
        prompt       (str, REQUIRED)  — The user's message.
        session_id   (str, optional)  — Thread ID for conversation continuity.
        user_id      (str, optional)  — Learner identifier.
        topic        (str, optional)  — Current subject/topic.
        level        (str, optional)  — beginner | intermediate | advanced.
        learning_style (str, optional) — visual | textual | example_first | mixed.
        pace         (str, optional)  — slow | medium | fast.
        mode         (str, optional)  — Professor mode: strict | convenience.
        knowledge_mode (str, optional) — internal_only | external_ok | external_only.
        feedback     (dict, optional) — HITL feedback to resume a paused graph.
            action        (str) — approve | revise | reroute | cancel.
            feedback_text (str) — User's feedback message.
            reroute_to    (str) — Target agent for reroute action.
    """
    prompt = payload.get("prompt", "")
    if not prompt and not payload.get("feedback"):
        return {
            "error": "No prompt provided.",
            "agent_name": "system",
            "response_text": "Please provide a prompt.",
        }

    session_id = payload.get("session_id", str(uuid4()))
    config = {"configurable": {"thread_id": session_id}}

    # Check for pending HITL interrupt
    snapshot = None
    try:
        snapshot = graph.get_state(config)
    except Exception:
        pass

    is_hitl_pending = bool(
        snapshot and snapshot.next and "human_feedback" in snapshot.next
    )

    # ── HITL feedback path ──
    if is_hitl_pending:
        from langgraph.types import Command

        # Explicit feedback dict in payload takes priority
        feedback_payload = payload.get("feedback")
        if feedback_payload and isinstance(feedback_payload, dict):
            feedback = {
                "action": feedback_payload.get("action", "approve"),
                "feedback_text": feedback_payload.get("feedback_text", ""),
                "reroute_to": feedback_payload.get("reroute_to"),
            }
        else:
            # Auto-classify the prompt as feedback
            current_agent = ""
            if snapshot.values:
                current_agent = (
                    snapshot.values.get("agent_output", {}).get("agent_name", "")
                )
            feedback = classify_hitl_feedback(prompt, current_agent)

        try:
            result = graph.invoke(Command(resume=feedback), config)
        except Exception as exc:
            logger.error("HITL resume failed: %s", exc)
            prev_output = (
                snapshot.values.get("agent_output", {}) if snapshot.values else {}
            )
            return {
                "session_id": session_id,
                "agent_name": prev_output.get("agent_name", "system"),
                "response_text": prev_output.get("response_text", f"Error: {exc}"),
                "citations": prev_output.get("citations", []),
                "awaiting_feedback": False,
                "error": str(exc),
            }
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
            "user_message": prompt,
            "user_profile": {
                "user_id": payload.get("user_id", "anonymous"),
                "level": payload.get("level", "intermediate"),
                "learning_style": payload.get("learning_style", "mixed"),
                "pace": payload.get("pace", "medium"),
            },
            "session": {
                "session_id": session_id,
                "topic": payload.get("topic", ""),
                "subject": payload.get("topic", ""),
                "mode": payload.get("mode", "strict"),
                "knowledge_mode": payload.get("knowledge_mode", "external_ok"),
                "require_human_review": payload.get("require_human_review", True),
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
            logger.error("Graph invocation failed: %s", exc)
            return {
                "session_id": session_id,
                "agent_name": "system",
                "response_text": f"Orchestration error: {exc}",
                "error": str(exc),
            }

    # ── Build response ──
    final = result.get("final_response") or {}
    routing = result.get("routing") or {}
    agent_output = result.get("agent_output") or {}
    rag_found = result.get("rag_found", False)
    rag_citations = result.get("rag_citations", [])

    # Check if graph is paused at HITL interrupt
    is_interrupted = (
        not final.get("response_text")
        and agent_output.get("response_text")
    )

    if is_interrupted:
        return {
            "session_id": session_id,
            "agent_name": agent_output.get("agent_name", "unknown"),
            "response_text": agent_output.get("response_text", ""),
            "structured_data": agent_output.get("structured_data", {}),
            "citations": agent_output.get("citations", []),
            "next_action": agent_output.get("next_action", "continue"),
            "route_used": routing.get("route", ""),
            "intent": routing.get("intent", ""),
            "awaiting_feedback": True,
            "rag_found": rag_found,
            "rag_citations_count": len(rag_citations),
        }

    return {
        "session_id": session_id,
        "agent_name": final.get("agent_name", "unknown"),
        "response_text": final.get("response_text", "No response generated."),
        "structured_data": final.get("structured_data", {}),
        "citations": final.get("citations", []),
        "next_action": final.get("next_action", "continue"),
        "route_used": routing.get("route", ""),
        "intent": routing.get("intent", ""),
        "awaiting_feedback": False,
        "rag_found": rag_found,
        "rag_citations_count": len(rag_citations),
    }


app.run()
