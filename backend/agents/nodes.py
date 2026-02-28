"""LangGraph node functions for the multi-agent orchestration graph.

Each node is a plain function: node(state: OrchestratorState) -> dict
that reads from the shared state and returns a partial state update.

Nodes wrap existing agent implementations without modifying them.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
from uuid import uuid4

from agents.graph_state import (
    AgentOutput,
    HumanFeedback,
    OrchestratorState,
    RoutingDecision,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent classification (rule-based, two-tier)
# ---------------------------------------------------------------------------

# Priority 1: Problem-solving signals (user has work to grade)
_PROBLEM_SOLVE_PATTERNS = [
    re.compile(
        r"\b(check|grade|evaluate|score|review)\s+(my|this|the)\s+"
        r"(work|answer|solution|attempt|steps?)",
        re.I,
    ),
    re.compile(r"\b(did i|is this|am i)\s+(get it right|correct|wrong|right)", re.I),
    re.compile(r"\bhere'?s?\s+my\s+(work|solution|attempt|answer)", re.I),
    re.compile(r"\b(verify|validate)\s+(my|this)", re.I),
    re.compile(r"\bhere['.]?s?\s+(my|the|is\s+my)\s+(work|solution|attempt|answer|steps?)", re.I),
    re.compile(r"\bhere\s+is\s+my\s+(work|solution|attempt|answer|steps?)", re.I),
]

# Priority 2: Problem generation signals
_PROBLEM_GEN_PATTERNS = [
    re.compile(
        r"\b(give|generate|create|make|write)\s+.*\b"
        r"(problem|exercise|practice|question|drill|quiz)",
        re.I,
    ),
    re.compile(r"\b(practice|drill|quiz)\s+(me|on|with|problems?)", re.I),
    re.compile(r"\bpractice\s+problems?\b", re.I),
    re.compile(r"\bmore\s+problems?\b", re.I),
    re.compile(r"\b(i\s+want|i\s+need)\s+(to\s+)?practice\b", re.I),
    re.compile(r"\bexercises?\s+(on|for|about)\b", re.I),
    re.compile(r"\bmore\s+exercises?\b", re.I),
]

# Priority 3: Planning signals
_PLANNER_PATTERNS = [
    re.compile(r"\b(study|learning)\s+(plan|schedule|roadmap)\b", re.I),
    re.compile(r"\bwhat\s+should\s+i\s+(study|learn|review)\b", re.I),
    re.compile(r"\bplan\s+(my|a)\s+(study|session|day)\b", re.I),
    re.compile(r"\bhow\s+should\s+i\s+prepare\b", re.I),
]

# Priority 4: Profile signals
_PROFILE_PATTERNS = [
    re.compile(
        r"\b(change|update|set|switch)\s+(my\s+)?"
        r"(level|pace|style|preference|learning)",
        re.I,
    ),
    re.compile(r"\bmy\s+profile\b", re.I),
]


def _classify_intent(
    user_message: str,
    session: dict[str, Any],
) -> RoutingDecision:
    """Rule-based intent classifier with regex pattern matching.

    Checks patterns in priority order. First match wins.
    Returns RoutingDecision with intent, route, confidence, reasoning.
    """
    checks: list[tuple[list[re.Pattern], str, str]] = [
        (_PROBLEM_SOLVE_PATTERNS, "problem_solve", "ta_problem_solve"),
        (_PROBLEM_GEN_PATTERNS, "problem_gen", "ta_problem_gen"),
        (_PLANNER_PATTERNS, "planning", "planner"),
        (_PROFILE_PATTERNS, "profile", "profile"),
    ]

    for patterns, intent, route in checks:
        for pattern in patterns:
            if pattern.search(user_message):
                return RoutingDecision(
                    intent=intent,
                    route=route,
                    confidence=0.85,
                    reasoning=f"Matched rule-based pattern for {intent}",
                )

    # Default: concept learning → professor
    return RoutingDecision(
        intent="concept_learning",
        route="professor",
        confidence=0.7,
        reasoning="No specific pattern matched; defaulting to concept learning",
    )


def _classify_intent_llm(
    user_message: str,
    session: dict[str, Any],
) -> RoutingDecision:
    """LLM-based intent classification fallback.

    Used when rule-based classification confidence is low or as a
    configurable alternative via config.agents.manager.classification_mode.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    from prompts.manager_prompt import (
        MANAGER_CLASSIFICATION_PROMPT,
        build_manager_user_prompt,
    )

    api_key = os.environ.get("FEATHERLESSAI_API_KEY")
    if not api_key:
        logger.warning("No FEATHERLESSAI_API_KEY for LLM classification; using rule-based")
        return _classify_intent(user_message, session)

    try:
        from config.config_loader import config

        llm = ChatOpenAI(
            model=config.get("llm.model", "Qwen/Qwen2.5-32B-Instruct"),
            base_url=config.get("llm.base_url", "https://api.featherless.ai/v1"),
            api_key=api_key,
            temperature=0,
        )

        user_prompt = build_manager_user_prompt(
            user_message=user_message,
            topic=session.get("topic", ""),
            level=session.get("level", ""),
        )

        response = llm.invoke([
            SystemMessage(content=MANAGER_CLASSIFICATION_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        raw = response.content.strip()
        # Strip markdown fences
        if raw.startswith("```"):
            first_nl = raw.find("\n")
            if first_nl != -1:
                raw = raw[first_nl + 1:]
            if raw.rstrip().endswith("```"):
                raw = raw.rstrip()[:-3].rstrip()

        parsed = json.loads(raw)

        intent_to_route = {
            "concept_learning": "professor",
            "problem_gen": "ta_problem_gen",
            "problem_solve": "ta_problem_solve",
            "planning": "planner",
            "profile": "profile",
            "general": "rag",
        }

        intent = parsed.get("intent", "concept_learning")
        return RoutingDecision(
            intent=intent,
            route=intent_to_route.get(intent, "professor"),
            confidence=float(parsed.get("confidence", 0.5)),
            reasoning=parsed.get("reasoning", "LLM classification"),
        )
    except Exception as exc:
        logger.warning("LLM classification failed: %s; falling back to rule-based", exc)
        return _classify_intent(user_message, session)


# ---------------------------------------------------------------------------
# Agent singletons (avoid re-initialization per request)
# ---------------------------------------------------------------------------

_rag_agent = None
_professor_agent = None


def _get_rag_agent():
    global _rag_agent
    if _rag_agent is None:
        from agents.rag_agent import RagAgent
        _rag_agent = RagAgent()
    return _rag_agent


def _get_professor_agent():
    global _professor_agent
    if _professor_agent is None:
        from agents.professor_agent import ProfessorAgent
        _professor_agent = ProfessorAgent()
    return _professor_agent


# ---------------------------------------------------------------------------
# Node: Manager
# ---------------------------------------------------------------------------

def manager_node(state: OrchestratorState) -> dict:
    """Entry point node. Classifies intent, decides routing, handles HITL feedback.

    Reads: user_message, user_profile, session, human_feedback, agent_output
    Writes: routing, route_history
    """
    # --- Handle human feedback if present ---
    feedback = state.get("human_feedback")
    if feedback and feedback.get("action"):
        action = feedback["action"]

        if action == "approve":
            return {
                "routing": RoutingDecision(
                    intent="feedback_approve",
                    route="respond",
                    confidence=1.0,
                    reasoning="User approved the response",
                ),
                "route_history": state.get("route_history", []) + ["respond"],
                "human_feedback": None,
            }

        if action == "reroute" and feedback.get("reroute_to"):
            target = feedback["reroute_to"]
            return {
                "routing": RoutingDecision(
                    intent="feedback_reroute",
                    route=target,
                    confidence=1.0,
                    reasoning=f"User requested reroute to {target}",
                ),
                "route_history": state.get("route_history", []) + [target],
                "human_feedback": None,
            }

        if action == "revise":
            # Re-invoke the same agent with feedback appended to message
            prev_history = state.get("route_history", [])
            previous_route = prev_history[-1] if prev_history else "professor"
            # Skip non-agent routes
            agent_routes = {"professor", "ta_problem_gen", "ta_problem_solve", "rag"}
            if previous_route not in agent_routes:
                previous_route = "professor"

            feedback_text = feedback.get("feedback_text", "")
            revised_message = state.get("user_message", "")
            if feedback_text:
                revised_message = f"{revised_message}\n\n[Student feedback: {feedback_text}]"

            return {
                "routing": RoutingDecision(
                    intent="feedback_revise",
                    route=previous_route,
                    confidence=1.0,
                    reasoning=f"User requested revision, re-routing to {previous_route}",
                ),
                "route_history": state.get("route_history", []) + [previous_route],
                "user_message": revised_message,
                "human_feedback": None,
            }

        # cancel or unknown action
        return {
            "routing": RoutingDecision(
                intent="feedback_cancel",
                route="respond",
                confidence=1.0,
                reasoning="User cancelled or unknown feedback action",
            ),
            "route_history": state.get("route_history", []) + ["respond"],
            "human_feedback": None,
        }

    # --- Check if a previous agent signaled re-routing ---
    agent_output = state.get("agent_output")
    if agent_output and agent_output.get("next_action") in (
        "route_problem_ta",
        "route_planner",
    ):
        next_action = agent_output["next_action"]
        route_map = {
            "route_problem_ta": ("problem_gen", "ta_problem_gen"),
            "route_planner": ("planning", "planner"),
        }
        intent, route = route_map[next_action]
        return {
            "routing": RoutingDecision(
                intent=intent,
                route=route,
                confidence=0.95,
                reasoning=f"Previous agent signaled {next_action}",
            ),
            "route_history": state.get("route_history", []) + [route],
        }

    # --- Fresh classification of user message ---
    user_message = state.get("user_message", "")
    session = state.get("session", {})

    if not user_message.strip():
        return {
            "routing": RoutingDecision(
                intent="error",
                route="respond",
                confidence=1.0,
                reasoning="Empty user message",
            ),
            "route_history": state.get("route_history", []) + ["respond"],
            "error": "Empty message provided.",
        }

    # Rule-based classification
    routing = _classify_intent(user_message, session)

    # Optional: LLM classification when rule-based confidence is low
    # Uncomment for hybrid mode:
    # if routing["confidence"] < 0.6:
    #     routing = _classify_intent_llm(user_message, session)

    route = routing.get("route", "professor")
    return {
        "routing": routing,
        "route_history": state.get("route_history", []) + [route],
    }


# ---------------------------------------------------------------------------
# Node: Professor
# ---------------------------------------------------------------------------

def professor_node(state: OrchestratorState) -> dict:
    """Invoke ProfessorAgent for concept explanation / Socratic tutoring.

    Reads: user_message, user_profile, session, agent_outputs_history
    Writes: agent_output, agent_outputs_history, rag_context, rag_citations, rag_found
    """
    from db.models import ProfessorMode, ProfessorProfile, ProfessorTurnRequest

    profile_data = state.get("user_profile", {})
    session = state.get("session", {})

    # Keep the raw user message for RAG retrieval (conversation history
    # would pollute FAISS search). Pass conversation context separately
    # by appending it AFTER the core question so the RAG query stays clean.
    user_message = state.get("user_message", "")
    history = state.get("agent_outputs_history", [])

    # Build the message for the professor: raw question first (for RAG),
    # then conversation context appended (for the LLM to consider).
    if history:
        context_parts = []
        for prev in history[-4:]:
            agent = prev.get("agent_name", "assistant")
            text = prev.get("response_text", "")
            if len(text) > 600:
                text = text[:600] + "..."
            context_parts.append(f"[Previous {agent} response]: {text}")
        conversation_ctx = "\n".join(context_parts)
        # The professor agent uses request.message for BOTH RAG search and
        # LLM prompt. RAG _enrich_query_for_search only uses the beginning
        # of the text, so put the raw question FIRST, then history after a
        # clear separator the LLM can use but FAISS will de-weight.
        professor_message = (
            f"{user_message}\n\n"
            f"--- conversation context (do NOT repeat previous answers) ---\n"
            f"{conversation_ctx}"
        )
    else:
        professor_message = user_message

    try:
        request = ProfessorTurnRequest(
            session_id=session.get("session_id", uuid4().hex),
            message=professor_message,
            topic=session.get("topic", "general"),
            mode=ProfessorMode(session.get("mode", "strict")),
            profile=ProfessorProfile(
                level=profile_data.get("level", "intermediate"),
                learning_style=profile_data.get("learning_style", "mixed"),
                pace=profile_data.get("pace", "medium"),
            ),
        )

        agent = _get_professor_agent()
        response = agent.run(request)

        # The ProfessorAgent calls RAG internally but doesn't expose
        # citations on the response (hardcoded []). Retrieve them
        # separately so the orchestrator can surface them to the UI.
        rag_citations = []
        rag_context = ""
        rag_found = False
        try:
            rag_agent = _get_rag_agent()
            rag_result = rag_agent.run({
                "prompt": state.get("user_message", ""),
                "caller": "professor",
                "subject": session.get("topic", ""),
                "mode": "internal_only",
                "level": profile_data.get("level", "intermediate"),
                "retrieve_only": True,
            })
            rag_citations = rag_result.get("citations", [])
            rag_context = rag_result.get("context", "")
            rag_found = rag_result.get("found", False)
        except Exception as rag_exc:
            logger.warning("Professor node RAG citation fetch failed: %s", rag_exc)

        output = AgentOutput(
            agent_name="professor",
            response_text=response.assistant_response,
            structured_data=response.model_dump(mode="json"),
            citations=rag_citations,
            next_action=response.next_action.value,
            error=None,
        )

        return {
            "agent_output": output,
            "agent_outputs_history": state.get("agent_outputs_history", []) + [output],
            "rag_context": rag_context,
            "rag_citations": rag_citations,
            "rag_found": rag_found,
        }

    except Exception as exc:
        logger.error("Professor node failed: %s", exc)
        output = AgentOutput(
            agent_name="professor",
            response_text=f"Professor agent encountered an error: {exc}",
            structured_data={},
            citations=[],
            next_action="continue",
            error=str(exc),
        )
        return {
            "agent_output": output,
            "agent_outputs_history": state.get("agent_outputs_history", []) + [output],
        }


# ---------------------------------------------------------------------------
# Node: TA Problem Generation
# ---------------------------------------------------------------------------

def ta_problem_gen_node(state: OrchestratorState) -> dict:
    """Invoke ProblemGenTATool for practice problem generation.

    Reads: user_message, user_profile, session
    Writes: agent_output, agent_outputs_history
    """
    from agents.TA_tools import ProblemGenTATool
    from db.models import (
        DifficultyBand,
        DifficultyCurve,
        KnowledgeMode,
        LearnerProfile,
        ProblemGenTARequest,
    )

    profile_data = state.get("user_profile", {})
    session = state.get("session", {})

    try:
        request = ProblemGenTARequest(
            request_id=uuid4().hex,
            user_id=profile_data.get("user_id", "anonymous"),
            session_id=session.get("session_id", uuid4().hex),
            topic=session.get("topic", "general"),
            profile=LearnerProfile(
                level=profile_data.get("level", "intermediate"),
                learning_style=profile_data.get("learning_style", "mixed"),
                pace=profile_data.get("pace", "medium"),
            ),
            mode=KnowledgeMode(session.get("knowledge_mode", "internal_only")),
            desired_difficulty_curve=DifficultyCurve(
                current=DifficultyBand.EASY,
                target=DifficultyBand.MEDIUM,
            ),
            num_problems=3,
        )

        tool = ProblemGenTATool()
        raw_result = tool._run(request.model_dump(mode="json"))
        result = json.loads(raw_result)

        if "error" in result:
            raise ValueError(result["error"])

        # Build user-friendly response text from generated problems
        problems = result.get("problems", [])
        response_parts = [f"Here are {len(problems)} practice problem(s):\n"]
        for i, p in enumerate(problems, 1):
            response_parts.append(
                f"**Problem {i}** ({p.get('difficulty', 'medium')}): "
                f"{p.get('statement', '')}"
            )
        response_text = "\n\n".join(response_parts)

        output = AgentOutput(
            agent_name="ta_problem_gen",
            response_text=response_text,
            structured_data=result,
            citations=result.get("citations", []),
            next_action="continue",
            error=None,
        )

        return {
            "agent_output": output,
            "agent_outputs_history": state.get("agent_outputs_history", []) + [output],
        }

    except Exception as exc:
        logger.error("TA Problem Gen node failed: %s", exc)
        output = AgentOutput(
            agent_name="ta_problem_gen",
            response_text=f"Problem generation encountered an error: {exc}",
            structured_data={},
            citations=[],
            next_action="continue",
            error=str(exc),
        )
        return {
            "agent_output": output,
            "agent_outputs_history": state.get("agent_outputs_history", []) + [output],
        }


# ---------------------------------------------------------------------------
# Node: TA Problem Solving
# ---------------------------------------------------------------------------

def ta_problem_solve_node(state: OrchestratorState) -> dict:
    """Invoke ProblemSolvingTATool for grading student work.

    Reads: user_message, user_profile, session, agent_outputs_history
    Writes: agent_output, agent_outputs_history

    Note: This node needs structured input (scan_parse, rubric, problem_ref).
    For Phase 0, it extracts what it can from the user message and prior
    problem generation output. Full structured input comes from the frontend
    in production.
    """
    from agents.TA_tools import ProblemSolvingTATool
    from db.models import (
        KnowledgeMode,
        LearnerProfile,
        ProblemReference,
        ProblemSolvingTARequest,
        RubricCriterion,
        ScanParseInput,
        StudentStep,
    )

    profile_data = state.get("user_profile", {})
    session = state.get("session", {})
    user_message = state.get("user_message", "")

    try:
        # Try to find problem context from prior problem_gen output
        topic = session.get("topic", "general")
        problem_id = session.get("current_problem_id")
        problem_statement = None

        for prev_output in reversed(state.get("agent_outputs_history", [])):
            if prev_output.get("agent_name") == "ta_problem_gen":
                gen_data = prev_output.get("structured_data", {})
                problems = gen_data.get("problems", [])
                if problems:
                    first_problem = problems[0]
                    problem_id = problem_id or first_problem.get("problem_id")
                    problem_statement = first_problem.get("statement")
                    topic = first_problem.get("topic", topic)
                break

        request = ProblemSolvingTARequest(
            request_id=uuid4().hex,
            user_id=profile_data.get("user_id", "anonymous"),
            attempt_id=uuid4().hex,
            session_id=session.get("session_id", uuid4().hex),
            profile=LearnerProfile(
                level=profile_data.get("level", "intermediate"),
                learning_style=profile_data.get("learning_style", "mixed"),
                pace=profile_data.get("pace", "medium"),
            ),
            mode=KnowledgeMode(session.get("knowledge_mode", "internal_only")),
            problem_ref=ProblemReference(
                problem_id=problem_id or uuid4().hex,
                statement=problem_statement,
                topic=topic,
            ),
            scan_parse=ScanParseInput(
                problem_statement=problem_statement,
                steps=[
                    StudentStep(
                        step_index=1,
                        content=user_message,
                    ),
                ],
                final_answer=user_message,
            ),
            rubric=[
                RubricCriterion(
                    criterion_id="method_selection",
                    description="Chooses an appropriate method.",
                    max_points=3,
                ),
                RubricCriterion(
                    criterion_id="procedure_execution",
                    description="Executes steps in a valid order.",
                    max_points=4,
                ),
                RubricCriterion(
                    criterion_id="justification",
                    description="Justifies key transitions.",
                    max_points=3,
                ),
            ],
        )

        tool = ProblemSolvingTATool()
        raw_result = tool._run(request.model_dump(mode="json"))
        result = json.loads(raw_result)

        if "error" in result:
            raise ValueError(result["error"])

        response_text = result.get("feedback_message", "Evaluation complete.")
        verdict = result.get("overall_verdict", "")
        score = result.get("partial_score", {})
        if verdict and score:
            response_text = (
                f"**Verdict**: {verdict.replace('_', ' ').title()}\n"
                f"**Score**: {score.get('earned_points', 0)}/{score.get('max_points', 0)} "
                f"({score.get('percent', 0)}%)\n\n"
                f"{response_text}"
            )

        output = AgentOutput(
            agent_name="ta_problem_solve",
            response_text=response_text,
            structured_data=result,
            citations=result.get("citations", []),
            next_action=result.get("recommended_next_action", "continue"),
            error=None,
        )

        return {
            "agent_output": output,
            "agent_outputs_history": state.get("agent_outputs_history", []) + [output],
        }

    except Exception as exc:
        logger.error("TA Problem Solve node failed: %s", exc)
        output = AgentOutput(
            agent_name="ta_problem_solve",
            response_text=f"Solution evaluation encountered an error: {exc}",
            structured_data={},
            citations=[],
            next_action="continue",
            error=str(exc),
        )
        return {
            "agent_output": output,
            "agent_outputs_history": state.get("agent_outputs_history", []) + [output],
        }


# ---------------------------------------------------------------------------
# Node: RAG (direct KB query without professor/TA framing)
# ---------------------------------------------------------------------------

def rag_node(state: OrchestratorState) -> dict:
    """Invoke RagAgent for direct knowledge base queries.

    Reads: user_message, session
    Writes: agent_output, agent_outputs_history, rag_context, rag_citations, rag_found
    """
    session = state.get("session", {})

    try:
        agent = _get_rag_agent()
        result = agent.run({
            "prompt": state.get("user_message", ""),
            "caller": "manager",
            "subject": session.get("subject", session.get("topic", "")),
            "mode": session.get("knowledge_mode", "internal_only"),
            "level": state.get("user_profile", {}).get("level", "intermediate"),
        })

        output = AgentOutput(
            agent_name="rag",
            response_text=result.get("answer", "No content found."),
            structured_data=result,
            citations=result.get("citations", []),
            next_action="continue",
            error=result.get("error"),
        )

        return {
            "agent_output": output,
            "agent_outputs_history": state.get("agent_outputs_history", []) + [output],
            "rag_context": result.get("context", result.get("answer", "")),
            "rag_citations": result.get("citations", []),
            "rag_found": result.get("found", False),
        }

    except Exception as exc:
        logger.error("RAG node failed: %s", exc)
        output = AgentOutput(
            agent_name="rag",
            response_text=f"Knowledge retrieval encountered an error: {exc}",
            structured_data={},
            citations=[],
            next_action="continue",
            error=str(exc),
        )
        return {
            "agent_output": output,
            "agent_outputs_history": state.get("agent_outputs_history", []) + [output],
            "rag_found": False,
        }


# ---------------------------------------------------------------------------
# Node: Respond (terminal -- assembles final API response)
# ---------------------------------------------------------------------------

def respond_node(state: OrchestratorState) -> dict:
    """Assemble the final response from the most recent agent output.

    Reads: agent_output, routing, error
    Writes: final_response
    """
    error = state.get("error")
    if error:
        return {
            "final_response": {
                "agent_name": "system",
                "response_text": f"An error occurred: {error}",
                "structured_data": {},
                "citations": [],
                "next_action": "continue",
            },
        }

    agent_output = state.get("agent_output")
    if not agent_output:
        # No agent was invoked -- stub response for unimplemented routes
        routing = state.get("routing", {})
        route = routing.get("route", "unknown")
        stub_routes = {"planner", "profile", "results", "future_proof"}

        if route in stub_routes:
            return {
                "final_response": {
                    "agent_name": route,
                    "response_text": (
                        f"The {route.replace('_', ' ').title()} agent is coming soon. "
                        "This feature is under development."
                    ),
                    "structured_data": {},
                    "citations": [],
                    "next_action": "continue",
                },
            }

        return {
            "final_response": {
                "agent_name": "system",
                "response_text": "No response was generated. Please try rephrasing your question.",
                "structured_data": {},
                "citations": [],
                "next_action": "continue",
            },
        }

    return {
        "final_response": {
            "agent_name": agent_output.get("agent_name", "unknown"),
            "response_text": agent_output.get("response_text", ""),
            "structured_data": agent_output.get("structured_data", {}),
            "citations": agent_output.get("citations", []),
            "next_action": agent_output.get("next_action", "continue"),
        },
    }


# ---------------------------------------------------------------------------
# Node: Human Feedback (HITL interrupt point)
# ---------------------------------------------------------------------------

def human_feedback_node(state: OrchestratorState) -> dict:
    """Interrupt point for human-in-the-loop feedback.

    Uses langgraph's interrupt() to pause graph execution. When resumed,
    the feedback value is written to state.human_feedback.

    The graph is compiled with interrupt_before=["human_feedback"], so
    execution pauses BEFORE this node runs. When Command(resume=value)
    is called, this node receives the value and writes it to state.
    """
    from langgraph.types import interrupt

    # Show current agent output to user for review
    feedback_value = interrupt({
        "type": "human_feedback_request",
        "agent_output": state.get("agent_output"),
        "prompt": "Review this response. Choose: approve, revise, or reroute.",
        "options": ["approve", "revise", "reroute", "cancel"],
    })

    # feedback_value is provided by Command(resume=...) when the user responds
    if isinstance(feedback_value, dict):
        return {
            "human_feedback": HumanFeedback(
                feedback_text=feedback_value.get("feedback_text", ""),
                action=feedback_value.get("action", "approve"),
                reroute_to=feedback_value.get("reroute_to"),
            ),
            "awaiting_feedback": False,
        }

    # Simple string action (e.g., just "approve")
    return {
        "human_feedback": HumanFeedback(
            feedback_text="",
            action=str(feedback_value) if feedback_value else "approve",
            reroute_to=None,
        ),
        "awaiting_feedback": False,
    }
