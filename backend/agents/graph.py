"""LangGraph orchestration graph for the multi-agent tutoring platform.

Builds, compiles, and exposes the StateGraph that routes user queries
through the Manager → specialist agents → response assembly pipeline.

Graph flow:
    START → manager → [route_from_manager]
                       ├→ professor → [route_after_agent] → respond | manager | human_feedback
                       ├→ ta_problem_gen → [route_after_agent] → respond | manager | human_feedback
                       ├→ ta_problem_solve → [route_after_agent] → respond | manager | human_feedback
                       ├→ rag → [route_after_agent] → respond | manager | human_feedback
                       └→ respond → END

    human_feedback (HITL interrupt) → manager (re-evaluate with feedback)

Note: human_feedback is only reachable via route_after_agent (when
session.require_human_review=True), never directly from the manager.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.graph_state import OrchestratorState
from agents.nodes import (
    human_feedback_node,
    manager_node,
    professor_node,
    rag_node,
    respond_node,
    ta_problem_gen_node,
    ta_problem_solve_node,
)

logger = logging.getLogger(__name__)

# Maximum times the same route can appear in route_history before we bail
_MAX_ROUTE_DEPTH = 3


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def route_from_manager(state: OrchestratorState) -> str:
    """Conditional edge: read the manager's routing decision and return the
    target node name.

    Includes loop detection: if the same route appears too many times in
    route_history, bail to 'respond' to prevent infinite loops.
    """
    routing = state.get("routing", {})
    route = routing.get("route", "respond")

    # Loop detection
    history = state.get("route_history", [])
    if history.count(route) >= _MAX_ROUTE_DEPTH:
        logger.warning(
            "Loop detected: route '%s' appeared %d times. Bailing to respond.",
            route, history.count(route),
        )
        return "respond"

    # Map future stub routes to respond
    stub_routes = {"planner", "profile", "results", "future_proof"}
    if route in stub_routes:
        return "respond"

    # Validate route is a known node
    # Note: human_feedback is NOT a valid manager route -- it's only
    # reachable via route_after_agent when session.require_human_review=True
    valid_routes = {
        "professor", "ta_problem_gen", "ta_problem_solve",
        "rag", "respond",
    }
    if route not in valid_routes:
        logger.warning("Unknown route '%s'; falling back to respond.", route)
        return "respond"

    return route


def route_after_agent(state: OrchestratorState) -> str:
    """Conditional edge: decide what happens after an agent node executes.

    Rules:
    1. If agent_output.next_action signals re-routing → back to manager
    2. If agent_output.error is set → respond with error
    3. If session.require_human_review → human_feedback (HITL interrupt)
    4. Default → respond (return result to user)
    """
    output = state.get("agent_output", {})
    session = state.get("session", {})
    next_action = output.get("next_action", "continue")

    # Agent explicitly requests re-routing (e.g., professor → TA)
    if next_action in ("route_problem_ta", "route_planner"):
        return "manager"

    # TA problem solving: escalate, easier_problem, or request_hint → back to manager
    # Manager will route to ta_problem_gen with adjusted difficulty
    if next_action in ("escalate", "easier_problem", "request_hint"):
        return "manager"

    # Error → respond immediately
    if output.get("error"):
        return "respond"

    # Human-in-the-loop review requested
    if session.get("require_human_review", False):
        return "human_feedback"

    # Default: return response to user
    return "respond"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_orchestration_graph() -> StateGraph:
    """Construct the StateGraph with all nodes and conditional edges."""
    graph = StateGraph(OrchestratorState)

    # --- Register nodes ---
    graph.add_node("manager", manager_node)
    graph.add_node("professor", professor_node)
    graph.add_node("ta_problem_gen", ta_problem_gen_node)
    graph.add_node("ta_problem_solve", ta_problem_solve_node)
    graph.add_node("rag", rag_node)
    graph.add_node("respond", respond_node)
    graph.add_node("human_feedback", human_feedback_node)

    # --- Entry edge ---
    graph.add_edge(START, "manager")

    # --- Manager → conditional routing to agents ---
    # Note: human_feedback is NOT reachable from manager. It's only
    # reachable from agent nodes via route_after_agent.
    graph.add_conditional_edges(
        "manager",
        route_from_manager,
        [
            "professor",
            "ta_problem_gen",
            "ta_problem_solve",
            "rag",
            "respond",
        ],
    )

    # --- Agent nodes → conditional routing (respond, manager, or HITL) ---
    agent_targets = ["respond", "manager", "human_feedback"]
    graph.add_conditional_edges("professor", route_after_agent, agent_targets)
    graph.add_conditional_edges("ta_problem_gen", route_after_agent, agent_targets)
    graph.add_conditional_edges("ta_problem_solve", route_after_agent, agent_targets)
    graph.add_conditional_edges("rag", route_after_agent, agent_targets)

    # --- Human feedback → always back to manager ---
    graph.add_edge("human_feedback", "manager")

    # --- Respond → terminal ---
    graph.add_edge("respond", END)

    return graph


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

def compile_graph(checkpointer: Any = None) -> CompiledStateGraph:
    """Build and compile the orchestration graph.

    Args:
        checkpointer: LangGraph checkpointer for state persistence.
                      Defaults to MemorySaver (in-memory, development only).
                      Use PostgresSaver for production.
    """
    graph_builder = build_orchestration_graph()

    if checkpointer is None:
        checkpointer = MemorySaver()

    compiled = graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_feedback"],
    )

    logger.info("Orchestration graph compiled successfully.")
    return compiled


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_compiled_graph: CompiledStateGraph | None = None


def get_graph() -> CompiledStateGraph:
    """Get or create the singleton compiled orchestration graph.

    Thread-safe via Python's GIL for the simple check-and-set pattern.
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = compile_graph()
    return _compiled_graph


def reset_graph() -> None:
    """Reset the singleton graph (useful for testing)."""
    global _compiled_graph
    _compiled_graph = None
