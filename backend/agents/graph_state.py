"""LangGraph shared state schema for the multi-agent orchestration graph.

Defines the OrchestratorState TypedDict that flows between all nodes in the
LangGraph StateGraph. Every node reads from and writes partial updates to
this shared state.

Design principles:
- TypedDict (not Pydantic) for native LangGraph compatibility.
- total=False: fields are populated incrementally as nodes execute.
- Agent-specific Pydantic models (ProfessorTurnRequest, etc.) are constructed
  inside node functions from these lightweight state dicts.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


# ---------------------------------------------------------------------------
# AgentRoute -- every possible destination the Manager can pick
# ---------------------------------------------------------------------------
AgentRoute = Literal[
    "professor",
    "ta_problem_gen",
    "ta_problem_solve",
    "rag",
    "planner",              # Phase 0 stub -- future
    "profile",              # Phase 0 stub -- future
    "results",              # Phase 3 stub -- future
    "future_proof",         # Phase 0 stub -- future
    "respond",              # Terminal: assemble and return response
    "human_feedback",       # Interrupt for human-in-the-loop
]


# ---------------------------------------------------------------------------
# Sub-state types
# ---------------------------------------------------------------------------
class UserProfile(TypedDict, total=False):
    """Learner profile carried through the graph.

    Maps to LearnerProfile / ProfessorProfile from db.models.
    """
    user_id: str
    level: str                  # beginner | intermediate | advanced
    learning_style: str         # visual | textual | example_first | mixed
    pace: str                   # slow | medium | fast


class SessionContext(TypedDict, total=False):
    """Per-session metadata that persists across turns."""
    session_id: str
    topic: str
    subject: str
    mode: str                   # strict | convenience (professor mode)
    knowledge_mode: str         # internal_only | external_ok | external_only
    turn_count: int
    current_problem_id: str     # set when in problem-solving flow
    require_human_review: bool  # HITL toggle


class RoutingDecision(TypedDict, total=False):
    """Manager's classification of the user query."""
    intent: str                 # concept_learning | problem_gen | problem_solve | planning | profile | general
    route: str                  # AgentRoute value -- which node to execute next
    confidence: float           # 0.0-1.0 confidence in classification
    reasoning: str              # short explanation for debugging


class AgentOutput(TypedDict, total=False):
    """Standardized output from any agent node."""
    agent_name: str
    response_text: str
    structured_data: dict[str, Any]     # agent-specific payload
    citations: list[dict[str, Any]]
    next_action: str                    # continue | route_problem_ta | route_planner | etc.
    error: str | None


class HumanFeedback(TypedDict, total=False):
    """Captured human-in-the-loop feedback."""
    feedback_text: str
    action: str                 # approve | revise | reroute | cancel
    reroute_to: str | None      # target AgentRoute if action == reroute


# ---------------------------------------------------------------------------
# OrchestratorState -- the main graph state
# ---------------------------------------------------------------------------
class OrchestratorState(TypedDict, total=False):
    """Top-level state flowing through the LangGraph orchestration graph.

    Every node reads from and writes partial updates to this state.
    LangGraph manages state persistence and checkpointing.
    """
    # --- User input ---
    user_message: str
    user_profile: UserProfile
    session: SessionContext

    # --- Routing ---
    routing: RoutingDecision
    route_history: list[str]            # visited nodes (for loop detection)

    # --- Agent outputs ---
    agent_output: AgentOutput           # most recent agent output
    agent_outputs_history: list[AgentOutput]  # full history this turn

    # --- RAG context (shared resource for professor/TA) ---
    rag_context: str
    rag_citations: list[dict[str, Any]]
    rag_found: bool

    # --- Human-in-the-loop ---
    human_feedback: HumanFeedback | None
    awaiting_feedback: bool

    # --- Final response ---
    final_response: dict[str, Any]

    # --- Error handling ---
    error: str | None
