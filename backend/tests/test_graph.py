"""Tests for the LangGraph multi-agent orchestration system.

Block 1: Intent classification (rule-based)
Block 2: Graph structure validation
Block 3: Manager node logic (fresh query + feedback handling)
Block 4: Route-after-agent logic
Block 5: Respond node assembly
Block 6: End-to-end graph invocation (mocked agents)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Ensure backend/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.graph_state import (
    AgentOutput,
    HumanFeedback,
    OrchestratorState,
    RoutingDecision,
    SessionContext,
    UserProfile,
)
from agents.nodes import (
    _classify_intent,
    manager_node,
    respond_node,
)
from agents.graph import (
    build_orchestration_graph,
    route_after_agent,
    route_from_manager,
)


# ===================================================================
# Helpers
# ===================================================================

def _base_state(**overrides) -> OrchestratorState:
    """Build a minimal valid state for testing."""
    state: dict[str, Any] = {
        "user_message": "Explain Bayes theorem",
        "user_profile": UserProfile(
            user_id="test-user",
            level="intermediate",
            learning_style="mixed",
            pace="medium",
        ),
        "session": SessionContext(
            session_id="test-session",
            topic="probability",
            subject="probability",
            mode="strict",
            knowledge_mode="internal_only",
            turn_count=0,
        ),
        "route_history": [],
        "agent_outputs_history": [],
    }
    state.update(overrides)
    return state


# ===================================================================
# Block 1: Intent Classification
# ===================================================================

class TestIntentClassification:
    """Test rule-based intent classification with regex patterns."""

    @pytest.mark.parametrize("query,expected_intent,expected_route", [
        # Concept learning → professor
        ("Explain Bayes theorem", "concept_learning", "professor"),
        ("What is eigenvalue decomposition?", "concept_learning", "professor"),
        ("Why does gradient descent work?", "concept_learning", "professor"),
        ("I need help understanding recursion", "concept_learning", "professor"),
        ("I want to learn about neural networks", "concept_learning", "professor"),
        ("How does backpropagation work?", "concept_learning", "professor"),
        # Problem generation → ta_problem_gen
        ("Give me practice problems on derivatives", "problem_gen", "ta_problem_gen"),
        ("I want to practice linear algebra", "problem_gen", "ta_problem_gen"),
        ("More exercises please", "problem_gen", "ta_problem_gen"),
        ("Quiz me on probability", "problem_gen", "ta_problem_gen"),
        ("Generate some exercises on calculus", "problem_gen", "ta_problem_gen"),
        ("More problems", "problem_gen", "ta_problem_gen"),
        # Problem solving → ta_problem_solve
        ("Check my solution for problem 3", "problem_solve", "ta_problem_solve"),
        ("Is this correct? x = 5", "problem_solve", "ta_problem_solve"),
        ("Grade my work", "problem_solve", "ta_problem_solve"),
        ("Verify my answer", "problem_solve", "ta_problem_solve"),
        ("Here is my work: step 1 apply chain rule", "problem_solve", "ta_problem_solve"),
        # Planning → planner
        ("Make me a study plan for the exam", "planning", "planner"),
        ("What should I study today?", "planning", "planner"),
        ("How should I prepare for the midterm?", "planning", "planner"),
        # Profile → profile
        ("Change my level to advanced", "profile", "profile"),
        ("Update my learning style", "profile", "profile"),
    ])
    def test_classify_intent(self, query, expected_intent, expected_route):
        result = _classify_intent(query, {})
        assert result["intent"] == expected_intent
        assert result["route"] == expected_route
        assert 0.0 <= result["confidence"] <= 1.0

    def test_default_is_concept_learning(self):
        """Ambiguous queries should default to concept_learning/professor."""
        result = _classify_intent("hello there", {})
        assert result["intent"] == "concept_learning"
        assert result["route"] == "professor"

    def test_problem_solve_priority_over_gen(self):
        """Problem-solve patterns should fire before problem-gen patterns."""
        result = _classify_intent("Check my solution for the practice problems", {})
        assert result["intent"] == "problem_solve"

    def test_empty_message(self):
        """Empty messages should still return a valid RoutingDecision."""
        result = _classify_intent("", {})
        assert result["intent"] == "concept_learning"
        assert result["route"] == "professor"


# ===================================================================
# Block 2: Graph Structure Validation
# ===================================================================

class TestGraphStructure:
    """Test that the orchestration graph has the correct shape."""

    def test_graph_builds(self):
        graph = build_orchestration_graph()
        assert graph is not None

    def test_graph_has_all_nodes(self):
        graph = build_orchestration_graph()
        expected_nodes = {
            "manager", "professor", "ta_problem_gen",
            "ta_problem_solve", "rag", "respond", "human_feedback",
        }
        assert set(graph.nodes.keys()) == expected_nodes

    def test_graph_compiles(self):
        from langgraph.checkpoint.memory import MemorySaver

        graph = build_orchestration_graph()
        compiled = graph.compile(
            checkpointer=MemorySaver(),
            interrupt_before=["human_feedback"],
        )
        assert compiled is not None

    def test_compiled_graph_has_start_and_end(self):
        from langgraph.checkpoint.memory import MemorySaver

        graph = build_orchestration_graph()
        compiled = graph.compile(
            checkpointer=MemorySaver(),
            interrupt_before=["human_feedback"],
        )
        graph_data = compiled.get_graph()
        node_ids = list(graph_data.nodes)
        assert "__start__" in node_ids
        assert "__end__" in node_ids


# ===================================================================
# Block 3: Manager Node Logic
# ===================================================================

class TestManagerNode:
    """Test manager_node routing decisions."""

    def test_fresh_query_routes_to_professor(self):
        state = _base_state(user_message="Explain Bayes theorem")
        result = manager_node(state)
        assert result["routing"]["route"] == "professor"
        assert result["routing"]["intent"] == "concept_learning"

    def test_fresh_query_routes_to_ta_gen(self):
        state = _base_state(user_message="Give me practice problems on calculus")
        result = manager_node(state)
        assert result["routing"]["route"] == "ta_problem_gen"
        assert result["routing"]["intent"] == "problem_gen"

    def test_fresh_query_routes_to_ta_solve(self):
        state = _base_state(user_message="Check my solution: x = 5")
        result = manager_node(state)
        assert result["routing"]["route"] == "ta_problem_solve"
        assert result["routing"]["intent"] == "problem_solve"

    def test_empty_message_routes_to_respond_with_error(self):
        state = _base_state(user_message="")
        result = manager_node(state)
        assert result["routing"]["route"] == "respond"
        assert result.get("error") == "Empty message provided."

    def test_feedback_approve_routes_to_respond(self):
        state = _base_state(
            human_feedback=HumanFeedback(
                action="approve",
                feedback_text="",
            ),
        )
        result = manager_node(state)
        assert result["routing"]["route"] == "respond"
        assert result["human_feedback"] is None

    def test_feedback_reroute(self):
        state = _base_state(
            human_feedback=HumanFeedback(
                action="reroute",
                feedback_text="Try the TA instead",
                reroute_to="ta_problem_gen",
            ),
        )
        result = manager_node(state)
        assert result["routing"]["route"] == "ta_problem_gen"
        assert result["human_feedback"] is None

    def test_feedback_revise_appends_context(self):
        state = _base_state(
            user_message="Explain Bayes theorem",
            human_feedback=HumanFeedback(
                action="revise",
                feedback_text="Give me more detail",
            ),
            route_history=["professor"],
        )
        result = manager_node(state)
        assert result["routing"]["route"] == "professor"
        assert "[Student feedback: Give me more detail]" in result["user_message"]

    def test_agent_reroute_signal(self):
        """When a previous agent signals route_problem_ta, manager re-routes."""
        state = _base_state(
            agent_output=AgentOutput(
                agent_name="professor",
                response_text="Let me route you to practice",
                next_action="route_problem_ta",
            ),
        )
        result = manager_node(state)
        assert result["routing"]["route"] == "ta_problem_gen"

    def test_route_history_grows(self):
        state = _base_state(route_history=["professor"])
        result = manager_node(state)
        assert len(result["route_history"]) > 1


# ===================================================================
# Block 4: Route-After-Agent Logic
# ===================================================================

class TestRouteAfterAgent:
    """Test route_after_agent conditional edge function."""

    def test_continue_goes_to_respond(self):
        state = _base_state(
            agent_output=AgentOutput(next_action="continue"),
        )
        assert route_after_agent(state) == "respond"

    def test_route_problem_ta_goes_to_manager(self):
        state = _base_state(
            agent_output=AgentOutput(next_action="route_problem_ta"),
        )
        assert route_after_agent(state) == "manager"

    def test_route_planner_goes_to_manager(self):
        state = _base_state(
            agent_output=AgentOutput(next_action="route_planner"),
        )
        assert route_after_agent(state) == "manager"

    def test_escalate_goes_to_manager(self):
        state = _base_state(
            agent_output=AgentOutput(next_action="escalate"),
        )
        assert route_after_agent(state) == "manager"

    def test_error_goes_to_respond(self):
        state = _base_state(
            agent_output=AgentOutput(next_action="continue", error="Something broke"),
        )
        assert route_after_agent(state) == "respond"

    def test_human_review_enabled(self):
        state = _base_state(
            agent_output=AgentOutput(next_action="continue"),
            session=SessionContext(
                session_id="s1",
                topic="math",
                require_human_review=True,
            ),
        )
        assert route_after_agent(state) == "human_feedback"

    def test_human_review_disabled(self):
        state = _base_state(
            agent_output=AgentOutput(next_action="continue"),
            session=SessionContext(
                session_id="s1",
                topic="math",
                require_human_review=False,
            ),
        )
        assert route_after_agent(state) == "respond"


# ===================================================================
# Block 5: Route-From-Manager Logic
# ===================================================================

class TestRouteFromManager:
    """Test route_from_manager conditional edge function."""

    def test_routes_to_professor(self):
        state = _base_state(routing=RoutingDecision(route="professor"))
        assert route_from_manager(state) == "professor"

    def test_routes_to_ta_gen(self):
        state = _base_state(routing=RoutingDecision(route="ta_problem_gen"))
        assert route_from_manager(state) == "ta_problem_gen"

    def test_stub_routes_go_to_respond(self):
        for stub in ["planner", "profile", "results", "future_proof"]:
            state = _base_state(routing=RoutingDecision(route=stub))
            assert route_from_manager(state) == "respond", f"Stub {stub} should route to respond"

    def test_unknown_route_goes_to_respond(self):
        state = _base_state(routing=RoutingDecision(route="nonexistent"))
        assert route_from_manager(state) == "respond"

    def test_loop_detection(self):
        """After 3 visits to the same route, bail to respond."""
        state = _base_state(
            routing=RoutingDecision(route="professor"),
            route_history=["professor", "professor", "professor"],
        )
        assert route_from_manager(state) == "respond"

    def test_no_loop_under_threshold(self):
        state = _base_state(
            routing=RoutingDecision(route="professor"),
            route_history=["professor", "professor"],
        )
        assert route_from_manager(state) == "professor"


# ===================================================================
# Block 6: Respond Node
# ===================================================================

class TestRespondNode:
    """Test respond_node final response assembly."""

    def test_assembles_from_agent_output(self):
        state = _base_state(
            agent_output=AgentOutput(
                agent_name="professor",
                response_text="Bayes theorem states...",
                structured_data={"strategy": "conceptual_explanation"},
                citations=[{"source_id": "kb-1", "title": "Prob Lecture"}],
                next_action="continue",
                error=None,
            ),
        )
        result = respond_node(state)
        final = result["final_response"]
        assert final["agent_name"] == "professor"
        assert "Bayes theorem" in final["response_text"]
        assert len(final["citations"]) == 1

    def test_error_state(self):
        state = _base_state(error="Something went wrong")
        result = respond_node(state)
        assert "error" in result["final_response"]["response_text"].lower()

    def test_no_agent_output(self):
        state = _base_state()
        result = respond_node(state)
        assert result["final_response"]["agent_name"] in ("system", "unknown")

    def test_stub_route_message(self):
        state = _base_state(
            routing=RoutingDecision(route="planner"),
        )
        # Remove agent_output so respond_node hits the stub path
        state.pop("agent_output", None)
        result = respond_node(state)
        assert "coming soon" in result["final_response"]["response_text"].lower()


# ===================================================================
# Block 7: End-to-End Graph (mocked agents)
# ===================================================================

class TestGraphE2E:
    """End-to-end graph tests with mocked agent implementations."""

    def _mock_professor_response(self):
        """Return a mock ProfessorTurnResponse."""
        from db.models import (
            ProfessorNextAction,
            ProfessorTurnResponse,
            ProfessorTurnStrategy,
        )
        return ProfessorTurnResponse(
            assistant_response="Great question! Bayes theorem connects prior beliefs with new evidence.",
            strategy=ProfessorTurnStrategy.CONCEPTUAL_EXPLANATION,
            revealed_final_answer=False,
            next_action=ProfessorNextAction.CONTINUE,
            citations=[],
        )

    @patch("agents.nodes._get_professor_agent")
    def test_concept_query_routes_to_professor(self, mock_get_prof):
        """A concept query should route through manager → professor → respond."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = self._mock_professor_response()
        mock_get_prof.return_value = mock_agent

        from agents.graph import compile_graph

        graph = compile_graph()
        state = _base_state(user_message="Explain Bayes theorem")
        config = {"configurable": {"thread_id": "test-e2e-1"}}

        result = graph.invoke(state, config)

        assert result["routing"]["route"] == "professor"
        assert result["routing"]["intent"] == "concept_learning"
        assert result["final_response"]["agent_name"] == "professor"
        assert "Bayes" in result["final_response"]["response_text"]
        mock_agent.run.assert_called_once()

    def test_stub_route_returns_coming_soon(self):
        """A planning query should route to planner stub → respond with 'coming soon'."""
        from agents.graph import compile_graph

        graph = compile_graph()
        state = _base_state(user_message="Make me a study plan for the exam")
        config = {"configurable": {"thread_id": "test-e2e-2"}}

        result = graph.invoke(state, config)

        assert result["routing"]["route"] == "planner"
        assert "coming soon" in result["final_response"]["response_text"].lower()

    @patch("agents.nodes._get_professor_agent")
    def test_hitl_interrupt(self, mock_get_prof):
        """With require_human_review=True, graph should pause at human_feedback."""
        mock_agent = MagicMock()
        mock_agent.run.return_value = self._mock_professor_response()
        mock_get_prof.return_value = mock_agent

        from agents.graph import compile_graph

        graph = compile_graph()
        state = _base_state(
            user_message="Explain Bayes theorem",
            session=SessionContext(
                session_id="test-hitl",
                topic="probability",
                subject="probability",
                mode="strict",
                knowledge_mode="internal_only",
                require_human_review=True,
            ),
        )
        config = {"configurable": {"thread_id": "test-e2e-hitl"}}

        # First invoke should pause at human_feedback
        result = graph.invoke(state, config)

        # Graph should have stopped before human_feedback node
        # The professor output should be in the state
        graph_state = graph.get_state(config)
        assert graph_state.next == ("human_feedback",)

    @patch("agents.nodes._get_professor_agent")
    def test_hitl_resume_approve(self, mock_get_prof):
        """After HITL interrupt, resuming with 'approve' should produce final_response."""
        from langgraph.types import Command

        mock_agent = MagicMock()
        mock_agent.run.return_value = self._mock_professor_response()
        mock_get_prof.return_value = mock_agent

        from agents.graph import compile_graph

        graph = compile_graph()
        state = _base_state(
            user_message="Explain Bayes theorem",
            session=SessionContext(
                session_id="test-hitl-approve",
                topic="probability",
                subject="probability",
                mode="strict",
                knowledge_mode="internal_only",
                require_human_review=True,
            ),
        )
        config = {"configurable": {"thread_id": "test-e2e-hitl-approve"}}

        # First invoke pauses at human_feedback
        graph.invoke(state, config)

        # Resume with approve
        result = graph.invoke(
            Command(resume={"action": "approve", "feedback_text": ""}),
            config,
        )

        assert result["final_response"]["agent_name"] == "professor"
        assert "Bayes" in result["final_response"]["response_text"]

    @patch("agents.nodes._get_professor_agent")
    def test_conversation_memory_across_turns(self, mock_get_prof):
        """agent_outputs_history should accumulate across multiple turns."""
        from db.models import (
            ProfessorNextAction,
            ProfessorTurnResponse,
            ProfessorTurnStrategy,
        )

        mock_agent = MagicMock()

        # First turn response
        mock_agent.run.return_value = ProfessorTurnResponse(
            assistant_response="Turn 1: Bayes theorem connects priors with evidence.",
            strategy=ProfessorTurnStrategy.CONCEPTUAL_EXPLANATION,
            revealed_final_answer=False,
            next_action=ProfessorNextAction.CONTINUE,
            citations=[],
        )
        mock_get_prof.return_value = mock_agent

        from agents.graph import compile_graph

        graph = compile_graph()
        thread_id = "test-memory-turns"
        config = {"configurable": {"thread_id": thread_id}}

        # Turn 1
        state1 = _base_state(user_message="Explain Bayes theorem")
        state1["agent_outputs_history"] = []
        result1 = graph.invoke(state1, config)
        assert result1["final_response"]["agent_name"] == "professor"
        assert len(result1.get("agent_outputs_history", [])) == 1

        # Turn 2 -- do NOT pass agent_outputs_history so checkpoint preserves it
        mock_agent.run.return_value = ProfessorTurnResponse(
            assistant_response="Turn 2: To explain it more simply...",
            strategy=ProfessorTurnStrategy.CONCEPTUAL_EXPLANATION,
            revealed_final_answer=False,
            next_action=ProfessorNextAction.CONTINUE,
            citations=[],
        )
        state2 = _base_state(user_message="Explain it more simply")
        # Critically: no agent_outputs_history key → checkpoint value preserved
        state2.pop("agent_outputs_history", None)
        # Reset per-turn fields
        state2["route_history"] = []
        state2["routing"] = {}
        state2["agent_output"] = {}
        state2["final_response"] = {}
        state2["error"] = None

        result2 = graph.invoke(state2, config)
        history = result2.get("agent_outputs_history", [])
        assert len(history) == 2, f"Expected 2 history entries, got {len(history)}"
        assert "Turn 1" in history[0]["response_text"]
        assert "Turn 2" in history[1]["response_text"]

        # Verify the professor saw conversation context (message_with_history)
        last_call_args = mock_agent.run.call_args
        request_msg = last_call_args[0][0].message
        assert "Conversation history" in request_msg or "Previous" in request_msg
