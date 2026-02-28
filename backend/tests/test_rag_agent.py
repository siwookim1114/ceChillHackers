"""
Document uploads are handled by the FastAPI layer, not the agent.
Run after uploading lecture slides via the API endpoint.

Run from the backend/ directory:
    pytest tests/test_rag_agent.py -v -s
"""
import json
from pathlib import Path

import pytest

from agents.rag_agent import RagAgent
from db.models import RagMode

# Helpers

def assert_valid_schema(result: dict, expected_mode: str, expected_caller: str) -> None:
    assert isinstance(result, dict), f"Result must be a dict, got {type(result)}"
    assert "answer" in result, "Missing 'answer' key"
    assert "citations" in result, "Missing 'citations' key"
    assert "found" in result, "Missing 'found' key"
    assert "mode" in result, "Missing 'mode' key"
    assert "caller" in result, "Missing 'caller' key"
    assert "tool_calls_count" in result, "Missing 'tool_calls_count' key"
    assert isinstance(result["answer"], str), "'answer' must be a string"
    assert isinstance(result["citations"], list), "'citations' must be a list"
    assert isinstance(result["found"], bool), "'found' must be a bool"
    assert result["mode"] == expected_mode, f"Mode mismatch: {result['mode']} != {expected_mode}"
    assert result["caller"] == expected_caller, f"Caller mismatch: {result['caller']} != {expected_caller}"
    assert isinstance(result["tool_calls_count"], int), "'tool_calls_count' must be int"
    assert result["tool_calls_count"] >= 0


# Fixture — one agent instance for the entire test session

@pytest.fixture(scope="session")
def agent():
    return RagAgent()


# BLOCK 1: Initialization

def test_agent_initializes(agent):
    """Agent must load LLM and core tools (+ optional exa) successfully."""
    assert agent.llm is not None, "LLM not initialized"
    tools = agent._get_tools()
    assert len(tools) >= 3, f"Expected at least 3 tools, got {len(tools)}"
    tool_names = {t.name for t in tools}
    assert {"retrieve_context", "check_ingestion_status", "list_available_documents"} <= tool_names
    # Exa tool is optional but should be present when EXA_API_KEY is set
    if agent.exa_tool is not None:
        assert "exa_web_search" in tool_names


def test_empty_prompt_returns_error(agent):
    """Empty prompt must short-circuit before hitting the LLM."""
    result = agent.run({"prompt": "  ", "caller": "professor"})
    assert result["found"] is False
    assert "error" in result
    assert result["tool_calls_count"] == 0


# BLOCK 2: Retrieve / content tests

def test_professor_retrieve_lecture_content(agent):
    """Professor asks for general lecture content — retrieve_context must be called."""
    result = agent.run({
        "prompt": "Explain the main concepts covered in the lecture slides.",
        "caller": "professor",
        "mode": "internal_only",
    })

    print("\n[PROFESSOR LECTURE RESULT]")
    print(json.dumps(result, indent=2, default=str))

    assert_valid_schema(result, expected_mode="internal_only", expected_caller="professor")
    assert result["tool_calls_count"] >= 1, (
        f"retrieve_context must be called, got tool_calls_count={result['tool_calls_count']}"
    )
    if result["found"]:
        assert len(result["citations"]) > 0, "found=True but no citations returned"
        for cite in result["citations"]:
            assert "index" in cite or "doc" in cite, f"Malformed citation: {cite}"


def test_professor_retrieve_bayes_theorem(agent):
    """
    Professor asks about Bayes' rule with subject=probability and statistics.
    Strictly validates:
      - retrieve_context called first (tool_calls_count >= 1)
      - If found: citations non-empty, answer contains Bayes/probability keywords
      - If not found: citations empty, answer explicitly says not found (no hallucination)
    """
    result = agent.run({
        "prompt": (
            "Explain Bayes' rule definition, interpretation, "
            "how to compute them, and why they matter in probabilistic analysis."
        ),
        "caller": "professor",
        "subject": "probability and statistics",
        "mode": "internal_only",
    })

    print("\n[Probability RESULT]")
    print(json.dumps(result, indent=2, default=str))

    assert_valid_schema(result, expected_mode="internal_only", expected_caller="professor")
    assert result["tool_calls_count"] >= 1, (
        f"retrieve_context must be called before answering. Got tool_calls_count={result['tool_calls_count']}"
    )

    if result["found"]:
        assert len(result["citations"]) > 0, (
            "found=True but citations list is empty — every KB-sourced fact must be cited"
        )
        for cite in result["citations"]:
            assert "doc" in cite or "index" in cite, f"Malformed citation: {cite}"

        answer_lower = result["answer"].lower()
        assert any(kw in answer_lower for kw in [
            "bayes", "posterior", "prior", "conditional", "probability"
        ]), f"Answer missing Bayes' rule keywords: {result['answer'][:300]}"
    else:
        assert result["citations"] == [], (
            f"found=False but citations non-empty: {result['citations']}"
        )
        assert (
            "not found" in result["answer"].lower()
            or "knowledge base" in result["answer"].lower()
        ), f"INTERNAL_ONLY: expected 'not found' message, got: {result['answer'][:300]}"


def test_ta_retrieve_practice_problems(agent):
    """TA asks for practice problems — retrieve_context must be called."""
    result = agent.run({
        "prompt": "Find practice problems and exercises from the lecture materials.",
        "caller": "ta",
        "mode": "internal_only",
    })

    print("\n[TA PRACTICE RESULT]")
    print(json.dumps(result, indent=2, default=str))

    assert_valid_schema(result, expected_mode="internal_only", expected_caller="ta")
    assert result["tool_calls_count"] >= 1, "TA agent must call retrieve_context"


def test_manager_list_documents(agent):
    """Manager lists available documents — list_available_documents must be called."""
    result = agent.run({
        "prompt": "List all documents available for the subject 'lecture_slides'.",
        "caller": "manager",
        "mode": "internal_only",
    })

    print("\n[LIST DOCUMENTS RESULT]")
    print(json.dumps(result, indent=2, default=str))

    assert result["tool_calls_count"] >= 1, "Manager must call list_available_documents"
    assert result.get("answer", "").strip(), "List response must not be empty"


# ===========================================================================
# BLOCK 3: Mode enforcement + schema robustness
# ===========================================================================

def test_internal_only_blocks_hallucination(agent):
    """
    Off-topic query (Byzantine architecture) guaranteed not in KB.
    INTERNAL_ONLY must return found=False, empty citations, no hallucination.
    """
    result = agent.run({
        "prompt": "Explain the history of Byzantine architecture in 15th century Constantinople.",
        "caller": "professor",
        "mode": "internal_only",
    })

    print("\n[HALLUCINATION GUARD RESULT]")
    print(json.dumps(result, indent=2, default=str))

    assert_valid_schema(result, expected_mode="internal_only", expected_caller="professor")
    assert result["tool_calls_count"] >= 1, "Agent must call retrieve_context even for unknown topics"
    assert result["found"] is False, (
        f"INTERNAL_ONLY: Byzantine architecture must not be found. Got: {result['answer'][:200]}"
    )
    assert result["citations"] == [], (
        f"citations must be [] when found=False, got {result['citations']}"
    )


def test_external_ok_uses_web_search(agent):
    """
    EXTERNAL_OK mode with off-KB topic: should trigger Exa web search.
    Verifies tool_calls_count >= 2 (KB retrieval + web search) and
    web citations are present with source_type='web'.
    """
    result = agent.run({
        "prompt": "What is the Pythagorean theorem and how is it proven?",
        "caller": "professor",
        "mode": "external_ok",
        "level": "intermediate",
    })

    print("\n[EXTERNAL_OK WEB SEARCH RESULT]")
    print(json.dumps(result, indent=2, default=str))

    assert_valid_schema(result, expected_mode="external_ok", expected_caller="professor")
    assert result["answer"].strip(), "answer must not be empty in EXTERNAL_OK mode"
    assert result["tool_calls_count"] >= 1, "Agent must still try retrieve_context first"

    # If exa tool is available, expect web search was called
    if agent.exa_tool is not None:
        assert result["tool_calls_count"] >= 2, (
            f"EXTERNAL_OK with off-KB topic should call both KB + web. "
            f"Got tool_calls_count={result['tool_calls_count']}"
        )
        # Check for web citations
        web_cites = [c for c in result["citations"] if c.get("source_type") == "web"]
        if web_cites:
            assert all("url" in wc for wc in web_cites), "Web citations must have 'url'"
            assert all(wc["index"].startswith("W") for wc in web_cites), (
                "Web citation indices must start with 'W'"
            )


def test_web_citations_labeled(agent):
    """
    Verify that citations carry source_type labels: 'kb' or 'web'.
    """
    # Internal-only should have only KB citations
    result_kb = agent.run({
        "prompt": "Explain probability concepts from the lecture.",
        "caller": "professor",
        "mode": "internal_only",
    })

    print("\n[KB CITATIONS LABELED]")
    print(json.dumps(result_kb["citations"][:3], indent=2, default=str))

    if result_kb["found"] and result_kb["citations"]:
        for cite in result_kb["citations"]:
            assert cite.get("source_type") == "kb", (
                f"KB citation missing source_type='kb': {cite}"
            )

    # External-ok with off-KB topic should have web citations
    if agent.exa_tool is not None:
        result_web = agent.run({
            "prompt": "Explain quantum entanglement and its implications.",
            "caller": "professor",
            "mode": "external_ok",
            "level": "advanced",
        })

        print("\n[WEB CITATIONS LABELED]")
        print(json.dumps(result_web["citations"][:3], indent=2, default=str))

        web_cites = [c for c in result_web["citations"] if c.get("source_type") == "web"]
        if web_cites:
            for wc in web_cites:
                assert "url" in wc, f"Web citation missing 'url': {wc}"
                assert "title" in wc, f"Web citation missing 'title': {wc}"


def test_level_beginner_professor(agent):
    """
    Beginner-level professor search should produce results tailored
    for introductory content. Uses EXTERNAL_OK to trigger web search.
    """
    if agent.exa_tool is None:
        pytest.skip("Exa tool not available")

    result = agent.run({
        "prompt": "What is machine learning?",
        "caller": "professor",
        "mode": "external_ok",
        "level": "beginner",
    })

    print("\n[BEGINNER PROFESSOR RESULT]")
    print(json.dumps(result, indent=2, default=str))

    assert_valid_schema(result, expected_mode="external_ok", expected_caller="professor")
    assert result["answer"].strip(), "Beginner professor query must return content"
    assert result["found"] is True, "Web search should find ML content"


def test_level_advanced_ta(agent):
    """
    Advanced-level TA search should produce challenging practice content.
    Uses EXTERNAL_OK to trigger web search.
    """
    if agent.exa_tool is None:
        pytest.skip("Exa tool not available")

    result = agent.run({
        "prompt": "Find challenging calculus integration problems.",
        "caller": "ta",
        "mode": "external_ok",
        "level": "advanced",
    })

    print("\n[ADVANCED TA RESULT]")
    print(json.dumps(result, indent=2, default=str))

    assert_valid_schema(result, expected_mode="external_ok", expected_caller="ta")
    assert result["answer"].strip(), "Advanced TA query must return content"


def test_internal_only_never_uses_web(agent):
    """
    INTERNAL_ONLY mode must NEVER trigger web search, even for off-KB topics.
    tool_calls_count should be exactly 1 (just KB retrieval).
    """
    result = agent.run({
        "prompt": "Explain quantum computing qubits and superposition.",
        "caller": "professor",
        "mode": "internal_only",
        "level": "intermediate",
    })

    print("\n[INTERNAL_ONLY NO WEB RESULT]")
    print(json.dumps(result, indent=2, default=str))

    assert_valid_schema(result, expected_mode="internal_only", expected_caller="professor")
    assert result["tool_calls_count"] == 1, (
        f"INTERNAL_ONLY must only call KB retrieval (1 tool call). "
        f"Got tool_calls_count={result['tool_calls_count']}"
    )
    # No web citations should be present
    web_cites = [c for c in result["citations"] if c.get("source_type") == "web"]
    assert len(web_cites) == 0, f"INTERNAL_ONLY must never have web citations: {web_cites}"


def test_output_schema_never_crashes(agent):
    """
    _build_output must always return a valid dict regardless of LLM response format.
    """
    result = agent.run({
        "prompt": "Summarize the key topics in the uploaded lecture materials.",
        "caller": "professor",
        "mode": "external_ok",
    })

    print("\n[SCHEMA ROBUSTNESS RESULT]")
    print(json.dumps(result, indent=2, default=str))

    assert isinstance(result, dict)
    assert "answer" in result
    assert result["answer"] is not None


# ===========================================================================
# BLOCK 4: EXTERNAL_ONLY mode
# ===========================================================================

def test_external_only_skips_kb(agent):
    """
    EXTERNAL_ONLY mode must skip KB retrieval and go straight to web search.
    tool_calls_count should be exactly 1 (only Exa, no KB retrieval).
    """
    if agent.exa_tool is None:
        pytest.skip("Exa tool not available")

    result = agent.run({
        "prompt": "Explain the fundamentals of quantum computing.",
        "caller": "professor",
        "mode": "external_only",
        "level": "intermediate",
    })

    print("\n[EXTERNAL_ONLY RESULT]")
    print(json.dumps(result, indent=2, default=str))

    assert_valid_schema(result, expected_mode="external_only", expected_caller="professor")
    assert result["tool_calls_count"] == 1, (
        f"EXTERNAL_ONLY should call only Exa (1 tool call). "
        f"Got tool_calls_count={result['tool_calls_count']}"
    )
    # All citations must be web-sourced
    for cite in result["citations"]:
        assert cite.get("source_type") == "web", (
            f"EXTERNAL_ONLY must only have web citations: {cite}"
        )
    # No KB citations
    kb_cites = [c for c in result["citations"] if c.get("source_type") == "kb"]
    assert len(kb_cites) == 0, f"EXTERNAL_ONLY must never have KB citations: {kb_cites}"


def test_external_only_without_exa_returns_error(agent):
    """
    EXTERNAL_ONLY mode without Exa tool must return an error, not fall back to LLM.
    """
    original_exa = agent.exa_tool
    agent.exa_tool = None

    try:
        result = agent.run({
            "prompt": "Explain neural networks.",
            "caller": "professor",
            "mode": "external_only",
        })

        print("\n[EXTERNAL_ONLY NO EXA RESULT]")
        print(json.dumps(result, indent=2, default=str))

        assert result["found"] is False
        assert "error" in result or "unavailable" in result["answer"].lower()
        assert result["tool_calls_count"] == 0
    finally:
        agent.exa_tool = original_exa


# ===========================================================================
# BLOCK 5: Dynamic mode detection (_detect_mode_upgrade) — unit tests
# ===========================================================================

class TestDynamicModeDetection:
    """Unit tests for _detect_mode_upgrade — no LLM or AWS calls needed."""

    # --- Upgrade INTERNAL_ONLY -> EXTERNAL_OK ---

    def test_more_examples_upgrades_to_external_ok(self):
        mode, upgraded = RagAgent._detect_mode_upgrade(
            "I want more examples not covered in lecture", RagMode.INTERNAL_ONLY
        )
        assert upgraded is True
        assert mode in (RagMode.EXTERNAL_OK, RagMode.EXTERNAL_ONLY)

    def test_better_explanation_upgrades_to_external_ok(self):
        mode, upgraded = RagAgent._detect_mode_upgrade(
            "I want a better explanation of eigenvalues", RagMode.INTERNAL_ONLY
        )
        assert mode == RagMode.EXTERNAL_OK
        assert upgraded is True

    def test_explain_more_simply_upgrades(self):
        mode, upgraded = RagAgent._detect_mode_upgrade(
            "Explain it more simply please", RagMode.INTERNAL_ONLY
        )
        assert mode == RagMode.EXTERNAL_OK
        assert upgraded is True

    def test_real_world_examples_upgrades(self):
        mode, upgraded = RagAgent._detect_mode_upgrade(
            "Give me real-world examples of Bayes theorem", RagMode.INTERNAL_ONLY
        )
        assert mode == RagMode.EXTERNAL_OK
        assert upgraded is True

    def test_other_sources_upgrades(self):
        mode, upgraded = RagAgent._detect_mode_upgrade(
            "What do other sources say about this?", RagMode.INTERNAL_ONLY
        )
        assert mode == RagMode.EXTERNAL_OK
        assert upgraded is True

    def test_additional_resources_upgrades(self):
        mode, upgraded = RagAgent._detect_mode_upgrade(
            "I need additional examples for this topic", RagMode.INTERNAL_ONLY
        )
        assert mode == RagMode.EXTERNAL_OK
        assert upgraded is True

    def test_search_the_web_upgrades(self):
        mode, upgraded = RagAgent._detect_mode_upgrade(
            "Search the web for more info on linear algebra", RagMode.INTERNAL_ONLY
        )
        assert mode == RagMode.EXTERNAL_OK
        assert upgraded is True

    # --- Upgrade to EXTERNAL_ONLY ---

    def test_not_on_slides_upgrades_to_external_only(self):
        mode, upgraded = RagAgent._detect_mode_upgrade(
            "I want examples not on the lecture slides", RagMode.INTERNAL_ONLY
        )
        assert mode == RagMode.EXTERNAL_ONLY
        assert upgraded is True

    def test_skip_lecture_upgrades_to_external_only(self):
        mode, upgraded = RagAgent._detect_mode_upgrade(
            "Skip the lecture material and find it online", RagMode.INTERNAL_ONLY
        )
        assert mode == RagMode.EXTERNAL_ONLY
        assert upgraded is True

    def test_dont_use_kb_upgrades_to_external_only(self):
        mode, upgraded = RagAgent._detect_mode_upgrade(
            "Don't use the knowledge base, search externally", RagMode.INTERNAL_ONLY
        )
        assert mode == RagMode.EXTERNAL_ONLY
        assert upgraded is True

    def test_web_only_upgrades_to_external_only(self):
        mode, upgraded = RagAgent._detect_mode_upgrade(
            "Find information from web only", RagMode.EXTERNAL_OK
        )
        assert mode == RagMode.EXTERNAL_ONLY
        assert upgraded is True

    def test_ignore_textbook_upgrades_to_external_only(self):
        mode, upgraded = RagAgent._detect_mode_upgrade(
            "Ignore the textbook and explain from other sources", RagMode.INTERNAL_ONLY
        )
        assert mode == RagMode.EXTERNAL_ONLY
        assert upgraded is True

    # --- No upgrade (should stay as-is) ---

    def test_normal_query_no_upgrade(self):
        mode, upgraded = RagAgent._detect_mode_upgrade(
            "Explain eigenvalues and eigenvectors", RagMode.INTERNAL_ONLY
        )
        assert mode == RagMode.INTERNAL_ONLY
        assert upgraded is False

    def test_already_external_ok_no_downgrade(self):
        mode, upgraded = RagAgent._detect_mode_upgrade(
            "Explain eigenvalues", RagMode.EXTERNAL_OK
        )
        assert mode == RagMode.EXTERNAL_OK
        assert upgraded is False

    def test_already_external_only_no_change(self):
        mode, upgraded = RagAgent._detect_mode_upgrade(
            "Not on the slides please", RagMode.EXTERNAL_ONLY
        )
        assert mode == RagMode.EXTERNAL_ONLY
        assert upgraded is False

    def test_never_downgrades_external_ok(self):
        """A normal query with EXTERNAL_OK must NOT downgrade to INTERNAL_ONLY."""
        mode, upgraded = RagAgent._detect_mode_upgrade(
            "What is a matrix?", RagMode.EXTERNAL_OK
        )
        assert mode == RagMode.EXTERNAL_OK
        assert upgraded is False


# ===========================================================================
# BLOCK 6: Dynamic detection end-to-end (integration)
# ===========================================================================

def test_dynamic_detection_triggers_web_search(agent):
    """
    A query with 'more examples' in INTERNAL_ONLY mode should be
    dynamically upgraded to EXTERNAL_OK and trigger web search.
    """
    if agent.exa_tool is None:
        pytest.skip("Exa tool not available")

    result = agent.run({
        "prompt": "Give me more examples of probability distributions beyond what's in the slides.",
        "caller": "professor",
        "mode": "internal_only",
        "level": "intermediate",
    })

    print("\n[DYNAMIC DETECTION → EXTERNAL_OK RESULT]")
    print(json.dumps(result, indent=2, default=str))

    # Mode should be upgraded (either external_ok or external_only)
    assert result["mode"] in ("external_ok", "external_only"), (
        f"Dynamic detection should upgrade mode, got mode={result['mode']}"
    )
    assert result["tool_calls_count"] >= 1


def test_dynamic_detection_external_only_skips_kb(agent):
    """
    A query explicitly saying 'not on the lecture slides' should
    trigger EXTERNAL_ONLY mode and skip KB entirely.
    """
    if agent.exa_tool is None:
        pytest.skip("Exa tool not available")

    result = agent.run({
        "prompt": "I want examples not on the lecture slides for Bayes' theorem.",
        "caller": "professor",
        "mode": "internal_only",
        "level": "intermediate",
    })

    print("\n[DYNAMIC DETECTION → EXTERNAL_ONLY RESULT]")
    print(json.dumps(result, indent=2, default=str))

    assert result["mode"] == "external_only", (
        f"Dynamic detection should upgrade to external_only, got mode={result['mode']}"
    )
    # Only 1 tool call (Exa only, no KB)
    assert result["tool_calls_count"] == 1
    # Only web citations
    kb_cites = [c for c in result["citations"] if c.get("source_type") == "kb"]
    assert len(kb_cites) == 0, f"EXTERNAL_ONLY must never have KB citations: {kb_cites}"
