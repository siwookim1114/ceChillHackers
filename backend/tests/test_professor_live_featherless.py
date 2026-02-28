from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Allow running `pytest backend/tests` from repository root.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agents.professor_agent import invoke_professor  # noqa: E402
from db.models import ProfessorTurnResponse  # noqa: E402


ROOT_DIR = BACKEND_ROOT.parent
LIVE_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "professor_payloads_live.json"
)


def _load_live_payloads() -> list[dict]:
    return json.loads(LIVE_FIXTURE_PATH.read_text(encoding="utf-8"))


def _has_required_env() -> bool:
    return bool(os.getenv("FEATHERLESS_API_KEY"))


def _live_test_enabled() -> bool:
    return os.getenv("RUN_LIVE_FEATHERLESS_TEST", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def test_professor_live_featherless_outputs_valid_schema_for_dummy_payloads(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Load local .env so the test can run from IDE without manual export.
    load_dotenv(ROOT_DIR / ".env", override=False)

    if not _live_test_enabled():
        pytest.skip("Set RUN_LIVE_FEATHERLESS_TEST=1 to run live Featherless integration test")

    if not _has_required_env():
        pytest.skip("Missing FEATHERLESS_API_KEY for live Featherless test")

    monkeypatch.setenv("PROFESSOR_USE_LIVE_FEATHERLESS", "1")

    payloads = _load_live_payloads()
    assert payloads, "Live fixture payload list must not be empty"

    with caplog.at_level(logging.WARNING):
        for payload in payloads:
            output = invoke_professor(payload)
            response = ProfessorTurnResponse.model_validate(output)

            assert isinstance(response.assistant_response, str)
            assert response.assistant_response.strip() != ""
            assert response.revealed_final_answer is False
            assert response.strategy.value in {
                "socratic_question",
                "conceptual_explanation",
                "procedural_explanation",
                "broken_down_questions",
            }
            assert response.next_action.value in {
                "continue",
                "route_problem_ta",
                "route_planner",
            }
            assert isinstance(response.citations, list)

    warning_text = caplog.text.lower()
    if "live featherless call failed" in warning_text:
        if (
            "connection error" in warning_text
            or "nodename nor servname" in warning_text
            or "temporary failure in name resolution" in warning_text
        ):
            pytest.skip(
                "Featherless network/DNS is unavailable in this environment."
            )
        if "429" in warning_text or "rate limit" in warning_text:
            pytest.skip(
                "Featherless account is rate-limited; live test cannot complete now."
            )
        if "403" in warning_text and (
            "error code: 1010" in warning_text or "access denied" in warning_text
        ):
            pytest.skip(
                "Featherless endpoint is blocked by upstream access policy "
                "(HTTP 403/1010). Run from an allowed network/environment."
            )
        pytest.fail(
            "Live Featherless path fell back to deterministic response. "
            f"Reason captured in logs: {caplog.text.strip()}"
        )
