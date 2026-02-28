from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Allow running from repository root.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_ROOT.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agents.professor_agent import invoke_professor  # noqa: E402
from agents.tools import ProfessorRespondTool  # noqa: E402
from config.config_loader import config  # noqa: E402
from db.models import (  # noqa: E402
    ProfessorTurnRequest,
    ProfessorTurnResponse,
    ProfessorTurnStrategy,
)


SCENARIO_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "professor_payloads_10.json"
)
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "professor_scenario_results.json"
)


def _load_scenarios() -> list[dict[str, Any]]:
    return json.loads(SCENARIO_FIXTURE_PATH.read_text(encoding="utf-8"))


def _print_result(index: int, total: int, result: dict[str, Any]) -> None:
    print(f"[{index}/{total}] session={result['session_id']} topic={result['topic']}")
    if result.get("error"):
        print(f"  error: {result['error']}")
        return
    if result.get("expected_strategy"):
        print(f"  expected_strategy: {result['expected_strategy']}")
    print(f"  strategy: {result['strategy']}")
    print(f"  next_action: {result['next_action']}")
    print(f"  revealed_final_answer: {result['revealed_final_answer']}")
    print(f"  citations_count: {result['citations_count']}")
    print(f"  assistant_response: {result['assistant_response']}")
    semantic_issues = result.get("semantic_issues", [])
    if semantic_issues:
        print(f"  semantic_issues: {semantic_issues}")


def _resolve_mode_default() -> str:
    mode_default = str(config.get("agents.professor.mode_default", "strict")).strip().lower()
    if mode_default in {"strict", "convenience"}:
        return mode_default
    return "strict"


def _apply_mode_default(payload: dict[str, Any]) -> dict[str, Any]:
    mode_value = payload.get("mode")
    if mode_value is not None and str(mode_value).strip():
        return payload
    normalized_payload = dict(payload)
    normalized_payload["mode"] = _resolve_mode_default()
    return normalized_payload


def main() -> int:
    load_dotenv(ROOT_DIR / ".env", override=False)

    if not os.getenv("FEATHERLESS_API_KEY", "").strip():
        print("Missing FEATHERLESS_API_KEY. Set it in .env or shell environment.")
        return 1

    # Force live Featherless path for this runner.
    os.environ["PROFESSOR_USE_LIVE_FEATHERLESS"] = "1"

    scenarios = _load_scenarios()
    if not scenarios:
        print(f"No scenarios found in {SCENARIO_FIXTURE_PATH}")
        return 1

    policy_tool = ProfessorRespondTool(config=config)
    results: list[dict[str, Any]] = []
    errors = 0
    semantic_failures = 0
    total = len(scenarios)

    for idx, payload in enumerate(scenarios, start=1):
        session_id = str(payload.get("session_id", f"scenario-{idx}"))
        topic = str(payload.get("topic", ""))
        mode = str(payload.get("mode", ""))
        try:
            effective_payload = _apply_mode_default(payload)
            request = ProfessorTurnRequest.model_validate(effective_payload)
            expected_strategy = policy_tool._choose_strategy(request).value
            session_id = request.session_id
            topic = request.topic
            mode = request.mode.value

            output = invoke_professor(payload)
            response = ProfessorTurnResponse.model_validate(output)
            semantic_issues: list[str] = []
            if request.mode.value == "strict" and response.strategy.value != expected_strategy:
                semantic_issues.append(
                    f"strategy_mismatch_strict(expected={expected_strategy}, actual={response.strategy.value})"
                )
            style_issue = policy_tool._assistant_response_style_issue(
                response.assistant_response,
                request,
                ProfessorTurnStrategy(response.strategy.value),
            )
            if style_issue:
                semantic_issues.append(style_issue)
            if semantic_issues:
                semantic_failures += 1

            result = {
                "session_id": session_id,
                "topic": topic,
                "mode": mode,
                "expected_strategy": expected_strategy,
                "strategy": response.strategy.value,
                "next_action": response.next_action.value,
                "revealed_final_answer": response.revealed_final_answer,
                "citations_count": len(response.citations),
                "assistant_response": response.assistant_response.strip(),
                "semantic_issues": semantic_issues,
            }
        except Exception as exc:  # pragma: no cover - live runner only
            errors += 1
            result = {
                "session_id": session_id,
                "topic": topic,
                "mode": mode,
                "error": f"{type(exc).__name__}: {str(exc)}",
            }

        results.append(result)
        _print_result(idx, total, result)

    output_path = Path(
        os.getenv("PROFESSOR_SCENARIO_OUTPUT_PATH", str(DEFAULT_OUTPUT_PATH))
    )
    output_path.write_text(
        json.dumps(
            {
                "results": results,
                "errors": errors,
                "semantic_failures": semantic_failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved results to {output_path}")

    if errors or semantic_failures:
        print(
            "Completed with "
            f"{errors} scenario error(s) and {semantic_failures} semantic failure(s)."
        )
        return 1

    print("Completed all scenarios successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
