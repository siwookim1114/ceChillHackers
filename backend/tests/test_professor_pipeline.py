from __future__ import annotations

import io
import json
import logging
import sys
import urllib.error
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

# Allow running `pytest backend/tests` from repository root.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agents.professor_agent import invoke_professor  # noqa: E402
from agents.tools import ProfessorRespondTool  # noqa: E402
from db.models import ProfessorTurnResponse  # noqa: E402


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "professor_payload_valid.json"
SCENARIO_FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "professor_payloads_10.json"
)


class DictConfig:
    """Minimal config stub compatible with Config.get(dot.path)."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def get(self, key: str, default: Any = None) -> Any:
        value: Any = self.data
        for part in key.split("."):
            if not isinstance(value, dict):
                return default
            if part not in value:
                return default
            value = value[part]
        return value


def build_professor_config(
    *,
    provider: str = "featherless",
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    mode_default: str = "strict",
    use_live_featherless: bool = False,
    runtime_timeout_sec: int = 15,
    runtime_retries: int = 0,
    fallback_on_live_error: bool = True,
    socratic_default: bool = True,
    citations_enabled: bool = True,
) -> DictConfig:
    return DictConfig(
        {
            "agents": {
                "professor": {
                    "mode_default": mode_default,
                    "llm": {
                        "provider": provider,
                        "model_id": model_id,
                        "temperature": 0.3,
                        "max_tokens": 800,
                    },
                    "runtime": {
                        "use_live_featherless": use_live_featherless,
                        "timeout_sec": runtime_timeout_sec,
                        "retries": runtime_retries,
                        "fallback_on_live_error": fallback_on_live_error,
                    },
                    "tutoring": {
                        "socratic_default": socratic_default,
                        "citations_enabled": citations_enabled,
                    },
                }
            },
            "providers": {
                "featherless": {
                    "base_url": "https://api.featherless.ai/v1",
                    "api_key_env": "FEATHERLESS_API_KEY",
                    "timeout_sec": 15,
                    "user_agent": "ceChillHackers-professor/1.0",
                }
            },
        }
    )


def load_valid_payload() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def load_scenario_payloads() -> list[dict[str, Any]]:
    return json.loads(SCENARIO_FIXTURE_PATH.read_text(encoding="utf-8"))


class _DummyHttpResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> "_DummyHttpResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        _ = exc_type
        _ = exc
        _ = tb
        return False


def _patch_live_response(
    monkeypatch: pytest.MonkeyPatch,
    content: Any,
) -> None:
    response_payload = {
        "choices": [
            {
                "message": {
                    "content": content,
                }
            }
        ]
    }

    def _fake_urlopen(*args: Any, **kwargs: Any) -> _DummyHttpResponse:
        _ = args
        _ = kwargs
        return _DummyHttpResponse(response_payload)

    monkeypatch.setattr("agents.tools.urllib.request.urlopen", _fake_urlopen)


def test_tool_pipeline_accepts_dict_input_and_returns_valid_schema() -> None:
    tool = ProfessorRespondTool(config=build_professor_config())
    response_json = tool._run(load_valid_payload())
    response = ProfessorTurnResponse.model_validate_json(response_json)

    assert isinstance(response, ProfessorTurnResponse)
    assert response.revealed_final_answer is False
    assert response.next_action.value == "continue"
    assert isinstance(response.citations, list)


def test_tool_pipeline_accepts_json_string_input() -> None:
    tool = ProfessorRespondTool(config=build_professor_config())
    payload_json = json.dumps(load_valid_payload())
    response_json = tool._run(payload_json)
    response = ProfessorTurnResponse.model_validate_json(response_json)

    assert response.assistant_response.strip() != ""
    assert response.revealed_final_answer is False


def test_tool_pipeline_rejects_payload_with_extra_field() -> None:
    tool = ProfessorRespondTool(config=build_professor_config())
    payload = load_valid_payload()
    payload["unexpected"] = "should_fail"

    with pytest.raises(ValidationError):
        tool._run(payload)


def test_live_mode_not_enabled_when_provider_is_not_featherless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROFESSOR_USE_LIVE_FEATHERLESS", "1")
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test-key")

    tool = ProfessorRespondTool(
        config=build_professor_config(
            provider="bedrock_converse",
            use_live_featherless=True,
        )
    )
    assert tool.use_live_featherless is False


def test_live_path_accepts_plain_json_string_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test-key")
    tool = ProfessorRespondTool(
        config=build_professor_config(use_live_featherless=True)
    )

    _patch_live_response(
        monkeypatch,
        json.dumps(
            {
                "assistant_response": "The core concept here means tracking how one quantity changes with another.",
                "strategy": "conceptual_explanation",
                "revealed_final_answer": False,
                "next_action": "continue",
                "citations": [],
            }
        ),
    )

    payload = load_valid_payload()
    payload["mode"] = "convenience"
    payload["message"] = "Can you explain the intuition behind derivatives?"
    payload["profile"]["level"] = "intermediate"
    response = ProfessorTurnResponse.model_validate_json(tool._run(payload))
    assert response.strategy.value == "conceptual_explanation"
    assert response.revealed_final_answer is False


def test_live_path_accepts_json_embedded_in_plain_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test-key")
    tool = ProfessorRespondTool(
        config=build_professor_config(use_live_featherless=True)
    )

    _patch_live_response(
        monkeypatch,
        (
            "Sure, here is your structured answer:\n"
            '{"assistant_response":"Step 1: identify what is known. Step 2: map it to the governing rule.",'
            '"strategy":"procedural_explanation","revealed_final_answer":false,'
            '"next_action":"continue","citations":[]}'
            "\nAsk if you need another step."
        ),
    )

    payload = load_valid_payload()
    payload["mode"] = "convenience"
    payload["message"] = "Please walk me through this step by step."
    payload["profile"]["level"] = "intermediate"
    payload["profile"]["learning_style"] = "example_first"
    response = ProfessorTurnResponse.model_validate_json(tool._run(payload))
    assert response.strategy.value == "procedural_explanation"
    assert response.revealed_final_answer is False


def test_live_path_normalizes_minor_schema_drift_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test-key")
    tool = ProfessorRespondTool(
        config=build_professor_config(use_live_featherless=True)
    )

    _patch_live_response(
        monkeypatch,
        json.dumps(
            {
                "assistant_response": "",
                "strategy": "explain",
                "revealed_final_answer": True,
                "next_action": "route_ta",
                "citations": [{"any": "shape"}],
            }
        ),
    )

    payload = load_valid_payload()
    payload["mode"] = "convenience"
    payload["message"] = "Can you explain the key concept behind derivatives?"
    payload["profile"]["level"] = "intermediate"
    response = ProfessorTurnResponse.model_validate_json(tool._run(payload))
    assert response.assistant_response.strip() != ""
    assert response.strategy.value == "conceptual_explanation"
    assert response.next_action.value == "route_problem_ta"
    assert response.revealed_final_answer is False
    assert response.citations == []


def test_live_path_maps_legacy_hint_and_encouragement_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test-key")
    tool = ProfessorRespondTool(
        config=build_professor_config(use_live_featherless=True)
    )

    payload = load_valid_payload()
    payload["mode"] = "convenience"

    _patch_live_response(
        monkeypatch,
        json.dumps(
            {
                "assistant_response": "What quantity are you solving for first? What relation links it to the final target?",
                "strategy": "hint",
                "revealed_final_answer": False,
                "next_action": "continue",
                "citations": [],
            }
        ),
    )
    hint_response = ProfessorTurnResponse.model_validate_json(tool._run(payload))
    assert hint_response.strategy.value == "broken_down_questions"

    _patch_live_response(
        monkeypatch,
        json.dumps(
            {
                "assistant_response": "In this concept, what assumption should you test first?",
                "strategy": "encouragement",
                "revealed_final_answer": False,
                "next_action": "continue",
                "citations": [],
            }
        ),
    )
    socratic_payload = load_valid_payload()
    socratic_payload["mode"] = "convenience"
    socratic_payload["message"] = "I want to challenge my reasoning."
    socratic_payload["profile"]["level"] = "advanced"
    socratic_payload["profile"]["learning_style"] = "mixed"
    encouragement_response = ProfessorTurnResponse.model_validate_json(tool._run(socratic_payload))
    assert encouragement_response.strategy.value == "socratic_question"


def test_live_path_enforces_selected_strategy_in_live_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test-key")
    tool = ProfessorRespondTool(
        config=build_professor_config(use_live_featherless=True)
    )

    _patch_live_response(
        monkeypatch,
        json.dumps(
            {
                "assistant_response": "The concept means relating the target quantity to a known rate before computing.",
                "strategy": "conceptual_explanation",
                "revealed_final_answer": False,
                "next_action": "continue",
                "citations": [],
            }
        ),
    )

    response = ProfessorTurnResponse.model_validate_json(tool._run(load_valid_payload()))
    assert response.strategy.value == "broken_down_questions"


def test_live_path_retries_once_on_semantic_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test-key")
    tool = ProfessorRespondTool(
        config=build_professor_config(
            use_live_featherless=True,
            fallback_on_live_error=False,
        )
    )

    payload = load_valid_payload()
    payload["mode"] = "convenience"
    call_counter = {"count": 0}

    def _semantic_flaky_urlopen(*args: Any, **kwargs: Any) -> _DummyHttpResponse:
        _ = args
        _ = kwargs
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            response_content = json.dumps(
                {
                    "assistant_response": payload["message"],
                    "strategy": "socratic_question",
                    "revealed_final_answer": False,
                    "next_action": "continue",
                    "citations": [],
                }
            )
        else:
            response_content = json.dumps(
                {
                    "assistant_response": "What is the unknown quantity? What equation can represent that unknown?",
                    "strategy": "broken_down_questions",
                    "revealed_final_answer": False,
                    "next_action": "continue",
                    "citations": [],
                }
            )
        return _DummyHttpResponse(
            {"choices": [{"message": {"content": response_content}}]}
        )

    monkeypatch.setattr("agents.tools.urllib.request.urlopen", _semantic_flaky_urlopen)

    response = ProfessorTurnResponse.model_validate_json(tool._run(payload))
    assert call_counter["count"] == 2
    assert response.strategy.value == "broken_down_questions"
    assert response.assistant_response != payload["message"]


def test_live_path_raises_when_semantic_validation_fails_after_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test-key")
    tool = ProfessorRespondTool(
        config=build_professor_config(
            use_live_featherless=True,
            fallback_on_live_error=False,
        )
    )

    payload = load_valid_payload()
    payload["mode"] = "convenience"
    _patch_live_response(
        monkeypatch,
        json.dumps(
            {
                "assistant_response": "socratic_question",
                "strategy": "socratic_question",
                "revealed_final_answer": False,
                "next_action": "continue",
                "citations": [],
            }
        ),
    )

    with pytest.raises(ValueError, match="semantic validation after retry"):
        tool._run(payload)


@pytest.mark.parametrize(
    ("message", "expected_strategy"),
    [
        ("Please walk me through this step by step.", "procedural_explanation"),
        ("Can you break this into an easier smaller version?", "broken_down_questions"),
        ("I keep mixing up derivative and slope.", "broken_down_questions"),
        ("Can you explain the intuition behind this concept?", "conceptual_explanation"),
    ],
)
def test_strategy_policy_selects_expected_type(
    message: str,
    expected_strategy: str,
) -> None:
    tool = ProfessorRespondTool(config=build_professor_config())
    payload = load_valid_payload()
    payload["message"] = message

    response = ProfessorTurnResponse.model_validate_json(tool._run(payload))
    assert response.strategy.value == expected_strategy


def test_scenario_fixture_semantics_with_deterministic_professor_path() -> None:
    tool = ProfessorRespondTool(
        config=build_professor_config(use_live_featherless=False)
    )
    expected_by_session = {
        "scenario-001": "broken_down_questions",
        "scenario-002": "procedural_explanation",
        "scenario-003": "conceptual_explanation",
        "scenario-004": "conceptual_explanation",
        "scenario-005": "conceptual_explanation",
        "scenario-006": "conceptual_explanation",
        "scenario-007": "conceptual_explanation",
        "scenario-008": "broken_down_questions",
        "scenario-009": "socratic_question",
        "scenario-010": "procedural_explanation",
    }
    strategy_labels = {
        "socratic_question",
        "conceptual_explanation",
        "procedural_explanation",
        "broken_down_questions",
    }

    payloads = load_scenario_payloads()
    assert len(payloads) == 10
    for payload in payloads:
        session_id = payload["session_id"]
        response = ProfessorTurnResponse.model_validate_json(tool._run(payload))

        assert response.strategy.value == expected_by_session[session_id]
        normalized_response = " ".join(response.assistant_response.strip().lower().split())
        normalized_message = " ".join(payload["message"].strip().lower().split())
        assert normalized_response
        assert normalized_response != normalized_message
        assert normalized_response not in strategy_labels
        assert len(normalized_response.split()) >= 6
        assert response.revealed_final_answer is False
        assert response.next_action.value == "continue"
        assert isinstance(response.citations, list)


def test_live_path_fallback_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test-key")
    tool = ProfessorRespondTool(
        config=build_professor_config(use_live_featherless=True)
    )

    def _raise_http_error(*args: Any, **kwargs: Any) -> Any:
        _ = args
        _ = kwargs
        raise urllib.error.HTTPError(
            url="https://api.featherless.ai/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"rate_limit"}'),
        )

    monkeypatch.setattr("agents.tools.urllib.request.urlopen", _raise_http_error)

    with caplog.at_level(logging.WARNING):
        output = ProfessorTurnResponse.model_validate_json(tool._run(load_valid_payload()))

    assert output.revealed_final_answer is False
    assert "live featherless call failed" in caplog.text.lower()


def test_live_path_raises_when_fallback_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test-key")
    tool = ProfessorRespondTool(
        config=build_professor_config(
            use_live_featherless=True,
            fallback_on_live_error=False,
        )
    )

    def _raise_connection_error(*args: Any, **kwargs: Any) -> Any:
        _ = args
        _ = kwargs
        raise urllib.error.URLError("network down")

    monkeypatch.setattr("agents.tools.urllib.request.urlopen", _raise_connection_error)

    with pytest.raises(RuntimeError, match="Featherless connection error"):
        tool._run(load_valid_payload())


def test_live_path_retries_transient_error_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test-key")
    tool = ProfessorRespondTool(
        config=build_professor_config(
            use_live_featherless=True,
            runtime_retries=2,
            fallback_on_live_error=False,
        )
    )

    attempt_counter = {"count": 0}

    def _flaky_urlopen(*args: Any, **kwargs: Any) -> _DummyHttpResponse:
        _ = args
        _ = kwargs
        attempt_counter["count"] += 1
        if attempt_counter["count"] < 3:
            raise urllib.error.HTTPError(
                url="https://api.featherless.ai/v1/chat/completions",
                code=503,
                msg="Service Unavailable",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"transient"}'),
            )
        return _DummyHttpResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                {
                    "assistant_response": "The concept means relating rate of change to local slope before computation.",
                    "strategy": "conceptual_explanation",
                    "revealed_final_answer": False,
                    "next_action": "continue",
                                    "citations": [],
                                }
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("agents.tools.urllib.request.urlopen", _flaky_urlopen)

    payload = load_valid_payload()
    payload["mode"] = "convenience"
    payload["message"] = "Can you explain the intuition behind derivatives?"
    payload["profile"]["level"] = "intermediate"
    payload["profile"]["learning_style"] = "textual"
    response = ProfessorTurnResponse.model_validate_json(tool._run(payload))
    assert attempt_counter["count"] == 3
    assert response.strategy.value == "conceptual_explanation"


def test_live_path_uses_runtime_timeout_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test-key")
    tool = ProfessorRespondTool(
        config=build_professor_config(
            use_live_featherless=True,
            runtime_timeout_sec=9,
            fallback_on_live_error=False,
        )
    )

    captured_timeout: dict[str, Any] = {"value": None}

    def _capture_urlopen(*args: Any, **kwargs: Any) -> _DummyHttpResponse:
        captured_timeout["value"] = kwargs.get("timeout")
        return _DummyHttpResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "assistant_response": "Let's do one small concept check.",
                                    "strategy": "conceptual_explanation",
                                    "revealed_final_answer": False,
                                    "next_action": "continue",
                                    "citations": [],
                                }
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("agents.tools.urllib.request.urlopen", _capture_urlopen)

    payload = load_valid_payload()
    payload["mode"] = "convenience"
    payload["message"] = "Can you explain what derivatives mean conceptually?"
    payload["profile"]["level"] = "intermediate"
    _ = ProfessorTurnResponse.model_validate_json(tool._run(payload))
    assert captured_timeout["value"] == 9


def test_invoke_professor_pipeline_returns_valid_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agents.professor_agent as professor_agent_module

    monkeypatch.setattr(
        professor_agent_module,
        "config",
        build_professor_config(socratic_default=False, citations_enabled=False),
    )
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test-key")
    _patch_live_response(
        monkeypatch,
        json.dumps(
            {
                "assistant_response": "Let's do one small step first.",
                "strategy": "conceptual_explanation",
                "revealed_final_answer": False,
                "next_action": "continue",
                "citations": [],
            }
        ),
    )

    response_payload = invoke_professor(load_valid_payload())
    response = ProfessorTurnResponse.model_validate(response_payload)

    assert response.strategy.value in {
        "socratic_question",
        "conceptual_explanation",
        "procedural_explanation",
        "broken_down_questions",
    }
    assert response.revealed_final_answer is False
    assert response.citations == []


def test_invoke_professor_uses_loaded_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agents.professor_agent as professor_agent_module

    monkeypatch.setattr(
        professor_agent_module,
        "config",
        build_professor_config(
            use_live_featherless=True,
            fallback_on_live_error=False,
        ),
    )
    monkeypatch.setattr(
        professor_agent_module,
        "load_system_prompt",
        lambda: "CUSTOM_SYSTEM_PROMPT_FOR_TEST",
    )
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test-key")

    captured_payload: dict[str, Any] = {}

    def _capture_request(request: Any, **kwargs: Any) -> _DummyHttpResponse:
        _ = kwargs
        captured_payload.update(json.loads(request.data.decode("utf-8")))
        return _DummyHttpResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "assistant_response": "What quantity is changing first? What relation links it to slope?",
                                    "strategy": "broken_down_questions",
                                    "revealed_final_answer": False,
                                    "next_action": "continue",
                                    "citations": [],
                                }
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("agents.tools.urllib.request.urlopen", _capture_request)

    payload = load_valid_payload()
    _ = invoke_professor(payload)

    assert captured_payload["messages"][0]["role"] == "system"
    assert (
        captured_payload["messages"][0]["content"]
        == "CUSTOM_SYSTEM_PROMPT_FOR_TEST"
    )


def test_invoke_professor_applies_config_mode_default_when_payload_mode_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agents.professor_agent as professor_agent_module

    monkeypatch.setattr(
        professor_agent_module,
        "config",
        build_professor_config(
            mode_default="convenience",
            use_live_featherless=True,
            fallback_on_live_error=False,
        ),
    )
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test-key")

    captured_payload: dict[str, Any] = {}

    def _capture_request(request: Any, **kwargs: Any) -> _DummyHttpResponse:
        _ = kwargs
        captured_payload.update(json.loads(request.data.decode("utf-8")))
        return _DummyHttpResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "assistant_response": "What quantity is changing first? What relation links it to slope?",
                                    "strategy": "broken_down_questions",
                                    "revealed_final_answer": False,
                                    "next_action": "continue",
                                    "citations": [],
                                }
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("agents.tools.urllib.request.urlopen", _capture_request)

    payload = load_valid_payload()
    payload.pop("mode", None)
    _ = invoke_professor(payload)

    assert "mode=convenience" in captured_payload["messages"][1]["content"]
