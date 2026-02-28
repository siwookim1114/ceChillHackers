"""Shared utility helpers for agent tools."""

import json
import logging
from typing import Union

logger = logging.getLogger(__name__)


def parse_tool_input(tool_input: Union[str, dict]) -> dict:
    """Normalize tool input to a dict.

    LangChain tool-calling may deliver input as either a JSON string
    or an already-parsed dict depending on the model provider.  This
    helper handles both cases and raises a clear error on malformed
    input so callers get an actionable message instead of a stack trace.
    """
    if isinstance(tool_input, dict):
        return tool_input
    if isinstance(tool_input, str):
        try:
            parsed = json.loads(tool_input)
        except json.JSONDecodeError as exc:
            logger.error("parse_tool_input received invalid JSON: %s", tool_input[:200])
            raise ValueError(f"Tool input is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise TypeError(f"Expected JSON object, got {type(parsed).__name__}")
        return parsed
    raise TypeError(f"tool_input must be str or dict, got {type(tool_input).__name__}")
