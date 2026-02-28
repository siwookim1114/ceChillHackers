import json
from typing import Union

def parse_tool_input(tool_input: Union[str, dict]) -> dict:
    if isinstance(tool_input, str):
        return json.loads(tool_input)
    return tool_input
