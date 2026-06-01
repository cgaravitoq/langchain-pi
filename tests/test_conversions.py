from __future__ import annotations

import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_pi.codex_conversions import (
    build_request_body,
    messages_to_responses,
    to_tool_calls,
    to_usage_metadata,
    tool_to_responses,
)
from langchain_pi.models import (
    clamp_thinking_level,
    get_supported_thinking_levels,
    thinking_wire_value,
)


def test_system_folds_into_instructions():
    messages = [
        SystemMessage(content="Be terse."),
        HumanMessage(content="Hi"),
    ]
    instructions, items = messages_to_responses(messages, "Top system.")
    assert instructions == "Top system.\n\nBe terse."
    assert len(items) == 1
    assert items[0] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "Hi"}],
    }


def test_default_instructions():
    instructions, _ = messages_to_responses([HumanMessage(content="x")], None)
    assert instructions == "You are a helpful assistant."


def test_image_block():
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:img"}},
        ]
    )
    _, items = messages_to_responses([msg], None)
    assert items[0]["content"] == [
        {"type": "input_text", "text": "look"},
        {"type": "input_image", "image_url": "data:img"},
    ]


def test_ai_tool_call_to_function_call():
    ai = AIMessage(
        content="calling",
        tool_calls=[
            {"name": "f", "args": {"a": 1}, "id": "call_1", "type": "tool_call"}
        ],
    )
    _, items = messages_to_responses([ai], None)
    assert items[0] == {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "calling"}],
    }
    assert items[1] == {
        "type": "function_call",
        "name": "f",
        "arguments": json.dumps({"a": 1}),
        "call_id": "call_1",
    }


def test_tool_message_to_function_call_output():
    tm = ToolMessage(content="result", tool_call_id="call_1")
    _, items = messages_to_responses([tm], None)
    assert items[0] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "result",
    }


def test_tool_to_responses_strict_null():
    tool = {
        "type": "function",
        "function": {
            "name": "f",
            "description": "d",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    out = tool_to_responses(tool)
    assert out["type"] == "function"
    assert out["name"] == "f"
    assert out["strict"] is None


def test_build_request_body_shape():
    body = build_request_body(
        model="gpt-5.3-codex",
        instructions="sys",
        input_items=[{"role": "user", "content": []}],
        tools=[{"type": "function", "name": "f"}],
        tool_choice={"type": "function", "name": "f"},
        reasoning_effort="medium",
        session_id="sess-1",
    )
    assert body["store"] is False
    assert body["stream"] is True
    assert body["text"] == {"verbosity": "low"}
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["tool_choice"] == {"type": "function", "name": "f"}
    assert body["parallel_tool_calls"] is True
    assert body["prompt_cache_key"] == "sess-1"
    assert body["reasoning"] == {"effort": "medium", "summary": "auto"}


def test_build_request_body_defaults_tool_choice_auto():
    body = build_request_body(
        model="gpt-5.3-codex",
        instructions="sys",
        input_items=[],
    )
    assert body["tool_choice"] == "auto"


def test_build_request_body_off_omits_reasoning():
    body = build_request_body(
        model="gpt-5.3-codex",
        instructions="sys",
        input_items=[],
        reasoning_effort="off",
    )
    assert "reasoning" not in body
    assert "prompt_cache_key" not in body


def test_thinking_wire_minimal_maps_to_low():
    assert thinking_wire_value("gpt-5.3-codex", "minimal") == "low"
    assert thinking_wire_value("gpt-5.3-codex", "high") == "high"
    assert thinking_wire_value("gpt-5.3-codex", "xhigh") == "xhigh"
    assert thinking_wire_value("gpt-5.3-codex", "off") is None


def test_supported_levels_and_clamp():
    assert get_supported_thinking_levels("gpt-5.5") == [
        "off",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    ]
    assert clamp_thinking_level("gpt-5.5", "medium") == "medium"


def test_to_tool_calls_parses_json_args():
    out = to_tool_calls(
        [{"name": "f", "arguments": '{"x": 1}', "call_id": "c1"}]
    )
    assert out[0] == {"name": "f", "args": {"x": 1}, "id": "c1", "type": "tool_call"}


def test_usage_metadata():
    usage = {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 3},
    }
    md = to_usage_metadata(usage)
    assert md["input_tokens"] == 10
    assert md["output_tokens"] == 5
    assert md["total_tokens"] == 15
    assert md["input_token_details"]["cache_read"] == 3
