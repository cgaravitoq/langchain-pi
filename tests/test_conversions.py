from __future__ import annotations

import json

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from open_langchain.codex_conversions import (
    build_request_body,
    messages_to_responses,
    to_tool_calls,
    to_usage_metadata,
    tool_to_responses,
)
from open_langchain.models import (
    OPENAI_CODEX_MODELS,
    calculate_cost,
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


def test_codex_catalog_matches_chatgpt_account_support():
    assert set(OPENAI_CODEX_MODELS) == {
        "gpt-5.3-codex-spark",
        "gpt-5.5",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-6-astra",
    }


def test_codex_5_6_pricing_and_context():
    expected = {
        "gpt-5.6-sol": ("GPT-5.6 Sol", 4, 20, 0.4, 5),
        "gpt-5.6-terra": ("GPT-5.6 Terra", 2, 12, 0.2, 2.5),
        "gpt-5.6-luna": ("GPT-5.6 Luna", 0.2, 1.2, 0.02, 0.25),
    }
    for model, (name, cost_in, cost_out, cache_read, cache_write) in expected.items():
        meta = OPENAI_CODEX_MODELS[model]
        assert meta["name"] == name
        assert meta["input"] == ["text", "image"]
        assert meta["context_window"] == 1050000
        assert meta["cost"] == {
            "input": cost_in,
            "output": cost_out,
            "cache_read": cache_read,
            "cache_write": cache_write,
        }
    assert OPENAI_CODEX_MODELS["gpt-5.5"]["context_window"] == 1050000


def test_codex_gpt_6_astra_matches_models_dev():
    astra = OPENAI_CODEX_MODELS["gpt-6-astra"]
    assert astra["name"] == "GPT-6 Astra"
    assert astra["context_window"] == 1050000
    assert astra["cost"] == {
        "input": 10,
        "output": 50,
        "cache_read": 1,
        "cache_write": 12.5,
    }


def test_calculate_cost_uses_catalog_pricing():
    cost = calculate_cost("gpt-5.6-luna", {"input": 1_000_000, "output": 1_000_000})
    assert cost == 2.2


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.5", {"input": 10, "output": 45, "cache_read": 1}),
        (
            "gpt-5.6-sol",
            {"input": 8, "output": 30, "cache_read": 0.8, "cache_write": 10},
        ),
        (
            "gpt-5.6-terra",
            {"input": 4, "output": 18, "cache_read": 0.4, "cache_write": 5},
        ),
        (
            "gpt-5.6-luna",
            {"input": 0.4, "output": 1.8, "cache_read": 0.04, "cache_write": 0.5},
        ),
        (
            "gpt-6-astra",
            {"input": 20, "output": 75, "cache_read": 2, "cache_write": 25},
        ),
    ],
)
def test_codex_tiers_match_models_dev(model, expected):
    assert OPENAI_CODEX_MODELS[model]["tiers"] == [
        {"context": 272000, "cost": expected}
    ]


def test_calculate_cost_stays_on_base_rate_at_the_tier_context():
    cost = calculate_cost("gpt-5.6-luna", {"input": 272000})
    assert cost == 0.054400000000000004


def test_calculate_cost_jumps_to_tier_rate_above_the_tier_context():
    cost = calculate_cost("gpt-5.6-luna", {"input": 272001})
    assert cost == 0.1088004


def test_calculate_cost_counts_cache_read_toward_the_prompt():
    base = calculate_cost("gpt-5.6-luna", {"input": 200000, "cache_read": 72000})
    tier = calculate_cost("gpt-5.6-luna", {"input": 200000, "cache_read": 72001})
    assert base == 0.04144
    assert tier == 0.08288004


def test_calculate_cost_ignores_tiers_for_models_without_them():
    cost = calculate_cost("gpt-5.3-codex-spark", {"input": 272001, "output": 10})
    assert cost == 0.47614175


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


def test_calculate_cost_counts_cache_write_toward_the_prompt():
    base = calculate_cost("gpt-5.6-terra", {"input": 271000, "cache_write": 1000})
    tier = calculate_cost("gpt-5.6-terra", {"input": 271000, "cache_write": 1001})
    assert base == pytest.approx(0.5445)
    assert tier == pytest.approx(1.089005)


def test_calculate_cost_ignores_output_tokens_for_the_tier():
    cost = calculate_cost("gpt-5.6-luna", {"input": 200000, "output": 100000})
    assert cost == pytest.approx(0.16)
