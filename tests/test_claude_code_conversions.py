from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from open_langchain.claude_code_conversions import (
    apply_claude_code_transforms,
    build_request_body,
    messages_to_anthropic,
    prefix_tool_name,
    repair_tool_pairs,
    tool_to_anthropic,
    unprefix_tool_name,
)

V = dict(version="2.1.112", entrypoint="sdk-cli")


def _transform(model, messages, **kwargs):
    system_texts, anthropic_messages = messages_to_anthropic(messages)
    body = build_request_body(model, system_texts, anthropic_messages, **kwargs)
    return apply_claude_code_transforms(body, **V)


def test_human_text_to_user_string():
    _, messages = messages_to_anthropic([HumanMessage(content="hi")])
    assert messages == [{"role": "user", "content": "hi"}]


def test_human_image_block():
    _, messages = messages_to_anthropic(
        [HumanMessage(content=[{"type": "image_url", "image_url": {"url": "u"}}])]
    )
    block = messages[0]["content"][0]
    assert block["type"] == "image"
    assert block["source"] == {"type": "url", "url": "u"}


def test_ai_tool_call_to_tool_use():
    msg = AIMessage(
        content="ok",
        tool_calls=[{"name": "f", "args": {"a": 1}, "id": "t1", "type": "tool_call"}],
    )
    _, messages = messages_to_anthropic([msg])
    blocks = messages[0]["content"]
    assert blocks[0] == {"type": "text", "text": "ok"}
    assert blocks[1] == {"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1}}


def test_ai_thinking_dropped_with_placeholder():
    msg = AIMessage(content=[{"type": "thinking", "thinking": "secret"}])
    _, messages = messages_to_anthropic([msg])
    assert messages[0]["content"] == [{"type": "text", "text": "(no content)"}]


def test_consecutive_tool_messages_merged():
    _, messages = messages_to_anthropic(
        [
            ToolMessage(content="a", tool_call_id="t1"),
            ToolMessage(content="b", tool_call_id="t2"),
        ]
    )
    assert len(messages) == 1
    assert [b["tool_use_id"] for b in messages[0]["content"]] == ["t1", "t2"]


def test_tool_to_anthropic_input_schema():
    tool = {
        "type": "function",
        "function": {
            "name": "f",
            "description": "d",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        },
    }
    out = tool_to_anthropic(tool)
    assert out["name"] == "f"
    assert out["input_schema"] == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }


def test_prefix_and_unprefix_roundtrip():
    assert prefix_tool_name("search") == "mcp_Search"
    assert unprefix_tool_name("mcp_Search") == "search"
    assert unprefix_tool_name("plain") == "plain"


def test_transforms_system_layout_and_move():
    body = _transform(
        "claude-sonnet-4-6",
        [SystemMessage(content="rules"), HumanMessage(content="hi")],
    )
    assert body["system"][0]["text"].startswith("x-anthropic-billing-header:")
    assert (
        body["system"][1]["text"]
        == "You are Claude Code, Anthropic's official CLI for Claude."
    )
    assert len(body["system"]) == 2
    assert "rules" in body["messages"][0]["content"]


def test_transforms_prefix_tools_and_history():
    tool = tool_to_anthropic(
        {"type": "function", "function": {"name": "search", "parameters": {}}}
    )
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "search", "args": {}, "id": "t1", "type": "tool_call"}],
    )
    body = _transform(
        "claude-sonnet-4-6",
        [HumanMessage(content="q"), msg, ToolMessage(content="r", tool_call_id="t1")],
        tools=[tool],
    )
    assert body["tools"][0]["name"] == "mcp_Search"
    assert body["messages"][1]["content"][0]["name"] == "mcp_Search"


def test_transforms_strip_effort_for_haiku():
    body = build_request_body(
        "claude-haiku-4-5", [], [{"role": "user", "content": "hi"}]
    )
    body["thinking"] = {"type": "enabled", "effort": "high"}
    apply_claude_code_transforms(body, **V)
    assert body["thinking"] == {"type": "enabled"}  # only the effort key is stripped


def test_transforms_drop_thinking_when_only_effort():
    body = build_request_body(
        "claude-haiku-4-5", [], [{"role": "user", "content": "hi"}]
    )
    body["thinking"] = {"effort": "high"}
    apply_claude_code_transforms(body, **V)
    assert "thinking" not in body


def test_repair_orphan_tool_use():
    repaired = repair_tool_pairs(
        [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "x", "name": "s", "input": {}}],
            },
            {"role": "user", "content": [{"type": "text", "text": "n"}]},
        ]
    )
    assert repaired[1]["content"] == [{"type": "text", "text": "(no content)"}]
    assert [m["role"] for m in repaired] == ["user", "assistant", "user"]


def test_repair_orphan_tool_result():
    repaired = repair_tool_pairs(
        [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        ]
    )
    assert repaired[0]["content"] == [{"type": "text", "text": "[tool result omitted]"}]


def test_build_body_adaptive_thinking_opus():
    body = build_request_body(
        "claude-opus-4-8", [], [{"role": "user", "content": "hi"}], reasoning="medium"
    )
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "medium"}
    assert "budget_tokens" not in str(body["thinking"])


def test_build_body_adaptive_thinking_opus_5():
    body = build_request_body(
        "claude-opus-5", [], [{"role": "user", "content": "hi"}], reasoning="high"
    )
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "high"}
    assert "budget_tokens" not in json.dumps(body)


def test_build_body_adaptive_thinking_claude_5_family():
    for model in ("claude-fable-5", "claude-sonnet-5"):
        body = build_request_body(
            model, [], [{"role": "user", "content": "hi"}], reasoning="medium"
        )
        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"] == {"effort": "medium"}
        assert "budget_tokens" not in json.dumps(body)


def test_build_body_adaptive_minimal_maps_to_low():
    body = build_request_body(
        "claude-opus-4-8", [], [{"role": "user", "content": "hi"}], reasoning="minimal"
    )
    assert body["output_config"] == {"effort": "low"}


def test_build_body_budget_tokens_sonnet():
    body = build_request_body(
        "claude-sonnet-4-6", [], [{"role": "user", "content": "hi"}], reasoning="high"
    )
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 20480}
    assert "output_config" not in body


def test_build_body_no_thinking_for_haiku():
    body = build_request_body(
        "claude-haiku-4-5", [], [{"role": "user", "content": "hi"}], reasoning="medium"
    )
    assert "thinking" not in body


def test_build_body_default_max_tokens():
    body = build_request_body(
        "claude-sonnet-4-6", [], [{"role": "user", "content": "hi"}]
    )
    assert body["max_tokens"] == 128000 // 3


def test_build_body_stop_sequences():
    body = build_request_body(
        "claude-sonnet-4-6", [], [{"role": "user", "content": "hi"}], stop=["STOP"]
    )
    assert body["stop_sequences"] == ["STOP"]


def test_empty_user_message_skipped():
    _, messages = messages_to_anthropic(
        [HumanMessage(content="   "), HumanMessage(content="hi")]
    )
    assert messages == [{"role": "user", "content": "hi"}]


def test_system_only_conversation_synthesizes_user():
    body = _transform("claude-sonnet-4-6", [SystemMessage(content="rules here")])
    assert body["messages"], "request must not be sent with zero messages"
    assert "rules here" in body["messages"][0]["content"]
