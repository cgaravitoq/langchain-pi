from __future__ import annotations

from langchain_core.messages import AIMessageChunk, HumanMessage

from langchain_pi.claude_code_chat_models import ChatClaudeCode


class FakeClient:
    def __init__(self, events):
        self.events = events
        self.last_body = None
        self.last_model = None
        self.last_tool_names = None

    def stream(self, body, *, model, tool_names=None):
        self.last_body = body
        self.last_model = model
        self.last_tool_names = tool_names
        yield from self.events


def _attach(llm: ChatClaudeCode, events) -> FakeClient:
    fake = FakeClient(events)
    llm._client = fake
    return fake


def test_generate_text_and_usage():
    llm = ChatClaudeCode(model="claude-sonnet-4-6")
    _attach(
        llm,
        [
            {"type": "text_delta", "delta": "Hel"},
            {"type": "text_delta", "delta": "lo"},
            {
                "type": "done",
                "stop_reason": "stop",
                "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            },
        ],
    )
    message = llm._generate([HumanMessage(content="hi")]).generations[0].message
    assert message.content == "Hello"
    assert message.usage_metadata["total_tokens"] == 6
    assert message.response_metadata["provider"] == "claude-code"
    assert message.response_metadata["model"] == "claude-sonnet-4-6"


def test_generate_tool_calls():
    llm = ChatClaudeCode(model="claude-sonnet-4-6")
    _attach(
        llm,
        [
            {
                "type": "tool_call",
                "tool_call": {"id": "c1", "name": "f", "arguments": {"a": 1}},
            },
            {"type": "done", "stop_reason": "toolUse", "usage": None},
        ],
    )
    message = llm._generate([HumanMessage(content="hi")]).generations[0].message
    assert message.tool_calls == [
        {"name": "f", "args": {"a": 1}, "id": "c1", "type": "tool_call"}
    ]


def test_stream_chunks():
    llm = ChatClaudeCode(model="claude-sonnet-4-6")
    _attach(
        llm,
        [
            {"type": "text_delta", "delta": "a"},
            {"type": "text_delta", "delta": "b"},
            {
                "type": "tool_call",
                "tool_call": {"id": "c1", "name": "f", "arguments": {"a": 1}},
            },
            {"type": "done", "stop_reason": "toolUse", "usage": {"input_tokens": 1}},
        ],
    )
    chunks = list(llm._stream([HumanMessage(content="hi")]))
    text = "".join(
        c.message.content for c in chunks if isinstance(c.message, AIMessageChunk)
    )
    assert text == "ab"
    tool_chunks = [tc for c in chunks for tc in (c.message.tool_call_chunks or [])]
    assert tool_chunks == [
        {
            "name": "f",
            "args": '{"a": 1}',
            "id": "c1",
            "index": 0,
            "type": "tool_call_chunk",
        }
    ]
    assert chunks[-1].message.usage_metadata["input_tokens"] == 1


def test_build_body_adaptive_thinking_opus():
    llm = ChatClaudeCode(model="claude-opus-4-8", reasoning="medium")
    fake = _attach(llm, [{"type": "done", "stop_reason": "stop", "usage": None}])
    llm._generate([HumanMessage(content="hi")])
    assert fake.last_body["thinking"] == {"type": "adaptive"}
    assert fake.last_body["output_config"] == {"effort": "medium"}
    assert "budget_tokens" not in str(fake.last_body.get("thinking"))
    assert fake.last_model == "claude-opus-4-8"


def test_build_body_budget_sonnet():
    llm = ChatClaudeCode(model="claude-sonnet-4-6", reasoning="medium")
    fake = _attach(llm, [{"type": "done", "stop_reason": "stop", "usage": None}])
    llm._generate([HumanMessage(content="hi")])
    assert fake.last_body["thinking"] == {"type": "enabled", "budget_tokens": 10240}
    assert "output_config" not in fake.last_body


def test_bind_tools_surface():
    llm = ChatClaudeCode(model="claude-sonnet-4-6")
    tool = {
        "type": "function",
        "function": {"name": "f", "description": "d", "parameters": {}},
    }
    bound = llm.bind_tools([tool], tool_choice="auto")
    tools = bound.kwargs["tools"]
    assert tools[0]["name"] == "f"  # mcp_ prefix is applied later, in transforms
    assert tools[0]["input_schema"] == {
        "type": "object",
        "properties": {},
        "required": [],
    }
    assert bound.kwargs["tool_choice"] == "auto"


def test_billing_header_in_built_body():
    llm = ChatClaudeCode(model="claude-sonnet-4-6")
    fake = _attach(llm, [{"type": "done", "stop_reason": "stop", "usage": None}])
    llm._generate([HumanMessage(content="hi")])
    assert fake.last_body["system"][0]["text"].startswith("x-anthropic-billing-header:")
    assert (
        fake.last_body["system"][1]["text"]
        == "You are Claude Code, Anthropic's official CLI for Claude."
    )


def test_system_field_moved_into_first_user():
    llm = ChatClaudeCode(model="claude-sonnet-4-6", system="You are a pirate.")
    fake = _attach(llm, [{"type": "done", "stop_reason": "stop", "usage": None}])
    llm._generate([HumanMessage(content="hi")])
    assert "You are a pirate." in fake.last_body["messages"][0]["content"]


def test_identifying_params_and_llm_type():
    llm = ChatClaudeCode(model="claude-opus-4-8", reasoning="high")
    assert llm._llm_type == "claude-code"
    assert llm._identifying_params == {
        "provider": "claude-code",
        "model": "claude-opus-4-8",
        "reasoning": "high",
    }
