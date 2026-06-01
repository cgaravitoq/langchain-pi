from __future__ import annotations

from langchain_core.messages import AIMessageChunk, HumanMessage

from langchain_pi.chat_models import ChatCodex


class FakeClient:
    def __init__(self, events):
        self.events = events
        self.last_body = None

    def stream(self, body, *, session_id=None):
        self.last_body = body
        yield from self.events


def _attach(llm: ChatCodex, events) -> FakeClient:
    fake = FakeClient(events)
    llm._client = fake
    return fake


def test_generate_text_and_usage():
    llm = ChatCodex(model="gpt-5.3-codex-spark", reasoning="minimal")
    _attach(
        llm,
        [
            {"type": "text_delta", "delta": "Hel"},
            {"type": "text_delta", "delta": "lo"},
            {
                "type": "done",
                "stop_reason": "completed",
                "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            },
        ],
    )
    result = llm._generate([HumanMessage(content="hi")])
    message = result.generations[0].message
    assert message.content == "Hello"
    assert message.usage_metadata["total_tokens"] == 6
    assert message.response_metadata["provider"] == "openai-codex"
    assert message.response_metadata["model"] == "gpt-5.3-codex-spark"


def test_generate_tool_calls():
    llm = ChatCodex(model="gpt-5.5")
    _attach(
        llm,
        [
            {
                "type": "tool_call",
                "tool_call": {
                    "type": "function_call",
                    "name": "f",
                    "arguments": '{"a": 1}',
                    "call_id": "c1",
                },
            },
            {"type": "done", "stop_reason": "completed", "usage": None},
        ],
    )
    message = llm._generate([HumanMessage(content="hi")]).generations[0].message
    assert message.tool_calls == [
        {"name": "f", "args": {"a": 1}, "id": "c1", "type": "tool_call"}
    ]


def test_stream_chunks():
    llm = ChatCodex(model="gpt-5.5")
    _attach(
        llm,
        [
            {"type": "text_delta", "delta": "a"},
            {"type": "text_delta", "delta": "b"},
            {"type": "done", "stop_reason": "completed", "usage": {"input_tokens": 1}},
        ],
    )
    chunks = list(llm._stream([HumanMessage(content="hi")]))
    text = "".join(
        c.message.content for c in chunks if isinstance(c.message, AIMessageChunk)
    )
    assert text == "ab"
    assert chunks[-1].message.usage_metadata["input_tokens"] == 1


def test_build_body_clamps_reasoning():
    llm = ChatCodex(model="gpt-5.3-codex", reasoning="minimal")
    body = llm._build_body([HumanMessage(content="hi")])
    assert body["reasoning"] == {"effort": "low", "summary": "auto"}
    assert body["model"] == "gpt-5.3-codex"


def test_bind_tools_surface():
    llm = ChatCodex(model="gpt-5.5")
    tool = {
        "type": "function",
        "function": {"name": "f", "description": "d", "parameters": {}},
    }
    bound = llm.bind_tools([tool], tool_choice="auto")
    tools = bound.kwargs["tools"]
    assert tools[0]["type"] == "function"
    assert tools[0]["strict"] is None
    assert bound.kwargs["tool_choice"] == "auto"


def test_build_body_preserves_tool_choice():
    llm = ChatCodex(model="gpt-5.5")
    body = llm._build_body(
        [HumanMessage(content="hi")],
        tools=[{"type": "function", "name": "f"}],
        tool_choice={"type": "function", "name": "f"},
    )
    assert body["tool_choice"] == {"type": "function", "name": "f"}


def test_identifying_params():
    llm = ChatCodex(model="gpt-5.4", reasoning="high")
    params = llm._identifying_params
    assert params == {
        "provider": "openai-codex",
        "model": "gpt-5.4",
        "reasoning": "high",
    }
    assert llm._llm_type == "openai-codex"
