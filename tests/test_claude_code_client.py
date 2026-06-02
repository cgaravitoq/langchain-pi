from __future__ import annotations

import json

import httpx
import pytest

from langchain_pi.claude_code_auth import ClaudeCodeAuth
from langchain_pi.claude_code_client import ClaudeCodeClient, build_headers


def test_build_headers_exact():
    headers = build_headers("tok", ["b1", "b2"], "2.1.112", "sdk-cli")
    assert headers["Authorization"] == "Bearer tok"
    assert "x-api-key" not in headers
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["anthropic-beta"] == "b1,b2"
    assert headers["anthropic-dangerous-direct-browser-access"] == "true"
    assert headers["x-app"] == "cli"
    assert headers["user-agent"] == "claude-cli/2.1.112 (external, sdk-cli)"
    assert headers["accept"] == "text/event-stream"


def test_user_agent_env_override(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_USER_AGENT", "custom/1.0")
    assert build_headers("t", [], "2.1.112", "sdk-cli")["user-agent"] == "custom/1.0"


def test_long_context_beta_gated(claude_creds_file):
    auth = ClaudeCodeAuth(str(claude_creds_file))
    creds = {"access": "tok"}
    default_betas = ClaudeCodeClient(auth)._headers(creds, "claude-opus-4-8")[
        "anthropic-beta"
    ]
    assert "context-1m-2025-08-07" not in default_betas
    assert "interleaved-thinking-2025-05-14" in default_betas  # base beta survives
    lc_betas = ClaudeCodeClient(auth, long_context=True)._headers(
        creds, "claude-opus-4-8"
    )["anthropic-beta"]
    assert "context-1m-2025-08-07" in lc_betas


def _sse(*events: dict) -> bytes:
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()


def _client(creds_file, handler) -> ClaudeCodeClient:
    transport = httpx.MockTransport(handler)
    httpx_client = httpx.Client(transport=transport)
    return ClaudeCodeClient(ClaudeCodeAuth(str(creds_file)), client=httpx_client)


def _text_stream():
    return _sse(
        {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 5, "output_tokens": 0}},
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hel"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "lo"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    )


def test_stream_text_and_done(claude_creds_file):
    client = _client(
        claude_creds_file, lambda r: httpx.Response(200, content=_text_stream())
    )
    events = list(client.stream({"model": "x"}, model="claude-sonnet-4-6"))
    assert [e["delta"] for e in events if e["type"] == "text_delta"] == ["Hel", "lo"]
    done = [e for e in events if e["type"] == "done"][0]
    assert done["stop_reason"] == "stop"
    assert done["usage"]["input_tokens"] == 5 and done["usage"]["output_tokens"] == 2
    assert done["usage"]["total_tokens"] == 7


def test_stream_tool_use(claude_creds_file):
    body = _sse(
        {"type": "message_start", "message": {"usage": {}}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "tu1", "name": "mcp_Search"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"q":'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '"hi"}'},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        {"type": "message_stop"},
    )
    client = _client(claude_creds_file, lambda r: httpx.Response(200, content=body))
    events = list(
        client.stream({"model": "x"}, model="claude-sonnet-4-6", tool_names=["search"])
    )
    tool = [e for e in events if e["type"] == "tool_call"][0]["tool_call"]
    assert tool == {"id": "tu1", "name": "search", "arguments": {"q": "hi"}}
    assert [e for e in events if e["type"] == "done"][0]["stop_reason"] == "toolUse"


def test_stream_thinking_delta(claude_creds_file):
    body = _sse(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "hmm"},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    )
    client = _client(claude_creds_file, lambda r: httpx.Response(200, content=body))
    events = list(client.stream({"model": "x"}, model="claude-opus-4-8"))
    assert [e["delta"] for e in events if e["type"] == "thinking_delta"] == ["hmm"]


def test_stream_multiline_data_reassembly(claude_creds_file):
    payload = json.dumps(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "x"},
        }
    )
    half = len(payload) // 2
    start = json.dumps(
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}
    )
    body = (
        f"data: {start}\n\n"
        f"data: {payload[:half]}\n"
        f"data: {payload[half:]}\n\n"
        'data: {"type": "message_stop"}\n\n'
    ).encode()
    client = _client(claude_creds_file, lambda r: httpx.Response(200, content=body))
    events = list(client.stream({"model": "x"}, model="claude-sonnet-4-6"))
    assert [e["delta"] for e in events if e["type"] == "text_delta"] == ["x"]


def test_stream_emits_done_without_message_stop(claude_creds_file):
    body = _sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 2},
        },
    )
    client = _client(claude_creds_file, lambda r: httpx.Response(200, content=body))
    events = list(client.stream({"model": "x"}, model="claude-sonnet-4-6"))
    done = [e for e in events if e["type"] == "done"][0]
    assert done["stop_reason"] == "stop"
    assert done["usage"]["output_tokens"] == 2


def test_error_event_raises(claude_creds_file):
    body = _sse({"type": "error", "error": {"message": "boom"}})
    client = _client(claude_creds_file, lambda r: httpx.Response(200, content=body))
    with pytest.raises(RuntimeError, match="boom"):
        list(client.stream({"model": "x"}, model="claude-sonnet-4-6"))


def test_http_error_raises(claude_creds_file):
    client = _client(
        claude_creds_file, lambda r: httpx.Response(400, text="bad request")
    )
    with pytest.raises(RuntimeError, match="bad request"):
        list(client.stream({"model": "x"}, model="claude-sonnet-4-6"))


def test_401_refresh_retry(claude_creds_file, monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(401, json={"error": "expired"})
        return httpx.Response(200, content=_text_stream())

    monkeypatch.setattr(
        "langchain_pi.claude_code_auth.httpx.post",
        lambda *a, **k: httpx.Response(
            200,
            json={"access_token": "fresh", "refresh_token": "r2", "expires_in": 3600},
        ),
    )
    client = _client(claude_creds_file, handler)
    events = list(client.stream({"model": "x"}, model="claude-sonnet-4-6"))
    assert calls["n"] == 2
    assert [e["delta"] for e in events if e["type"] == "text_delta"] == ["Hel", "lo"]
