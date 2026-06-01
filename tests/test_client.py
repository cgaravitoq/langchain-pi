from __future__ import annotations

import json

import httpx
import pytest

from langchain_pi.auth import CodexAuth
from langchain_pi.client import (
    CodexClient,
    CodexUsageLimitError,
    build_headers,
    resolve_url,
)
from tests.conftest import make_jwt


def test_resolve_url():
    assert (
        resolve_url("https://chatgpt.com/backend-api")
        == "https://chatgpt.com/backend-api/codex/responses"
    )
    assert (
        resolve_url("https://x/codex")
        == "https://x/codex/responses"
    )
    assert resolve_url("https://x/codex/responses") == "https://x/codex/responses"


def test_build_headers_exact():
    headers = build_headers("tok", "acct", session_id="sess")
    assert headers["Authorization"] == "Bearer tok"
    assert headers["chatgpt-account-id"] == "acct"
    assert headers["originator"] == "pi"
    assert headers["OpenAI-Beta"] == "responses=experimental"
    assert headers["accept"] == "text/event-stream"
    assert headers["content-type"] == "application/json"
    assert headers["session-id"] == "sess"
    assert headers["x-client-request-id"] == "sess"
    assert headers["User-Agent"].startswith("pi (")


def test_build_headers_no_session():
    headers = build_headers("tok", "acct")
    assert "session-id" not in headers
    assert "x-client-request-id" not in headers


def _sse(*events: dict) -> bytes:
    frames = []
    for event in events:
        frames.append(f"data: {json.dumps(event)}\n\n")
    frames.append("data: [DONE]\n\n")
    return "".join(frames).encode()


def _make_client(auth_file, handler) -> CodexClient:
    transport = httpx.MockTransport(handler)
    httpx_client = httpx.Client(transport=transport)
    return CodexClient(CodexAuth(str(auth_file)), client=httpx_client)


def test_stream_text_and_done(auth_file):
    body = _sse(
        {"type": "response.output_text.delta", "delta": "Hel"},
        {"type": "response.output_text.delta", "delta": "lo"},
        {
            "type": "response.completed",
            "response": {"status": "completed", "usage": {"input_tokens": 1}},
        },
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = _make_client(auth_file, handler)
    events = list(client.stream({"model": "gpt-5.5"}))
    deltas = [e["delta"] for e in events if e["type"] == "text_delta"]
    assert deltas == ["Hel", "lo"]
    done = [e for e in events if e["type"] == "done"]
    assert done and done[0]["usage"] == {"input_tokens": 1}


def test_stream_multiline_data_reassembly(auth_file):
    payload = {"type": "response.output_text.delta", "delta": "x"}
    raw = json.dumps(payload)
    half = len(raw) // 2
    body = (
        f"data: {raw[:half]}\n"
        f"data: {raw[half:]}\n\n"
        'data: {"type": "response.completed", "response": {}}\n\n'
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = _make_client(auth_file, handler)
    events = list(client.stream({"model": "gpt-5.5"}))
    assert events[0] == {"type": "text_delta", "delta": "x"}


def test_stream_tool_call(auth_file):
    body = _sse(
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "name": "f",
                "arguments": '{"a":1}',
                "call_id": "c1",
            },
        },
        {"type": "response.completed", "response": {}},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = _make_client(auth_file, handler)
    events = list(client.stream({"model": "gpt-5.5"}))
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert tool_calls[0]["tool_call"]["name"] == "f"


def test_usage_limit_429(auth_file):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"error": {"code": "usage_limit_reached", "resets_at": 12345}},
        )

    client = _make_client(auth_file, handler)
    with pytest.raises(CodexUsageLimitError) as exc:
        list(client.stream({"model": "gpt-5.5"}))
    assert exc.value.resets_at == 12345


def test_error_event_raises(auth_file):
    body = _sse({"type": "error", "code": "boom", "message": "bad"})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    client = _make_client(auth_file, handler)
    with pytest.raises(RuntimeError, match="bad"):
        list(client.stream({"model": "gpt-5.5"}))


def test_401_refresh_retry(auth_file, monkeypatch):
    new_access = make_jwt("acct_after_refresh")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(401, json={"error": "expired"})
        return httpx.Response(
            200,
            content=_sse(
                {"type": "response.output_text.delta", "delta": "ok"},
                {"type": "response.completed", "response": {}},
            ),
        )

    def refresh_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": new_access,
                "refresh_token": "r2",
                "expires_in": 3600,
            },
        )

    def fake_post(url, **kwargs):
        with httpx.Client(transport=httpx.MockTransport(refresh_handler)) as c:
            return c.post(url, **kwargs)

    monkeypatch.setattr("langchain_pi.auth.httpx.post", fake_post)

    client = _make_client(auth_file, handler)
    events = list(client.stream({"model": "gpt-5.5"}))
    assert calls["n"] == 2
    assert [e["delta"] for e in events if e["type"] == "text_delta"] == ["ok"]
