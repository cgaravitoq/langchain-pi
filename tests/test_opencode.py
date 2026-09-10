from __future__ import annotations

from typing import Any, Callable

import httpx
import pytest

from open_langchain.opencode import ZEN_BASE_URL, ZEN_GO_BASE_URL, ChatOpencode

COMPLETION: dict[str, Any] = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

Build = Callable[..., tuple[ChatOpencode, list[httpx.Request]]]


@pytest.fixture(autouse=True)
def no_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)


@pytest.fixture
def wire() -> Any:
    clients: list[httpx.Client] = []

    def build(**kwargs: Any) -> tuple[ChatOpencode, list[httpx.Request]]:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=COMPLETION)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        clients.append(client)
        return ChatOpencode(http_client=client, **kwargs), requests

    yield build

    for client in clients:
        client.close()


def test_free_tier_sends_session_and_user_agent_on_wire(wire: Build):
    model, requests = wire(model="nemotron-3.5-lightning-free")
    assert model.invoke("hi").content == "ok"
    request = requests[0]
    assert request.url == httpx.URL(f"{ZEN_BASE_URL}/chat/completions")
    assert (
        request.headers["x-opencode-session"]
        == model.default_headers["x-opencode-session"]
    )
    assert request.headers["User-Agent"].startswith("open-langchain/")
    assert request.headers["authorization"] == ""


def test_go_tier_sends_session_and_user_agent_on_wire(wire: Build):
    model, requests = wire(model="minimax-m3", tier="go", api_key="test-key")
    assert model.invoke("hi").content == "ok"
    request = requests[0]
    assert request.url == httpx.URL(f"{ZEN_GO_BASE_URL}/chat/completions")
    assert (
        request.headers["x-opencode-session"]
        == model.default_headers["x-opencode-session"]
    )
    assert request.headers["User-Agent"].startswith("open-langchain/")
    assert request.headers["authorization"] == "Bearer test-key"


@pytest.mark.parametrize("tier", ["zen", "go"])
def test_session_id_stable_within_instance_and_unique_across_instances(
    tier: str, wire: Build
):
    model, requests = wire(model="nemotron-3.5-lightning-free", tier=tier)
    model.invoke("hi")
    model.invoke("hi")
    session_ids = {request.headers["x-opencode-session"] for request in requests}
    assert len(requests) == 2
    assert len(session_ids) == 1
    session_id = session_ids.pop()
    assert session_id
    other, _ = wire(model="nemotron-3.5-lightning-free", tier=tier)
    assert other.default_headers["x-opencode-session"] != session_id


def test_user_supplied_default_headers_win(wire: Build):
    model, requests = wire(
        model="m",
        default_headers={
            "x-opencode-session": "custom-session",
            "User-Agent": "custom/9.9",
        },
    )
    assert model.invoke("hi").content == "ok"
    assert requests[0].headers["x-opencode-session"] == "custom-session"
    assert requests[0].headers["User-Agent"] == "custom/9.9"


def test_user_supplied_default_headers_win_case_insensitively(wire: Build):
    model, requests = wire(
        model="m",
        default_headers={
            "X-OpenCode-Session": "custom-session",
            "user-agent": "custom/9.9",
        },
    )
    assert model.invoke("hi").content == "ok"
    assert requests[0].headers["x-opencode-session"] == "custom-session"
    assert requests[0].headers["user-agent"] == "custom/9.9"


@pytest.mark.parametrize("name", ["Authorization", "authorization", "AUTHORIZATION"])
def test_anonymous_branch_blanks_user_supplied_authorization(wire: Build, name: str):
    model, requests = wire(model="m", default_headers={name: "Bearer leaked"})
    assert model.invoke("hi").content == "ok"
    assert requests[0].headers.get_list("authorization") == [""]
