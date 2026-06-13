from __future__ import annotations

import json
import threading
import time

import httpx
import pytest

from open_langchain.claude_code_auth import (
    ClaudeCodeAuth,
    ClaudeCodeAuthError,
    _parse_blob,
)


def _future_ms(minutes: int = 60) -> int:
    return int(time.time() * 1000) + minutes * 60_000


def _write(path, access="cc_access", refresh="cc_refresh", expires=None, wrapper=True):
    blob = {
        "accessToken": access,
        "refreshToken": refresh,
        "expiresAt": expires if expires is not None else _future_ms(),
    }
    path.write_text(json.dumps({"claudeAiOauth": blob} if wrapper else blob))


def test_parse_camel_case():
    creds = _parse_blob(
        json.dumps({"accessToken": "a", "refreshToken": "r", "expiresAt": 123})
    )
    assert creds == {"access": "a", "refresh": "r", "expires_at": 123}


def test_parse_snake_case():
    creds = _parse_blob(
        json.dumps({"access_token": "a", "refresh_token": "r", "expires_at": 123})
    )
    assert creds == {"access": "a", "refresh": "r", "expires_at": 123}


def test_parse_wrapper():
    creds = _parse_blob(
        json.dumps(
            {"claudeAiOauth": {"accessToken": "a", "refreshToken": "r", "expiresAt": 1}}
        )
    )
    assert creds == {"access": "a", "refresh": "r", "expires_at": 1}


def test_parse_rejects_missing_fields():
    assert _parse_blob(json.dumps({"accessToken": "a", "refreshToken": "r"})) is None
    assert _parse_blob(json.dumps({"accessToken": "a", "expiresAt": 1})) is None
    assert _parse_blob("not json") is None


def test_missing_file_raises(tmp_path):
    auth = ClaudeCodeAuth(str(tmp_path / "nope.json"))
    with pytest.raises(ClaudeCodeAuthError):
        auth.get_access_token()


def test_no_refresh_when_fresh(claude_creds_file, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not refresh a fresh token")

    monkeypatch.setattr("open_langchain.claude_code_auth.httpx.post", boom)
    auth = ClaudeCodeAuth(str(claude_creds_file))
    assert auth.get_access_token() == "cc_access"


def test_refresh_posts_oauth_and_writes_back(tmp_path, monkeypatch):
    path = tmp_path / ".credentials.json"
    _write(path, expires=int(time.time() * 1000) - 1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    # preserve an unrelated key in the file
    data = json.loads(path.read_text())
    data["other"] = {"keep": True}
    path.write_text(json.dumps(data))

    seen = {}

    def fake_post(url, **kwargs):
        seen["url"] = url
        seen["data"] = kwargs.get("data")
        return httpx.Response(
            200,
            json={
                "access_token": "new_access",
                "refresh_token": "r2",
                "expires_in": 3600,
            },
        )

    monkeypatch.setattr("open_langchain.claude_code_auth.httpx.post", fake_post)
    auth = ClaudeCodeAuth(str(path))
    token = auth.get_access_token()

    assert token == "new_access"
    assert seen["url"] == "https://claude.ai/v1/oauth/token"
    assert seen["data"]["grant_type"] == "refresh_token"
    assert seen["data"]["client_id"] == "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    assert seen["data"]["refresh_token"] == "cc_refresh"

    on_disk = json.loads(path.read_text())
    assert on_disk["claudeAiOauth"]["accessToken"] == "new_access"
    assert on_disk["claudeAiOauth"]["refreshToken"] == "r2"  # rotation persisted
    assert on_disk["other"] == {"keep": True}
    assert (path.stat().st_mode & 0o777) == 0o600


def test_refresh_falls_back_to_cli(tmp_path, monkeypatch):
    path = tmp_path / ".credentials.json"
    _write(path, expires=int(time.time() * 1000) - 1000)

    monkeypatch.setattr(
        "open_langchain.claude_code_auth.httpx.post",
        lambda *a, **k: httpx.Response(400, json={}),
    )

    def fake_cli(self):
        _write(path, access="cli_access", refresh="cli_refresh")
        return None

    monkeypatch.setattr(ClaudeCodeAuth, "_refresh_via_cli", fake_cli)
    auth = ClaudeCodeAuth(str(path))
    assert auth.get_access_token() == "cli_access"


def test_refresh_failure_raises(tmp_path, monkeypatch):
    path = tmp_path / ".credentials.json"
    _write(path, expires=int(time.time() * 1000) - 1000)
    monkeypatch.setattr(
        "open_langchain.claude_code_auth.httpx.post",
        lambda *a, **k: httpx.Response(400, json={}),
    )
    monkeypatch.setattr(
        ClaudeCodeAuth, "_refresh_via_cli", lambda self: "claude CLI not found on PATH"
    )
    auth = ClaudeCodeAuth(str(path))
    with pytest.raises(ClaudeCodeAuthError, match="re-authenticate"):
        auth.get_access_token()


def test_concurrent_refresh_dedupes(tmp_path, monkeypatch):
    path = tmp_path / ".credentials.json"
    _write(path, expires=int(time.time() * 1000) - 1000)

    calls = {"n": 0}

    def fake_post(url, **kwargs):
        calls["n"] += 1
        time.sleep(0.05)
        return httpx.Response(
            200,
            json={
                "access_token": "shared_new",
                "refresh_token": "r2",
                "expires_in": 3600,
            },
        )

    monkeypatch.setattr("open_langchain.claude_code_auth.httpx.post", fake_post)

    auth = ClaudeCodeAuth(str(path))
    expired = {
        "access": "cc_access",
        "refresh": "cc_refresh",
        "expires_at": int(time.time() * 1000) - 1000,
    }
    results = []
    barrier = threading.Barrier(5)

    def worker():
        barrier.wait()
        results.append(auth.refresh(dict(expired))["access"])

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls["n"] == 1  # only one OAuth POST despite 5 concurrent refreshers
    assert results == ["shared_new"] * 5
