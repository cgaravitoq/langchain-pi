from __future__ import annotations

import json
import subprocess
import sys
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


def _keychain_json(access="kc_access", refresh="kc_refresh", expires=None) -> str:
    return json.dumps(
        {
            "mcpOAuth": {"srv": {"accessToken": "keep"}},
            "claudeAiOauth": {
                "accessToken": access,
                "refreshToken": refresh,
                "expiresAt": expires if expires is not None else _future_ms(),
            },
        }
    )


class FakeSecurity:
    def __init__(self, blob, account="cgaravitoq") -> None:
        self.blob = blob
        self.account = account
        self.calls: list[list[str]] = []
        self.written: list[list[str]] = []

    def __call__(self, args, **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(args)
        assert args[0] == "security", args
        if args[1] == "find-generic-password":
            if self.blob is None:
                return subprocess.CompletedProcess(args, 44, "", "not found")
            if "-w" in args:
                return subprocess.CompletedProcess(args, 0, self.blob, "")
            return subprocess.CompletedProcess(
                args, 0, f'    "acct"<blob>="{self.account}"\n', ""
            )
        if args[1] == "add-generic-password":
            self.written.append(args)
            self.blob = args[args.index("-w") + 1]
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(args)


def _mock_security(monkeypatch, blob) -> FakeSecurity:
    fake = FakeSecurity(blob)
    monkeypatch.setattr(
        "open_langchain.claude_code_auth.subprocess.run", fake, raising=True
    )
    return fake


def _home_creds(tmp_path):
    path = tmp_path / ".claude" / ".credentials.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_reads_credentials_from_keychain(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "darwin")
    _mock_security(monkeypatch, _keychain_json())
    auth = ClaudeCodeAuth()
    assert auth.get_access_token() == "kc_access"


def test_falls_back_to_file_on_keychain_miss(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "darwin")
    _mock_security(monkeypatch, None)
    path = _home_creds(tmp_path)
    _write(path, access="file_access")
    assert ClaudeCodeAuth().get_access_token() == "file_access"


def test_explicit_creds_path_skips_keychain(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "darwin")
    fake = _mock_security(monkeypatch, _keychain_json(access="kc_access"))
    path = _home_creds(tmp_path)
    _write(path, access="file_access")
    assert ClaudeCodeAuth(str(path)).get_access_token() == "file_access"
    assert fake.calls == []


def test_non_darwin_skips_keychain(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "linux")
    fake = _mock_security(monkeypatch, _keychain_json())
    with pytest.raises(ClaudeCodeAuthError):
        ClaudeCodeAuth().get_access_token()
    assert fake.calls == []


def test_refresh_writes_back_to_keychain(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "darwin")
    fake = _mock_security(
        monkeypatch, _keychain_json(expires=int(time.time() * 1000) - 1000)
    )
    monkeypatch.setattr(
        "open_langchain.claude_code_auth.httpx.post",
        lambda url, **kwargs: httpx.Response(
            200,
            json={
                "access_token": "new_access",
                "refresh_token": "r2",
                "expires_in": 3600,
            },
        ),
    )
    auth = ClaudeCodeAuth()
    assert auth.get_access_token() == "new_access"

    assert len(fake.written) == 1
    args = fake.written[0]
    assert args[:2] == ["security", "add-generic-password"]
    assert args[args.index("-s") + 1] == "Claude Code-credentials"
    assert args[args.index("-a") + 1] == "cgaravitoq"
    payload = args[args.index("-w") + 1]
    assert "\n" not in payload  # `security -w` reads back hex-encoded otherwise
    stored = json.loads(payload)
    assert stored["mcpOAuth"] == {"srv": {"accessToken": "keep"}}
    assert stored["claudeAiOauth"]["accessToken"] == "new_access"
    assert stored["claudeAiOauth"]["refreshToken"] == "r2"
    assert stored["claudeAiOauth"]["expiresAt"] > int(time.time() * 1000)
    assert not _home_creds(tmp_path).exists()


def test_refresh_writes_back_to_file(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "darwin")
    fake = _mock_security(monkeypatch, None)
    path = _home_creds(tmp_path)
    _write(path, expires=int(time.time() * 1000) - 1000)
    monkeypatch.setattr(
        "open_langchain.claude_code_auth.httpx.post",
        lambda url, **kwargs: httpx.Response(
            200,
            json={
                "access_token": "new_access",
                "refresh_token": "r2",
                "expires_in": 3600,
            },
        ),
    )
    assert ClaudeCodeAuth().get_access_token() == "new_access"
    assert fake.written == []
    on_disk = json.loads(path.read_text())
    assert on_disk["claudeAiOauth"]["accessToken"] == "new_access"
    assert on_disk["claudeAiOauth"]["refreshToken"] == "r2"


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


def test_falls_back_to_file_when_security_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "darwin")

    def no_security(*args, **kwargs):
        raise FileNotFoundError("security")

    monkeypatch.setattr(
        "open_langchain.claude_code_auth.subprocess.run", no_security, raising=True
    )
    path = _home_creds(tmp_path)
    _write(path, access="file_access")
    assert ClaudeCodeAuth().get_access_token() == "file_access"


def test_write_back_target_survives_a_concurrent_file_read(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "darwin")
    fake = _mock_security(
        monkeypatch, _keychain_json(expires=int(time.time() * 1000) - 1000)
    )
    path = _home_creds(tmp_path)
    _write(path, access="file_access")
    auth = ClaudeCodeAuth()

    def fake_post(url, **kwargs):
        # a concurrent caller resolves the file source while we hold the lock
        blob, fake.blob = fake.blob, None
        assert auth.read()["access"] == "file_access"
        fake.blob = blob
        return httpx.Response(
            200,
            json={
                "access_token": "new_access",
                "refresh_token": "r2",
                "expires_in": 3600,
            },
        )

    monkeypatch.setattr("open_langchain.claude_code_auth.httpx.post", fake_post)
    assert auth.get_access_token() == "new_access"

    assert len(fake.written) == 1
    stored = json.loads(fake.written[0][fake.written[0].index("-w") + 1])
    assert stored["claudeAiOauth"]["accessToken"] == "new_access"
    assert json.loads(path.read_text())["claudeAiOauth"]["accessToken"] == "file_access"
