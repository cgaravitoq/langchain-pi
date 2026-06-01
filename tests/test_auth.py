from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from langchain_pi.auth import (
    CodexAuth,
    CodexAuthError,
    extract_account_id,
    resolve_auth_path,
)
from tests.conftest import make_jwt


def test_resolve_path_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path))
    assert resolve_auth_path() == tmp_path / "auth.json"


def test_resolve_path_default(monkeypatch):
    monkeypatch.delenv("PI_CODING_AGENT_DIR", raising=False)
    monkeypatch.delenv("PI_CONFIG_DIR_NAME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/home/x")))
    assert resolve_auth_path() == Path("/home/x/.pi/agent/auth.json")


def test_resolve_path_explicit(tmp_path):
    p = tmp_path / "custom" / "auth.json"
    assert resolve_auth_path(str(p)) == p


def test_extract_account_id():
    jwt = make_jwt("acct_999")
    assert extract_account_id(jwt) == "acct_999"


def test_extract_account_id_invalid():
    assert extract_account_id("not-a-jwt") is None


def test_get_access_token_happy_path(auth_file):
    auth = CodexAuth(str(auth_file))
    cred = auth.get_access_token()
    assert cred["access"].count(".") == 2
    assert auth.account_id(cred) == "acct_test_123"


def test_missing_entry_raises(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"claude-code": {"type": "oauth"}}))
    auth = CodexAuth(str(path))
    with pytest.raises(CodexAuthError):
        auth.get_credential()


def test_missing_file_raises(tmp_path):
    auth = CodexAuth(str(tmp_path / "nope.json"))
    with pytest.raises(CodexAuthError):
        auth.get_credential()


def test_refresh_writes_back(auth_file, monkeypatch):
    expired = int(time.time() * 1000) - 1000
    data = json.loads(auth_file.read_text())
    data["openai-codex"]["expires"] = expired
    auth_file.write_text(json.dumps(data, indent=2))

    new_access = make_jwt("acct_refreshed")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("https://auth.openai.com/oauth/token")
        body = request.content.decode()
        assert "grant_type=refresh_token" in body
        return httpx.Response(
            200,
            json={
                "access_token": new_access,
                "refresh_token": "new_refresh",
                "expires_in": 3600,
            },
        )

    def fake_post(url, **kwargs):
        transport = httpx.MockTransport(handler)
        with httpx.Client(transport=transport) as client:
            return client.post(url, **kwargs)

    monkeypatch.setattr("langchain_pi.auth.httpx.post", fake_post)

    auth = CodexAuth(str(auth_file))
    cred = auth.get_access_token()
    assert cred["access"] == new_access
    assert cred["accountId"] == "acct_refreshed"

    on_disk = json.loads(auth_file.read_text())
    assert on_disk["openai-codex"]["access"] == new_access
    assert on_disk["openai-codex"]["refresh"] == "new_refresh"
    assert "claude-code" in on_disk  # other providers preserved
    assert (auth_file.stat().st_mode & 0o777) == 0o600
