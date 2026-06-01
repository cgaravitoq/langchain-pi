from __future__ import annotations

import base64
import hashlib

import httpx
import pytest

from langchain_pi.oauth import (
    CodexOAuth,
    CodexOAuthError,
    create_state,
    generate_pkce,
    parse_authorization_input,
)
from tests.conftest import make_jwt


def test_generate_pkce():
    verifier, challenge = generate_pkce()
    assert "=" not in verifier and "=" not in challenge
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    assert challenge == expected


def test_create_state():
    state = create_state()
    assert len(state) == 32
    int(state, 16)


def test_parse_full_url():
    code = parse_authorization_input(
        "http://localhost:1455/auth/callback?code=abc&state=s1", "s1"
    )
    assert code == "abc"


def test_parse_hash_form():
    assert parse_authorization_input("abc#s1", "s1") == "abc"


def test_parse_code_eq_form():
    assert parse_authorization_input("code=abc&state=s1", "s1") == "abc"


def test_parse_raw_code():
    assert parse_authorization_input("rawcode", "s1") == "rawcode"


def test_parse_state_mismatch():
    with pytest.raises(CodexOAuthError, match="State mismatch"):
        parse_authorization_input("abc#wrong", "s1")


def test_exchange_code(tmp_path, monkeypatch):
    access = make_jwt("acct_x")

    def handler(request: httpx.Request) -> httpx.Response:
        assert "grant_type=authorization_code" in request.content.decode()
        return httpx.Response(
            200,
            json={
                "access_token": access,
                "refresh_token": "r",
                "expires_in": 3600,
            },
        )

    def fake_post(url, **kwargs):
        with httpx.Client(transport=httpx.MockTransport(handler)) as c:
            return c.post(url, **kwargs)

    monkeypatch.setattr("langchain_pi.oauth.httpx.post", fake_post)
    oauth = CodexOAuth(auth_path=str(tmp_path / "auth.json"))
    cred = oauth.exchange_code("code", "verifier", "redirect")
    assert cred["access"] == access
    assert cred["accountId"] == "acct_x"


def test_device_login_poll_loop(tmp_path, monkeypatch):
    access = make_jwt("acct_dev")
    state = {"poll": 0}

    def post(url, **kwargs):
        if url.endswith("/usercode"):
            return _resp({"device_auth_id": "d1", "user_code": "WXYZ", "interval": 0})
        if url.endswith("/deviceauth/token"):
            state["poll"] += 1
            if state["poll"] == 1:
                return _resp({"error": "deviceauth_authorization_pending"}, 400)
            if state["poll"] == 2:
                return _resp({"error": "slow_down"}, 400)
            return _resp(
                {"authorization_code": "ac", "code_verifier": "cv"}, 200
            )
        return _resp(
            {"access_token": access, "refresh_token": "r", "expires_in": 3600}, 200
        )

    def _resp(json_body, status=200):
        return httpx.Response(status, json=json_body)

    monkeypatch.setattr(
        "langchain_pi.oauth.httpx.post",
        lambda url, **kwargs: post(str(url), **kwargs),
    )
    monkeypatch.setattr("langchain_pi.oauth.time.sleep", lambda s: None)

    oauth = CodexOAuth(auth_path=str(tmp_path / "auth.json"))
    prompts = []
    cred = oauth.device_login(
        on_prompt=lambda code, uri: prompts.append((code, uri))
    )
    assert state["poll"] == 3
    assert cred["accountId"] == "acct_dev"
    assert prompts[0][0] == "WXYZ"
