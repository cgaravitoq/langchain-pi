from __future__ import annotations

import base64
import json
import time

import httpx
import pytest
from pytest_socket import disable_socket


def pytest_runtest_setup() -> None:
    disable_socket()


def make_jwt(account_id: str = "acct_test_123") -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    payload_obj = {"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}
    payload = (
        base64.urlsafe_b64encode(json.dumps(payload_obj).encode()).rstrip(b"=").decode()
    )
    return f"{header}.{payload}.sig"


@pytest.fixture
def account_id() -> str:
    return "acct_test_123"


@pytest.fixture
def access_jwt(account_id: str) -> str:
    return make_jwt(account_id)


@pytest.fixture
def auth_file(tmp_path, access_jwt):
    path = tmp_path / "auth.json"
    expires = int(time.time() * 1000) + 3_600_000
    path.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "type": "oauth",
                    "access": access_jwt,
                    "refresh": "refresh_token_value",
                    "expires": expires,
                    "accountId": "acct_test_123",
                },
                "claude-code": {
                    "type": "oauth",
                    "access": "claude_access",
                    "refresh": "claude_refresh",
                    "expires": expires,
                },
            },
            indent=2,
        )
    )
    return path


@pytest.fixture
def mock_transport_factory():
    def factory(handler):
        return httpx.MockTransport(handler)

    return factory


@pytest.fixture
def claude_creds_file(tmp_path):
    path = tmp_path / ".credentials.json"
    expires = int(time.time() * 1000) + 3_600_000
    path.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "cc_access",
                    "refreshToken": "cc_refresh",
                    "expiresAt": expires,
                }
            },
            indent=2,
        )
    )
    return path
