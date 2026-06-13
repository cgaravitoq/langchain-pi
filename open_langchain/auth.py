from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Optional

import httpx

from .constants import (
    CLIENT_ID,
    JWT_CLAIM_PATH,
    PROVIDER_ID,
    REFRESH_TIMEOUT,
    TOKEN_URL,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None  # type: ignore[assignment]


class CodexAuthError(Exception):
    """Raised when no usable openai-codex credential is available."""


def _decode_jwt_payload(token: str) -> Optional[dict]:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        raw = base64.urlsafe_b64decode(payload.encode())
        return json.loads(raw)
    except Exception:
        return None


def extract_account_id(access_token: str) -> Optional[str]:
    payload = _decode_jwt_payload(access_token)
    if not payload:
        return None
    auth = payload.get(JWT_CLAIM_PATH)
    if not isinstance(auth, dict):
        return None
    account_id = auth.get("chatgpt_account_id")
    if isinstance(account_id, str) and account_id:
        return account_id
    return None


def resolve_auth_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    agent_dir = os.environ.get("PI_CODING_AGENT_DIR")
    if agent_dir:
        return Path(agent_dir).expanduser() / "auth.json"
    config_dir = os.environ.get("PI_CONFIG_DIR_NAME", ".pi")
    return Path.home() / config_dir / "agent" / "auth.json"


class CodexAuth:
    def __init__(self, auth_path: Optional[str] = None) -> None:
        self.path = resolve_auth_path(auth_path)

    def load(self) -> dict:
        if not self.path.exists():
            raise CodexAuthError(
                f"No auth.json found at {self.path}. Run `codex-login` to sign in."
            )
        return json.loads(self.path.read_text())

    def get_credential(self) -> dict:
        credentials = self.load()
        entry = credentials.get(PROVIDER_ID)
        if not entry:
            raise CodexAuthError(
                f"No `{PROVIDER_ID}` credential in {self.path}. "
                "Run `codex-login` to sign in with your ChatGPT subscription."
            )
        return entry

    def get_access_token(self) -> dict:
        credential = self.get_credential()
        now_ms = int(time.time() * 1000)
        if now_ms >= int(credential.get("expires", 0)):
            credential = self.refresh()
        return credential

    def account_id(self, credential: Optional[dict] = None) -> str:
        credential = credential or self.get_credential()
        stored = credential.get("accountId")
        if isinstance(stored, str) and stored:
            return stored
        account_id = extract_account_id(credential.get("access", ""))
        if not account_id:
            raise CodexAuthError("Failed to extract accountId from token")
        return account_id

    def refresh(self) -> dict:
        credentials = self.load()
        entry = credentials.get(PROVIDER_ID)
        if not entry:
            raise CodexAuthError(
                f"No `{PROVIDER_ID}` credential in {self.path}. Run `codex-login`."
            )
        response = httpx.post(
            TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": entry["refresh"],
                "client_id": CLIENT_ID,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=REFRESH_TIMEOUT,
        )
        if response.status_code >= 400:
            raise CodexAuthError(
                f"OpenAI Codex token refresh failed ({response.status_code}): "
                f"{response.text}"
            )
        data = response.json()
        access = data.get("access_token")
        refresh_token = data.get("refresh_token")
        expires_in = data.get("expires_in")
        if not access or not refresh_token or not isinstance(expires_in, (int, float)):
            raise CodexAuthError("Token refresh response missing fields")
        credential = {
            "type": "oauth",
            "access": access,
            "refresh": refresh_token,
            "expires": int(time.time() * 1000) + int(expires_in) * 1000,
            "accountId": extract_account_id(access),
        }
        credentials[PROVIDER_ID] = credential
        self._write(credentials)
        return credential

    def write_credential(self, credential: dict) -> None:
        try:
            credentials = self.load()
        except CodexAuthError:
            credentials = {}
        credentials[PROVIDER_ID] = {"type": "oauth", **credential}
        self._write(credentials)

    def _write(self, credentials: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        lock_path = self.path.with_name(self.path.name + ".lock")
        lock_file = open(lock_path, "w")
        try:
            if fcntl is not None:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                except OSError:
                    pass
            self.path.write_text(json.dumps(credentials, indent=2))
            os.chmod(self.path, 0o600)
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            lock_file.close()
            try:
                lock_path.unlink()
            except OSError:
                pass
