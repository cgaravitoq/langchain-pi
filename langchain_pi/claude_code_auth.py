from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import httpx

OAUTH_TOKEN_URL = "https://claude.ai/v1/oauth/token"
OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
REFRESH_LEEWAY_MS = 60_000
REFRESH_TIMEOUT = 30.0
CLI_TIMEOUT = 20.0


class ClaudeCodeAuthError(Exception):
    """Raised when no usable Claude Code OAuth credential is available."""


# Module-level lock so concurrent callers share one refresh. The OAuth server
# rotates the refresh token on success, so parallel refreshes with the same
# token would race and all but one would fail.
_refresh_lock = threading.Lock()


def _now_ms() -> int:
    return int(time.time() * 1000)


def credentials_file_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".claude" / ".credentials.json"


def _parse_blob(raw: str) -> Optional[dict]:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    nested = parsed.get("claudeAiOauth")
    data = nested if isinstance(nested, dict) else parsed
    access = data.get("accessToken") or data.get("access_token")
    refresh = data.get("refreshToken") or data.get("refresh_token")
    expires = data.get("expiresAt")
    if expires is None:
        expires = data.get("expires_at")
    if (
        not access
        or not refresh
        or isinstance(expires, bool)
        or not isinstance(expires, (int, float))
    ):
        return None
    return {"access": access, "refresh": refresh, "expires_at": int(expires)}


class ClaudeCodeAuth:
    def __init__(self, creds_path: Optional[str] = None) -> None:
        self.path = credentials_file_path(creds_path)

    def read(self) -> Optional[dict]:
        if not self.path.exists():
            return None
        try:
            return _parse_blob(self.path.read_text())
        except OSError:
            return None

    def get_credentials(self) -> dict:
        creds = self.read()
        if not creds:
            raise ClaudeCodeAuthError(
                f"Claude Code credentials not found at {self.path}. "
                "Run `claude` to authenticate first."
            )
        if creds["expires_at"] > _now_ms() + REFRESH_LEEWAY_MS:
            return creds
        return self.refresh(creds)

    def get_access_token(self) -> str:
        return self.get_credentials()["access"]

    def refresh(self, current: Optional[dict] = None, *, force: bool = False) -> dict:
        with _refresh_lock:
            disk = self.read()
            latest = disk or current
            if not latest:
                raise ClaudeCodeAuthError(
                    f"Claude Code credentials not found at {self.path}. "
                    "Run `claude` to authenticate first."
                )
            fresh = latest["expires_at"] > _now_ms() + REFRESH_LEEWAY_MS
            # A concurrent caller (or the proactive path) may have already rotated
            # the token while we waited for the lock; reuse it instead of POSTing
            # the now-stale refresh token a second time.
            rotated = bool(
                force and current and disk and disk["refresh"] != current["refresh"]
            )
            if fresh and (not force or rotated):
                return latest

            oauth = self._refresh_via_oauth(latest["refresh"])
            if oauth and oauth["expires_at"] > _now_ms() + REFRESH_LEEWAY_MS:
                try:
                    self._write_back(oauth)
                except OSError:
                    pass
                return oauth

            cli_reason = self._refresh_via_cli()
            reread = self.read()
            if reread and reread["expires_at"] > _now_ms() + REFRESH_LEEWAY_MS:
                return reread

            suffix = f" ({cli_reason})" if cli_reason else ""
            raise ClaudeCodeAuthError(
                f"Failed to refresh Claude Code credentials{suffix}. "
                "Run `claude` once to re-authenticate."
            )

    def _refresh_via_oauth(self, refresh_token: str) -> Optional[dict]:
        try:
            response = httpx.post(
                OAUTH_TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": OAUTH_CLIENT_ID,
                    "refresh_token": refresh_token,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=REFRESH_TIMEOUT,
            )
        except httpx.HTTPError:
            return None
        if response.status_code >= 400:
            return None
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError):
            return None
        access = data.get("access_token")
        if not access:
            return None
        try:
            ttl = float(data.get("expires_in"))
        except (TypeError, ValueError):
            ttl = 0.0
        if not ttl > 0:
            ttl = 36000.0
        return {
            "access": access,
            "refresh": data.get("refresh_token") or refresh_token,
            "expires_at": _now_ms() + int(ttl * 1000),
        }

    def _refresh_via_cli(self) -> Optional[str]:
        cli = shutil.which("claude")
        if not cli:
            return "claude CLI not found on PATH"
        try:
            subprocess.run(
                [cli, "-p", ".", "--model", "haiku"],
                timeout=CLI_TIMEOUT,
                env={**os.environ, "TERM": "dumb"},
                cwd=tempfile.gettempdir(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
            )
            return None
        except subprocess.TimeoutExpired:
            return "claude CLI refresh timed out after 20s"
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or b""
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", "replace")
            first = next((line for line in stderr.splitlines() if line.strip()), "")
            return (
                f"claude CLI: {first}"
                if first
                else f"claude CLI failed ({exc.returncode})"
            )
        except OSError as exc:
            return f"claude CLI: {exc}"

    def _write_back(self, creds: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        existing: dict = {}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text())
                if isinstance(loaded, dict):
                    existing = loaded
            except (OSError, json.JSONDecodeError, ValueError):
                existing = {}
        updated = {
            **existing,
            "claudeAiOauth": {
                "accessToken": creds["access"],
                "refreshToken": creds["refresh"],
                "expiresAt": creds["expires_at"],
            },
        }
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{_now_ms()}.tmp")
        try:
            tmp.write_text(json.dumps(updated, indent=2))
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
