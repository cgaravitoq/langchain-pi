from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
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
KEYCHAIN_SERVICE = "Claude Code-credentials"


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


def _keychain_read() -> Optional[str]:
    result = subprocess.run(
        ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _keychain_account() -> str:
    result = subprocess.run(
        ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE],
        capture_output=True,
        text=True,
    )
    match = re.search(r'"acct"<blob>="(.*)"', result.stdout or "")
    if result.returncode != 0 or match is None:
        raise ClaudeCodeAuthError(
            f"Failed to read the account of the {KEYCHAIN_SERVICE!r} keychain item. "
            "Run `claude` once to re-authenticate."
        )
    return match.group(1)


def _merge_blob(existing: Optional[str], creds: dict) -> dict:
    blob: dict = {}
    if existing:
        try:
            parsed = json.loads(existing)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            blob = parsed
    blob["claudeAiOauth"] = {
        "accessToken": creds["access"],
        "refreshToken": creds["refresh"],
        "expiresAt": creds["expires_at"],
    }
    return blob


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
        self._explicit_path = creds_path is not None
        self._source = "file"

    def read(self) -> Optional[dict]:
        if not self._explicit_path and sys.platform == "darwin":
            raw = _keychain_read()
            creds = _parse_blob(raw) if raw else None
            if creds:
                self._source = "keychain"
                return creds
        if not self.path.exists():
            return None
        try:
            creds = _parse_blob(self.path.read_text())
        except OSError:
            return None
        if creds:
            self._source = "file"
        return creds

    def get_credentials(self) -> dict:
        creds = self.read()
        if not creds:
            raise ClaudeCodeAuthError(self._missing_message())
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
                raise ClaudeCodeAuthError(self._missing_message())
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
                    self._store(oauth)
                except (OSError, subprocess.SubprocessError):
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

    def _missing_message(self) -> str:
        where = str(self.path)
        if not self._explicit_path and sys.platform == "darwin":
            where = f"the macOS keychain or {self.path}"
        return (
            f"Claude Code credentials not found in {where}. "
            "Run `claude` to authenticate first."
        )

    def _store(self, creds: dict) -> None:
        if self._source == "keychain":
            self._write_back_keychain(creds)
        else:
            self._write_back(creds)

    def _write_back_keychain(self, creds: dict) -> None:
        account = _keychain_account()
        # `security -w` hex-encodes any stored password containing control
        # characters, so the blob must stay on a single line.
        updated = json.dumps(
            _merge_blob(_keychain_read(), creds), separators=(",", ":")
        )
        subprocess.run(
            [
                "security",
                "add-generic-password",
                "-U",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                account,
                "-w",
                updated,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    def _write_back(self, creds: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        existing: Optional[str] = None
        if self.path.exists():
            try:
                existing = self.path.read_text()
            except OSError:
                existing = None
        updated = json.dumps(_merge_blob(existing, creds), indent=2)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{_now_ms()}.tmp")
        try:
            tmp.write_text(updated)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
