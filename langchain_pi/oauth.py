from __future__ import annotations

import base64
import hashlib
import os
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from .auth import CodexAuth, extract_account_id
from .constants import (
    AUTHORIZE_URL,
    CALLBACK_HOST,
    CALLBACK_PATH,
    CALLBACK_PORT,
    CLIENT_ID,
    DEVICE_CODE_TIMEOUT_SECONDS,
    DEVICE_POLL_TIMEOUT,
    DEVICE_REDIRECT_URI,
    DEVICE_TOKEN_URL,
    DEVICE_USER_CODE_URL,
    DEVICE_VERIFICATION_URI,
    MINIMUM_INTERVAL_MS,
    ORIGINATOR,
    PKCE_METHOD,
    REDIRECT_URI,
    REFRESH_TIMEOUT,
    SCOPE,
    TOKEN_URL,
)
from .oauth_pages import SUCCESS_HTML, error_html


class CodexOAuthError(Exception):
    pass


def generate_pkce() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode()
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    return verifier, challenge


def create_state() -> str:
    return os.urandom(16).hex()


def parse_authorization_input(raw: str, expected_state: str) -> str:
    raw = raw.strip()
    code: Optional[str] = None
    state: Optional[str] = None

    parsed = urlparse(raw)
    if parsed.scheme and parsed.query:
        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]
    elif "#" in raw:
        code, _, state = raw.partition("#")
    elif "code=" in raw:
        params = parse_qs(raw)
        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]
    else:
        code = raw

    if state and state != expected_state:
        raise CodexOAuthError("State mismatch")
    if not code:
        raise CodexOAuthError("Missing authorization code")
    return code


def _read_token_response(response: httpx.Response, operation: str) -> dict:
    if response.status_code >= 400:
        raise CodexOAuthError(
            f"OpenAI Codex token {operation} failed "
            f"({response.status_code}): {response.text}"
        )
    data = response.json()
    access = data.get("access_token")
    refresh = data.get("refresh_token")
    expires_in = data.get("expires_in")
    if not access or not refresh or not isinstance(expires_in, (int, float)):
        raise CodexOAuthError("Token response missing fields")
    account_id = extract_account_id(access)
    if not account_id:
        raise CodexOAuthError("Failed to extract accountId from token")
    return {
        "access": access,
        "refresh": refresh,
        "expires": int(time.time() * 1000) + int(expires_in) * 1000,
        "accountId": account_id,
    }


class CodexOAuth:
    def __init__(self, auth_path: Optional[str] = None, originator: str = ORIGINATOR):
        self.auth = CodexAuth(auth_path)
        self.originator = originator

    def exchange_code(self, code: str, verifier: str, redirect_uri: str) -> dict:
        response = httpx.post(
            TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=REFRESH_TIMEOUT,
        )
        return _read_token_response(response, "exchange")

    def _authorize_url(self, challenge: str, state: str) -> str:
        params = {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": PKCE_METHOD,
            "state": state,
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "originator": self.originator,
        }
        return f"{AUTHORIZE_URL}?{urlencode(params)}"

    def browser_login(self) -> dict:
        verifier, challenge = generate_pkce()
        state = create_state()
        url = self._authorize_url(challenge, state)

        code = self._run_callback_server(state, url)
        if code is None:
            webbrowser.open(url)
            raw = input(
                "Paste the authorization code (or the full redirect URL): "
            ).strip()
            code = parse_authorization_input(raw, state)

        credential = self.exchange_code(code, verifier, REDIRECT_URI)
        self.auth.write_credential(credential)
        return credential

    def _run_callback_server(self, state: str, url: str) -> Optional[str]:
        result: dict[str, Optional[str]] = {"code": None}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # noqa: A003
                pass

            def _send(self, status: int, html: str) -> None:
                body = html.encode()
                self.send_response(status)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path != CALLBACK_PATH:
                    self._send(404, error_html("Not found."))
                    return
                params = parse_qs(parsed.query)
                if params.get("state", [None])[0] != state:
                    self._send(400, error_html("State mismatch."))
                    return
                code = params.get("code", [None])[0]
                if not code:
                    self._send(400, error_html("Missing authorization code."))
                    return
                result["code"] = code
                self._send(200, SUCCESS_HTML)

        try:
            server = HTTPServer((CALLBACK_HOST, CALLBACK_PORT), Handler)
        except OSError:
            return None

        try:
            webbrowser.open(url)
            while result["code"] is None:
                server.handle_request()
        finally:
            server.server_close()
        return result["code"]

    def _start_device_auth(self) -> dict:
        response = httpx.post(
            DEVICE_USER_CODE_URL,
            json={"client_id": CLIENT_ID},
            headers={"Content-Type": "application/json"},
            timeout=DEVICE_POLL_TIMEOUT,
        )
        if response.status_code == 404:
            raise CodexOAuthError("Device code login is not enabled for this server.")
        if response.status_code >= 400:
            raise CodexOAuthError(
                f"Device auth failed ({response.status_code}): {response.text}"
            )
        data = response.json()
        interval = data.get("interval")
        if isinstance(interval, str):
            interval = float(interval.strip())
        device_auth_id = data.get("device_auth_id")
        user_code = data.get("user_code")
        if not device_auth_id or not user_code or interval is None:
            raise CodexOAuthError("Device auth response missing fields")
        return {
            "device_auth_id": device_auth_id,
            "user_code": user_code,
            "interval": float(interval),
        }

    def _poll_device(self, device_auth_id: str, user_code: str) -> dict:
        response = httpx.post(
            DEVICE_TOKEN_URL,
            json={"device_auth_id": device_auth_id, "user_code": user_code},
            headers={"Content-Type": "application/json"},
            timeout=DEVICE_POLL_TIMEOUT,
        )
        if response.status_code < 400:
            data = response.json()
            authorization_code = data.get("authorization_code")
            code_verifier = data.get("code_verifier")
            if not authorization_code or not code_verifier:
                return {"status": "failed", "message": "missing authorization fields"}
            return {
                "status": "complete",
                "authorization_code": authorization_code,
                "code_verifier": code_verifier,
            }
        if response.status_code in (403, 404):
            return {"status": "pending"}
        try:
            error = response.json().get("error")
        except Exception:
            error = None
        code = error.get("code") if isinstance(error, dict) else error
        if code == "deviceauth_authorization_pending":
            return {"status": "pending"}
        if code == "slow_down":
            return {"status": "slow_down"}
        return {"status": "failed", "message": str(error or response.text)}

    def device_login(self, on_prompt=None) -> dict:
        start = self._start_device_auth()
        if on_prompt is not None:
            on_prompt(start["user_code"], DEVICE_VERIFICATION_URI)

        deadline = time.time() + DEVICE_CODE_TIMEOUT_SECONDS
        interval_ms = max(
            MINIMUM_INTERVAL_MS, int((start["interval"] or 5) * 1000)
        )
        saw_slow_down = False
        while time.time() < deadline:
            result = self._poll_device(start["device_auth_id"], start["user_code"])
            status = result["status"]
            if status == "complete":
                credential = self.exchange_code(
                    result["authorization_code"],
                    result["code_verifier"],
                    DEVICE_REDIRECT_URI,
                )
                self.auth.write_credential(credential)
                return credential
            if status == "failed":
                raise CodexOAuthError(result.get("message") or "Device login failed")
            if status == "slow_down":
                saw_slow_down = True
                interval_ms = max(MINIMUM_INTERVAL_MS, interval_ms + 5000)
            remaining_ms = max(0, int((deadline - time.time()) * 1000))
            time.sleep(min(interval_ms, remaining_ms) / 1000)

        if saw_slow_down:
            raise CodexOAuthError(
                "Device flow timed out (possible clock drift after slow_down)."
            )
        raise CodexOAuthError("Device flow timed out")
