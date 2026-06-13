from __future__ import annotations

import json
import platform
from typing import Iterator, Optional

import httpx

from .auth import CodexAuth
from .constants import (
    DEFAULT_CODEX_BASE_URL,
    ORIGINATOR,
    REQUEST_TIMEOUT,
)

_USAGE_LIMIT_CODES = {
    "usage_limit_reached",
    "usage_not_included",
    "rate_limit_exceeded",
}

_ARCH_MAP = {"x86_64": "x64", "aarch64": "arm64", "amd64": "x64", "arm64": "arm64"}


class CodexUsageLimitError(Exception):
    def __init__(self, message: str, resets_at: Optional[int] = None) -> None:
        super().__init__(message)
        self.resets_at = resets_at


def _user_agent() -> str:
    system = platform.system().lower()
    release = platform.release()
    arch = _ARCH_MAP.get(platform.machine().lower(), platform.machine().lower())
    return f"pi ({system} {release}; {arch})"


def build_headers(
    access: str, account_id: str, session_id: Optional[str] = None
) -> dict:
    headers = {
        "Authorization": f"Bearer {access}",
        "chatgpt-account-id": account_id,
        "originator": ORIGINATOR,
        "User-Agent": _user_agent(),
        "OpenAI-Beta": "responses=experimental",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }
    if session_id:
        headers["session-id"] = session_id
        headers["x-client-request-id"] = session_id
    return headers


def resolve_url(base: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/codex/responses"):
        return base
    if base.endswith("/codex"):
        return base + "/responses"
    return base + "/codex/responses"


def _normalize_event(payload: dict) -> Optional[dict]:
    etype = payload.get("type")
    if etype in ("response.output_text.delta", "response.text.delta"):
        return {"type": "text_delta", "delta": payload.get("delta", "")}
    if etype == "response.output_item.done":
        item = payload.get("item") or {}
        if item.get("type") == "function_call":
            return {"type": "tool_call", "tool_call": item}
        return None
    if etype in ("response.completed", "response.done", "response.incomplete"):
        response = payload.get("response") or {}
        return {
            "type": "done",
            "usage": response.get("usage"),
            "stop_reason": response.get("status") or etype,
        }
    if etype in ("error", "response.failed"):
        if etype == "response.failed":
            error = (payload.get("response") or {}).get("error") or {}
        else:
            error = payload
        return {
            "type": "error",
            "code": error.get("code"),
            "message": error.get("message") or "Codex request failed",
            "resets_at": error.get("resets_at"),
        }
    return None


def _iter_sse(response: httpx.Response) -> Iterator[dict]:
    data_lines: list[str] = []
    for line in response.iter_lines():
        if line == "":
            if data_lines:
                raw = "".join(data_lines)
                data_lines = []
                if raw.strip() == "[DONE]":
                    continue
                try:
                    yield json.loads(raw)
                except json.JSONDecodeError:
                    continue
            continue
        if line.startswith("data:"):
            value = line[len("data:") :]
            if value.startswith(" "):
                value = value[1:]
            data_lines.append(value)
    if data_lines:
        raw = "".join(data_lines)
        if raw.strip() != "[DONE]":
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                pass


class CodexClient:
    def __init__(
        self,
        auth: CodexAuth,
        base_url: str = DEFAULT_CODEX_BASE_URL,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.auth = auth
        self.url = resolve_url(base_url)
        self._client = client or httpx.Client(timeout=REQUEST_TIMEOUT)

    def _headers(self, session_id: Optional[str]) -> dict:
        credential = self.auth.get_access_token()
        return build_headers(
            credential["access"], self.auth.account_id(credential), session_id
        )

    def stream(
        self, body: dict, *, session_id: Optional[str] = None
    ) -> Iterator[dict]:
        content = json.dumps(body).encode()
        headers = self._headers(session_id)
        retried = False
        while True:
            try:
                with self._client.stream(
                    "POST", self.url, headers=headers, content=content
                ) as response:
                    if response.status_code == 401 and not retried:
                        response.read()
                        self.auth.refresh()
                        headers = self._headers(session_id)
                        retried = True
                        continue
                    if response.status_code >= 400:
                        self._raise_for_status(response)
                    for payload in _iter_sse(response):
                        event = _normalize_event(payload)
                        if event is None:
                            continue
                        if event["type"] == "error":
                            self._raise_event_error(event)
                        yield event
                        if event["type"] == "done":
                            return
                return
            finally:
                pass

    def _raise_for_status(self, response: httpx.Response) -> None:
        body = response.read()
        text = body.decode(errors="replace")
        if response.status_code == 429:
            self._maybe_usage_limit(text)
        raise RuntimeError(
            f"Codex request failed ({response.status_code}): {text}"
        )

    def _maybe_usage_limit(self, text: str) -> None:
        code = None
        resets_at = None
        try:
            data = json.loads(text)
            error = data.get("error") or data
            code = error.get("code")
            resets_at = error.get("resets_at")
        except (json.JSONDecodeError, AttributeError):
            pass
        if code in _USAGE_LIMIT_CODES:
            raise CodexUsageLimitError(
                "You hit your ChatGPT usage limit.", resets_at=resets_at
            )

    def _raise_event_error(self, event: dict) -> None:
        if event.get("code") in _USAGE_LIMIT_CODES:
            raise CodexUsageLimitError(
                "You hit your ChatGPT usage limit.",
                resets_at=event.get("resets_at"),
            )
        raise RuntimeError(event.get("message") or "Codex request failed")
