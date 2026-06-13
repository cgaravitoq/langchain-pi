from __future__ import annotations

import json
import os
from typing import Iterator, Optional

import httpx

from .claude_code_auth import ClaudeCodeAuth
from .claude_code_conversions import unprefix_tool_name
from .claude_code_models import CC_VERSION, compute_betas

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
REQUEST_TIMEOUT = 600.0
LONG_CONTEXT_BETA = "context-1m-2025-08-07"


def build_headers(access: str, betas: list[str], version: str, entrypoint: str) -> dict:
    user_agent = os.environ.get(
        "ANTHROPIC_USER_AGENT", f"claude-cli/{version} (external, {entrypoint})"
    )
    return {
        "Authorization": f"Bearer {access}",
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": ",".join(betas),
        "anthropic-dangerous-direct-browser-access": "true",
        "x-app": "cli",
        "user-agent": user_agent,
        "accept": "text/event-stream",
        "content-type": "application/json",
    }


def _map_stop_reason(reason: Optional[str]) -> Optional[str]:
    if reason in ("end_turn", "pause_turn", "stop_sequence"):
        return "stop"
    if reason == "max_tokens":
        return "length"
    if reason == "tool_use":
        return "toolUse"
    if reason is None:
        return None
    return "error"


def _resolve_tool_name(name: str, tool_names: Optional[list[str]]) -> str:
    stripped = unprefix_tool_name(name)
    if tool_names:
        lower = stripped.lower()
        for original in tool_names:
            if original.lower() == lower:
                return original
    return stripped


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


class ClaudeCodeClient:
    def __init__(
        self,
        auth: ClaudeCodeAuth,
        base_url: str = DEFAULT_ANTHROPIC_BASE_URL,
        entrypoint: Optional[str] = None,
        long_context: bool = False,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self.auth = auth
        self.url = base_url.rstrip("/") + "/v1/messages"
        self.entrypoint = entrypoint or os.environ.get(
            "CLAUDE_CODE_ENTRYPOINT", "sdk-cli"
        )
        self.long_context = long_context
        self._client = client or httpx.Client(timeout=REQUEST_TIMEOUT)

    def _headers(self, creds: dict, model: str) -> dict:
        version = os.environ.get("ANTHROPIC_CLI_VERSION", CC_VERSION)
        betas = compute_betas(model)
        # The 1M-context beta classifies the request as long-context, which the
        # Claude Code subscription bills against extra credits; keep it opt-in.
        if not self.long_context:
            betas = [b for b in betas if b != LONG_CONTEXT_BETA]
        return build_headers(creds["access"], betas, version, self.entrypoint)

    def stream(
        self, body: dict, *, model: str, tool_names: Optional[list[str]] = None
    ) -> Iterator[dict]:
        content = json.dumps(body).encode()
        creds = self.auth.get_credentials()
        headers = self._headers(creds, model)
        retried = False
        while True:
            with self._client.stream(
                "POST", self.url, headers=headers, content=content
            ) as response:
                if response.status_code == 401 and not retried:
                    response.read()
                    creds = self.auth.refresh(creds, force=True)
                    headers = self._headers(creds, model)
                    retried = True
                    continue
                if response.status_code >= 400:
                    self._raise_for_status(response)
                yield from self._iter_events(response, tool_names)
                return

    def _iter_events(
        self, response: httpx.Response, tool_names: Optional[list[str]]
    ) -> Iterator[dict]:
        blocks: dict[int, dict] = {}
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read": 0,
            "cache_write": 0,
        }
        stop_reason: Optional[str] = None

        for payload in _iter_sse(response):
            etype = payload.get("type")
            if etype == "message_start":
                u = (payload.get("message") or {}).get("usage") or {}
                usage["input_tokens"] = u.get("input_tokens", 0) or 0
                usage["output_tokens"] = u.get("output_tokens", 0) or 0
                usage["cache_read"] = u.get("cache_read_input_tokens", 0) or 0
                usage["cache_write"] = u.get("cache_creation_input_tokens", 0) or 0
            elif etype == "content_block_start":
                idx = payload.get("index")
                block = payload.get("content_block") or {}
                btype = block.get("type")
                if btype == "tool_use":
                    blocks[idx] = {
                        "type": "tool_use",
                        "id": block.get("id"),
                        "name": _resolve_tool_name(block.get("name", ""), tool_names),
                        "partial_json": "",
                    }
                else:
                    blocks[idx] = {"type": btype}
            elif etype == "content_block_delta":
                idx = payload.get("index")
                delta = payload.get("delta") or {}
                dtype = delta.get("type")
                if dtype == "text_delta":
                    yield {"type": "text_delta", "delta": delta.get("text", "")}
                elif dtype == "thinking_delta":
                    yield {"type": "thinking_delta", "delta": delta.get("thinking", "")}
                elif dtype == "input_json_delta" and idx in blocks:
                    blocks[idx]["partial_json"] += delta.get("partial_json", "")
            elif etype == "content_block_stop":
                idx = payload.get("index")
                block = blocks.get(idx)
                if block and block.get("type") == "tool_use":
                    try:
                        args = json.loads(block["partial_json"] or "{}")
                    except (json.JSONDecodeError, ValueError):
                        args = {}
                    yield {
                        "type": "tool_call",
                        "index": idx,
                        "tool_call": {
                            "id": block.get("id"),
                            "name": block.get("name"),
                            "arguments": args,
                        },
                    }
            elif etype == "message_delta":
                delta = payload.get("delta") or {}
                if delta.get("stop_reason"):
                    stop_reason = _map_stop_reason(delta["stop_reason"])
                out = (payload.get("usage") or {}).get("output_tokens")
                if isinstance(out, int):
                    usage["output_tokens"] = out
            elif etype == "error":
                error = payload.get("error") or {}
                raise RuntimeError(error.get("message") or "Anthropic request failed")

        # Emit the terminal event once the SSE stream is exhausted, regardless of
        # whether a message_stop frame arrived (matches the TS/canonical ports).
        usage["total_tokens"] = (
            usage["input_tokens"]
            + usage["output_tokens"]
            + usage["cache_read"]
            + usage["cache_write"]
        )
        yield {"type": "done", "usage": dict(usage), "stop_reason": stop_reason}

    def _raise_for_status(self, response: httpx.Response) -> None:
        text = response.read().decode(errors="replace")
        raise RuntimeError(f"Anthropic request failed ({response.status_code}): {text}")
