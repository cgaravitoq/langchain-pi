"""LangChain <-> Pi conversions. The LangChain-facing mirror of the TS
pi-conversions.ts. Message context shaping that needs the resolved pi Model
(assistant api/provider/id stamping) lives in the Node sidecar; here we only
extract a neutral IR from LangChain messages plus tool/usage/metadata mapping."""

from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
        else:
            parts.append(json.dumps(block))
    return "\n".join(parts)


def apply_stop(text: str, stop: Optional[list[str]]) -> str:
    if not stop:
        return text
    idxs = [i for i in (text.find(s) for s in stop) if i >= 0]
    return text[: min(idxs)] if idxs else text


def tool_to_pi(tool: Any) -> dict:
    fn = convert_to_openai_tool(tool)["function"]
    return {
        "name": fn["name"],
        "description": fn.get("description") or "",
        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
    }


def _assistant(message: AIMessage) -> dict:
    tool_calls = [
        {"id": c.get("id") or "", "name": c["name"], "arguments": c.get("args") or {}}
        for c in (message.tool_calls or [])
    ]
    return {
        "role": "assistant",
        "text": _content_to_text(message.content),
        "toolCalls": tool_calls,
    }


def _tool_result(message: ToolMessage) -> dict:
    return {
        "role": "toolResult",
        "toolCallId": message.tool_call_id,
        "toolName": message.name or "",
        "text": _content_to_text(message.content),
        "isError": message.status == "error",
    }


def messages_to_pi_payload(
    messages: list[BaseMessage], system: Optional[str]
) -> tuple[Optional[str], list[dict]]:
    system_prompts = [system] if system else []
    history: list[dict] = []
    for message in messages:
        kind = message.type
        if kind in ("system", "developer"):
            system_prompts.append(_content_to_text(message.content))
        elif kind == "ai":
            history.append(_assistant(message))  # type: ignore[arg-type]
        elif kind == "tool":
            history.append(_tool_result(message))  # type: ignore[arg-type]
        else:
            history.append({"role": "user", "text": _content_to_text(message.content)})
    prompts = [p for p in system_prompts if p]
    return ("\n\n".join(prompts) if prompts else None), history


def to_tool_calls(raw: list[dict]) -> list[dict]:
    return [
        {
            "name": tc["name"],
            "args": tc.get("arguments") or {},
            "id": tc.get("id"),
            "type": "tool_call",
        }
        for tc in raw
    ]


def to_usage_metadata(usage: Optional[dict]) -> Optional[dict]:
    if not usage:
        return None
    return {
        "input_tokens": usage.get("input", 0),
        "output_tokens": usage.get("output", 0),
        "total_tokens": usage.get("totalTokens", 0),
        "input_token_details": {
            "cache_read": usage.get("cacheRead", 0),
            "cache_creation": usage.get("cacheWrite", 0),
        },
    }


def to_response_metadata(
    provider: str, model: str, stop_reason: Optional[str], usage: Optional[dict]
) -> dict:
    metadata: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "stopReason": stop_reason,
    }
    if usage:
        metadata["usage"] = usage
        metadata["cost"] = usage.get("cost")
    return metadata
