from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from .models import thinking_wire_value

DEFAULT_INSTRUCTIONS = "You are a helpful assistant."


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


def tool_to_responses(tool: Any) -> dict:
    fn = convert_to_openai_tool(tool)["function"]
    return {
        "type": "function",
        "name": fn["name"],
        "description": fn.get("description") or "",
        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        "strict": None,
    }


def _user_content(content: Any) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    blocks: list[dict] = []
    for block in content:
        if isinstance(block, str):
            blocks.append({"type": "input_text", "text": block})
        elif isinstance(block, dict):
            btype = block.get("type")
            if btype == "text":
                blocks.append({"type": "input_text", "text": block.get("text", "")})
            elif btype == "image_url":
                image_url = block.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                blocks.append({"type": "input_image", "image_url": url})
            elif btype in ("input_text", "input_image"):
                blocks.append(block)
            else:
                blocks.append({"type": "input_text", "text": json.dumps(block)})
    return blocks


def _assistant_items(message: AIMessage) -> list[dict]:
    items: list[dict] = []
    text = _content_to_text(message.content)
    if text:
        items.append(
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        )
    for call in message.tool_calls or []:
        items.append(
            {
                "type": "function_call",
                "name": call["name"],
                "arguments": json.dumps(call.get("args") or {}),
                "call_id": call.get("id") or "",
            }
        )
    return items


def _tool_result_item(message: ToolMessage) -> dict:
    return {
        "type": "function_call_output",
        "call_id": message.tool_call_id,
        "output": _content_to_text(message.content),
    }


def messages_to_responses(
    messages: list[BaseMessage], system: Optional[str]
) -> tuple[str, list[dict]]:
    instructions_parts = [system] if system else []
    input_items: list[dict] = []
    for message in messages:
        kind = message.type
        if kind in ("system", "developer"):
            instructions_parts.append(_content_to_text(message.content))
        elif kind == "ai":
            input_items.extend(_assistant_items(message))  # type: ignore[arg-type]
        elif kind == "tool":
            input_items.append(_tool_result_item(message))  # type: ignore[arg-type]
        else:
            input_items.append(
                {"role": "user", "content": _user_content(message.content)}
            )
    parts = [p for p in instructions_parts if p]
    instructions = "\n\n".join(parts) if parts else DEFAULT_INSTRUCTIONS
    return instructions, input_items


def build_request_body(
    model: str,
    instructions: str,
    input_items: list[dict],
    tools: Optional[list[dict]] = None,
    tool_choice: Any = "auto",
    reasoning_effort: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict:
    body: dict[str, Any] = {
        "model": model,
        "store": False,
        "stream": True,
        "instructions": instructions,
        "input": input_items,
        "text": {"verbosity": "low"},
        "include": ["reasoning.encrypted_content"],
        "tool_choice": tool_choice,
        "parallel_tool_calls": True,
    }
    if session_id:
        body["prompt_cache_key"] = session_id
    if tools:
        body["tools"] = tools
    wire = thinking_wire_value(model, reasoning_effort) if reasoning_effort else None
    if wire is not None:
        body["reasoning"] = {"effort": wire, "summary": "auto"}
    return body


def to_tool_calls(raw: list[dict]) -> list[dict]:
    tool_calls = []
    for tc in raw:
        args = tc.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                args = {}
        tool_calls.append(
            {
                "name": tc.get("name"),
                "args": args or {},
                "id": tc.get("call_id") or tc.get("id"),
                "type": "tool_call",
            }
        )
    return tool_calls


def to_usage_metadata(usage: Optional[dict]) -> Optional[dict]:
    if not usage:
        return None
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
        "input_token_details": {
            "cache_read": cached,
            "cache_creation": 0,
        },
    }


def to_response_metadata(
    model: str, stop_reason: Optional[str], usage: Optional[dict]
) -> dict:
    metadata: dict[str, Any] = {
        "provider": "openai-codex",
        "model": model,
        "stopReason": stop_reason,
    }
    if usage:
        metadata["usage"] = usage
    return metadata
