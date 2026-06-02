from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from .claude_code_models import (
    CLAUDE_CODE_MODELS,
    DEFAULT_BUDGETS,
    get_model_override,
)
from .claude_code_signing import build_billing_header_value, sanitize_surrogates

TOOL_PREFIX = "mcp_"
SYSTEM_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
BILLING_PREFIX = "x-anthropic-billing-header"


def prefix_tool_name(name: str) -> str:
    return f"{TOOL_PREFIX}{name[:1].upper()}{name[1:]}"


def unprefix_tool_name(name: str) -> str:
    if not name.startswith(TOOL_PREFIX):
        return name
    rest = name[len(TOOL_PREFIX) :]
    return rest[:1].lower() + rest[1:]


def to_effort(level: str) -> str:
    return "low" if level == "minimal" else level


def _flatten_text(content: Any) -> str:
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


def _assistant_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            parts.append(block["text"])
    return "\n".join(parts)


def _user_blocks(content: list) -> list[dict]:
    blocks: list[dict] = []
    for block in content:
        if isinstance(block, str):
            blocks.append({"type": "text", "text": sanitize_surrogates(block)})
        elif isinstance(block, dict):
            btype = block.get("type")
            if btype == "text":
                blocks.append(
                    {"type": "text", "text": sanitize_surrogates(block.get("text", ""))}
                )
            elif btype == "image":
                blocks.append(block)
            elif btype == "image_url":
                image_url = block.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
            else:
                blocks.append({"type": "text", "text": json.dumps(block)})
    return blocks


def _user_message(message: BaseMessage) -> Optional[dict]:
    content = message.content
    if isinstance(content, str):
        text = sanitize_surrogates(content)
        return {"role": "user", "content": text} if text.strip() else None
    blocks = _user_blocks(content)
    return {"role": "user", "content": blocks} if blocks else None


def _assistant_message(message: AIMessage) -> dict:
    blocks: list[dict] = []
    text = _assistant_text(message.content)
    if text.strip():
        blocks.append({"type": "text", "text": sanitize_surrogates(text)})
    for call in message.tool_calls or []:
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id") or "",
                "name": call["name"],
                "input": call.get("args") or {},
            }
        )
    # Drop thinking blocks (never re-sent: their signature is bound to the
    # original turn). Keep a placeholder so role alternation survives.
    if not blocks:
        blocks.append({"type": "text", "text": "(no content)"})
    return {"role": "assistant", "content": blocks}


def _tool_result_block(message: ToolMessage) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": message.tool_call_id,
        "content": sanitize_surrogates(_flatten_text(message.content)),
        "is_error": message.status == "error",
    }


def messages_to_anthropic(
    messages: list[BaseMessage],
) -> tuple[list[str], list[dict]]:
    system_texts: list[str] = []
    result: list[dict] = []
    tool_user: Optional[dict] = None

    for message in messages:
        kind = message.type
        if kind == "tool":
            block = _tool_result_block(message)  # type: ignore[arg-type]
            if tool_user is not None:
                tool_user["content"].append(block)
            else:
                tool_user = {"role": "user", "content": [block]}
                result.append(tool_user)
            continue
        tool_user = None
        if kind in ("system", "developer"):
            system_texts.append(_flatten_text(message.content))
        elif kind == "ai":
            result.append(_assistant_message(message))  # type: ignore[arg-type]
        else:
            user = _user_message(message)
            if user is not None:
                result.append(user)

    if result:
        last = result[-1]
        if (
            last.get("role") == "user"
            and isinstance(last.get("content"), list)
            and last["content"]
        ):
            last["content"][-1]["cache_control"] = {"type": "ephemeral"}

    return system_texts, result


def tool_to_anthropic(tool: Any) -> dict:
    fn = convert_to_openai_tool(tool)["function"]
    params = fn.get("parameters") or {"type": "object", "properties": {}}
    return {
        "name": fn["name"],
        "description": fn.get("description") or "",
        "input_schema": {
            "type": "object",
            "properties": params.get("properties") or {},
            "required": params.get("required") or [],
        },
    }


def repair_tool_pairs(messages: list[dict]) -> list[dict]:
    tool_use_ids: set[str] = set()
    tool_result_ids: set[str] = set()
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                tool_use_ids.add(block["id"])
            if block.get("type") == "tool_result" and isinstance(
                block.get("tool_use_id"), str
            ):
                tool_result_ids.add(block["tool_use_id"])

    orphan_uses = tool_use_ids - tool_result_ids
    orphan_results = tool_result_ids - tool_use_ids
    if not orphan_uses and not orphan_results:
        return messages

    repaired: list[dict] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            repaired.append(message)
            continue
        filtered = []
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("id") in orphan_uses
            ):
                continue
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and block.get("tool_use_id") in orphan_results
            ):
                continue
            filtered.append(block)
        if not filtered:
            placeholder = (
                {"type": "text", "text": "[tool result omitted]"}
                if message.get("role") == "user"
                else {"type": "text", "text": "(no content)"}
            )
            repaired.append({**message, "content": [placeholder]})
        else:
            repaired.append({**message, "content": filtered})
    return repaired


def apply_claude_code_transforms(body: dict, *, version: str, entrypoint: str) -> dict:
    billing_header = build_billing_header_value(
        body.get("messages") or [], version, entrypoint
    )

    raw_system = body.get("system")
    if isinstance(raw_system, str):
        system = [{"type": "text", "text": raw_system}] if raw_system.strip() else []
    elif isinstance(raw_system, list):
        system = list(raw_system)
    else:
        system = []

    system = [
        e
        for e in system
        if not (isinstance(e.get("text"), str) and e["text"].startswith(BILLING_PREFIX))
    ]

    has_identity = any(
        isinstance(e.get("text"), str) and e["text"].startswith(SYSTEM_IDENTITY)
        for e in system
    )

    split: list[dict] = []
    for entry in system:
        text = entry.get("text") if isinstance(entry.get("text"), str) else ""
        if text.startswith(SYSTEM_IDENTITY) and len(text) > len(SYSTEM_IDENTITY):
            rest = text[len(SYSTEM_IDENTITY) :].lstrip("\n")
            identity_props = {
                k: v for k, v in entry.items() if k not in ("text", "cache_control")
            }
            rest_props = {k: v for k, v in entry.items() if k != "text"}
            split.append({**identity_props, "type": "text", "text": SYSTEM_IDENTITY})
            if rest:
                split.append({**rest_props, "type": "text", "text": rest})
        else:
            split.append(entry)
    system = split

    if not has_identity:
        system.insert(0, {"type": "text", "text": SYSTEM_IDENTITY})

    kept: list[dict] = []
    moved: list[str] = []
    for entry in system:
        text = entry.get("text") if isinstance(entry.get("text"), str) else ""
        if text.startswith(BILLING_PREFIX) or text.startswith(SYSTEM_IDENTITY):
            kept.append(entry)
        elif text:
            moved.append(text)

    messages = body.get("messages")
    if moved and isinstance(messages, list):
        prefix = "\n\n".join(moved)
        first_user = next((m for m in messages if m.get("role") == "user"), None)
        if first_user is not None:
            content = first_user.get("content")
            if isinstance(content, str):
                first_user["content"] = prefix + "\n\n" + content
            elif isinstance(content, list):
                content.insert(0, {"type": "text", "text": prefix})
        else:
            # No user turn to carry the system text; synthesize one so the prompt
            # is not dropped and the request is not sent with zero messages.
            messages.insert(0, {"role": "user", "content": prefix})

    kept.insert(0, {"type": "text", "text": billing_header})
    body["system"] = kept

    override = get_model_override(body.get("model") or "")
    thinking = body.get("thinking")
    if (
        override
        and override.get("disable_effort")
        and isinstance(thinking, dict)
        and "effort" in thinking
    ):
        del thinking["effort"]
        if not thinking:
            body.pop("thinking", None)

    tools = body.get("tools")
    if isinstance(tools, list):
        body["tools"] = [
            {**t, "name": prefix_tool_name(t["name"])} if t.get("name") else t
            for t in tools
        ]

    if isinstance(messages, list):
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            message["content"] = [
                {**b, "name": prefix_tool_name(b["name"])}
                if isinstance(b, dict)
                and b.get("type") == "tool_use"
                and isinstance(b.get("name"), str)
                else b
                for b in content
            ]
        body["messages"] = repair_tool_pairs(messages)

    return body


def build_request_body(
    model_id: str,
    system_texts: list[str],
    messages: list[dict],
    *,
    max_tokens: Optional[int] = None,
    tools: Optional[list[dict]] = None,
    reasoning: Optional[str] = None,
    thinking_budgets: Optional[dict] = None,
    stop: Optional[list[str]] = None,
) -> dict:
    model = CLAUDE_CODE_MODELS.get(model_id, {})
    system: list[dict] = [
        {
            "type": "text",
            "text": SYSTEM_IDENTITY,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    for text in system_texts:
        if text:
            system.append(
                {
                    "type": "text",
                    "text": sanitize_surrogates(text),
                    "cache_control": {"type": "ephemeral"},
                }
            )

    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens or (model.get("max_tokens", 4096) // 3),
        "stream": True,
        "system": system,
    }
    if tools:
        body["tools"] = tools
    if stop:
        body["stop_sequences"] = stop

    if reasoning and reasoning != "off" and model.get("reasoning"):
        override = get_model_override(model_id)
        if override and override.get("adaptive_thinking"):
            body["thinking"] = {"type": "adaptive"}
            body["output_config"] = {"effort": to_effort(reasoning)}
        else:
            budgets = thinking_budgets or {}
            budget = budgets.get(reasoning) or DEFAULT_BUDGETS.get(reasoning) or 10240
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}

    return body
