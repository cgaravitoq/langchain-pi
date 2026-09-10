from __future__ import annotations

from typing import Optional

OPENAI_CODEX_DEFAULTS = {
    "api": "openai-codex-responses",
    "provider": "openai-codex",
    "base_url": "https://chatgpt.com/backend-api",
    "reasoning": True,
    "thinking_level_map": {"xhigh": "xhigh", "minimal": "low"},
    "max_tokens": 128000,
}

OPENAI_CODEX_MODELS = {
    "gpt-5.3-codex-spark": {
        "name": "GPT-5.3 Codex Spark",
        "input": ["text"],
        "context_window": 128000,
        "cost": {"input": 1.75, "output": 14, "cache_read": 0.175, "cache_write": 0},
    },
    "gpt-5.5": {
        "name": "GPT-5.5",
        "input": ["text", "image"],
        "context_window": 1050000,
        "cost": {"input": 5, "output": 30, "cache_read": 0.5, "cache_write": 0},
    },
    "gpt-5.6-sol": {
        "name": "GPT-5.6 Sol",
        "input": ["text", "image"],
        "context_window": 1050000,
        "cost": {"input": 4, "output": 20, "cache_read": 0.4, "cache_write": 5},
    },
    "gpt-5.6-terra": {
        "name": "GPT-5.6 Terra",
        "input": ["text", "image"],
        "context_window": 1050000,
        "cost": {"input": 2, "output": 12, "cache_read": 0.2, "cache_write": 2.5},
    },
    "gpt-5.6-luna": {
        "name": "GPT-5.6 Luna",
        "input": ["text", "image"],
        "context_window": 1050000,
        "cost": {"input": 0.2, "output": 1.2, "cache_read": 0.02, "cache_write": 0.25},
    },
    "gpt-6-astra": {
        "name": "GPT-6 Astra",
        "input": ["text", "image"],
        "context_window": 1050000,
        "cost": {"input": 10, "output": 50, "cache_read": 1, "cache_write": 12.5},
    },
}

EXTENDED_THINKING_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh"]


def _model_meta(model: str) -> dict:
    meta = dict(OPENAI_CODEX_DEFAULTS)
    meta.update(OPENAI_CODEX_MODELS.get(model, {}))
    return meta


def get_supported_thinking_levels(model: str) -> list[str]:
    meta = _model_meta(model)
    if not meta.get("reasoning"):
        return ["off"]
    level_map = meta.get("thinking_level_map") or {}
    supported: list[str] = []
    for level in EXTENDED_THINKING_LEVELS:
        if level in level_map and level_map[level] is None:
            continue
        if level == "xhigh" and level not in level_map:
            continue
        supported.append(level)
    return supported


def clamp_thinking_level(model: str, level: str) -> str:
    available = get_supported_thinking_levels(model)
    if level in available:
        return level
    try:
        idx = EXTENDED_THINKING_LEVELS.index(level)
    except ValueError:
        return available[0] if available else "off"
    for candidate in EXTENDED_THINKING_LEVELS[idx:]:
        if candidate in available:
            return candidate
    for candidate in reversed(EXTENDED_THINKING_LEVELS[:idx]):
        if candidate in available:
            return candidate
    return available[0] if available else "off"


def thinking_wire_value(model: str, level: str) -> Optional[str]:
    """Map a pi canonical level to the value sent in the request. off -> None (omit)."""
    if level == "off":
        return None
    level_map = _model_meta(model).get("thinking_level_map") or {}
    mapped = level_map.get(level)
    if mapped is None and level not in level_map:
        return level
    return mapped


def calculate_cost(model: str, usage: dict) -> float:
    cost = _model_meta(model).get("cost") or {}
    total = 0.0
    for key in ("input", "output", "cache_read", "cache_write"):
        per_million = cost.get(key, 0) or 0
        tokens = usage.get(key, 0) or 0
        total += per_million / 1_000_000 * tokens
    return total
