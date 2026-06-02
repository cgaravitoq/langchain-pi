from __future__ import annotations

from typing import Optional

PROVIDER_ID = "claude-code"
PROVIDER_NAME = "Claude Code (OAuth)"

CC_VERSION = "2.1.112"

BASE_BETAS = [
    "claude-code-20250219",
    "oauth-2025-04-20",
    "interleaved-thinking-2025-05-14",
    "prompt-caching-scope-2026-01-05",
    "context-management-2025-06-27",
    "advisor-tool-2026-03-01",
]

LONG_CONTEXT_BETAS = [
    "context-1m-2025-08-07",
    "interleaved-thinking-2025-05-14",
]

# Insertion order matters: get_model_override returns the first key that is a
# substring of the lowercased model id.
MODEL_OVERRIDES: dict[str, dict] = {
    "haiku": {"exclude": ["interleaved-thinking-2025-05-14"], "disable_effort": True},
    "4-6": {"long_context": True, "add": ["effort-2025-11-24"]},
    "4-7": {
        "long_context": True,
        "add": ["effort-2025-11-24"],
        "adaptive_thinking": True,
    },
    "4-8": {
        "long_context": True,
        "add": ["effort-2025-11-24"],
        "adaptive_thinking": True,
    },
}

DEFAULT_BUDGETS = {
    "minimal": 1024,
    "low": 4096,
    "medium": 10240,
    "high": 20480,
    "xhigh": 32768,
    "max": 64000,
}

CLAUDE_CODE_MODELS: dict[str, dict] = {
    "claude-opus-4-8": {
        "name": "Claude Opus 4.8 (Claude Code)",
        "reasoning": True,
        "input": ["text", "image"],
        "cost": {"input": 5, "output": 25, "cache_read": 0.5, "cache_write": 6.25},
        "context_window": 1000000,
        "max_tokens": 128000,
    },
    "claude-opus-4-7": {
        "name": "Claude Opus 4.7 (Claude Code)",
        "reasoning": True,
        "input": ["text", "image"],
        "cost": {"input": 5, "output": 25, "cache_read": 0.5, "cache_write": 6.25},
        "context_window": 1000000,
        "max_tokens": 128000,
    },
    "claude-sonnet-4-6": {
        "name": "Claude Sonnet 4.6 (Claude Code)",
        "reasoning": True,
        "input": ["text", "image"],
        "cost": {"input": 3, "output": 15, "cache_read": 0.3, "cache_write": 3.75},
        "context_window": 1000000,
        "max_tokens": 128000,
    },
    "claude-haiku-4-5": {
        "name": "Claude Haiku 4.5 (Claude Code)",
        "reasoning": False,
        "input": ["text", "image"],
        "cost": {"input": 1, "output": 5, "cache_read": 0.1, "cache_write": 1.25},
        "context_window": 200000,
        "max_tokens": 64000,
    },
}


def get_model_override(model_id: str) -> Optional[dict]:
    lower = model_id.lower()
    for pattern, override in MODEL_OVERRIDES.items():
        if pattern in lower:
            return override
    return None


def compute_betas(model_id: str) -> list[str]:
    override = get_model_override(model_id)
    betas = list(BASE_BETAS)
    if override and override.get("long_context"):
        betas += LONG_CONTEXT_BETAS
    if override and override.get("exclude"):
        betas = [b for b in betas if b not in override["exclude"]]
    if override and override.get("add"):
        betas += override["add"]
    return list(dict.fromkeys(betas))
