from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel


def create_chat(provider: str, model: str, **kwargs: Any) -> BaseChatModel:
    """Route supported native providers to their chat model."""
    if provider == "openai-codex":
        from .chat_models import ChatCodex

        return ChatCodex(model=model, **kwargs)
    if provider in ("opencode", "opencode-go"):
        from .opencode import ChatOpencode

        tier = "go" if provider == "opencode-go" else "zen"
        return ChatOpencode(model, tier=tier, **kwargs)
    raise ValueError(f"Unsupported provider: {provider}")
