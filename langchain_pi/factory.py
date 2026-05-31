from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel


def create_chat(provider: str, model: str, **kwargs: Any) -> BaseChatModel:
    """Route by provider: opencode / opencode-go → ChatOpencode (native Zen),
    everything else → ChatPi (via pi)."""
    if provider in ("opencode", "opencode-go"):
        from .opencode import ChatOpencode

        tier = "go" if provider == "opencode-go" else "zen"
        return ChatOpencode(model, tier=tier, **kwargs)
    from .chat_models import ChatPi

    return ChatPi(provider=provider, model=model, **kwargs)
