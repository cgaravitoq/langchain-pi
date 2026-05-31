from __future__ import annotations

import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI

ZEN_BASE_URL = "https://opencode.ai/zen/v1"
ZEN_GO_BASE_URL = "https://opencode.ai/zen/go/v1"


class ChatOpencode(ChatOpenAI):
    """opencode/Zen chat model via the OpenAI-compatible endpoint (no pi)."""

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        tier: str = "zen",
        **kwargs: Any,
    ) -> None:
        key = api_key or os.environ.get("OPENCODE_API_KEY")
        base = ZEN_GO_BASE_URL if tier == "go" else ZEN_BASE_URL
        if key:
            super().__init__(model=model, api_key=key, base_url=base, **kwargs)
        else:
            # No key → anonymous free tier (blank Authorization); paid needs a key.
            headers = {**kwargs.pop("default_headers", {}), "Authorization": ""}
            super().__init__(
                model=model,
                api_key="anonymous",
                base_url=base,
                default_headers=headers,
                **kwargs,
            )
