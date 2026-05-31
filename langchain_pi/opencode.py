"""Native opencode / OpenCode Zen chat model — talks the OpenAI-compatible
chat/completions endpoint directly (no pi, no opencode binary).

With no key, free models work via the anonymous IP-rate-limited trial; with a
key (auth.json / OPENCODE_API_KEY / explicit) it unlocks paid models and higher
limits. Requires the optional ``langchain-openai`` dependency (``[opencode]`` extra)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from langchain_openai import ChatOpenAI

ZEN_BASE_URL = "https://opencode.ai/zen/v1"
ZEN_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
_AUTH_PATH = Path.home() / ".local/share/opencode/auth.json"


def read_opencode_key(auth_path: Optional[str] = None) -> Optional[str]:
    """The key opencode auto-provisions on login, read from its auth.json."""
    try:
        data = json.loads(Path(auth_path or _AUTH_PATH).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    entry = data.get("opencode-go")
    return entry.get("key") if isinstance(entry, dict) else None


class ChatOpencode(ChatOpenAI):
    """opencode/Zen chat model. Free models (deepseek-v4-flash-free, big-pickle,
    mimo-v2.5-free, nemotron-3-super-free) work with no key via the anonymous
    IP-limited trial; paid models require a key."""

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        tier: str = "zen",
        **kwargs: Any,
    ) -> None:
        key = api_key or os.environ.get("OPENCODE_API_KEY") or read_opencode_key()
        base = ZEN_GO_BASE_URL if tier == "go" else ZEN_BASE_URL
        if key:
            # Normal auth: unlocks paid models + higher limits.
            super().__init__(model=model, api_key=key, base_url=base, **kwargs)
        else:
            # Anonymous free tier: the gateway serves free models on a blank
            # Authorization header. The SDK rejects an empty key, so we pass a
            # placeholder and override the header to "".
            headers = {**kwargs.pop("default_headers", {}), "Authorization": ""}
            super().__init__(
                model=model,
                api_key="anonymous",
                base_url=base,
                default_headers=headers,
                **kwargs,
            )
