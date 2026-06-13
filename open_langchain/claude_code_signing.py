from __future__ import annotations

import hashlib
from typing import Any

BILLING_SALT = "59cf53e54c78"


def sanitize_surrogates(text: str) -> str:
    """Replace every UTF-16 surrogate code unit with U+FFFD, matching the JS
    extension's ``/[\\uD800-\\uDFFF]/g`` replace. Astral chars (emoji) collapse to
    two U+FFFD exactly like the canonical implementation, so the resulting string
    is BMP-only and code-point indexing equals JS UTF-16 code-unit indexing."""
    units = text.encode("utf-16-le", "surrogatepass")
    out = []
    for i in range(0, len(units), 2):
        cu = units[i] | (units[i + 1] << 8)
        out.append("�" if 0xD800 <= cu <= 0xDFFF else chr(cu))
    return "".join(out)


def _extract_first_user_message_text(messages: list[dict]) -> str:
    user_msg = next((m for m in messages if m.get("role") == "user"), None)
    if not user_msg:
        return ""
    content = user_msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        block = next(
            (b for b in content if isinstance(b, dict) and b.get("type") == "text"),
            None,
        )
        if block and block.get("text"):
            return block["text"]
    return ""


def _compute_cch(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:5]


def _compute_version_suffix(text: str, version: str) -> str:
    sampled = "".join(text[i] if i < len(text) else "0" for i in (4, 7, 20))
    digest = hashlib.sha256((BILLING_SALT + sampled + version).encode("utf-8"))
    return digest.hexdigest()[:3]


def build_billing_header_value(
    messages: list[dict[str, Any]], version: str, entrypoint: str
) -> str:
    text = sanitize_surrogates(_extract_first_user_message_text(messages))
    suffix = _compute_version_suffix(text, version)
    cch = _compute_cch(text)
    return (
        f"x-anthropic-billing-header: cc_version={version}.{suffix}; "
        f"cc_entrypoint={entrypoint}; cch={cch};"
    )
