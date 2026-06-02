from __future__ import annotations

import hashlib
import re

from langchain_pi.claude_code_signing import (
    BILLING_SALT,
    _compute_cch,
    _compute_version_suffix,
    _extract_first_user_message_text,
    build_billing_header_value,
    sanitize_surrogates,
)

HEADER_RE = re.compile(
    r"^x-anthropic-billing-header: cc_version=\d+\.\d+\.\d+\.[a-f0-9]{3}; "
    r"cc_entrypoint=[^;]+; cch=[a-f0-9]{5};$"
)


def test_extract_first_user_text_string():
    messages = [
        {"role": "assistant", "content": "ignored"},
        {"role": "user", "content": "hello"},
    ]
    assert _extract_first_user_message_text(messages) == "hello"


def test_extract_first_user_text_blocks():
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert _extract_first_user_message_text(messages) == "hi"


def test_extract_first_user_text_empty():
    assert _extract_first_user_message_text([]) == ""
    assert (
        _extract_first_user_message_text([{"role": "assistant", "content": "x"}]) == ""
    )


def test_cch_is_sha256_prefix_5():
    assert _compute_cch("hello") == hashlib.sha256(b"hello").hexdigest()[:5]
    assert _compute_cch("") == "e3b0c"


def test_version_suffix_samples_4_7_20_and_pads():
    # short message: every sampled index is out of range -> padded with "0"
    expected = hashlib.sha256((BILLING_SALT + "000" + "2.1.112").encode()).hexdigest()[
        :3
    ]
    assert _compute_version_suffix("abc", "2.1.112") == expected


def test_version_suffix_uses_sampled_chars():
    text = "0123456789012345678901"
    sampled = text[4] + text[7] + text[20]
    expected = hashlib.sha256(
        (BILLING_SALT + sampled + "2.1.112").encode()
    ).hexdigest()[:3]
    assert _compute_version_suffix(text, "2.1.112") == expected


def test_build_billing_header_format_and_deterministic():
    value = build_billing_header_value(
        [{"role": "user", "content": "hello"}], "2.1.112", "sdk-cli"
    )
    assert HEADER_RE.match(value)
    assert value == build_billing_header_value(
        [{"role": "user", "content": "hello"}], "2.1.112", "sdk-cli"
    )


def test_build_billing_header_empty_is_deterministic():
    value = build_billing_header_value([], "2.1.112", "sdk-cli")
    assert HEADER_RE.match(value)
    assert "cch=e3b0c;" in value


def test_only_first_user_text_influences_output():
    base = [
        {"role": "assistant", "content": "ignored"},
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
    ]
    changed = [
        {"role": "assistant", "content": "changed"},
        {"role": "user", "content": "first"},
        {"role": "user", "content": "changed"},
    ]
    assert build_billing_header_value(base, "2.1.112", "sdk-cli") == (
        build_billing_header_value(changed, "2.1.112", "sdk-cli")
    )


def test_sanitize_surrogates_astral_parity():
    # Matches the JS extension: each UTF-16 surrogate code unit (incl. emoji
    # halves) collapses to U+FFFD, so indexing stays BMP-aligned.
    assert sanitize_surrogates("\U0001f600") == "��"
    assert sanitize_surrogates("hi") == "hi"


def test_billing_header_astral_uses_utf16_sampling():
    # An emoji-prefixed first message must hash the sanitized (BMP-collapsed)
    # text, not raw code points.
    messages = [{"role": "user", "content": "\U0001f600abcdefghij"}]
    value = build_billing_header_value(messages, "2.1.112", "sdk-cli")
    sanitized = sanitize_surrogates("\U0001f600abcdefghij")
    expected_cch = hashlib.sha256(sanitized.encode()).hexdigest()[:5]
    assert f"cch={expected_cch};" in value
