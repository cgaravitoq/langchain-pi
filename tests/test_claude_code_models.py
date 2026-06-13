from __future__ import annotations

from open_langchain.claude_code_models import (
    BASE_BETAS,
    CLAUDE_CODE_MODELS,
    compute_betas,
    get_model_override,
)


def test_compute_betas_opus_4_8():
    betas = compute_betas("claude-opus-4-8")
    assert "context-1m-2025-08-07" in betas
    assert "effort-2025-11-24" in betas
    assert len(betas) == len(set(betas))


def test_compute_betas_sonnet_4_6():
    betas = compute_betas("claude-sonnet-4-6")
    assert "context-1m-2025-08-07" in betas
    assert "effort-2025-11-24" in betas


def test_compute_betas_haiku_excludes_interleaved_and_effort():
    betas = compute_betas("claude-haiku-4-5")
    assert "interleaved-thinking-2025-05-14" not in betas
    assert "effort-2025-11-24" not in betas
    assert "context-1m-2025-08-07" not in betas


def test_compute_betas_unknown_model_is_base():
    assert compute_betas("gpt-5.5") == list(BASE_BETAS)


def test_get_model_override_first_match_wins():
    assert get_model_override("claude-opus-4-8")["adaptive_thinking"] is True
    assert get_model_override("claude-opus-4-7")["adaptive_thinking"] is True
    assert "adaptive_thinking" not in get_model_override("claude-sonnet-4-6")
    assert get_model_override("claude-haiku-4-5")["disable_effort"] is True
    assert get_model_override("gpt-foo") is None


def test_models_registry_metadata():
    assert set(CLAUDE_CODE_MODELS) == {
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    }
    opus = CLAUDE_CODE_MODELS["claude-opus-4-8"]
    assert opus["context_window"] == 1000000 and opus["max_tokens"] == 128000
    assert opus["reasoning"] is True
    haiku = CLAUDE_CODE_MODELS["claude-haiku-4-5"]
    assert haiku["context_window"] == 200000 and haiku["max_tokens"] == 64000
    assert haiku["reasoning"] is False
