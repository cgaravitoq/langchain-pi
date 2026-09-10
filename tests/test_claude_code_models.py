from __future__ import annotations

from open_langchain.claude_code_models import (
    BASE_BETAS,
    CC_VERSION,
    CLAUDE_CODE_MODELS,
    MODEL_OVERRIDES,
    compute_betas,
    get_model_override,
)


def test_cc_version_is_recent_enough_for_fable_5_1():
    assert CC_VERSION == "2.1.267"


def test_compute_betas_opus_4_8():
    betas = compute_betas("claude-opus-4-8")
    assert "context-1m-2025-08-07" in betas
    assert "effort-2025-11-24" in betas
    assert len(betas) == len(set(betas))


def test_compute_betas_sonnet_4_6():
    betas = compute_betas("claude-sonnet-4-6")
    assert "context-1m-2025-08-07" in betas
    assert "effort-2025-11-24" in betas


def test_compute_betas_claude_5_has_no_long_context():
    for model_id in (
        "claude-opus-5",
        "claude-fable-5",
        "claude-fable-5-1",
        "claude-sonnet-5",
    ):
        betas = compute_betas(model_id)
        assert "context-1m-2025-08-07" not in betas
        assert betas == list(BASE_BETAS)


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


def test_get_model_override_claude_5_is_adaptive_without_long_context():
    for model_id in (
        "claude-opus-5",
        "claude-fable-5",
        "claude-fable-5-1",
        "claude-mythos-5",
        "claude-sonnet-5",
    ):
        override = get_model_override(model_id)
        assert override["adaptive_thinking"] is True
        assert "long_context" not in override


def test_get_model_override_sonnet_4_6_does_not_match_claude_5():
    assert get_model_override("claude-sonnet-4-6") is MODEL_OVERRIDES["4-6"]


def test_get_model_override_fable_5_covers_fable_5_1_by_substring():
    assert get_model_override("claude-fable-5-1") is MODEL_OVERRIDES["fable-5"]


def test_models_registry_metadata():
    assert set(CLAUDE_CODE_MODELS) == {
        "claude-opus-5",
        "claude-fable-5",
        "claude-fable-5-1",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    }
    opus = CLAUDE_CODE_MODELS["claude-opus-4-8"]
    assert opus["context_window"] == 1000000 and opus["max_tokens"] == 128000
    assert opus["reasoning"] is True
    haiku = CLAUDE_CODE_MODELS["claude-haiku-4-5"]
    assert haiku["context_window"] == 200000 and haiku["max_tokens"] == 64000
    assert haiku["reasoning"] is False


def test_claude_5_registry_metadata():
    opus = CLAUDE_CODE_MODELS["claude-opus-5"]
    assert opus["name"] == "Claude Opus 5 (Claude Code)"
    assert opus["cost"] == {
        "input": 5,
        "output": 25,
        "cache_read": 0.5,
        "cache_write": 6.25,
    }
    fable = CLAUDE_CODE_MODELS["claude-fable-5"]
    assert fable["name"] == "Claude Fable 5 (Claude Code)"
    assert fable["cost"] == {
        "input": 10,
        "output": 50,
        "cache_read": 1,
        "cache_write": 12.5,
    }
    fable_5_1 = CLAUDE_CODE_MODELS["claude-fable-5-1"]
    assert fable_5_1["name"] == "Claude Fable 5.1 (Claude Code)"
    assert fable_5_1["cost"] == {
        "input": 10,
        "output": 50,
        "cache_read": 0.25,
        "cache_write": 12.5,
    }
    sonnet = CLAUDE_CODE_MODELS["claude-sonnet-5"]
    assert sonnet["name"] == "Claude Sonnet 5 (Claude Code)"
    assert sonnet["cost"] == {
        "input": 2,
        "output": 10,
        "cache_read": 0.2,
        "cache_write": 2.5,
    }
    for model in (opus, fable, fable_5_1, sonnet):
        assert model["reasoning"] is True
        assert model["input"] == ["text", "image"]
        assert model["context_window"] == 1000000 and model["max_tokens"] == 128000
