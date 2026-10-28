"""Registry wiring tests for the qwen_voice_design engine.

Covers TTS_ENGINES, get_tts_backend_for_engine, and get_model_load_func --
the same dispatch surface that had a real bug for TADA (model_size silently
ignored because it fell through to the generic branch instead of getting
its own, fixed in ac3fafd). These tests exercise the *named* branches
directly rather than only checking that the end result happens to be
right.
"""

from unittest.mock import AsyncMock, patch

import pytest

from backend.backends import (
    TTS_ENGINES,
    ModelConfig,
    engine_has_model_sizes,
    engine_retries_runaway,
    get_model_config,
    get_model_load_func,
    get_tts_backend_for_engine,
    get_tts_model_configs,
    reset_backends,
)
from backend.backends.qwen_voice_design_backend import QwenVoiceDesignBackend


@pytest.fixture(autouse=True)
def _reset():
    reset_backends()
    yield
    reset_backends()


def test_qwen_voice_design_is_registered_in_tts_engines():
    assert "qwen_voice_design" in TTS_ENGINES


def test_get_tts_backend_for_engine_returns_qwen_voice_design_backend():
    backend = get_tts_backend_for_engine("qwen_voice_design")
    assert isinstance(backend, QwenVoiceDesignBackend)


def test_get_tts_backend_for_engine_returns_same_instance_on_repeat_calls():
    backend1 = get_tts_backend_for_engine("qwen_voice_design")
    backend2 = get_tts_backend_for_engine("qwen_voice_design")
    assert backend1 is backend2


@pytest.mark.asyncio
async def test_get_model_load_func_dispatches_through_named_qwen_voice_design_branch():
    """The dispatch must go through an explicit `config.engine ==
    "qwen_voice_design"` branch in get_model_load_func, not the generic
    fallthrough -- this is the exact shape of bug ac3fafd fixed for TADA.
    """
    config = ModelConfig(
        model_name="qwen-voice-design-1.7B",
        display_name="Qwen VoiceDesign 1.7B",
        engine="qwen_voice_design",
        hf_repo_id="mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
        model_size="1.7B",
    )

    reset_backends()
    mock_backend = QwenVoiceDesignBackend()
    mock_backend.load_model = AsyncMock()

    with patch(
        "backend.backends.get_tts_backend_for_engine",
        return_value=mock_backend,
    ) as mock_get_backend:
        load_func = get_model_load_func(config)
        await load_func()

    # The named branch calls get_tts_backend_for_engine(config.engine) and
    # invokes .load_model(config.model_size) -- confirm both, not just that
    # *a* callable came back.
    mock_get_backend.assert_called_once_with("qwen_voice_design")
    mock_backend.load_model.assert_called_once_with("1.7B")


# ── ModelConfig registration (drives /models/status, /models/download,
#    /models/{name}/unload -- all 400/404 without an entry here) ──────


def test_qwen_voice_design_has_a_model_config():
    config = get_model_config("qwen-voice-design-1.7B")
    assert config is not None
    assert config.engine == "qwen_voice_design"
    assert config.hf_repo_id == "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
    assert config.model_size == "1.7B"


def test_qwen_voice_design_is_single_size():
    assert engine_has_model_sizes("qwen_voice_design") is False


def test_qwen_voice_design_retries_runaway():
    """Same mlx-audio qwen3_tts decode loop as the Base engine (ac3fafd's
    EOS-miss runaway fix applies here too), and MLX-only, so this is
    unconditionally True -- no backend_type branch needed."""
    assert engine_retries_runaway("qwen_voice_design") is True


def test_qwen_voice_design_appears_exactly_once_in_tts_model_configs():
    matches = [c for c in get_tts_model_configs() if c.engine == "qwen_voice_design"]
    assert len(matches) == 1
