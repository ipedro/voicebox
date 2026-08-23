"""Registry wiring tests for the qwen_custom_voice engine's platform branch.

qwen_custom_voice is an EXISTING engine that previously always resolved
to the PyTorch/CPU-only QwenCustomVoiceBackend regardless of platform.
This adds the same backend_type branching pattern the "qwen" (Base)
engine already uses in get_tts_backend_for_engine(): MLX on Apple
Silicon, PyTorch everywhere else -- same engine identifier, same API
surface, only the backend selection changes.

Note: at the time this file was written, "qwen"'s own MLX/PyTorch
platform branch in get_tts_backend_for_engine() had no existing test
coverage anywhere in the suite (confirmed via grep for MLXTTSBackend /
PyTorchTTSBackend / get_tts_backend_for_engine("qwen") across tests/).
This file writes that kind of test from scratch for qwen_custom_voice;
qwen's equivalent branch remains untested (informational -- not fixed
here, out of scope for this task).
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
from backend.backends.qwen_custom_voice_backend import QwenCustomVoiceBackend
from backend.backends.qwen_custom_voice_mlx_backend import QwenCustomVoiceMLXBackend


@pytest.fixture(autouse=True)
def _reset():
    reset_backends()
    yield
    reset_backends()


# ── 1. Platform branching in get_tts_backend_for_engine ────────────────


def test_get_tts_backend_for_engine_returns_mlx_backend_on_apple_silicon():
    with patch("backend.backends.get_backend_type", return_value="mlx"):
        backend = get_tts_backend_for_engine("qwen_custom_voice")
    assert isinstance(backend, QwenCustomVoiceMLXBackend)


def test_get_tts_backend_for_engine_returns_pytorch_backend_elsewhere():
    with patch("backend.backends.get_backend_type", return_value="pytorch"):
        backend = get_tts_backend_for_engine("qwen_custom_voice")
    assert isinstance(backend, QwenCustomVoiceBackend)


def test_get_tts_backend_for_engine_returns_same_instance_on_repeat_calls():
    with patch("backend.backends.get_backend_type", return_value="mlx"):
        backend1 = get_tts_backend_for_engine("qwen_custom_voice")
        backend2 = get_tts_backend_for_engine("qwen_custom_voice")
    assert backend1 is backend2


def test_qwen_custom_voice_is_registered_in_tts_engines():
    assert "qwen_custom_voice" in TTS_ENGINES


# ── 2. get_model_load_func regression check (unchanged, must still work) ─


@pytest.mark.asyncio
async def test_get_model_load_func_dispatches_through_named_qwen_custom_voice_branch():
    config = ModelConfig(
        model_name="qwen-custom-voice-1.7B",
        display_name="Qwen CustomVoice 1.7B",
        engine="qwen_custom_voice",
        hf_repo_id="mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
        model_size="1.7B",
    )

    reset_backends()
    mock_backend = QwenCustomVoiceMLXBackend()
    mock_backend.load_model = AsyncMock()

    with patch(
        "backend.backends.get_tts_backend_for_engine",
        return_value=mock_backend,
    ) as mock_get_backend:
        load_func = get_model_load_func(config)
        await load_func()

    mock_get_backend.assert_called_once_with("qwen_custom_voice")
    mock_backend.load_model.assert_called_once_with("1.7B")


# ── 3. _get_qwen_custom_voice_configs(): repo ids / retries_runaway ─────


def test_configs_use_mlx_repos_and_retries_runaway_on_mlx():
    with patch("backend.backends.get_backend_type", return_value="mlx"):
        configs = get_tts_model_configs()

    cv_configs = {c.model_size: c for c in configs if c.engine == "qwen_custom_voice"}
    assert set(cv_configs) == {"1.7B", "0.6B"}
    assert cv_configs["1.7B"].hf_repo_id == "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"
    assert cv_configs["0.6B"].hf_repo_id == "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"
    assert cv_configs["1.7B"].retries_runaway is True
    assert cv_configs["0.6B"].retries_runaway is True


def test_configs_use_pytorch_repos_and_no_retries_runaway_on_pytorch():
    with patch("backend.backends.get_backend_type", return_value="pytorch"):
        configs = get_tts_model_configs()

    cv_configs = {c.model_size: c for c in configs if c.engine == "qwen_custom_voice"}
    assert set(cv_configs) == {"1.7B", "0.6B"}
    assert cv_configs["1.7B"].hf_repo_id == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert cv_configs["0.6B"].hf_repo_id == "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    assert cv_configs["1.7B"].retries_runaway is False
    assert cv_configs["0.6B"].retries_runaway is False


def test_qwen_custom_voice_has_a_model_config():
    config = get_model_config("qwen-custom-voice-1.7B")
    assert config is not None
    assert config.engine == "qwen_custom_voice"
    assert config.model_size == "1.7B"


def test_qwen_custom_voice_supports_two_model_sizes():
    assert engine_has_model_sizes("qwen_custom_voice") is True


def test_engine_retries_runaway_reflects_current_platform():
    with patch("backend.backends.get_backend_type", return_value="mlx"):
        assert engine_retries_runaway("qwen_custom_voice") is True
    with patch("backend.backends.get_backend_type", return_value="pytorch"):
        assert engine_retries_runaway("qwen_custom_voice") is False
