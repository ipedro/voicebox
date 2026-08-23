"""Unit tests for the Qwen CustomVoice MLX backend.

qwen_custom_voice is an EXISTING engine that already works today via a
PyTorch/CPU-only backend (backend/backends/qwen_custom_voice_backend.py,
class QwenCustomVoiceBackend). This is a behavior-transparent backend
swap for Apple Silicon -- same engine identifier, same API surface, same
stored preset_voice_id values working unchanged. Any behavioral
divergence between the MLX and PyTorch code paths for the same engine is
a bug, not a feature (unlike the separate qwen_voice_design engine, which
was greenfield and could raise on an unmapped language with no fallback
grounding).

These tests mock mlx_audio entirely -- no real checkpoint is downloaded
or loaded.
"""

import ast
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from backend.backends.qwen_custom_voice_backend import QWEN_CV_DEFAULT_SPEAKER
from backend.backends.qwen_custom_voice_mlx_backend import (
    QWEN_CV_MLX_HF_REPOS,
    QwenCustomVoiceMLXBackend,
)

BACKEND_SOURCE_PATH = Path(__file__).resolve().parents[1] / "backends" / "qwen_custom_voice_mlx_backend.py"


def _make_loaded_backend() -> tuple[QwenCustomVoiceMLXBackend, MagicMock]:
    """Build a backend with a fake model already "loaded" (bypasses load_model_async)."""
    backend = QwenCustomVoiceMLXBackend()
    mock_model = MagicMock()
    backend.model = mock_model
    backend._current_model_size = "1.7B"
    return backend, mock_model


def _fake_result(audio, sample_rate=24000):
    result = MagicMock()
    result.audio = audio
    result.sample_rate = sample_rate
    return result


# ── 1. Speaker pass-through (zero transformation) ──────────────────────


@pytest.mark.parametrize("speaker_id", ["Ryan", "Uncle_Fu"])
@pytest.mark.asyncio
async def test_preset_speaker_reaches_generate_custom_voice_unchanged(speaker_id):
    """Preset speakers are stored capitalized (e.g. "Ryan", "Uncle_Fu") in
    the production DB. The real MLX checkpoint's speaker-matching code
    lowercases internally before lookup, so these must reach
    generate_custom_voice() completely unchanged -- no case
    transformation, no remapping in our code.
    """
    backend, mock_model = _make_loaded_backend()
    mock_model.generate_custom_voice.return_value = iter([_fake_result(np.zeros(10, dtype=np.float32))])

    await backend.generate("hello", {"preset_voice_id": speaker_id}, language="en")

    _args, kwargs = mock_model.generate_custom_voice.call_args
    assert kwargs["speaker"] == speaker_id


# ── 2. Default speaker ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_preset_voice_id_uses_default_speaker():
    backend, mock_model = _make_loaded_backend()
    mock_model.generate_custom_voice.return_value = iter([_fake_result(np.zeros(10, dtype=np.float32))])

    await backend.generate("hello", {}, language="en")

    _args, kwargs = mock_model.generate_custom_voice.call_args
    assert kwargs["speaker"] == QWEN_CV_DEFAULT_SPEAKER == "Ryan"


# ── 3. Language: mirror PyTorch's silent .get(language, "auto") fallback ─
#
# This is the OPPOSITE assertion from qwen_voice_design's test suite,
# deliberately. qwen_voice_design has no reference audio or preset
# speaker to fall back on, so an unmapped language raises there. This
# engine (qwen_custom_voice) already ships in production on PyTorch with
# a silent fallback to "auto" for unmapped codes (see
# qwen_custom_voice_backend.py's `LANGUAGE_CODE_TO_NAME.get(language,
# "auto")`) -- the MLX backend must match that, not VoiceDesign's raise,
# or the exact same request would behave differently depending on which
# platform happened to pick the backend. Do NOT "fix" this to raise to
# match VoiceDesign.


@pytest.mark.asyncio
async def test_unmapped_language_falls_back_to_auto_without_raising():
    backend, mock_model = _make_loaded_backend()
    mock_model.generate_custom_voice.return_value = iter([_fake_result(np.zeros(10, dtype=np.float32))])

    await backend.generate("hello", {"preset_voice_id": "Ryan"}, language="he")

    _args, kwargs = mock_model.generate_custom_voice.call_args
    assert kwargs["language"] == "auto"


@pytest.mark.asyncio
async def test_mapped_language_reaches_generate_custom_voice():
    backend, mock_model = _make_loaded_backend()
    mock_model.generate_custom_voice.return_value = iter([_fake_result(np.zeros(10, dtype=np.float32))])

    await backend.generate("hello", {"preset_voice_id": "Ryan"}, language="pt")

    _args, kwargs = mock_model.generate_custom_voice.call_args
    assert kwargs["language"] == "portuguese"


# ── 3b. instruct kwarg omission for falsy values (parity with PyTorch's
#        `if instruct: kwargs["instruct"] = instruct` guard) ──────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("falsy_instruct", [None, ""])
async def test_falsy_instruct_is_omitted_not_passed_empty(falsy_instruct):
    """qwen_custom_voice_backend.py (PyTorch) only sets kwargs["instruct"]
    when instruct is truthy -- an explicit "" is never sent. Mirror that
    exactly rather than passing instruct=None/instruct="" unconditionally,
    so the two backends send an identical kwarg set for the same call.
    """
    backend, mock_model = _make_loaded_backend()
    mock_model.generate_custom_voice.return_value = iter([_fake_result(np.zeros(10, dtype=np.float32))])

    await backend.generate("hello", {"preset_voice_id": "Ryan"}, language="en", instruct=falsy_instruct)

    _args, kwargs = mock_model.generate_custom_voice.call_args
    assert "instruct" not in kwargs


@pytest.mark.asyncio
async def test_truthy_instruct_is_passed_through():
    backend, mock_model = _make_loaded_backend()
    mock_model.generate_custom_voice.return_value = iter([_fake_result(np.zeros(10, dtype=np.float32))])

    await backend.generate("hello", {"preset_voice_id": "Ryan"}, language="en", instruct="speak angrily")

    _args, kwargs = mock_model.generate_custom_voice.call_args
    assert kwargs["instruct"] == "speak angrily"


# ── 4. create_voice_prompt() returns real dict, not NotImplementedError ──


@pytest.mark.asyncio
async def test_create_voice_prompt_returns_real_default_preset_dict():
    backend = QwenCustomVoiceMLXBackend()

    voice_prompt, was_cached = await backend.create_voice_prompt("unused/path.wav", "unused text")

    assert voice_prompt == {
        "voice_type": "preset",
        "preset_engine": "qwen_custom_voice",
        "preset_voice_id": "Ryan",
    }
    assert was_cached is False


# ── 5. model_size ────────────────────────────────────────────────────────


def test_both_model_sizes_exist():
    assert set(QWEN_CV_MLX_HF_REPOS) == {"1.7B", "0.6B"}


def test_1_7b_resolves_to_correct_mlx_repo():
    backend = QwenCustomVoiceMLXBackend()
    assert backend._get_model_path("1.7B") == "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"


def test_0_6b_resolves_to_correct_mlx_repo():
    backend = QwenCustomVoiceMLXBackend()
    assert backend._get_model_path("0.6B") == "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"


@pytest.mark.parametrize("bad_size", ["3B", "1B", "not-a-size"])
def test_get_model_path_rejects_unsupported_size(bad_size):
    backend = QwenCustomVoiceMLXBackend()
    with pytest.raises(ValueError, match=bad_size.replace(".", r"\.")):
        backend._get_model_path(bad_size)


@pytest.mark.asyncio
async def test_load_model_rejects_unsupported_size():
    backend = QwenCustomVoiceMLXBackend()
    with pytest.raises(ValueError, match=r"3B"):
        await backend.load_model_async("3B")
    assert backend.model is None


@pytest.mark.asyncio
async def test_load_model_loads_1_7b_via_mlx_audio():
    backend = QwenCustomVoiceMLXBackend()
    sentinel_model = MagicMock()

    with patch("mlx_audio.tts.utils.load", return_value=sentinel_model) as mock_load:
        await backend.load_model_async("1.7B")

    mock_load.assert_called_once_with(QWEN_CV_MLX_HF_REPOS["1.7B"])
    assert backend.model is sentinel_model
    assert backend.is_loaded()


def test_default_model_size_is_1_7b():
    backend = QwenCustomVoiceMLXBackend()
    assert backend.model_size == "1.7B"


# ── 6. Chunk concatenation ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multiple_chunks_are_concatenated_into_one_array():
    backend, mock_model = _make_loaded_backend()
    chunk_a = np.full(5, 0.1, dtype=np.float32)
    chunk_b = np.full(7, 0.2, dtype=np.float32)
    mock_model.generate_custom_voice.return_value = iter(
        [_fake_result(chunk_a, sample_rate=22050), _fake_result(chunk_b, sample_rate=22050)]
    )

    audio, sample_rate = await backend.generate("hello", {"preset_voice_id": "Ryan"}, language="en")

    assert isinstance(audio, np.ndarray)
    assert len(audio) == len(chunk_a) + len(chunk_b)
    np.testing.assert_allclose(audio, np.concatenate([chunk_a, chunk_b]))
    assert sample_rate == 22050


# ── 7. MLX-thread routing (not asyncio.to_thread) ─────────────────────


def test_module_never_imports_asyncio_to_thread():
    """Static check: this file must never reach for asyncio.to_thread.

    MLX binds its Metal stream to whichever OS thread first touches it;
    routing MLX-touching calls through the shared asyncio.to_thread() pool
    can hand a later call to a fresh thread and crash the process (see
    commits be13d3a / b98d006). Every MLX call here must go through the
    dedicated run_on_mlx_thread executor instead.
    """
    source = BACKEND_SOURCE_PATH.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name != "asyncio" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "asyncio"
        if isinstance(node, ast.Attribute) and node.attr == "to_thread":
            pytest.fail("qwen_custom_voice_mlx_backend.py must not call asyncio.to_thread()")


def test_module_does_not_import_torch_at_module_level():
    """The QWEN_CUSTOM_VOICES/QWEN_CV_DEFAULT_SPEAKER constants must be
    imported lazily (inside create_voice_prompt), not at module top,
    so this MLX-only file doesn't pull in qwen_custom_voice_backend.py's
    module-level `import torch` at import time -- same pattern
    services/profiles.py already uses in _get_preset_voice_ids().
    """
    source = BACKEND_SOURCE_PATH.read_text()
    tree = ast.parse(source)
    for node in tree.body:  # only module-level statements
        if isinstance(node, ast.Import):
            assert all(alias.name != "torch" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "torch"
            assert node.module != "backend.backends.qwen_custom_voice_backend"


@pytest.mark.asyncio
async def test_load_model_routes_through_run_on_mlx_thread():
    backend = QwenCustomVoiceMLXBackend()

    with patch(
        "backend.backends.qwen_custom_voice_mlx_backend._run_on_mlx_thread",
        new_callable=AsyncMock,
    ) as mock_run:
        mock_run.return_value = None
        await backend.load_model_async("1.7B")

    mock_run.assert_awaited_once()
    args, _kwargs = mock_run.call_args
    assert args[0] == backend._load_model_sync
    assert args[1] == "1.7B"


@pytest.mark.asyncio
async def test_generate_routes_through_run_on_mlx_thread():
    backend, mock_model = _make_loaded_backend()
    mock_model.generate_custom_voice.return_value = iter([_fake_result(np.zeros(3, dtype=np.float32))])

    with patch(
        "backend.backends.qwen_custom_voice_mlx_backend._run_on_mlx_thread",
        new_callable=AsyncMock,
    ) as mock_run:
        mock_run.return_value = (np.zeros(3, dtype=np.float32), 24000)
        await backend.generate("hello", {"preset_voice_id": "Ryan"}, language="en")

    mock_run.assert_awaited_once()
