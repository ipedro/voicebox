"""Unit tests for the Qwen VoiceDesign MLX backend.

Qwen3-TTS VoiceDesign (mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16)
generates speech from a text description of a voice instead of cloning
from reference audio. These tests mock mlx_audio entirely -- no real
checkpoint is downloaded or loaded.
"""

import ast
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from backend.backends import LANGUAGE_CODE_TO_NAME
from backend.backends.qwen_voice_design_backend import (
    QWEN_VOICE_DESIGN_HF_REPOS,
    QwenVoiceDesignBackend,
)

BACKEND_SOURCE_PATH = Path(__file__).resolve().parents[1] / "backends" / "qwen_voice_design_backend.py"

SUPPORTED_LANGUAGES = ["zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"]
UNSUPPORTED_LANGUAGES = ["he", "ar", "tr"]


def _make_loaded_backend() -> tuple[QwenVoiceDesignBackend, MagicMock]:
    """Build a backend with a fake model already "loaded" (bypasses load_model_async)."""
    backend = QwenVoiceDesignBackend()
    mock_model = MagicMock()
    backend.model = mock_model
    backend._current_model_size = "1.7B"
    return backend, mock_model


def _fake_result(audio, sample_rate=24000):
    result = MagicMock()
    result.audio = audio
    result.sample_rate = sample_rate
    return result


# ── 1. Language map ─────────────────────────────────────────────────


@pytest.mark.parametrize("language", SUPPORTED_LANGUAGES)
@pytest.mark.asyncio
async def test_supported_language_reaches_generate_voice_design(language):
    backend, mock_model = _make_loaded_backend()
    mock_model.generate_voice_design.return_value = iter([_fake_result(np.zeros(10, dtype=np.float32))])

    await backend.generate("hello", {"design_prompt": "a warm voice"}, language=language)

    _args, kwargs = mock_model.generate_voice_design.call_args
    assert kwargs["language"] == LANGUAGE_CODE_TO_NAME[language]


@pytest.mark.parametrize("language", UNSUPPORTED_LANGUAGES)
@pytest.mark.asyncio
async def test_unsupported_language_raises_and_never_calls_model(language):
    backend, mock_model = _make_loaded_backend()

    with pytest.raises(ValueError, match=language):
        await backend.generate("hello", {"design_prompt": "a warm voice"}, language=language)

    mock_model.generate_voice_design.assert_not_called()


@pytest.mark.asyncio
async def test_unsupported_language_error_lists_supported_languages():
    backend, _mock_model = _make_loaded_backend()

    with pytest.raises(ValueError, match="he") as excinfo:
        await backend.generate("hello", {"design_prompt": "a warm voice"}, language="he")

    message = str(excinfo.value)
    for code in SUPPORTED_LANGUAGES:
        assert code in message


# ── 2. design_prompt / instruct composition ─────────────────────────


@pytest.mark.asyncio
async def test_design_prompt_alone_reaches_instruct_arg():
    backend, mock_model = _make_loaded_backend()
    mock_model.generate_voice_design.return_value = iter([_fake_result(np.zeros(10, dtype=np.float32))])

    await backend.generate("hello", {"design_prompt": "a warm adult female voice"}, language="en")

    _args, kwargs = mock_model.generate_voice_design.call_args
    assert kwargs["instruct"] == "a warm adult female voice"


@pytest.mark.asyncio
async def test_design_prompt_and_instruct_are_combined():
    backend, mock_model = _make_loaded_backend()
    mock_model.generate_voice_design.return_value = iter([_fake_result(np.zeros(10, dtype=np.float32))])

    await backend.generate(
        "hello",
        {"design_prompt": "a warm adult female voice"},
        language="en",
        instruct="speak angrily",
    )

    _args, kwargs = mock_model.generate_voice_design.call_args
    assert "a warm adult female voice" in kwargs["instruct"]
    assert "speak angrily" in kwargs["instruct"]


@pytest.mark.asyncio
async def test_missing_design_prompt_and_instruct_raises():
    backend, mock_model = _make_loaded_backend()

    with pytest.raises(ValueError, match="design_prompt"):
        await backend.generate("hello", {}, language="en")

    mock_model.generate_voice_design.assert_not_called()


# ── 3. model_size validation ─────────────────────────────────────────


def test_only_1_7b_model_size_exists():
    assert set(QWEN_VOICE_DESIGN_HF_REPOS) == {"1.7B"}


@pytest.mark.parametrize("bad_size", ["0.6B", "3B", "1B", "not-a-size"])
def test_get_model_path_rejects_unsupported_size(bad_size):
    backend = QwenVoiceDesignBackend()
    with pytest.raises(ValueError, match=bad_size.replace(".", r"\.")):
        backend._get_model_path(bad_size)


@pytest.mark.asyncio
async def test_load_model_rejects_unsupported_size():
    backend = QwenVoiceDesignBackend()
    with pytest.raises(ValueError, match=r"0\.6B"):
        await backend.load_model_async("0.6B")
    assert backend.model is None


@pytest.mark.asyncio
async def test_load_model_loads_1_7b_via_mlx_audio():
    backend = QwenVoiceDesignBackend()
    sentinel_model = MagicMock()

    with patch("mlx_audio.tts.utils.load", return_value=sentinel_model) as mock_load:
        await backend.load_model_async("1.7B")

    mock_load.assert_called_once_with(QWEN_VOICE_DESIGN_HF_REPOS["1.7B"])
    assert backend.model is sentinel_model
    assert backend.is_loaded()


# ── 5. Chunk concatenation ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multiple_chunks_are_concatenated_into_one_array():
    backend, mock_model = _make_loaded_backend()
    chunk_a = np.full(5, 0.1, dtype=np.float32)
    chunk_b = np.full(7, 0.2, dtype=np.float32)
    mock_model.generate_voice_design.return_value = iter(
        [_fake_result(chunk_a, sample_rate=22050), _fake_result(chunk_b, sample_rate=22050)]
    )

    audio, sample_rate = await backend.generate("hello", {"design_prompt": "a calm voice"}, language="en")

    assert isinstance(audio, np.ndarray)
    assert len(audio) == len(chunk_a) + len(chunk_b)
    np.testing.assert_allclose(audio, np.concatenate([chunk_a, chunk_b]))
    assert sample_rate == 22050


# ── 6. MLX-thread routing (not asyncio.to_thread) ─────────────────────


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
            pytest.fail("qwen_voice_design_backend.py must not call asyncio.to_thread()")


@pytest.mark.asyncio
async def test_load_model_routes_through_run_on_mlx_thread():
    backend = QwenVoiceDesignBackend()

    with patch(
        "backend.backends.qwen_voice_design_backend._run_on_mlx_thread",
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
    mock_model.generate_voice_design.return_value = iter([_fake_result(np.zeros(3, dtype=np.float32))])

    with patch(
        "backend.backends.qwen_voice_design_backend._run_on_mlx_thread",
        new_callable=AsyncMock,
    ) as mock_run:
        mock_run.return_value = (np.zeros(3, dtype=np.float32), 24000)
        await backend.generate("hello", {"design_prompt": "a calm voice"}, language="en")

    mock_run.assert_awaited_once()
