"""Tests for the voicebox.transcribe MCP tool's Opus handling.

``_transcribe_file`` used to call ``load_audio`` only to compute duration,
then hand the *original* file path straight to ``whisper.transcribe()``.
That works for WAV/FLAC/MP3/Vorbis (what the STT backend's miniaudio
decoder reads natively) but fails for anything else -- e.g. Opus -- because
the STT backend never gets a re-encoded WAV the way the ``/transcribe`` HTTP
route already provides. These tests pin the fix: ``_transcribe_file`` now
runs input through ``prepare_for_stt`` and passes *that* path to whisper.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.mcp_server import tools
from backend.services import transcribe as transcribe_service

SR = 24000


def _tone(duration_s: float, amp: float = 0.3, freq: float = 220.0) -> np.ndarray:
    n = int(duration_s * SR)
    t = np.arange(n, dtype=np.float32) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class _FakeWhisper:
    """Stand-in STT backend that records the path it's asked to transcribe."""

    model_size = "base"

    def __init__(self):
        self.transcribed_paths: list[str] = []

    def is_loaded(self) -> bool:
        return True

    def _is_model_cached(self, model_size: str) -> bool:
        return True

    async def transcribe(self, audio_path: str, language, model_size) -> str:
        self.transcribed_paths.append(audio_path)
        return "hello world"


@pytest.fixture
def fake_whisper(monkeypatch):
    fake = _FakeWhisper()
    monkeypatch.setattr(transcribe_service, "get_whisper_model", lambda: fake)
    return fake


@pytest.mark.asyncio
async def test_transcribe_file_reencodes_opus_before_whisper(tmp_path, fake_whisper):
    audio = _tone(1.0)
    opus_path = tmp_path / "clip.opus"
    sf.write(str(opus_path), audio, SR, format="OGG", subtype="OPUS")

    result = await tools._transcribe_file(opus_path, None, None)

    assert len(fake_whisper.transcribed_paths) == 1
    transcribed_path = fake_whisper.transcribed_paths[0]
    # The whole point: whisper must never see the raw .opus path directly --
    # its miniaudio decoder can't read it. It must get a re-encoded WAV.
    assert transcribed_path.endswith(".wav")
    assert transcribed_path != str(opus_path)
    assert result["text"] == "hello world"

    # prepare_for_stt's temp WAV must be cleaned up, not leaked.
    assert not Path(transcribed_path).exists()


@pytest.mark.asyncio
async def test_transcribe_file_passes_wav_through_unchanged(tmp_path, fake_whisper):
    audio = _tone(1.0)
    wav_path = tmp_path / "clip.wav"
    sf.write(str(wav_path), audio, SR)

    await tools._transcribe_file(wav_path, None, None)

    assert fake_whisper.transcribed_paths == [str(wav_path)]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
