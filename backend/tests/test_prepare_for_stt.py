"""
Unit tests for :func:`backend.utils.audio.prepare_for_stt`.

The STT backend (mlx_audio.stt / miniaudio) only decodes WAV/FLAC/MP3/Vorbis
natively -- not Opus. ``prepare_for_stt`` decodes any input librosa/soundfile
can read and, unless it's already a WAV, re-encodes the PCM to a temp WAV so
the STT backend always gets something it can open.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.utils.audio import prepare_for_stt

SR = 24000


def _tone(duration_s: float, amp: float = 0.3, freq: float = 220.0) -> np.ndarray:
    n = int(duration_s * SR)
    t = np.arange(n, dtype=np.float32) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.mark.asyncio
async def test_wav_input_passes_through_unchanged(tmp_path):
    audio = _tone(1.0)
    path = tmp_path / "clip.wav"
    sf.write(str(path), audio, SR)

    out_audio, out_sr, stt_path, is_temp = await prepare_for_stt(str(path))

    assert is_temp is False
    assert stt_path == str(path)
    assert out_sr == SR
    assert len(out_audio) == pytest.approx(len(audio), abs=SR * 0.05)


@pytest.mark.asyncio
async def test_opus_input_is_reencoded_to_temp_wav(tmp_path):
    audio = _tone(1.5)
    path = tmp_path / "clip.opus"
    sf.write(str(path), audio, SR, format="OGG", subtype="OPUS")

    out_audio, out_sr, stt_path, is_temp = await prepare_for_stt(str(path))

    try:
        assert is_temp is True
        assert stt_path != str(path)
        assert stt_path.endswith(".wav")
        assert Path(stt_path).exists()
        assert out_sr == SR
        # Opus is lossy so exact sample count won't match -- just check the
        # decoded length is in the right ballpark.
        assert len(out_audio) / out_sr == pytest.approx(1.5, abs=0.2)

        # The temp WAV on disk should itself decode back to ~1.5s of audio.
        written, written_sr = sf.read(stt_path)
        assert written_sr == SR
        assert len(written) / written_sr == pytest.approx(1.5, abs=0.2)
    finally:
        Path(stt_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
