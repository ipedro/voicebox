"""Tests for Opus handling in :func:`backend.services.captures.create_capture`.

``create_capture`` had its own suffix allowlist that was missing ``.opus``,
so a ``.opus`` upload got relabeled to ``.wav`` on disk while the bytes
stayed raw Opus. That mislabeling then made the code take the
``suffix == ".wav"`` passthrough branch instead of the transcode branch,
handing whisper a file that isn't actually a WAV. These tests pin the fix:
``.opus`` uploads must go through the transcode-to-canonical-WAV branch.
"""

import io

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import config
from backend.database import Base
from backend.services import captures as captures_service

SR = 24000


def _tone(duration_s: float, amp: float = 0.3, freq: float = 220.0) -> np.ndarray:
    n = int(duration_s * SR)
    t = np.arange(n, dtype=np.float32) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _opus_bytes(duration_s: float = 1.0) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, _tone(duration_s), SR, format="OGG", subtype="OPUS")
    return buf.getvalue()


class _FakeWhisper:
    model_size = "base"

    def __init__(self):
        self.transcribed_paths: list[str] = []

    async def transcribe(self, audio_path: str, language, model_size) -> str:
        self.transcribed_paths.append(audio_path)
        return "hello world"


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_local()
    yield db
    db.close()


@pytest.fixture
def captures_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "get_captures_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def fake_whisper(monkeypatch):
    fake = _FakeWhisper()
    monkeypatch.setattr(captures_service, "get_whisper_model", lambda: fake)
    return fake


@pytest.mark.asyncio
async def test_opus_upload_is_transcoded_to_wav(test_db, captures_dir, fake_whisper):
    response = await captures_service.create_capture(
        audio_bytes=_opus_bytes(),
        filename="recording.opus",
        source="file",
        language=None,
        stt_model=None,
        db=test_db,
    )

    # NOTE: with the pre-fix suffix allowlist, `suffix` is forced to ".wav"
    # *before* `raw_path` is even computed -- so the buggy passthrough
    # branch and the correct transcode branch both end up naming the file
    # "{capture_id}.wav". The path string can't distinguish the two
    # branches; only the actual file *content* can. libsndfile detects
    # format from the header, not the extension, so sf.info().format
    # reports the true container: "OGG" if the passthrough branch left raw
    # Opus bytes sitting under a .wav name (the bug), "WAV" only if the
    # transcode branch actually ran sf.write(..., format="WAV") (the fix).
    assert response.audio_path.endswith(".wav")

    assert len(fake_whisper.transcribed_paths) == 1
    transcribed_path = fake_whisper.transcribed_paths[0]
    assert transcribed_path.endswith(".wav")
    assert sf.info(transcribed_path).format == "WAV"

    written, written_sr = sf.read(transcribed_path)
    assert written_sr == SR
    assert len(written) / written_sr == pytest.approx(1.0, abs=0.2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
