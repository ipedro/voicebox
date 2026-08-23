"""
Audio processing utilities.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf


def normalize_audio(
    audio: np.ndarray,
    target_db: float = -20.0,
    peak_limit: float = 0.85,
) -> np.ndarray:
    """
    Normalize audio to target loudness with peak limiting.
    
    Args:
        audio: Input audio array
        target_db: Target RMS level in dB
        peak_limit: Peak limit (0.0-1.0)
        
    Returns:
        Normalized audio array
    """
    # Convert to float32
    audio = audio.astype(np.float32)
    
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio**2))
    
    # Calculate target RMS
    target_rms = 10**(target_db / 20)
    
    # Apply gain
    if rms > 0:
        gain = target_rms / rms
        audio = audio * gain
    
    # Peak limiting
    audio = np.clip(audio, -peak_limit, peak_limit)
    
    return audio


def load_audio(
    path: str,
    sample_rate: int = 24000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file with normalization.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate
        mono: Convert to mono
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio, sr = librosa.load(path, sr=sample_rate, mono=mono)
    return audio, sr


def save_audio(
    audio: np.ndarray,
    path: str,
    sample_rate: int = 24000,
) -> None:
    """
    Save audio file with atomic write and error handling.

    Writes to a temporary file first, then atomically renames to the
    target path.  This prevents corrupted/partial WAV files if the
    process is interrupted mid-write.

    Args:
        audio: Audio array
        path: Output path
        sample_rate: Sample rate

    Raises:
        OSError: If file cannot be written
    """
    from pathlib import Path
    import os

    temp_path = f"{path}.tmp"
    try:
        # Ensure parent directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file first (explicit format since .tmp
        # extension is not recognised by soundfile)
        sf.write(temp_path, audio, sample_rate, format='WAV')

        # Atomic rename to final path
        os.replace(temp_path, path)

    except Exception as e:
        # Clean up temp file on failure
        try:
            if Path(temp_path).exists():
                Path(temp_path).unlink()
        except Exception:
            pass  # Best effort cleanup

        raise OSError(f"Failed to save audio to {path}: {e}") from e


async def prepare_for_stt(path: str) -> tuple[np.ndarray, int, str, bool]:
    """
    Decode ``path`` and, unless it's already a WAV, re-encode it to a temp
    WAV the STT backend's decoder can read.

    The STT backend (mlx_audio.stt -> miniaudio) only decodes WAV/FLAC/MP3/
    Vorbis directly -- not every format librosa/soundfile can read (e.g.
    Opus). Decode once here via ``load_audio`` (librosa, with its
    audioread/ffmpeg fallback for exotic containers) and, if the source
    isn't a WAV, write that decoded PCM back out as WAV so the STT backend
    always gets something it can open natively.

    Args:
        path: Path to the source audio file.

    Returns:
        Tuple of (audio, sample_rate, stt_path, is_temp). ``stt_path`` is
        ``path`` unchanged when no re-encode was needed (``is_temp=False``),
        or a freshly created temp WAV (``is_temp=True``) that the caller
        owns and must remove (e.g. in a ``finally`` block).
    """
    # This is plain CPU-bound librosa/soundfile decoding, not an MLX call --
    # it must not go through run_on_mlx_thread/mlx_executor (backends/base.py).
    # Routing ordinary audio decoding through that single-worker executor
    # would serialize it behind MLX inference for no reason.
    audio, sr = await asyncio.to_thread(load_audio, path)

    if Path(path).suffix.lower() == ".wav":
        return audio, sr, path, False

    # A dedicated tempfile, not a path derived from `path` (e.g. `path +
    # ".stt.wav"`) -- this helper is also called with arbitrary caller-
    # supplied paths (MCP tool), which could point anywhere, including a
    # read-only directory. Writing beside the source file isn't safe there.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        stt_path = tmp.name

    try:
        await asyncio.to_thread(save_audio, audio, stt_path, sr)
    except Exception:
        # Caller only learns about (and cleans up) stt_path on a successful
        # return -- if the write itself fails, the empty temp file would
        # otherwise leak.
        Path(stt_path).unlink(missing_ok=True)
        raise
    return audio, sr, stt_path, True


def has_tts_runaway(
    audio: np.ndarray,
    sample_rate: int = 24000,
    frame_ms: int = 20,
    silence_threshold_db: float = -40.0,
    max_internal_silence_ms: int = 2000,
    noise_window_ms: int = 300,
    noise_zcr_ratio: float = 2.5,
    min_noise_run_ms: int = 1200,
) -> bool:
    """Detect a TTS model that missed EOS and kept generating.

    Two independent shapes both indicate this failure:
      1. speech -> long internal silence -> more output. The model
         restarted after a false stop. Leading and trailing silence do
         not count because they are not bounded by non-silent audio.
      2. speech -> a sustained run of noise-like audio with NO silence
         gap at all. The model overran straight into codec garbage: RMS
         energy stays high (it's not silence) but zero-crossing rate is
         far above this clip's own speech baseline, since noise lacks
         the phoneme-driven ZCR variation of real speech. Shape (1)'s
         energy-only check can't see this at all.
    """
    if _has_internal_silence_gap(audio, sample_rate, frame_ms, silence_threshold_db, max_internal_silence_ms):
        return True
    return _has_noise_tail(audio, sample_rate, noise_window_ms, silence_threshold_db, noise_zcr_ratio, min_noise_run_ms)


def _has_internal_silence_gap(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int,
    silence_threshold_db: float,
    max_internal_silence_ms: int,
) -> bool:
    frame_len = int(sample_rate * frame_ms / 1000)
    if frame_len == 0 or len(audio) < frame_len:
        return False

    n_frames = len(audio) // frame_len
    threshold_linear = 10 ** (silence_threshold_db / 20)
    max_silence_frames = int(max_internal_silence_ms / frame_ms)
    seen_speech = False
    consecutive_silence = 0

    for i in range(n_frames):
        frame = audio[i * frame_len : (i + 1) * frame_len]
        is_speech = np.sqrt(np.mean(frame**2)) >= threshold_linear
        if is_speech:
            if seen_speech and consecutive_silence >= max_silence_frames:
                return True
            seen_speech = True
            consecutive_silence = 0
        elif seen_speech:
            consecutive_silence += 1

    return False


def _has_noise_tail(
    audio: np.ndarray,
    sample_rate: int,
    window_ms: int,
    silence_threshold_db: float,
    noise_zcr_ratio: float,
    min_run_ms: int,
) -> bool:
    """Flag a sustained run of anomalously high zero-crossing-rate audio.

    Zero-crossing rate is measured over windows wide enough (300ms) to
    average out the frame-to-frame jitter that makes ZCR noisy at 20ms
    resolution — at that short a grain, codec-noise ZCR spikes for only
    2-3 frames at a time and never reads as "sustained" even though the
    underlying corruption is continuous.
    """
    win_len = int(sample_rate * window_ms / 1000)
    if win_len == 0 or len(audio) < win_len:
        return False

    n_windows = len(audio) // win_len
    threshold_linear = 10 ** (silence_threshold_db / 20)
    min_run_windows = max(1, int(min_run_ms / window_ms))

    # Baseline ZCR is established from this clip's own first second or so
    # of real speech, since corrupted output only ever appears after some
    # genuine speech has already been generated.
    baseline_zcrs: list[float] = []
    baseline_zcr: Optional[float] = None
    consecutive_noise = 0

    for i in range(n_windows):
        window = audio[i * win_len : (i + 1) * win_len]
        rms = np.sqrt(np.mean(window**2))
        if rms < threshold_linear:
            consecutive_noise = 0
            continue

        zcr = np.mean(np.abs(np.diff(np.sign(window)))) / 2

        if baseline_zcr is None:
            baseline_zcrs.append(zcr)
            if len(baseline_zcrs) >= min_run_windows:
                baseline_zcr = float(np.median(baseline_zcrs))
            continue

        if zcr >= max(baseline_zcr * noise_zcr_ratio, 0.2):
            consecutive_noise += 1
            if consecutive_noise >= min_run_windows:
                return True
        else:
            consecutive_noise = 0

    return False


def trim_tts_output(
    audio: np.ndarray,
    sample_rate: int = 24000,
    frame_ms: int = 20,
    silence_threshold_db: float = -40.0,
    min_silence_ms: int = 200,
    max_internal_silence_ms: int = 1000,
    fade_ms: int = 30,
) -> np.ndarray:
    """
    Trim trailing silence and post-silence hallucination from TTS output.

    Chatterbox sometimes produces ``[speech][silence][hallucinated noise]``.
    This detects internal silence gaps longer than *max_internal_silence_ms*
    and cuts the audio at that boundary, then trims trailing silence and
    applies a short cosine fade-out.

    Args:
        audio: Input audio array (mono float32)
        sample_rate: Sample rate in Hz
        frame_ms: Frame size for RMS energy calculation
        silence_threshold_db: dB threshold below which a frame is silence
        min_silence_ms: Minimum trailing silence to keep
        max_internal_silence_ms: Cut after any silence gap longer than this
        fade_ms: Cosine fade-out duration in ms

    Returns:
        Trimmed audio array
    """
    frame_len = int(sample_rate * frame_ms / 1000)
    if frame_len == 0 or len(audio) < frame_len:
        return audio

    n_frames = len(audio) // frame_len
    threshold_linear = 10 ** (silence_threshold_db / 20)

    # Compute per-frame RMS
    rms = np.array(
        [
            np.sqrt(np.mean(audio[i * frame_len : (i + 1) * frame_len] ** 2))
            for i in range(n_frames)
        ]
    )
    is_speech = rms >= threshold_linear

    # Find first speech frame
    first_speech = 0
    for i, s in enumerate(is_speech):
        if s:
            first_speech = max(0, i - 1)  # keep 1 frame padding
            break

    # Walk forward from first speech; cut at long internal silence gaps
    max_silence_frames = int(max_internal_silence_ms / frame_ms)
    consecutive_silence = 0
    cut_frame = n_frames

    for i in range(first_speech, n_frames):
        if is_speech[i]:
            consecutive_silence = 0
        else:
            consecutive_silence += 1
            if consecutive_silence >= max_silence_frames:
                cut_frame = i - consecutive_silence + 1
                break

    # Trim trailing silence from the cut point
    min_silence_frames = int(min_silence_ms / frame_ms)
    end_frame = cut_frame
    while end_frame > first_speech and not is_speech[end_frame - 1]:
        end_frame -= 1
    # Keep a short tail
    end_frame = min(end_frame + min_silence_frames, cut_frame)

    # Convert frames back to samples
    start_sample = first_speech * frame_len
    end_sample = min(end_frame * frame_len, len(audio))

    trimmed = audio[start_sample:end_sample].copy()

    # Cosine fade-out
    fade_samples = int(sample_rate * fade_ms / 1000)
    if fade_samples > 0 and len(trimmed) > fade_samples:
        fade = np.cos(np.linspace(0, np.pi / 2, fade_samples)) ** 2
        trimmed[-fade_samples:] *= fade

    return trimmed


def preprocess_reference_audio(
    audio: np.ndarray,
    sample_rate: int,
    peak_target: float = 0.95,
    trim_top_db: float = 40.0,
    edge_padding_ms: int = 100,
) -> np.ndarray:
    """
    Clean up a reference-audio sample before validation/storage.

    Removes DC offset, trims leading/trailing silence, and caps the peak so a
    slightly-hot recording doesn't get rejected downstream as "clipping". The
    goal is to accept reasonable real-world recordings — not to repair badly
    distorted ones. True clipping artifacts inside the waveform can't be
    recovered by peak scaling and will still sound bad.

    Args:
        audio: Mono audio array.
        sample_rate: Sample rate of ``audio`` in Hz.
        peak_target: Peak amplitude cap in [0, 1]. Applied only if the input
            peak exceeds this value.
        trim_top_db: Silence threshold for edge trimming, in dB below peak.
            40 dB sits below normal speech dynamic range (≈30 dB) so soft
            trailing syllables are preserved, while still catching obvious
            leading/trailing silence. Lower values are more aggressive;
            librosa's own default is 60.
        edge_padding_ms: Milliseconds of padding to add back at each edge
            *only if* trimming shortened the waveform, so TTS engines have a
            brief silence to anchor on without ever making the output longer
            than the input.

    Returns:
        Preprocessed audio array (float32).
    """
    audio = audio.astype(np.float32, copy=False)

    if audio.size == 0:
        return audio

    audio = audio - float(np.mean(audio))

    trimmed, _ = librosa.effects.trim(audio, top_db=trim_top_db)
    if 0 < trimmed.size < audio.size:
        pad_each = int(sample_rate * edge_padding_ms / 1000)
        # Never pad past the original length — for near-max-duration uploads
        # an unconditional pad would push them over the 30 s ceiling and
        # trigger a spurious "too long" rejection.
        headroom = (audio.size - trimmed.size) // 2
        pad = min(pad_each, max(headroom, 0))
        if pad > 0:
            trimmed = np.pad(trimmed, (pad, pad), mode="constant")
        audio = trimmed

    peak = float(np.abs(audio).max())
    if peak > peak_target and peak > 0:
        audio = audio * (peak_target / peak)

    return audio


def validate_reference_audio(
    audio_path: str,
    min_duration: float = 2.0,
    max_duration: float = 30.0,
    min_rms: float = 0.01,
) -> Tuple[bool, Optional[str]]:
    """
    Validate reference audio for voice cloning.

    Args:
        audio_path: Path to audio file
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        min_rms: Minimum RMS level

    Returns:
        Tuple of (is_valid, error_message)
    """
    result = validate_and_load_reference_audio(
        audio_path, min_duration, max_duration, min_rms
    )
    return (result[0], result[1])


def validate_and_load_reference_audio(
    audio_path: str,
    min_duration: float = 2.0,
    max_duration: float = 30.0,
    min_rms: float = 0.01,
) -> Tuple[bool, Optional[str], Optional[np.ndarray], Optional[int]]:
    """
    Validate and load reference audio in a single pass.

    Applies :func:`preprocess_reference_audio` before checks so that
    slightly-hot recordings aren't rejected as clipping. Duration and RMS
    checks run on the preprocessed waveform.

    Returns:
        Tuple of (is_valid, error_message, audio_array, sample_rate)
    """
    try:
        audio, sr = load_audio(audio_path)
        audio = preprocess_reference_audio(audio, sr)
        duration = len(audio) / sr

        if duration < min_duration:
            return False, f"Audio too short (minimum {min_duration} seconds)", None, None
        if duration > max_duration:
            return False, f"Audio too long (maximum {max_duration} seconds)", None, None

        rms = np.sqrt(np.mean(audio**2))
        if rms < min_rms:
            return False, "Audio is too quiet or silent", None, None

        return True, None, audio, sr
    except Exception as e:
        return False, f"Error validating audio: {str(e)}", None, None
