"""
Qwen3-TTS CustomVoice backend implementation (MLX).

Wraps the MLX conversion of the Qwen3-TTS-12Hz CustomVoice checkpoints
(``mlx-community/Qwen3-TTS-12Hz-{1.7B,0.6B}-CustomVoice-bf16``) -- an
Apple-Silicon-accelerated drop-in for the existing PyTorch/CPU-only
``QwenCustomVoiceBackend`` (backend/backends/qwen_custom_voice_backend.py).

This is a BEHAVIOR-TRANSPARENT backend swap, not a new engine: same
engine identifier (``qwen_custom_voice``), same API surface, same stored
``preset_voice_id`` values working unchanged. Unlike the separate
``qwen_voice_design`` engine (a brand-new engine with no prior contract),
any behavioral divergence between this MLX path and the existing
PyTorch path for the same engine is a bug, not a feature. The only
observable difference for any given request should be speed on Apple
Silicon.

Mirrors QwenVoiceDesignBackend's / MLXTTSBackend's structure for
load/generate/thread-routing: loads via ``mlx_audio.tts.utils.load``,
wrapped in ``model_load_progress``, with both load and generate routed
through the single dedicated ``run_on_mlx_thread`` executor rather than
``asyncio.to_thread`` -- see the comment above ``run_on_mlx_thread`` in
``backend/backends/base.py`` for why a second/shared executor would
reintroduce a real MLX thread-affinity crash (commits be13d3a / b98d006).

Preset speakers: identical between MLX and PyTorch (same 9 speakers).
The real MLX checkpoint's speaker-matching code lowercases internally
before lookup (``speaker.lower() in config.spk_id``), so the existing
capitalized ``preset_voice_id`` values already stored in production
(e.g. "Ryan", "Uncle_Fu") reach ``generate_custom_voice(speaker=...)``
completely unchanged -- no translation needed here.

Language handling: mirrors the PyTorch backend's existing
``LANGUAGE_CODE_TO_NAME.get(language, "auto")`` silent-fallback pattern
-- this is a known-imprecise pattern (a silent fallback to "auto" for any
unrecognized code) shared with mlx_backend.py and pytorch_backend.py,
being kept here ONLY for platform-parity with the existing PyTorch
qwen_custom_voice backend, not because it's ideal. A real hardening pass
across all of these is a separate follow-up. Do NOT change this to raise
on an unmapped language to match qwen_voice_design's pattern -- that
engine is greenfield with no fallback grounding; this one already has
real production data depending on the silent fallback behaving the same
regardless of which platform picked the backend.
"""

import logging

import numpy as np

from . import LANGUAGE_CODE_TO_NAME
from .base import (
    combine_voice_prompts as _combine_voice_prompts,
    is_model_cached,
    model_load_progress,
    run_on_mlx_thread as _run_on_mlx_thread,
)

logger = logging.getLogger(__name__)

# HuggingFace repo IDs per model size -- MLX bf16 conversions of the same
# checkpoints QWEN_CV_HF_REPOS (qwen_custom_voice_backend.py) points at
# for PyTorch. Both sizes confirmed to exist on Hugging Face.
QWEN_CV_MLX_HF_REPOS = {
    "1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
    "0.6B": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
}


class QwenCustomVoiceMLXBackend:
    """Qwen3-TTS CustomVoice backend (MLX) -- preset speakers with instruct control."""

    def __init__(self, model_size: str = "1.7B"):
        self.model = None
        self.model_size = model_size
        self._current_model_size: str | None = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        if model_size not in QWEN_CV_MLX_HF_REPOS:
            available = ", ".join(sorted(QWEN_CV_MLX_HF_REPOS))
            raise ValueError(
                f"Unsupported model size '{model_size}' for Qwen CustomVoice (MLX). "
                f"Available sizes: {available}."
            )
        return QWEN_CV_MLX_HF_REPOS[model_size]

    def _is_model_cached(self, model_size: str | None = None) -> bool:
        size = model_size or self.model_size
        return is_model_cached(
            self._get_model_path(size),
            weight_extensions=(".safetensors", ".bin", ".npz"),
        )

    async def load_model_async(self, model_size: str | None = None) -> None:
        """Lazy load the MLX CustomVoice model."""
        if model_size is None:
            model_size = self.model_size

        # Validate before touching threads/state so a bad size fails fast.
        self._get_model_path(model_size)

        if self.model is not None and self._current_model_size == model_size:
            return

        if self.model is not None and self._current_model_size != model_size:
            self.unload_model()

        await _run_on_mlx_thread(self._load_model_sync, model_size)

    # Alias for compatibility with the TTSBackend protocol
    load_model = load_model_async

    def _load_model_sync(self, model_size: str) -> None:
        """Synchronous model loading -- runs on the dedicated MLX thread."""
        model_path = self._get_model_path(model_size)
        model_name = f"qwen-custom-voice-{model_size}"
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(model_name, is_cached):
            from mlx_audio.tts.utils import load

            logger.info("Loading Qwen CustomVoice (MLX) model %s...", model_size)
            self.model = load(model_path)

        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("Qwen CustomVoice (MLX) model %s loaded successfully", model_size)

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._current_model_size = None
            logger.info("Qwen CustomVoice (MLX) model unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        """
        Create voice prompt for CustomVoice.

        CustomVoice doesn't use reference audio -- it uses preset
        speakers. When called for a cloned profile (fallback), uses the
        default speaker. For preset profiles, the voice_prompt dict is
        built by the profile service and bypasses this method entirely
        (create_voice_prompt_for_profile in backend/services/profiles.py).

        Mirrors QwenCustomVoiceBackend.create_voice_prompt exactly --
        this is not a stub. QWEN_CUSTOM_VOICES / QWEN_CV_DEFAULT_SPEAKER
        are imported lazily here (not at module top) so this MLX-only
        module doesn't pull in qwen_custom_voice_backend.py's
        module-level `import torch` at import time -- same pattern
        services/profiles.py already uses in _get_preset_voice_ids().
        """
        from .qwen_custom_voice_backend import QWEN_CV_DEFAULT_SPEAKER

        return {
            "voice_type": "preset",
            "preset_engine": "qwen_custom_voice",
            "preset_voice_id": QWEN_CV_DEFAULT_SPEAKER,
        }, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        return await _combine_voice_prompts(audio_paths, reference_texts)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: int | None = None,
        instruct: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio using Qwen CustomVoice (MLX).

        Args:
            text: Text to synthesize
            voice_prompt: Dict with preset_voice_id (speaker name).
                Passed straight through to generate_custom_voice(speaker=...)
                with zero transformation -- see module docstring.
            language: Language code (zh, en, ja, ko, etc.). An unmapped
                code silently falls back to "auto" (mirrors the PyTorch
                backend -- see module docstring).
            seed: Random seed for reproducibility
            instruct: Natural language instruction for style control
                      (e.g. "Speak in an angry tone", "Very happy")

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        await self.load_model_async(None)

        speaker = voice_prompt.get("preset_voice_id") or self._default_speaker()

        logger.info("Generating audio (CustomVoice, MLX) for text: %s", text)

        def _generate_sync():
            """Run synchronous generation on the dedicated MLX thread."""
            if seed is not None:
                import mlx.core as mx

                np.random.seed(seed)
                mx.random.seed(seed)

            # Known-imprecise pattern (see module docstring): silently
            # falls back to "auto" for any unrecognized code, mirroring
            # pytorch_backend.py / mlx_backend.py / the existing PyTorch
            # qwen_custom_voice backend -- kept for platform parity, not
            # because it's ideal.
            lang = LANGUAGE_CODE_TO_NAME.get(language, "auto")

            audio_chunks = []
            sample_rate = 24000

            for result in self.model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=lang,
                instruct=instruct,
            ):
                audio_chunks.append(np.array(result.audio))
                sample_rate = result.sample_rate

            if audio_chunks:
                audio = np.concatenate([np.asarray(chunk, dtype=np.float32) for chunk in audio_chunks])
            else:
                audio = np.array([], dtype=np.float32)

            return audio, sample_rate

        audio, sample_rate = await _run_on_mlx_thread(_generate_sync)

        return audio, sample_rate

    @staticmethod
    def _default_speaker() -> str:
        from .qwen_custom_voice_backend import QWEN_CV_DEFAULT_SPEAKER

        return QWEN_CV_DEFAULT_SPEAKER
