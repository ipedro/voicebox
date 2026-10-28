"""
Qwen3-TTS VoiceDesign backend implementation (MLX).

Wraps ``mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16``, which
generates speech from a natural-language TEXT DESCRIPTION of a voice
("a warm adult female voice, clear articulation, moderate pace") instead
of cloning from reference audio or picking a preset speaker. It only
ships at one size -- 1.7B.

Mirrors MLXTTSBackend's structure (``backend/backends/mlx_backend.py``):
loads via ``mlx_audio.tts.utils.load`` and routes every MLX-touching call
through the single dedicated ``run_on_mlx_thread`` executor rather than
``asyncio.to_thread`` -- see the comment above ``run_on_mlx_thread`` in
``backend/backends/base.py`` for why a second/shared executor would
reintroduce a real MLX thread-affinity crash (commits be13d3a / b98d006).

Languages supported: zh, en, ja, ko, de, fr, ru, pt, es, it (the same set
LANGUAGE_CODE_TO_NAME already covers). Unlike the cloning/preset backends,
an unmapped language here is NOT silently coerced to "auto" -- VoiceDesign
has no reference audio or preset speaker to fall back on, so skipping the
model's language-conditioning branch produces badly-accented, wrong-
sounding output with no error at all. We raise instead.
"""

import logging

import numpy as np

from . import LANGUAGE_CODE_TO_NAME
from .base import (
    is_model_cached,
    model_load_progress,
    run_on_mlx_thread as _run_on_mlx_thread,
)

logger = logging.getLogger(__name__)

# Only 1.7B exists for VoiceDesign -- no other size is published.
QWEN_VOICE_DESIGN_HF_REPOS = {
    "1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
}


class QwenVoiceDesignBackend:
    """Qwen3-TTS VoiceDesign backend -- voice defined by a text description."""

    def __init__(self, model_size: str = "1.7B"):
        self.model = None
        self.model_size = model_size
        self._current_model_size: str | None = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        if model_size not in QWEN_VOICE_DESIGN_HF_REPOS:
            available = ", ".join(sorted(QWEN_VOICE_DESIGN_HF_REPOS))
            raise ValueError(
                f"Unsupported model size '{model_size}' for Qwen VoiceDesign. "
                f"Only {available} is available."
            )
        return QWEN_VOICE_DESIGN_HF_REPOS[model_size]

    def _is_model_cached(self, model_size: str | None = None) -> bool:
        size = model_size or self.model_size
        return is_model_cached(
            self._get_model_path(size),
            weight_extensions=(".safetensors", ".bin", ".npz"),
        )

    async def load_model_async(self, model_size: str | None = None) -> None:
        """Lazy load the MLX VoiceDesign model."""
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
        model_name = f"qwen-voice-design-{model_size}"
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(model_name, is_cached):
            from mlx_audio.tts.utils import load

            logger.info("Loading Qwen VoiceDesign model %s...", model_size)
            self.model = load(model_path)

        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("Qwen VoiceDesign model %s loaded successfully", model_size)

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._current_model_size = None
            logger.info("Qwen VoiceDesign model unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        """
        VoiceDesign has no reference-audio cloning path. Profiles with
        voice_type="designed" never reach this method -- ``create_voice_prompt_for_profile``
        (backend/services/profiles.py) short-circuits to a
        ``{"voice_type": "designed", "design_prompt": ...}`` dict instead.
        Present only to satisfy the TTSBackend protocol shape.
        """
        raise NotImplementedError(
            "Qwen VoiceDesign does not support reference-audio cloning. "
            "Use a profile with voice_type='designed' and a design_prompt."
        )

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        raise NotImplementedError("Qwen VoiceDesign does not support combining reference audio.")

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: int | None = None,
        instruct: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from a text description of the target voice.

        Args:
            text: Text to synthesize
            voice_prompt: Dict with "design_prompt" -- the base voice
                description (see create_voice_prompt_for_profile).
            language: Language code, must be one of LANGUAGE_CODE_TO_NAME's
                keys -- unlike other backends, an unmapped code raises
                instead of silently falling back to "auto" (see module
                docstring).
            seed: Random seed for reproducibility
            instruct: Optional live per-call delivery-style instruction
                (e.g. "speak angrily"), layered on TOP of design_prompt.
                design_prompt defines the base voice; the model's own
                `instruct=` parameter takes a single string, so when both
                are present we concatenate them ("<design_prompt> <instruct>").
                When only one is present, that one is used alone.

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if language not in LANGUAGE_CODE_TO_NAME:
            supported = ", ".join(sorted(LANGUAGE_CODE_TO_NAME))
            raise ValueError(
                f"Unsupported language '{language}' for Qwen VoiceDesign. "
                f"Supported languages: {supported}."
            )
        lang = LANGUAGE_CODE_TO_NAME[language]

        design_prompt = (voice_prompt or {}).get("design_prompt")
        if design_prompt and instruct:
            composed_instruct = f"{design_prompt} {instruct}"
        elif design_prompt:
            composed_instruct = design_prompt
        elif instruct:
            composed_instruct = instruct
        else:
            raise ValueError(
                "Qwen VoiceDesign requires a design_prompt (voice_prompt['design_prompt']) "
                "or an instruct string describing the target voice."
            )

        await self.load_model_async(None)

        logger.info("Generating audio (VoiceDesign) for text: %s", text)

        def _generate_sync():
            """Run synchronous generation on the dedicated MLX thread."""
            if seed is not None:
                import mlx.core as mx

                np.random.seed(seed)
                mx.random.seed(seed)

            audio_chunks = []
            sample_rate = 24000

            for result in self.model.generate_voice_design(text=text, instruct=composed_instruct, language=lang):
                audio_chunks.append(np.array(result.audio))
                sample_rate = result.sample_rate

            if audio_chunks:
                audio = np.concatenate([np.asarray(chunk, dtype=np.float32) for chunk in audio_chunks])
            else:
                audio = np.array([], dtype=np.float32)

            return audio, sample_rate

        audio, sample_rate = await _run_on_mlx_thread(_generate_sync)

        return audio, sample_rate
