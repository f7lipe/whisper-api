import os
import tempfile
import time
from pathlib import Path

from app.services.base import BaseWhisperService
from app.schemas.transcription import TranscriptionResponse, Segment
from app.config import settings

# Map short model names → mlx-community HuggingFace repo IDs
# Full list: https://huggingface.co/collections/mlx-community/whisper-663256f9964fbb1177db93dc
_MLX_MODEL_MAP: dict[str, str] = {
    "tiny":              "mlx-community/whisper-tiny-mlx",
    "tiny.en":           "mlx-community/whisper-tiny.en-mlx",
    "base":              "mlx-community/whisper-base-mlx",
    "small":             "mlx-community/whisper-small-mlx",
    "medium":            "mlx-community/whisper-medium-mlx",
    "large":             "mlx-community/whisper-large-mlx",
    "large-v2":          "mlx-community/whisper-large-v2-mlx",
    "large-v3":          "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo":    "mlx-community/whisper-large-v3-turbo",
    "distil-large-v3":   "mlx-community/distil-whisper-large-v3",
}


def _resolve_repo(model_name: str) -> str:
    """Accept short names (e.g. 'large-v3') or full HF repo IDs."""
    return _MLX_MODEL_MAP.get(model_name, model_name)


class MlxWhisperService(BaseWhisperService):
    """Whisper backend powered by mlx-whisper (Apple MLX, MIT).

    Runs natively on Apple Silicon via the MPS/Metal backend — no conversion
    to CTranslate2 format needed. Models are downloaded from mlx-community on
    the Hugging Face Hub and cached locally on first use.

    mlx_whisper.transcribe() returns the same dict format as openai-whisper,
    so segment extraction is identical.
    """

    def __init__(self) -> None:
        self._repo: str | None = None      # resolved HF repo / local path
        self._model_name: str | None = None  # as provided by the caller

    def load(self, model_name: str) -> None:
        import mlx_whisper  # noqa: F401 — import eagerly to surface InstallError early

        repo = _resolve_repo(model_name)
        print(f"[mlx-whisper] Warming up '{model_name}' ({repo})…")
        # mlx-whisper is lazy: the model downloads/loads on the first transcribe()
        # call.  We store the repo so transcribe() can pass it in.
        self._repo = repo
        self._model_name = model_name
        print("[mlx-whisper] Ready (model will be downloaded on first transcription if needed).")

    def unload(self) -> None:
        self._repo = None
        self._model_name = None

    @property
    def is_ready(self) -> bool:
        return self._repo is not None

    def transcribe(self, audio_bytes: bytes, filename: str) -> TranscriptionResponse:
        if not self._repo:
            raise RuntimeError("Model not loaded")

        import mlx_whisper

        suffix = Path(filename).suffix or ".webm"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            start = time.perf_counter()
            result = mlx_whisper.transcribe(
                tmp_path,
                path_or_hf_repo=self._repo,
                language=settings.whisper_language,
                task="transcribe",
                word_timestamps=False,
            )
            duration = time.perf_counter() - start

            raw_segments = result.get("segments", [])
            segments = [
                Segment(
                    id=s["id"],
                    start=s["start"],
                    end=s["end"],
                    text=s["text"].strip(),
                )
                for s in raw_segments
            ]

            return TranscriptionResponse(
                text=result["text"].strip(),
                language=result.get("language", settings.whisper_language),
                duration=round(duration, 3),
                segments=segments or None,
            )
        finally:
            os.unlink(tmp_path)


mlx_whisper_service = MlxWhisperService()
