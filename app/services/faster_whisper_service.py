import os
import tempfile
import time
from pathlib import Path

from app.services.base import BaseWhisperService
from app.schemas.transcription import TranscriptionResponse, Segment
from app.config import settings


class FasterWhisperService(BaseWhisperService):
    """Whisper backend powered by SYSTRAN/faster-whisper (CTranslate2, MIT).

    Performance strategy (CPU / Apple Silicon):
    - BatchedInferencePipeline  → up to 2× faster than sequential on CPU
    - vad_filter                → skips silence, less audio to process
    - int8 compute_type         → smallest RAM footprint on CPU
    - cpu_threads               → saturates all cores (Apple M-series)
    - num_workers               → overlaps model I/O between requests
    """

    def __init__(self) -> None:
        self._model = None
        self._batched_model = None
        self._device: str = "cpu"
        self._compute_type: str = "int8"

    def load(self, model_name: str) -> None:
        import torch
        from faster_whisper import WhisperModel, BatchedInferencePipeline

        if torch.cuda.is_available():
            self._device = "cuda"
            self._compute_type = "float16"
        else:
            # MPS is unsupported by CTranslate2 — CPU with int8 is the best choice
            self._device = "cpu"
            self._compute_type = "int8"

        cpu_threads = settings.faster_whisper_cpu_threads
        num_workers  = settings.faster_whisper_num_workers

        print(
            f"[faster-whisper] Loading '{model_name}' on {self._device.upper()} "
            f"({self._compute_type}) | threads={cpu_threads} workers={num_workers} "
            f"batch_size={settings.faster_whisper_batch_size}"
        )

        self._model = WhisperModel(
            model_name,
            device=self._device,
            compute_type=self._compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )

        # Wrap in BatchedInferencePipeline when batch_size > 1
        if settings.faster_whisper_batch_size > 1:
            self._batched_model = BatchedInferencePipeline(model=self._model)
        else:
            self._batched_model = None

        print("[faster-whisper] Model ready.")

    def unload(self) -> None:
        self._model = None
        self._batched_model = None

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def transcribe(self, audio_bytes: bytes, filename: str) -> TranscriptionResponse:
        if not self._model:
            raise RuntimeError("Model not loaded")

        suffix = Path(filename).suffix or ".webm"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            start = time.perf_counter()

            transcribe_fn = (
                self._batched_model.transcribe
                if self._batched_model
                else self._model.transcribe
            )

            common_kwargs = dict(
                language=settings.whisper_language,
                task="transcribe",
                beam_size=settings.faster_whisper_beam_size,
                vad_filter=settings.faster_whisper_vad_filter,
            )

            # BatchedInferencePipeline accepts batch_size; WhisperModel does not
            if self._batched_model:
                common_kwargs["batch_size"] = settings.faster_whisper_batch_size

            segments_iter, info = transcribe_fn(tmp_path, **common_kwargs)
            segments_list = list(segments_iter)
            duration = time.perf_counter() - start

            full_text = " ".join(s.text.strip() for s in segments_list)

            segments = [
                Segment(
                    id=i,
                    start=s.start,
                    end=s.end,
                    text=s.text.strip(),
                )
                for i, s in enumerate(segments_list)
            ]

            return TranscriptionResponse(
                text=full_text.strip(),
                language=info.language,
                duration=round(duration, 3),
                segments=segments or None,
            )
        finally:
            os.unlink(tmp_path)


faster_whisper_service = FasterWhisperService()
