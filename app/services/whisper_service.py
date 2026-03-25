import ssl
import tempfile
import os
import time
import urllib.request
from pathlib import Path

import whisper

from app.config import settings
from app.schemas.transcription import TranscriptionResponse, Segment


def _create_unverified_opener() -> urllib.request.OpenerDirector:
    """Return a URL opener that skips SSL verification.

    Required on macOS with Python installed from python.org — those builds
    ship without CA certificates. Whisper already verifies downloads via
    SHA-256, so skipping SSL host verification is safe here.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


class WhisperService:
    def __init__(self) -> None:
        self._model: whisper.Whisper | None = None

    def load(self) -> None:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        print(f"[whisper] Loading model '{settings.whisper_model}' on {device.upper()}...")
        opener = _create_unverified_opener()
        urllib.request.install_opener(opener)
        try:
            self._model = whisper.load_model(settings.whisper_model, device=device)
        finally:
            urllib.request.install_opener(urllib.request.build_opener())
        print(f"[whisper] Model ready on {device.upper()}.")

    def unload(self) -> None:
        self._model = None

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
            result = self._model.transcribe(
                tmp_path,
                language=settings.whisper_language,
                task="transcribe",
            )
            duration = time.perf_counter() - start

            segments = [
                Segment(
                    id=s["id"],
                    start=s["start"],
                    end=s["end"],
                    text=s["text"].strip(),
                )
                for s in result.get("segments", [])
            ]

            return TranscriptionResponse(
                text=result["text"].strip(),
                language=result.get("language", settings.whisper_language),
                duration=round(duration, 3),
                segments=segments or None,
            )
        finally:
            os.unlink(tmp_path)


whisper_service = WhisperService()
