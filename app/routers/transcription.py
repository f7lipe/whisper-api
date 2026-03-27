from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Literal
import asyncio
import concurrent.futures

from app.schemas.transcription import TranscriptionResponse
from app.services.whisper_service import whisper_service
from app.services.faster_whisper_service import faster_whisper_service
from app.config import settings

router = APIRouter(prefix="/api", tags=["transcription"])

# Thread pool for blocking model-load calls
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
# Per-backend loading lock — prevents double-loading when concurrent requests arrive
_load_locks: dict[str, asyncio.Lock] = {}

BACKEND_INFO = {
    "openai-whisper": {
        "label": "OpenAI Whisper",
        "description": "Modelo original da OpenAI. Alta acurácia, mais memória.",
        "models": ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"],
        "default_model": settings.whisper_model,
        "license": "MIT",
        "author": "OpenAI",
    },
    "faster-whisper": {
        "label": "Faster Whisper",
        "description": "Reimplementação CTranslate2 pela SYSTRAN. Até 4× mais rápido, menos memória.",
        "models": ["tiny", "base", "small", "medium", "large-v2", "large-v3", "distil-large-v3"],
        "default_model": settings.faster_whisper_model,
        "license": "MIT",
        "author": "SYSTRAN",
    },
}


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "backends": {
            "openai-whisper": {"ready": whisper_service.is_ready},
            "faster-whisper": {"ready": faster_whisper_service.is_ready},
        },
    }


@router.get("/backends")
def list_backends() -> dict:
    return BACKEND_INFO


async def _ensure_loaded(backend: str) -> None:
    """Load a backend model lazily and thread-safely on first use."""
    loop = asyncio.get_running_loop()

    # Initialise the lock on first call (can't create at module level with asyncio)
    if backend not in _load_locks:
        _load_locks[backend] = asyncio.Lock()

    service = whisper_service if backend == "openai-whisper" else faster_whisper_service
    if service.is_ready:
        return

    async with _load_locks[backend]:
        if service.is_ready:  # double-checked after acquiring lock
            return
        model_name = (
            settings.whisper_model if backend == "openai-whisper"
            else settings.faster_whisper_model
        )
        print(f"[lazy] Loading backend '{backend}' with model '{model_name}' on first request…")
        await loop.run_in_executor(_executor, service.load, model_name)


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    backend: Literal["openai-whisper", "faster-whisper"] = Query(
        default="openai-whisper",
        description="Whisper backend to use for transcription",
    ),
) -> TranscriptionResponse:
    # Lazy-load the selected backend on first use (blocks until model is ready)
    await _ensure_loaded(backend)
    service = whisper_service if backend == "openai-whisper" else faster_whisper_service

    allowed_types = {
        "audio/webm", "audio/wav", "audio/mpeg", "audio/mp4",
        "audio/ogg", "audio/flac", "audio/x-m4a", "video/webm",
    }
    base_type = (file.content_type or "").split(";")[0].strip()
    if base_type and base_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}",
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        return service.transcribe(audio_bytes, file.filename or "audio.webm")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
