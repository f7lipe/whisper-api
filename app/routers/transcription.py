from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Literal
import asyncio
import concurrent.futures

from app.schemas.transcription import TranscriptionResponse
from app.services.whisper_service import whisper_service
from app.services.faster_whisper_service import faster_whisper_service
from app.services.mlx_whisper_service import mlx_whisper_service
from app.config import settings

router = APIRouter(prefix="/api", tags=["transcription"])

# Thread pool for blocking model-load calls
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
# Per-backend loading lock — prevents double-loading when concurrent requests arrive
_load_locks: dict[str, asyncio.Lock] = {}

BACKEND_INFO = {
    "openai-whisper": {
        "label": "OpenAI Whisper",
        "description": "Modelo original da OpenAI. Alta acurácia, roda em MPS no Mac.",
        "models": ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"],
        "default_model": settings.whisper_model,
        "license": "MIT",
        "author": "OpenAI",
    },
    "faster-whisper": {
        "label": "Faster Whisper",
        "description": "CTranslate2 pela SYSTRAN. Até 4× mais rápido, roda em CPU (int8).",
        "models": ["tiny", "base", "small", "medium", "large-v2", "large-v3", "distil-large-v3"],
        "default_model": settings.faster_whisper_model,
        "license": "MIT",
        "author": "SYSTRAN",
    },
    "mlx-whisper": {
        "label": "MLX Whisper",
        "description": "Apple MLX — roda em Metal/MPS nativamente. Melhor custo-benefício no Apple Silicon.",
        "models": [
            "tiny", "base", "small", "medium",
            "large", "large-v2", "large-v3", "large-v3-turbo", "distil-large-v3",
        ],
        "default_model": settings.mlx_whisper_model,
        "license": "MIT",
        "author": "Apple MLX Community",
    },
}

# Convenience map: backend key → service instance
_SERVICES = {
    "openai-whisper": whisper_service,
    "faster-whisper":  faster_whisper_service,
    "mlx-whisper":     mlx_whisper_service,
}

# Default model per backend when none is specified in the request
_DEFAULT_MODELS = {
    "openai-whisper": lambda: settings.whisper_model,
    "faster-whisper":  lambda: settings.faster_whisper_model,
    "mlx-whisper":     lambda: settings.mlx_whisper_model,
}

BackendKey = Literal["openai-whisper", "faster-whisper", "mlx-whisper"]


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "backends": {
            key: {"ready": svc.is_ready, "model": getattr(svc, "_current_model", None)}
            for key, svc in _SERVICES.items()
        },
    }


@router.get("/backends")
def list_backends() -> dict:
    return BACKEND_INFO


async def _ensure_loaded(backend: str, model_name: str) -> None:
    """Load (or reload) a backend model lazily and thread-safely.

    If the service is already loaded with a DIFFERENT model, it is unloaded
    first so the newly selected model takes effect.
    """
    loop = asyncio.get_running_loop()

    if backend not in _load_locks:
        _load_locks[backend] = asyncio.Lock()

    service = _SERVICES[backend]

    # Fast path: already loaded with the right model
    if service.is_ready and getattr(service, "_current_model", None) == model_name:
        return

    async with _load_locks[backend]:
        # Double-checked inside the lock
        if service.is_ready and getattr(service, "_current_model", None) == model_name:
            return
        if service.is_ready:
            print(f"[lazy] Reloading '{backend}' — switching model to '{model_name}'…")
            service.unload()
        else:
            print(f"[lazy] Loading '{backend}' with model '{model_name}' on first request…")
        await loop.run_in_executor(_executor, service.load, model_name)
        service._current_model = model_name


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    backend: BackendKey = Query(
        default="openai-whisper",
        description="Whisper backend to use for transcription",
    ),
    model: str | None = Query(
        default=None,
        description="Model name to use. Defaults to the value configured in settings.",
    ),
) -> TranscriptionResponse:
    if backend not in _SERVICES:
        raise HTTPException(status_code=400, detail=f"Unknown backend: {backend}")

    resolved_model = model or _DEFAULT_MODELS[backend]()
    await _ensure_loaded(backend, resolved_model)
    service = _SERVICES[backend]

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
