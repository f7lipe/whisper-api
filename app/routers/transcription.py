from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.transcription import TranscriptionResponse
from app.services.whisper_service import whisper_service

router = APIRouter(prefix="/api", tags=["transcription"])


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_ready": whisper_service.is_ready,
    }


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)) -> TranscriptionResponse:
    if not whisper_service.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    allowed_types = {
        "audio/webm", "audio/wav", "audio/mpeg", "audio/mp4",
        "audio/ogg", "audio/flac", "audio/x-m4a", "video/webm",
    }
    # Browsers may append codec params (e.g. "audio/webm;codecs=opus") — strip them.
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
        return whisper_service.transcribe(audio_bytes, file.filename or "audio.webm")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
