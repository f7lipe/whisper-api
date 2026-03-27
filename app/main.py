from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers.transcription import router
from app.services.whisper_service import whisper_service
from app.services.faster_whisper_service import faster_whisper_service
from app.services.mlx_whisper_service import mlx_whisper_service
from app.config import settings

BASE_DIR = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Backends use lazy loading — they will load on the first transcription request.
    yield
    whisper_service.unload()
    faster_whisper_service.unload()
    mlx_whisper_service.unload()


app = FastAPI(
    title="Whisper Transcription API",
    description="Transcrição de áudio para Português do Brasil via OpenAI Whisper & faster-whisper",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
