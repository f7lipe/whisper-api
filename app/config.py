import os
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_cpu_threads() -> int:
    """Use all physical cores available — ideal for Apple Silicon."""
    return os.cpu_count() or 4


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── OpenAI Whisper ────────────────────────────────────────────────
    whisper_model: str = "turbo"

    # ── faster-whisper ────────────────────────────────────────────────
    faster_whisper_model: str = "large-v3"

    # beam_size: quality vs speed trade-off (1 = fastest, 5 = best quality)
    faster_whisper_beam_size: int = 5

    # batch_size > 1 enables BatchedInferencePipeline — big speed gain on CPU
    # Set to 1 to disable batching (lower latency on short clips)
    faster_whisper_batch_size: int = 8

    # CPU threads for CTranslate2 (defaults to all logical cores)
    faster_whisper_cpu_threads: int = _default_cpu_threads()

    # Number of async workers for model I/O overlap
    faster_whisper_num_workers: int = 2

    # VAD filter: strips silence before transcription (faster + cleaner output)
    faster_whisper_vad_filter: bool = True

    # ── Shared ────────────────────────────────────────────────────────
    whisper_language: str = "pt"
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
