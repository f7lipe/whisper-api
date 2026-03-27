from abc import ABC, abstractmethod

from app.schemas.transcription import TranscriptionResponse


class BaseWhisperService(ABC):
    """Abstract contract for all Whisper backend implementations."""

    @abstractmethod
    def load(self, model_name: str) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @property
    @abstractmethod
    def is_ready(self) -> bool: ...

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, filename: str) -> TranscriptionResponse: ...
