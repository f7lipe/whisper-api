from pydantic import BaseModel


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    text: str
    language: str
    duration: float
    segments: list[Segment] | None = None
