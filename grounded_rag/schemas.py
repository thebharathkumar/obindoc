from __future__ import annotations
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    page: int
    section: str
    text: str
    char_span: tuple[int, int]


class RetrievedChunk(BaseModel):
    chunk: Chunk
    cosine: float
    section_match: float
    retrieval_confidence: float


class Citation(BaseModel):
    page: int
    section: str
    snippet: str


class Claim(BaseModel):
    text: str
    citation: Citation | None = None


class GenerationOutput(BaseModel):
    claims: list[Claim] = Field(default_factory=list)


class VerifierResult(BaseModel):
    claim_text: str
    grounded: bool
    reason: str
    confidence: float
