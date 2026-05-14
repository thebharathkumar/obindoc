from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Callable

import faiss
import numpy as np

from grounded_rag.schemas import Chunk, RetrievedChunk

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_STOPWORDS = {"the", "a", "an", "of", "is", "what", "which", "in", "to", "for",
              "by", "and", "or", "as", "at", "on", "with", "are", "be", "this"}

Encoder = Callable[[list[str]], np.ndarray]


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS}


def section_match_score(query: str, chunk: Chunk) -> float:
    q = _tokens(query)
    s = _tokens(chunk.section)
    return 1.0 if q & s else 0.0


class _SentenceTransformerEncoder:
    """Lazy SentenceTransformer wrapper used in production."""

    def __init__(self, name: str = EMBED_MODEL_NAME) -> None:
        self._name = name
        self._model = None

    def __call__(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._name)
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype("float32")


class Retriever:
    """In-memory FAISS retriever with section keyword boost.

    The encoder is injectable: production uses SentenceTransformer, tests can
    pass a deterministic fake to avoid model downloads.
    """

    def __init__(self, encoder: Encoder | None = None) -> None:
        self._encoder: Encoder = encoder or _SentenceTransformerEncoder()
        self._index: faiss.Index | None = None
        self._chunks: list[Chunk] = []

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._encoder(texts)
        if vecs.dtype != np.float32:
            vecs = vecs.astype("float32")
        return vecs

    def build_index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        if not self._chunks:
            return
        vecs = self._embed([c.text for c in self._chunks])
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        self._index = index

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        if self._index is None or not self._chunks:
            return []
        qv = self._embed([query])
        scores, idxs = self._index.search(qv, min(k, len(self._chunks)))
        out: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            chunk = self._chunks[int(idx)]
            cosine = float(score)
            sm = section_match_score(query, chunk)
            conf = 0.7 * cosine + 0.3 * sm
            out.append(RetrievedChunk(
                chunk=chunk,
                cosine=cosine,
                section_match=sm,
                retrieval_confidence=conf,
            ))
        return out

    def save(self, dirpath: Path) -> None:
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        assert self._index is not None
        faiss.write_index(self._index, str(dirpath / "index.faiss"))
        meta = [c.model_dump() for c in self._chunks]
        (dirpath / "chunks.json").write_text(json.dumps(meta), encoding="utf-8")

    def load(self, dirpath: Path) -> None:
        dirpath = Path(dirpath)
        self._index = faiss.read_index(str(dirpath / "index.faiss"))
        meta = json.loads((dirpath / "chunks.json").read_text(encoding="utf-8"))
        self._chunks = [Chunk(**m) for m in meta]
