from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Callable

import faiss
import numpy as np

from grounded_rag.schemas import Chunk, RetrievedChunk

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_CACHE_DIR = Path(".onnx_cache/onnx")
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


class _OnnxMiniLmEncoder:
    """Local ONNX runtime for all-MiniLM-L6-v2.

    Used when the Hugging Face Hub is unreachable. Expects a directory laid out
    like Chroma's published tarball: model.onnx + tokenizer.json next to each other.
    Same embedding space as the SentenceTransformers version.
    """

    def __init__(self, model_dir: Path = ONNX_CACHE_DIR) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self._sess = ort.InferenceSession(str(model_dir / "model.onnx"))
        self._tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tok.enable_truncation(max_length=256)
        self._tok.enable_padding()

    def __call__(self, texts: list[str]) -> np.ndarray:
        encs = self._tok.encode_batch(texts)
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        ttype = np.zeros_like(ids)
        hidden = self._sess.run(
            None,
            {"input_ids": ids, "attention_mask": mask, "token_type_ids": ttype},
        )[0]
        mask_f = mask[..., None].astype(np.float32)
        summed = (hidden * mask_f).sum(axis=1)
        counts = mask_f.sum(axis=1)
        counts[counts == 0] = 1.0
        pooled = summed / counts
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (pooled / norms).astype("float32")


def _default_encoder() -> Encoder:
    """Prefer local ONNX cache (no network), else lazy SentenceTransformer."""
    if ONNX_CACHE_DIR.exists() and (ONNX_CACHE_DIR / "model.onnx").exists():
        return _OnnxMiniLmEncoder(ONNX_CACHE_DIR)
    return _SentenceTransformerEncoder()


class Retriever:
    """In-memory FAISS retriever with section keyword boost.

    The encoder is injectable: production uses SentenceTransformer, tests can
    pass a deterministic fake to avoid model downloads.
    """

    def __init__(self, encoder: Encoder | None = None) -> None:
        self._encoder: Encoder = encoder or _default_encoder()
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
