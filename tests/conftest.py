from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest


@dataclass
class _FakeMessage:
    """Mimics the shape returned by anthropic.messages.create()."""
    content: list[Any]

    @classmethod
    def text(cls, body: str) -> "_FakeMessage":
        return cls(content=[type("Block", (), {"type": "text", "text": body})()])


@dataclass
class FakeAnthropicClient:
    """Drop-in replacement for anthropic.Anthropic during tests."""
    responses: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        outer = self

        class _Messages:
            def create(self, **kwargs: Any) -> _FakeMessage:
                outer.calls.append(kwargs)
                if not outer.responses:
                    raise AssertionError("FakeAnthropicClient ran out of canned responses")
                return _FakeMessage.text(outer.responses.pop(0))

        self.messages = _Messages()


@pytest.fixture
def fake_client() -> FakeAnthropicClient:
    return FakeAnthropicClient()


def _hash_token(tok: str, dim: int) -> int:
    h = hashlib.md5(tok.encode("utf-8")).hexdigest()
    return int(h, 16) % dim


def hashing_encoder(dim: int = 128):
    """Deterministic bag-of-tokens encoder used in tests.

    Avoids network access to the SentenceTransformer model. Two inputs that
    share tokens land near each other in cosine space; disjoint inputs are
    near-orthogonal.
    """
    pattern = re.compile(r"[a-z0-9]+")

    def encode(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in pattern.findall(t.lower()):
                out[i, _hash_token(tok, dim)] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype("float32")

    return encode


@pytest.fixture
def encoder():
    return hashing_encoder()
