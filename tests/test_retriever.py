from __future__ import annotations
from pathlib import Path

import pytest

from grounded_rag.retriever import Retriever, section_match_score
from grounded_rag.schemas import Chunk


def _mk(text: str, section: str, idx: int = 0, page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=f"c{idx}",
        page=page,
        section=section,
        text=text,
        char_span=(0, len(text)),
    )


def test_section_match_score_hits_on_keyword() -> None:
    chunk = _mk("anything", section="Expense Ratio Comparison")
    assert section_match_score("what is the expense ratio?", chunk) == 1.0


def test_section_match_score_miss() -> None:
    chunk = _mk("anything", section="Sector Diversification")
    assert section_match_score("what is the 10-year return?", chunk) == 0.0


def test_retriever_returns_top_k_with_confidence(encoder) -> None:
    chunks = [
        _mk("Vanguard S&P 500 ETF expense ratio is 0.03%", section="Expense ratio comparison", idx=1),
        _mk("Information Technology 26.8%", section="Sector Diversification", idx=2),
        _mk("Top holdings Apple Microsoft", section="Ten largest holdings", idx=3),
    ]
    r = Retriever(encoder=encoder)
    r.build_index(chunks)
    results = r.retrieve("what is the expense ratio?", k=2)
    assert len(results) == 2
    assert results[0].chunk.section.lower().startswith("expense")
    for res in results:
        assert 0.0 <= res.retrieval_confidence <= 1.0
        assert abs(res.retrieval_confidence
                   - (0.7 * res.cosine + 0.3 * res.section_match)) < 1e-6


def test_retriever_roundtrip(tmp_path: Path, encoder) -> None:
    chunks = [_mk("Expense ratio is 0.03%", section="Expense ratio", idx=1)]
    r = Retriever(encoder=encoder)
    r.build_index(chunks)
    r.save(tmp_path)
    r2 = Retriever(encoder=encoder)
    r2.load(tmp_path)
    assert r2.retrieve("expense ratio", k=1)[0].chunk.chunk_id == "c1"
