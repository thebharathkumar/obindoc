from __future__ import annotations
from pathlib import Path

import pytest

from grounded_rag.parser import parse_pdf, chunk_text
from grounded_rag.schemas import Chunk

VOO = Path("data/voo.pdf")


def test_chunk_text_word_window() -> None:
    text = " ".join(f"w{i}" for i in range(900))
    chunks = chunk_text(text, page=1, section="S", target_tokens=300, overlap=30)
    # 900 words at 300 target with 30 overlap, walk size 270 -> ceil(900/270) = 4 chunks
    assert 3 <= len(chunks) <= 5
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].page == 1
    assert chunks[0].section == "S"
    # char_span end exceeds start
    for c in chunks:
        a, b = c.char_span
        assert 0 <= a < b <= len(text)


@pytest.mark.skipif(not VOO.exists(), reason="data/voo.pdf not present")
def test_parse_pdf_returns_chunks() -> None:
    chunks = parse_pdf(VOO)
    assert len(chunks) >= 4
    # Pages observed
    pages = {c.page for c in chunks}
    assert pages.issubset({1, 2})
    # Sections detected and non-empty
    sections = {c.section for c in chunks}
    assert "" not in sections
    # All chunk_ids unique
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


@pytest.mark.skipif(not VOO.exists(), reason="data/voo.pdf not present")
def test_parse_pdf_finds_known_text() -> None:
    chunks = parse_pdf(VOO)
    blob = " ".join(c.text for c in chunks).lower()
    assert "expense ratio" in blob
    assert "vanguard" in blob
    assert "s&p 500" in blob


@pytest.mark.skipif(not VOO.exists(), reason="data/voo.pdf not present")
def test_parse_pdf_section_detection() -> None:
    chunks = parse_pdf(VOO)
    sections = {c.section for c in chunks}
    # Real sections must be detected.
    assert "Quick facts" in sections
    assert "Sector Diversification" in sections
    # Bullets and tickers must NOT become section headers.
    for bad in [
        "• Seeks to track the performance of the S&P 500 Index.",
        "VOO",
        "consider it carefully before investing.",
    ]:
        assert bad not in sections, f"unexpected section header: {bad!r}"
