from __future__ import annotations
import json

from grounded_rag.verifier import (
    verify_claim, verify_all, mean_confidence, locate_chunk,
)
from grounded_rag.schemas import (
    Chunk, Claim, Citation, RetrievedChunk, VerifierResult,
)


def _chunk(text: str, page: int = 1, section: str = "Quick facts",
           cid: str = "c") -> Chunk:
    return Chunk(chunk_id=cid, page=page, section=section, text=text,
                 char_span=(0, len(text)))


def _retr(chunk: Chunk, conf: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, cosine=conf, section_match=0.0,
                          retrieval_confidence=conf)


def test_verify_claim_grounded(fake_client) -> None:
    fake_client.responses.append(json.dumps(
        {"grounded": True, "reason": "snippet matches", "confidence": 0.95}
    ))
    claim = Claim(text="Expense ratio is 0.03%",
                  citation=Citation(page=1, section="Quick facts", snippet="Expense ratio 0.03%"))
    chunk = _chunk("Expense ratio 0.03% Dividend schedule Quarterly")
    result = verify_claim(claim, chunk, client=fake_client)
    assert result.grounded
    assert result.confidence == 0.95


def test_verify_claim_ungrounded_when_no_citation(fake_client) -> None:
    claim = Claim(text="CEO birthday is unknown", citation=None)
    result = verify_claim(claim, chunk=None, client=fake_client)
    assert not result.grounded
    assert result.confidence == 0.0
    assert "no citation" in result.reason
    assert fake_client.calls == []


def test_verify_claim_snippet_not_in_retrieved_short_circuits(fake_client) -> None:
    """If locate_chunk returns None, do not call Claude; return a high-confidence
    not-grounded verdict."""
    claim = Claim(text="hallucinated",
                  citation=Citation(page=1, section="x", snippet="never appears"))
    result = verify_claim(claim, chunk=None, client=fake_client)
    assert not result.grounded
    assert result.confidence == 0.9
    assert "not found" in result.reason
    assert fake_client.calls == []


def test_verify_all_uses_snippet_to_pick_chunk(fake_client) -> None:
    """The verifier must read from the chunk that contains the cited snippet,
    even when another retrieved chunk shares the same page."""
    # Two retrieved chunks on page 2. Top-ranked has the wrong content;
    # second one contains the cited snippet.
    wrong = _chunk("Expense ratio comparison VOO 0.03% peer average 0.74%",
                   page=2, section="Expense ratio comparison", cid="wrong")
    right = _chunk("Sector Diversification Information Technology 26.8% Health Care 15.2%",
                   page=2, section="Sector Diversification", cid="right")
    retrieved = [_retr(wrong, conf=0.7), _retr(right, conf=0.3)]

    # The verifier should see "right" chunk text, not "wrong".
    fake_client.responses.append(json.dumps(
        {"grounded": True, "reason": "ok", "confidence": 0.95}
    ))
    claim = Claim(
        text="Information Technology has the largest sector allocation at 26.8%",
        citation=Citation(page=2, section="Sector Diversification",
                          snippet="Information Technology 26.8%"),
    )
    results = verify_all([claim], retrieved, client=fake_client)
    assert len(fake_client.calls) == 1
    user_msg = fake_client.calls[0]["messages"][0]["content"]
    assert "Sector Diversification" in user_msg
    assert "Information Technology 26.8%" in user_msg
    assert "Expense ratio comparison VOO 0.03%" not in user_msg
    assert results[0].grounded


def test_locate_chunk_returns_none_when_snippet_missing() -> None:
    retrieved = [_retr(_chunk("nothing relevant here"))]
    claim = Claim(text="x", citation=Citation(page=1, section="x", snippet="absent"))
    assert locate_chunk(claim, retrieved) is None


def test_locate_chunk_prefers_matching_section_on_tie() -> None:
    a = _chunk("foo bar baz", page=1, section="Alpha", cid="a")
    b = _chunk("foo bar baz", page=1, section="Beta",  cid="b")
    retrieved = [_retr(a, conf=0.4), _retr(b, conf=0.9)]
    claim = Claim(text="x",
                  citation=Citation(page=1, section="Alpha", snippet="foo bar"))
    # Both contain the snippet; section match should win over higher confidence.
    assert locate_chunk(claim, retrieved).chunk_id == "a"


def test_verify_all_with_multiple_page_chunks(fake_client) -> None:
    """Original bug repro: two page-2 chunks, citation snippet only in one of them."""
    fake_client.responses.append(json.dumps(
        {"grounded": True, "reason": "ok", "confidence": 0.9}
    ))
    a = _chunk("expense ratios and fees", page=2, section="Expense ratio comparison", cid="a")
    b = _chunk("Information Technology 26.8%", page=2, section="Sector Diversification", cid="b")
    retrieved = [_retr(a, 0.8), _retr(b, 0.3)]
    claim = Claim(text="IT is 26.8%",
                  citation=Citation(page=2, section="Sector Diversification",
                                    snippet="Information Technology 26.8%"))
    results = verify_all([claim], retrieved, client=fake_client)
    assert results[0].grounded
    # The prompt must have included chunk b's text.
    assert "Information Technology" in fake_client.calls[0]["messages"][0]["content"]


def test_mean_confidence_handles_empty() -> None:
    assert mean_confidence([]) == 0.0
    r1 = VerifierResult(claim_text="a", grounded=True, reason="", confidence=0.8)
    r2 = VerifierResult(claim_text="b", grounded=False, reason="", confidence=0.2)
    assert abs(mean_confidence([r1, r2]) - 0.5) < 1e-6
