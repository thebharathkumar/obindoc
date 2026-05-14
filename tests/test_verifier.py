from __future__ import annotations
import json

from grounded_rag.verifier import verify_claim, verify_all, mean_confidence
from grounded_rag.schemas import Chunk, Claim, Citation, VerifierResult


def _chunk(text: str, page: int = 1, section: str = "Quick facts") -> Chunk:
    return Chunk(chunk_id="c", page=page, section=section, text=text, char_span=(0, len(text)))


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
    # The fake client must not have been called.
    assert fake_client.calls == []


def test_verify_all_runs_per_claim(fake_client) -> None:
    fake_client.responses.extend([
        json.dumps({"grounded": True, "reason": "ok", "confidence": 0.9}),
        json.dumps({"grounded": False, "reason": "snippet missing", "confidence": 0.2}),
    ])
    claims = [
        Claim(text="A", citation=Citation(page=1, section="x", snippet="foo")),
        Claim(text="B", citation=Citation(page=1, section="x", snippet="bar")),
    ]
    chunks_by_page = {1: _chunk("foo and other text")}
    results = verify_all(claims, chunks_by_page, client=fake_client)
    assert [r.grounded for r in results] == [True, False]


def test_mean_confidence_handles_empty() -> None:
    assert mean_confidence([]) == 0.0
    r1 = VerifierResult(claim_text="a", grounded=True, reason="", confidence=0.8)
    r2 = VerifierResult(claim_text="b", grounded=False, reason="", confidence=0.2)
    assert abs(mean_confidence([r1, r2]) - 0.5) < 1e-6
