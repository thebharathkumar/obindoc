from __future__ import annotations
import re
from typing import Any

from grounded_rag.generator import _get_client, call_claude, _extract_json
from grounded_rag.schemas import Chunk, Claim, RetrievedChunk, VerifierResult

JUDGE_SYSTEM = """You are a strict grounding verifier. Decide whether a single claim is supported by the cited source chunk.

Respond with a single JSON object:
{"grounded": <bool>, "reason": "<one sentence>", "confidence": <float between 0 and 1>}

Grounded means: the claim text is directly entailed by the source chunk text, not just plausible. Set confidence high when the snippet appears verbatim in the chunk and the claim restates that fact.
"""

_WS = re.compile(r"\s+")


def _normalize(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def locate_chunk(claim: Claim, retrieved: list[RetrievedChunk]) -> Chunk | None:
    """Find the retrieved chunk that contains claim.citation.snippet.

    Match is substring after whitespace+case normalization. If multiple chunks
    contain the snippet, prefer one whose (section, page) matches the citation;
    tiebreak by highest retrieval_confidence. Returns None if no retrieved chunk
    contains the snippet.
    """
    if claim.citation is None or not retrieved:
        return None
    needle = _normalize(claim.citation.snippet)
    if not needle:
        return None
    matches = [r for r in retrieved if needle in _normalize(r.chunk.text)]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0].chunk
    cite = claim.citation
    exact = [r for r in matches
             if r.chunk.page == cite.page
             and _normalize(r.chunk.section) == _normalize(cite.section)]
    pool = exact if exact else matches
    pool.sort(key=lambda r: r.retrieval_confidence, reverse=True)
    return pool[0].chunk


def _judge_prompt(claim: Claim, chunk: Chunk) -> str:
    assert claim.citation is not None
    return (
        f"CLAIM: {claim.text}\n"
        f"CITED SNIPPET: {claim.citation.snippet}\n"
        f"CITED LOCATION: page {claim.citation.page}, section {claim.citation.section!r}\n\n"
        f"FULL SOURCE CHUNK (page {chunk.page}, section {chunk.section!r}):\n"
        f"{chunk.text}\n\n"
        "Respond with the JSON object only."
    )


def verify_claim(claim: Claim, chunk: Chunk | None,
                 client: Any | None = None) -> VerifierResult:
    if claim.citation is None:
        return VerifierResult(
            claim_text=claim.text,
            grounded=False,
            reason="claim has no citation",
            confidence=0.0,
        )
    if chunk is None:
        return VerifierResult(
            claim_text=claim.text,
            grounded=False,
            reason="cited snippet not found in retrieved context",
            confidence=0.9,
        )
    client = client or _get_client()
    text = call_claude(client, JUDGE_SYSTEM, _judge_prompt(claim, chunk), max_tokens=256)
    data = _extract_json(text)
    return VerifierResult(
        claim_text=claim.text,
        grounded=bool(data.get("grounded", False)),
        reason=str(data.get("reason", "")),
        confidence=float(data.get("confidence", 0.0)),
    )


def verify_all(claims: list[Claim], retrieved: list[RetrievedChunk],
               client: Any | None = None) -> list[VerifierResult]:
    out: list[VerifierResult] = []
    for c in claims:
        chunk = locate_chunk(c, retrieved)
        out.append(verify_claim(c, chunk, client=client))
    return out


def mean_confidence(results: list[VerifierResult]) -> float:
    if not results:
        return 0.0
    return sum(r.confidence for r in results) / len(results)
