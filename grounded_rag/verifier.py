from __future__ import annotations
from typing import Any

from grounded_rag.generator import _get_client, call_claude, _extract_json
from grounded_rag.schemas import Chunk, Claim, VerifierResult

JUDGE_SYSTEM = """You are a strict grounding verifier. Decide whether a single claim is supported by the cited source chunk.

Respond with a single JSON object:
{"grounded": <bool>, "reason": "<one sentence>", "confidence": <float between 0 and 1>}

Grounded means: the claim text is directly entailed by the source chunk text, not just plausible. Set confidence high when the snippet appears verbatim in the chunk and the claim restates that fact.
"""


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
    if claim.citation is None or chunk is None:
        return VerifierResult(
            claim_text=claim.text,
            grounded=False,
            reason="claim has no citation",
            confidence=0.0,
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


def verify_all(claims: list[Claim], chunks_by_page: dict[int, Chunk],
               client: Any | None = None) -> list[VerifierResult]:
    out: list[VerifierResult] = []
    for c in claims:
        chunk = chunks_by_page.get(c.citation.page) if c.citation else None
        out.append(verify_claim(c, chunk, client=client))
    return out


def mean_confidence(results: list[VerifierResult]) -> float:
    if not results:
        return 0.0
    return sum(r.confidence for r in results) / len(results)
