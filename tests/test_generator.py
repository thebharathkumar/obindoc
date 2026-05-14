from __future__ import annotations
import json

import pytest

from grounded_rag.generator import generate, build_prompt, _extract_json
from grounded_rag.schemas import Chunk, RetrievedChunk, GenerationOutput


def _retrieved(text: str, section: str = "Quick facts", page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id="c1", page=page, section=section, text=text, char_span=(0, len(text))),
        cosine=0.8, section_match=1.0, retrieval_confidence=0.83,
    )


def test_build_prompt_includes_query_and_chunks() -> None:
    chunks = [_retrieved("Expense ratio is 0.03%")]
    msg = build_prompt("what is the expense ratio?", chunks)
    assert "0.03%" in msg
    assert "Quick facts" in msg
    assert "what is the expense ratio?" in msg
    assert "JSON" in msg


def test_extract_json_pulls_first_object() -> None:
    text = 'noise before {"claims": [{"text": "x", "citation": {"page": 1, "section": "s", "snippet": "y"}}]} noise'
    obj = _extract_json(text)
    assert obj["claims"][0]["text"] == "x"


def test_generate_parses_valid_response(fake_client) -> None:
    fake_client.responses.append(json.dumps({
        "claims": [
            {"text": "Expense ratio is 0.03%",
             "citation": {"page": 1, "section": "Quick facts", "snippet": "Expense ratio 0.03%"}}
        ]
    }))
    out = generate(
        query="what is the expense ratio?",
        retrieved=[_retrieved("Expense ratio is 0.03%")],
        client=fake_client,
    )
    assert isinstance(out, GenerationOutput)
    assert len(out.claims) == 1
    assert out.claims[0].citation.page == 1


def test_generate_allows_ungrounded(fake_client) -> None:
    fake_client.responses.append(json.dumps({
        "claims": [{"text": "Cannot be determined from source", "citation": None}]
    }))
    out = generate(
        query="what is the CEO's birthday?",
        retrieved=[_retrieved("Expense ratio is 0.03%")],
        client=fake_client,
    )
    assert out.claims[0].citation is None
