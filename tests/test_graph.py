from __future__ import annotations
import json
from pathlib import Path

from grounded_rag.graph import run_query, _ALL_NODE_NAMES
from grounded_rag.retriever import Retriever
from grounded_rag.schemas import Chunk
from grounded_rag.spans import verify_chain


def _build_retriever(encoder) -> Retriever:
    chunks = [
        Chunk(chunk_id="c1", page=1, section="Quick facts",
              text="Expense ratio 0.03% Dividend schedule Quarterly",
              char_span=(0, 47)),
        Chunk(chunk_id="c2", page=2, section="Sector Diversification",
              text="Information Technology 26.8% Health Care 15.2%", char_span=(0, 46)),
    ]
    r = Retriever(encoder=encoder)
    r.build_index(chunks)
    return r


def test_run_query_emits_one_span_per_node(tmp_path: Path, fake_client, encoder) -> None:
    fake_client.responses.extend([
        json.dumps({"claims": [{"text": "Expense ratio is 0.03%",
                                "citation": {"page": 1, "section": "Quick facts",
                                             "snippet": "Expense ratio 0.03%"}}]}),
        json.dumps({"grounded": True, "reason": "snippet present", "confidence": 0.9}),
    ])
    result = run_query(
        query="what is the expense ratio?",
        retriever=_build_retriever(encoder),
        traces_dir=tmp_path,
        client=fake_client,
    )
    trace_path = tmp_path / f"{result['trace_id']}.jsonl"
    assert trace_path.exists()
    lines = trace_path.read_text().splitlines()
    names = [json.loads(line)["name"] for line in lines]
    assert names == _ALL_NODE_NAMES


def test_run_query_chain_intact(tmp_path: Path, fake_client, encoder) -> None:
    fake_client.responses.extend([
        json.dumps({"claims": [{"text": "Expense ratio is 0.03%",
                                "citation": {"page": 1, "section": "Quick facts",
                                             "snippet": "Expense ratio 0.03%"}}]}),
        json.dumps({"grounded": True, "reason": "ok", "confidence": 0.88}),
    ])
    result = run_query(
        query="expense ratio",
        retriever=_build_retriever(encoder),
        traces_dir=tmp_path,
        client=fake_client,
    )
    ok, msg = verify_chain(tmp_path / f"{result['trace_id']}.jsonl")
    assert ok, msg


def test_run_query_returns_answer_and_confidence(tmp_path: Path, fake_client, encoder) -> None:
    fake_client.responses.extend([
        json.dumps({"claims": [{"text": "Expense ratio is 0.03%",
                                "citation": {"page": 1, "section": "Quick facts",
                                             "snippet": "Expense ratio 0.03%"}}]}),
        json.dumps({"grounded": True, "reason": "ok", "confidence": 0.9}),
    ])
    result = run_query(
        query="expense ratio",
        retriever=_build_retriever(encoder),
        traces_dir=tmp_path,
        client=fake_client,
    )
    assert "0.03%" in result["answer"]
    assert abs(result["generation_confidence"] - 0.9) < 1e-6
