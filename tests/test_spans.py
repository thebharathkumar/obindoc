from __future__ import annotations
import json
from pathlib import Path

import pytest

from grounded_rag.spans import (
    Span,
    ChainedLogger,
    compute_hash,
    verify_chain,
    GENESIS_HASH,
)


def _make_span(name: str, prev_hash: str, **overrides) -> Span:
    base = dict(
        span_id="s1",
        parent_id=None,
        trace_id="t1",
        name=name,
        start="2026-05-14T12:00:00Z",
        end="2026-05-14T12:00:01Z",
        input={"q": "hi"},
        output={"a": "ok"},
        reasoning="",
        prev_hash=prev_hash,
        this_hash="",
    )
    base.update(overrides)
    s = Span(**base)
    s.this_hash = compute_hash(s, prev_hash)
    return s


def test_compute_hash_is_deterministic() -> None:
    s1 = _make_span("retrieve", GENESIS_HASH)
    s2 = _make_span("retrieve", GENESIS_HASH)
    assert s1.this_hash == s2.this_hash
    assert len(s1.this_hash) == 64


def test_hash_changes_when_body_changes() -> None:
    s1 = _make_span("retrieve", GENESIS_HASH)
    s2 = _make_span("retrieve", GENESIS_HASH, output={"a": "tampered"})
    assert s1.this_hash != s2.this_hash


def test_chained_logger_links_spans(tmp_path: Path) -> None:
    log_path = tmp_path / "trace.jsonl"
    logger = ChainedLogger(log_path)
    a = logger.append(name="retrieve", parent_id=None, trace_id="t1",
                      span_id="a", start="t0", end="t1",
                      input={"q": "hi"}, output={"k": 1}, reasoning="r1")
    b = logger.append(name="generate", parent_id="a", trace_id="t1",
                      span_id="b", start="t1", end="t2",
                      input={"k": 1}, output={"claims": []}, reasoning="r2")
    assert b.prev_hash == a.this_hash
    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["this_hash"] == a.this_hash


def test_verify_chain_passes_on_intact_file(tmp_path: Path) -> None:
    log_path = tmp_path / "trace.jsonl"
    logger = ChainedLogger(log_path)
    logger.append(name="a", parent_id=None, trace_id="t1", span_id="1",
                  start="t0", end="t1", input={}, output={}, reasoning="")
    logger.append(name="b", parent_id="1", trace_id="t1", span_id="2",
                  start="t1", end="t2", input={}, output={}, reasoning="")
    ok, message = verify_chain(log_path)
    assert ok, message


def test_verify_chain_detects_tampering(tmp_path: Path) -> None:
    log_path = tmp_path / "trace.jsonl"
    logger = ChainedLogger(log_path)
    logger.append(name="a", parent_id=None, trace_id="t1", span_id="1",
                  start="t0", end="t1", input={}, output={"v": 1}, reasoning="")
    logger.append(name="b", parent_id="1", trace_id="t1", span_id="2",
                  start="t1", end="t2", input={}, output={}, reasoning="")
    # Tamper: rewrite the first line's output without updating hashes.
    lines = log_path.read_text().splitlines()
    first = json.loads(lines[0])
    first["output"] = {"v": 999}
    lines[0] = json.dumps(first)
    log_path.write_text("\n".join(lines) + "\n")
    ok, message = verify_chain(log_path)
    assert not ok
    assert "hash mismatch" in message.lower()
