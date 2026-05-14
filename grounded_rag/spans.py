from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

GENESIS_HASH = "0" * 64


class Span(BaseModel):
    span_id: str
    parent_id: str | None
    trace_id: str
    name: str
    start: str
    end: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    reasoning: str = ""
    prev_hash: str
    this_hash: str = ""


def _canonical_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def compute_hash(span: Span, prev_hash: str) -> str:
    """Hash spec: sha256(prev_hash + canonical_json(body))
    where body excludes prev_hash and this_hash to avoid double-counting.
    """
    body = span.model_dump(exclude={"prev_hash", "this_hash"})
    payload = (prev_hash + _canonical_json(body)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ChainedLogger:
    """Append-only writer for a single trace's JSONL file.

    Computes this_hash on append, persists prev_hash linkage.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._prev_hash = self._last_hash()

    def _last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return GENESIS_HASH
        last_line = ""
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    last_line = line
        if not last_line:
            return GENESIS_HASH
        return json.loads(last_line)["this_hash"]

    def append(self, **fields: Any) -> Span:
        fields.setdefault("prev_hash", self._prev_hash)
        span = Span(**fields)
        span.this_hash = compute_hash(span, span.prev_hash)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(span.model_dump_json() + "\n")
        self._prev_hash = span.this_hash
        return span


def verify_chain(path: Path) -> tuple[bool, str]:
    """Walk a JSONL trace file and confirm every span's hash is consistent.

    Returns (ok, message). Message is informational on success, diagnostic on failure.
    """
    path = Path(path)
    if not path.exists():
        return False, f"trace file not found: {path}"
    prev = GENESIS_HASH
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            span = Span(**data)
            if span.prev_hash != prev:
                return False, f"line {i}: prev_hash mismatch (expected {prev[:8]}, got {span.prev_hash[:8]})"
            expected = compute_hash(span, span.prev_hash)
            if expected != span.this_hash:
                return False, f"line {i}: hash mismatch (recomputed {expected[:8]}, stored {span.this_hash[:8]})"
            prev = span.this_hash
            count += 1
    return True, f"chain ok: {count} spans verified"
