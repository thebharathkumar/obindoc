from __future__ import annotations
import json
import os
import re
import time
from typing import Any

import anthropic

from grounded_rag.schemas import GenerationOutput, RetrievedChunk

DEFAULT_MODEL = "claude-sonnet-4-5"
TIMEOUT_S = 60.0
MAX_RETRIES = 3

_client_cache: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    """Single Anthropic client used by both generator and verifier."""
    global _client_cache
    if _client_cache is None:
        _client_cache = anthropic.Anthropic(timeout=TIMEOUT_S)
    return _client_cache


def _model_name() -> str:
    return os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)


def call_claude(client: Any, system: str, user: str, max_tokens: int = 1024) -> str:
    """Single-shot Claude call with retry and exponential backoff."""
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=_model_name(),
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    return block.text
            return ""
        except (anthropic.APIError, anthropic.APIConnectionError, anthropic.RateLimitError) as exc:
            last_err = exc
            if attempt == MAX_RETRIES - 1:
                break
            time.sleep(2 ** attempt)
    assert last_err is not None
    raise last_err


SYSTEM_PROMPT = """You are a careful research assistant. Answer the user's question using ONLY the provided source chunks.

Output must be a single JSON object with this exact shape:
{"claims": [{"text": "<one factual claim>", "citation": {"page": <int>, "section": "<string>", "snippet": "<exact phrase from the chunk>"}}]}

Rules:
1. Every claim must cite a specific page, section, and verbatim snippet from the chunks.
2. If a claim cannot be grounded in the provided chunks, set its citation to null and say so in the claim text.
3. Do NOT invent facts not present in the chunks.
4. Output ONLY the JSON object, no prose, no markdown fences.
"""


def build_prompt(query: str, retrieved: list[RetrievedChunk]) -> str:
    lines = [f"QUESTION: {query}", "", "SOURCE CHUNKS:"]
    for i, r in enumerate(retrieved, start=1):
        lines.append(
            f"[chunk {i}] page={r.chunk.page} section={r.chunk.section!r} "
            f"retrieval_confidence={r.retrieval_confidence:.3f}\n{r.chunk.text}\n"
        )
    lines.append("Respond with the JSON object only.")
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object from a string."""
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object found in response: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced braces in response")


def generate(query: str, retrieved: list[RetrievedChunk],
             client: Any | None = None) -> GenerationOutput:
    client = client or _get_client()
    user_msg = build_prompt(query, retrieved)
    text = call_claude(client, SYSTEM_PROMPT, user_msg, max_tokens=1024)
    data = _extract_json(text)
    return GenerationOutput.model_validate(data)
