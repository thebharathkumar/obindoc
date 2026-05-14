from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import pytest


@dataclass
class _FakeMessage:
    """Mimics the shape returned by anthropic.messages.create()."""
    content: list[Any]

    @classmethod
    def text(cls, body: str) -> "_FakeMessage":
        return cls(content=[type("Block", (), {"type": "text", "text": body})()])


@dataclass
class FakeAnthropicClient:
    """Drop-in replacement for anthropic.Anthropic during tests."""
    responses: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        outer = self

        class _Messages:
            def create(self, **kwargs: Any) -> _FakeMessage:
                outer.calls.append(kwargs)
                if not outer.responses:
                    raise AssertionError("FakeAnthropicClient ran out of canned responses")
                return _FakeMessage.text(outer.responses.pop(0))

        self.messages = _Messages()


@pytest.fixture
def fake_client() -> FakeAnthropicClient:
    return FakeAnthropicClient()
