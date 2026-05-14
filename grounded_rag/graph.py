from __future__ import annotations
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from grounded_rag.generator import generate
from grounded_rag.retriever import Retriever
from grounded_rag.schemas import GenerationOutput, RetrievedChunk, VerifierResult
from grounded_rag.spans import ChainedLogger
from grounded_rag.verifier import mean_confidence, verify_all

_ALL_NODE_NAMES = ["retrieve", "generate", "verify", "assemble"]


class _State(TypedDict, total=False):
    query: str
    trace_id: str
    retrieved: list[RetrievedChunk]
    generation: GenerationOutput
    verification: list[VerifierResult]
    answer: str
    generation_confidence: float


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def build_graph(retriever: Retriever, logger: ChainedLogger,
                client: Any | None = None) -> Any:
    """Compile a 4-node graph where retriever/logger/client are bound by closure."""

    def emit(state: _State, name: str, parent: str | None, span_id: str,
             start: str, end: str, payload_in: dict[str, Any],
             payload_out: dict[str, Any], reasoning: str) -> None:
        logger.append(
            span_id=span_id,
            parent_id=parent,
            trace_id=state["trace_id"],
            name=name,
            start=start,
            end=end,
            input=payload_in,
            output=payload_out,
            reasoning=reasoning,
        )

    def node_retrieve(state: _State) -> _State:
        start = _now()
        results = retriever.retrieve(state["query"], k=5)
        end = _now()
        emit(
            state, "retrieve", parent=None, span_id="retrieve",
            start=start, end=end,
            payload_in={"query": state["query"], "k": 5},
            payload_out={
                "n": len(results),
                "ids": [r.chunk.chunk_id for r in results],
                "confidences": [r.retrieval_confidence for r in results],
            },
            reasoning="cosine + section_match boost",
        )
        return {"retrieved": results}

    def node_generate(state: _State) -> _State:
        start = _now()
        out = generate(state["query"], state["retrieved"], client=client)
        end = _now()
        emit(
            state, "generate", parent="retrieve", span_id="generate",
            start=start, end=end,
            payload_in={"query": state["query"], "n_chunks": len(state["retrieved"])},
            payload_out={"claims": [c.model_dump() for c in out.claims]},
            reasoning="JSON-schema constrained citation generation",
        )
        return {"generation": out}

    def node_verify(state: _State) -> _State:
        start = _now()
        results = verify_all(state["generation"].claims, state["retrieved"], client=client)
        end = _now()
        emit(
            state, "verify", parent="generate", span_id="verify",
            start=start, end=end,
            payload_in={"n_claims": len(state["generation"].claims)},
            payload_out={"results": [r.model_dump() for r in results]},
            reasoning="per-claim LLM-as-judge",
        )
        return {"verification": results}

    def node_assemble(state: _State) -> _State:
        start = _now()
        claims = state["generation"].claims
        verdicts = state["verification"]
        parts: list[str] = []
        for c, v in zip(claims, verdicts):
            prefix = "[grounded]" if v.grounded else "[unverified]"
            if c.citation:
                cite = f"(p.{c.citation.page}, {c.citation.section})"
            else:
                cite = "(no citation)"
            parts.append(f"{prefix} {c.text} {cite}")
        answer = "\n".join(parts) if parts else "No answer produced."
        conf = mean_confidence(verdicts)
        end = _now()
        emit(
            state, "assemble", parent="verify", span_id="assemble",
            start=start, end=end,
            payload_in={"n_claims": len(claims)},
            payload_out={"answer": answer, "generation_confidence": conf},
            reasoning="join verified claims with citation suffix",
        )
        return {"answer": answer, "generation_confidence": conf}

    g: StateGraph = StateGraph(_State)
    g.add_node("retrieve", node_retrieve)
    g.add_node("generate", node_generate)
    g.add_node("verify", node_verify)
    g.add_node("assemble", node_assemble)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "verify")
    g.add_edge("verify", "assemble")
    g.add_edge("assemble", END)
    return g.compile()


def run_query(query: str, retriever: Retriever, traces_dir: Path,
              client: Any | None = None) -> dict[str, Any]:
    traces_dir = Path(traces_dir)
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_id = uuid.uuid4().hex[:12]
    logger = ChainedLogger(traces_dir / f"{trace_id}.jsonl")
    graph = build_graph(retriever=retriever, logger=logger, client=client)
    final = graph.invoke({"query": query, "trace_id": trace_id})
    return {
        "trace_id": trace_id,
        "answer": final["answer"],
        "generation_confidence": final["generation_confidence"],
        "claims": final["generation"].claims,
        "verification": final["verification"],
    }
