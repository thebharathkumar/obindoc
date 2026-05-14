from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

from grounded_rag.graph import run_query
from grounded_rag.parser import parse_pdf
from grounded_rag.retriever import Retriever
from grounded_rag.spans import verify_chain

load_dotenv()

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Grounded RAG over a single PDF.")

INDEX_DIR = Path(".faiss_cache")
TRACES_DIR = Path("traces")

CANNED_QUESTIONS = [
    "What is the expense ratio of VOO?",
    "What is the ETF total net assets as of June 30 2022?",
    "Which sector has the largest allocation?",
    "What is the 10-year return?",
    "What is the top holding by weight?",
]


@app.command()
def index(pdf: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True)) -> None:
    """Parse a PDF and persist a FAISS index."""
    chunks = parse_pdf(pdf)
    typer.echo(f"parsed {len(chunks)} chunks from {pdf}")
    retriever = Retriever()
    retriever.build_index(chunks)
    retriever.save(INDEX_DIR)
    typer.echo(f"index saved to {INDEX_DIR}/")


@app.command()
def ask(query: str = typer.Argument(...)) -> None:
    """Ask a single question against the indexed PDF."""
    retriever = Retriever()
    retriever.load(INDEX_DIR)
    result = run_query(query=query, retriever=retriever, traces_dir=TRACES_DIR)
    typer.echo(f"trace_id: {result['trace_id']}")
    typer.echo(f"confidence: {result['generation_confidence']:.3f}")
    typer.echo("")
    typer.echo(result["answer"])


@app.command("verify-log")
def verify_log(trace_file: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    """Walk a JSONL trace file and confirm the hash chain is intact."""
    ok, message = verify_chain(trace_file)
    typer.echo(message)
    if not ok:
        raise typer.Exit(code=1)


@app.command()
def eval(out_dir: Optional[Path] = typer.Option(None, help="Directory to write trace files into.")) -> None:
    """Run the five canned questions and print a summary table."""
    retriever = Retriever()
    retriever.load(INDEX_DIR)
    target = out_dir or TRACES_DIR
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in CANNED_QUESTIONS:
        r = run_query(query=q, retriever=retriever, traces_dir=target)
        rows.append({
            "question": q,
            "trace_id": r["trace_id"],
            "confidence": r["generation_confidence"],
            "answer": r["answer"],
        })
    typer.echo(json.dumps(rows, indent=2))


if __name__ == "__main__":
    app()
