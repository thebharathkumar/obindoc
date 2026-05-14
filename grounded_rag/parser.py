from __future__ import annotations
import re
from pathlib import Path
from statistics import median

import fitz  # PyMuPDF

from grounded_rag.schemas import Chunk

# Trailing footnote markers like "Sector Diversification  5" -> "Sector Diversification".
_FOOTNOTE_TAIL = re.compile(r"\s+\d{1,2}\s*$")


def _normalize_header(text: str) -> str:
    return _FOOTNOTE_TAIL.sub("", text.strip())


_BULLET_CHARS = {"•", "*", "-", "–", "·"}  # bullet, asterisk, hyphen, en dash, middot
_SENTENCE_PUNCT = {".", ",", ";", ":", "?", "!"}


def _is_header(text: str, size: float, flags: int, page_median: float) -> bool:
    """Heuristic: large font OR bold-and-short, with sentence/bullet rejections."""
    text = text.strip()
    if not text or len(text) > 80:
        return False
    if len(text) < 4:
        return False
    if text[0] in _BULLET_CHARS:
        return False
    if text[-1] in _SENTENCE_PUNCT:
        return False
    if text.isupper() and len(text) < 5:
        return False
    bold = bool(flags & 16)  # PyMuPDF flag bit for bold
    if size >= 1.1 * page_median:
        return True
    if bold and len(text) <= 60:
        return True
    return False


def _extract_page(page: fitz.Page) -> tuple[str, list[tuple[int, str]]]:
    """Return (full_page_text, list of (char_offset, section_name) header markers).

    Walks blocks/lines in reading order. Headers are detected via font heuristics.
    """
    blob = ""
    headers: list[tuple[int, str]] = []
    sizes: list[float] = []
    raw = page.get_text("dict")
    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sizes.append(span.get("size", 0.0))
    page_median = median(sizes) if sizes else 0.0

    for block in raw.get("blocks", []):
        for line in block.get("lines", []):
            line_text = "".join(span.get("text", "") for span in line.get("spans", []))
            stripped = line_text.strip()
            if not stripped:
                continue
            first_span = line["spans"][0]
            if _is_header(stripped, first_span.get("size", 0.0),
                          first_span.get("flags", 0), page_median):
                headers.append((len(blob), _normalize_header(stripped)))
            blob += line_text + "\n"
    return blob, headers


def chunk_text(text: str, page: int, section: str,
               target_tokens: int = 300, overlap: int = 30,
               id_prefix: str = "") -> list[Chunk]:
    """Word-based chunking with overlap. Token count is approximated by word count.

    char_span is reported against a whitespace-normalized form of the input.
    For this PDF the difference is negligible and downstream code only needs
    a stable, monotonic offset per chunk.
    """
    words = text.split()
    if not words:
        return []
    chunks: list[Chunk] = []
    step = max(1, target_tokens - overlap)
    cursor = 0
    idx = 0
    while cursor < len(words):
        window = words[cursor:cursor + target_tokens]
        if not window:
            break
        joined = " ".join(window)
        prefix = " ".join(words[:cursor])
        start = len(prefix) + (1 if prefix else 0)
        end = start + len(joined)
        suffix = f"#{idx}" if id_prefix else ""
        chunks.append(Chunk(
            chunk_id=f"{id_prefix}p{page}:{start}-{end}{suffix}",
            page=page,
            section=section or "(unsectioned)",
            text=joined,
            char_span=(start, end),
        ))
        cursor += step
        idx += 1
    return chunks


def parse_pdf(path: Path) -> list[Chunk]:
    """Parse a PDF into Chunks with page and detected section metadata."""
    path = Path(path)
    doc = fitz.open(path)
    all_chunks: list[Chunk] = []
    for i, page in enumerate(doc, start=1):
        blob, headers = _extract_page(page)
        if not blob.strip():
            continue
        boundaries = headers + [(len(blob), "")]
        current_section = "(intro)"
        last_off = 0
        seg_idx = 0
        for off, name in boundaries:
            segment = blob[last_off:off]
            if segment.strip():
                all_chunks.extend(chunk_text(
                    segment, page=i, section=current_section,
                    id_prefix=f"s{seg_idx}-",
                ))
                seg_idx += 1
            current_section = name
            last_off = off
    doc.close()
    return [c for c in all_chunks if not _is_header_only(c)]


def _is_header_only(chunk: Chunk) -> bool:
    text = chunk.text.strip()
    section = chunk.section.strip()
    if text == section:
        return True
    if text.startswith(section) and len(text) - len(section) < 40:
        return True
    return False
