"""Markdown-aware chunking for the handbook corpus.

Two-stage strategy:
1. Split on level-2/level-3 headers, keeping a section breadcrumb as metadata.
2. Window any oversized section down to ``max_chars`` with ``overlap``,
   preferring paragraph boundaries so no chunk exceeds the retrieval sweet spot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADER_RE = re.compile(r"^(#{2,3})\s+(.*\S)\s*$")
_TITLE_RE = re.compile(r"^#\s+(.*\S)\s*$")


@dataclass
class Chunk:
    chunk_id: str
    file: str  # corpus-relative path, e.g. "java/concurrency.md"
    section: str  # breadcrumb, e.g. "Java concurrency > Deadlock checklist"
    text: str


def _windows(text: str, max_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Prefer a paragraph break in the last quarter of the window.
            cut = text.rfind("\n\n", start, end)
            if cut > start + max_chars // 2:
                end = cut
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def chunk_markdown(
    text: str, rel_path: str, max_chars: int = 1500, overlap: int = 200
) -> list[Chunk]:
    """Split one markdown document into chunks with file/section metadata."""
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    title: str | None = None
    h2: str | None = None
    h3: str | None = None
    buf: list[str] = []
    sections: list[tuple[str, str]] = []  # (breadcrumb, body)

    def flush() -> None:
        if not buf:
            return
        body = "\n".join(buf).strip()
        buf.clear()
        if not body:
            return
        crumb = " > ".join(b for b in (title, h2, h3) if b) or rel_path
        sections.append((crumb, body))

    for raw in text.splitlines():
        m = _HEADER_RE.match(raw)
        if m:
            flush()
            if m.group(1) == "##":
                h2, h3 = m.group(2).strip(), None
            else:
                h3 = m.group(2).strip()
            buf.append(raw)
            continue
        mt = _TITLE_RE.match(raw)
        if mt and title is None:
            title = mt.group(1).strip()
        buf.append(raw)
    flush()

    chunks: list[Chunk] = []
    for i, (breadcrumb, body) in enumerate(sections):
        for j, piece in enumerate(_windows(body, max_chars, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{rel_path}#{i}.{j}",
                    file=rel_path,
                    section=breadcrumb,
                    text=piece,
                )
            )
    return chunks
