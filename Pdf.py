"""PDF extraction and chunking.

Chunks are retrieval units sized for the embedding model, not pages. Page
numbers are recorded as provenance so a citation can say "page 12" or
"pages 6-7" when a chunk straddles a break.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

try:  # PyMuPDF renamed its import from `fitz` to `pymupdf`; `fitz` still
      # works but warns, and will eventually be removed.
    import pymupdf as fitz
except ImportError:  # PyMuPDF < 1.24.3
    import fitz

from Config import ChunkingConfig

# Rough conversion. A real tokenizer would be more accurate but adds a heavy
# dependency to both processes; 4 chars/token is close enough for bge-m3 on
# European languages, and chunk size is not a precision-critical parameter.
CHARS_PER_TOKEN = 4

_WS = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")

# C0/C1 control characters other than tab and newline. PyMuPDF emits \x0c at
# page breaks as a matter of course, and NUL bytes turn up in text extracted
# from PDFs with broken embedded font encodings. Both make llama.cpp's
# tokenizer throw std::out_of_range, which surfaces as an opaque
# "_Map_base::at" 500.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
# Unicode replacement char and PDF private-use glyphs from broken CMaps.
_JUNK = re.compile(r"[\ufffd\ue000-\uf8ff]")
# Headings: numbered sections, or a short ALL-CAPS / Title Case line.
_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+\S.{0,80}|[A-Z][A-Z0-9 \-/]{3,60})$"
)


@dataclass(slots=True)
class Page:
    number: int          # 1-based
    text: str


@dataclass(slots=True)
class Chunk:
    chunk_index: int
    content: str
    page_start: int
    page_end: int
    section: str | None
    token_count: int


def clean_text(text: str) -> str:
    """Strip anything that will not survive tokenisation.

    Applied at extraction so no downstream stage has to think about it.
    """
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub(" ", text)
    text = _JUNK.sub("", text)
    # Format characters: zero-width space, BOM, soft hyphen, directional
    # marks. PDF generators scatter these through justified text. They are
    # invisible, they carry no meaning for an embedding, and a tokenizer
    # without byte fallback throws on the ones it has no piece for.
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    text = _WS.sub(" ", text)
    return _MULTI_NL.sub("\n\n", text).strip()


def is_embeddable(text: str) -> bool:
    """Whether a string will produce at least one real token.

    Whitespace-only input normalises away to nothing, and a server asked to
    embed nothing has no result to return.
    """
    return bool(text) and any(ch.isalnum() for ch in text)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract_pages(path: Path) -> list[Page]:
    pages: list[Page] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text") or "")
            if is_embeddable(text):
                pages.append(Page(number=i, text=text))
    return pages


def page_count(path: Path) -> int:
    with fitz.open(path) as doc:
        return doc.page_count


def extract_title_by_font(path: Path) -> str | None:
    """Largest-font text block on page 1.

    Fallback for scientific papers when the network lookup fails. Works
    surprisingly well on preprints, less well on scanned documents.
    """
    try:
        with fitz.open(path) as doc:
            if doc.page_count == 0:
                return None
            blocks = doc[0].get_text("dict")["blocks"]
    except Exception:
        return None

    best_size = 0.0
    best_text = ""
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                size = span.get("size", 0.0)
                text = span.get("text", "").strip()
                if not text or len(text) < 6:
                    continue
                if size > best_size:
                    best_size, best_text = size, text
                elif abs(size - best_size) < 0.5 and best_text:
                    # Same visual size: likely a wrapped title line.
                    best_text = f"{best_text} {text}"

    best_text = best_text.strip()
    return best_text if 6 <= len(best_text) <= 300 else None


def _paragraphs(pages: list[Page]) -> list[tuple[str, int, str | None]]:
    """Flatten to (paragraph, page_number, current_section)."""
    out: list[tuple[str, int, str | None]] = []
    section: str | None = None
    for page in pages:
        for para in page.text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            first_line = para.split("\n", 1)[0].strip()
            if _HEADING.match(first_line) and len(first_line) < 90:
                section = first_line
            out.append((para, page.number, section))
    return out


def chunk_pages(pages: list[Page], config: ChunkingConfig) -> list[Chunk]:
    """Greedy paragraph packing with overlap.

    Paragraphs accumulate until the target size is reached. A chunk carries the
    page of its first and last paragraph, so page_end > page_start exactly when
    the chunk crosses a page boundary.
    """
    target = config.target_tokens * CHARS_PER_TOKEN
    overlap = config.overlap_tokens * CHARS_PER_TOKEN
    minimum = config.min_tokens * CHARS_PER_TOKEN

    paras = _paragraphs(pages)
    chunks: list[Chunk] = []
    buf: list[tuple[str, int, str | None]] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        content = clean_text("\n\n".join(p for p, _, _ in buf))
        if len(content) >= minimum and is_embeddable(content):
            chunks.append(
                Chunk(
                    chunk_index=len(chunks),
                    content=content,
                    page_start=buf[0][1],
                    page_end=buf[-1][1],
                    section=next((s for _, _, s in buf if s), None),
                    token_count=len(content) // CHARS_PER_TOKEN,
                )
            )
        # Carry the tail forward so a procedure split across chunks is not lost.
        tail: list[tuple[str, int, str | None]] = []
        tail_len = 0
        for item in reversed(buf):
            if tail_len >= overlap:
                break
            tail.insert(0, item)
            tail_len += len(item[0])
        buf = tail if tail_len < target else []
        buf_len = sum(len(p) for p, _, _ in buf)

    for para, page_no, section in paras:
        # A single oversized paragraph (a spec table, usually) becomes its own
        # chunk rather than dragging the whole buffer over target.
        if len(para) > target * 1.5:
            flush()
            buf = []
            buf_len = 0
            for piece in _split_long(para, target):
                if not is_embeddable(piece):
                    continue
                chunks.append(
                    Chunk(
                        chunk_index=len(chunks),
                        content=piece,
                        page_start=page_no,
                        page_end=page_no,
                        section=section,
                        token_count=len(piece) // CHARS_PER_TOKEN,
                    )
                )
            continue

        buf.append((para, page_no, section))
        buf_len += len(para)
        if buf_len >= target:
            flush()

    flush()
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i
    return chunks


def _split_long(text: str, target: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    cur = ""
    for sentence in sentences:
        if cur and len(cur) + len(sentence) > target:
            out.append(cur.strip())
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    if cur:
        out.append(cur.strip())
    return out or [text[:target]]
