"""PDF extraction. Chunking and the shared contract live in Extract.py.

Re-exports `Page`, `Chunk`, `clean_text`, `is_embeddable`, `file_hash` and
`chunk_pages` so existing imports keep working during the transition.
"""

from __future__ import annotations

from pathlib import Path

try:  # PyMuPDF renamed its import from `fitz` to `pymupdf`; `fitz` still
      # works but warns, and will eventually be removed.
    import pymupdf as fitz
except ImportError:  # PyMuPDF < 1.24.3
    import fitz

from Extract import (
    Chunk,
    Extractor,
    Page,
    chunk_pages,
    clean_text,
    file_hash,
    is_embeddable,
    register,
)


class PdfExtractor(Extractor):
    suffixes = frozenset({".pdf"})
    kind = "pdf"

    def extract_pages(self, path: Path) -> list[Page]:
        pages: list[Page] = []
        with fitz.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                text = clean_text(page.get_text("text") or "")
                if is_embeddable(text):
                    pages.append(Page(number=i, text=text))
        return pages

    def page_count(self, path: Path) -> int | None:
        with fitz.open(path) as doc:
            return doc.page_count


extractor = PdfExtractor()
register(extractor)


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
