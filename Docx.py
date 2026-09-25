"""DOCX extraction (zip + word/document.xml; no third-party dependency).

Word has no reliable fixed pagination in the file itself, so a "page" here is
either the whole document or the segments between explicit page breaks
(`<w:br w:type="page"/>` / lastRenderedPageBreak). `page_numbering` in the
chunk metadata tells the citation layer which case applies; when it is "none",
citations should quote `section` instead of a page number.

Headings come from the paragraph style name (`Heading1`, `Otsikko 1`, ...),
which is why style matching is language-agnostic: it looks for the leading
digits after the localized prefix.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from Extract import (
    Extractor,
    Page,
    clean_text,
    doc_properties,
    is_embeddable,
    register,
)

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# "Heading 2", "heading2", "Otsikko 2", " heading 2 " -> 2
_STYLE_HEADING = re.compile(r"^(?:heading|otsikko)\s*(\d)$", re.IGNORECASE)


def _para_style(p: ElementTree.Element) -> str:
    val = p.find(f"{_W}pPr/{_W}pStyle")
    return val.get(f"{_W}val", "") if val is not None else ""


def _para_text(p: ElementTree.Element) -> str:
    # w:tab -> space, w:br -> newline (soft breaks are not paragraph breaks),
    # everything else concatenated; w:instrText (field codes) is skipped by
    # only collecting w:t.
    parts: list[str] = []
    for node in p.iter():
        tag = node.tag
        if tag == f"{_W}t":
            parts.append(node.text or "")
        elif tag == f"{_W}tab":
            parts.append(" ")
        elif tag == f"{_W}br":
            parts.append("\n")
    return "".join(parts).strip()


def _cell_text(tc: ElementTree.Element) -> str:
    texts = (_para_text(p) for p in tc.iter(f"{_W}p"))
    return " ".join(t for t in texts if t)


def _table_text(tbl: ElementTree.Element) -> str:
    rows: list[str] = []
    for tr in tbl.findall(f"{_W}tr"):
        cells = [_cell_text(tc) for tc in tr.findall(f"{_W}tc")]
        row = " | ".join(c for c in cells if c)
        if row:
            rows.append(row)
    return "\n".join(rows)


class DocxExtractor(Extractor):
    suffixes = frozenset({".docx"})
    kind = "docx"

    def extract_pages(self, path: Path) -> list[Page]:
        try:
            with zipfile.ZipFile(path) as zf:
                xml = zf.read("word/document.xml")
        except (OSError, KeyError, zipfile.BadZipFile) as exc:
            raise ValueError(f"not a readable .docx: {exc}") from exc
        try:
            body = ElementTree.fromstring(xml).find(f"{_W}body")
        except ElementTree.ParseError as exc:
            raise ValueError(f"corrupt word/document.xml: {exc}") from exc
        if body is None:
            return []

        # Block items in document order: paragraphs and tables alternate.
        items: list[tuple[str, str, int]] = []  # (kind, text, break_here)
        for el in body:
            if el.tag == f"{_W}p":
                style = _para_style(el)
                text = _para_text(el)
                if not text:
                    # Empty paragraphs still carry page breaks.
                    if self._has_page_break(el):
                        items.append(("break", "", 1))
                    continue
                heading = _STYLE_HEADING.match(style.strip())
                kind = "heading" if heading else "para"
                breaks = 1 if self._has_page_break(el) else 0
                items.append((kind, text, breaks))
            elif el.tag == f"{_W}tbl":
                text = _table_text(el)
                if text:
                    items.append(("table", text, 0))

        return self._paginate(items)

    @staticmethod
    def _has_page_break(p: ElementTree.Element) -> bool:
        for br in p.iter(f"{_W}br"):
            if br.get(f"{_W}type") == "page":
                return True
        # Word's repagination marker; treat it as authoritative when present.
        for br in p.iter(f"{_W}lastRenderedPageBreak"):
            return True
        return False

    def _paginate(self, items: list[tuple[str, str, int]]) -> list[Page]:
        pages: list[Page] = []
        buf: list[str] = []
        section: str | None = None
        page_no = 1

        def flush() -> None:
            nonlocal buf, page_no
            if not buf:
                return
            text = clean_text("\n\n".join(buf))
            if is_embeddable(text):
                pages.append(Page(number=page_no, text=text, section=section))
                page_no += 1
            buf = []

        # A document that never mentions page breaks stays one synthetic page.
        has_breaks = any(b for _, _, b in items)
        for kind, text, break_after in items:
            if kind == "break":
                flush()
                continue
            if kind == "heading":
                flush()  # a new heading starts a new section
                section = text[:200]
            buf.append(text)
            if break_after and has_breaks:
                flush()
        flush()
        return pages

    def doc_properties(self, path: Path) -> dict:
        return doc_properties(path)


extractor = DocxExtractor()
register(extractor)
