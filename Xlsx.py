"""XLSX extraction via openpyxl.

One `Page` per worksheet (page number = sheet ordinal, sheet name kept in
`section`). There is no real pagination, so citations should say "sheet X".
Rows are linearised to `Header: value | Header: value` when the sheet's first
non-empty row looks like a header, else to `cell | cell` positional rows, so
every value keeps the words that name it — that pairing is what makes a budget
sheet retrievable.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl

from Extract import (
    Extractor,
    Page,
    clean_text,
    doc_properties,
    is_embeddable,
    register,
)

_MAX_COLS = 30          # absurdly wide sheets only waste embedding tokens
_MAX_CELL_LEN = 500     # one cell holding a whole essay is a document, not a cell


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _looks_like_header(row: list[str]) -> bool:
    filled = [c for c in row if c]
    if len(filled) < 2:
        return False
    # A header row is short labels, not numbers or prose.
    return all(len(c) <= 60 for c in filled) and not any(
        c.replace(",", ".").replace(" ", "").isdigit() for c in filled
    )


def _sheet_pages(ws, sheet_no: int) -> list[Page]:
    rows: list[list[str]] = []
    for row in ws.iter_rows(values_only=True):
        cells = [_fmt(v)[:_MAX_CELL_LEN] for v in row[:_MAX_COLS]]
        if any(cells):
            rows.append(cells)
    if not rows:
        return []

    header: list[str] | None = None
    body = rows
    if _looks_like_header(rows[0]):
        header = rows[0]
        body = rows[1:]
        # A header-only sheet still gets its header text indexed.
        if not body:
            text = " | ".join(c for c in header if c)
            return [Page(number=sheet_no, text=text, section=ws.title)]

    paragraphs: list[str] = []
    for cells in body:
        if header:
            pairs = [
                f"{h}: {v}" for h, v in zip(header, cells) if v and h
            ]
            line = " | ".join(pairs) if pairs else " | ".join(c for c in cells if c)
        else:
            line = " | ".join(c for c in cells if c)
        if line.strip():
            paragraphs.append(line)

    text = clean_text("\n".join(paragraphs))
    if not is_embeddable(text):
        return []
    # Chunking splits on blank lines; join rows with a blank line so one
    # enormous sheet does not become one giant "paragraph".
    text = clean_text("\n\n".join(paragraphs))
    return [Page(number=sheet_no, text=text, section=ws.title)]


class XlsxExtractor(Extractor):
    suffixes = frozenset({".xlsx", ".xlsm"})
    kind = "xlsx"

    def extract_pages(self, path: Path) -> list[Page]:
        try:
            wb = openpyxl.load_workbook(
                path, read_only=True, data_only=True, keep_vba=False
            )
        except Exception as exc:
            raise ValueError(f"not a readable .xlsx: {exc}") from exc
        try:
            pages: list[Page] = []
            for i, ws in enumerate(wb.worksheets, start=1):
                if ws.sheet_state != "visible":
                    continue  # hidden sheets are usually scratch data
                pages.extend(_sheet_pages(ws, len(pages) + 1))
            return pages
        finally:
            wb.close()

    def doc_properties(self, path: Path) -> dict:
        return doc_properties(path)


extractor = XlsxExtractor()
register(extractor)
