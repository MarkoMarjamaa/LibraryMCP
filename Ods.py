"""ODS extraction (zip + content.xml; no third-party dependency).

Same model as Xlsx: one Page per visible table, table name in `section`,
rows linearised with header pairing when the first row looks like a header.
"""

from __future__ import annotations

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

_TABLE = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}table"
_NESTED = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}nested-table"
_ROW = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}table-row"
_CELL = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}table-cell"
_PROTECTED = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}protected-rows"
_STYLE = "{urn:oasis:names:tc:opendocument:xmlns:style:1.0}family"
_TEXT_P = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}p"
_MAX_COLS = 30


def _cell_text(cell: ElementTree.Element) -> str:
    parts = [p.itertext() for p in cell.findall(_TEXT_P)]
    return " ".join(" ".join(t for t in it).strip() for it in parts).strip()


def _repeat(el: ElementTree.Element, attr: str, default: int = 1) -> int:
    try:
        return max(1, int(el.get(attr, default)))
    except (TypeError, ValueError):
        return default


def _rows(table: ElementTree.Element) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table.findall(_ROW):
        if row.find(_NESTED) is not None:
            continue  # chart/annotation payloads, not data
        cells: list[str] = []
        for cell in row.findall(_CELL):
            # Covered cells (merged spans) are absent from the XML; expand
            # number-columns-repeated but cap it so a styled-to-hell sheet
            # with 16384 empty columns does not explode.
            rep = min(_repeat(cell, "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}number-columns-repeated"), _MAX_COLS)
            text = _cell_text(cell)
            cells.extend([text] * rep)
            if len(cells) >= _MAX_COLS:
                break
        rows.append(cells[:_MAX_COLS])
    # Drop trailing all-empty rows (ODS pads dimensions generously).
    while rows and not any(c.strip() for c in rows[-1]):
        rows.pop()
    return [r for r in rows if any(c.strip() for c in r)]


def _looks_like_header(row: list[str]) -> bool:
    filled = [c for c in row if c.strip()]
    if len(filled) < 2:
        return False
    return all(len(c) <= 60 for c in filled) and not any(
        c.replace(",", ".").replace(" ", "").isdigit() for c in filled
    )


def _table_to_text(rows: list[list[str]]) -> str:
    header: list[str] | None = None
    body = rows
    if _looks_like_header(rows[0]):
        header = rows[0]
        body = rows[1:]
        if not body:
            return " | ".join(c for c in header if c.strip())

    lines: list[str] = []
    for cells in body:
        if header:
            pairs = [f"{h}: {v}" for h, v in zip(header, cells) if v.strip() and h.strip()]
            line = " | ".join(pairs) if pairs else " | ".join(c for c in cells if c.strip())
        else:
            line = " | ".join(c for c in cells if c.strip())
        if line.strip():
            lines.append(line)
    return "\n\n".join(lines)


class OdsExtractor(Extractor):
    suffixes = frozenset({".ods"})
    kind = "ods"

    def extract_pages(self, path: Path) -> list[Page]:
        try:
            with zipfile.ZipFile(path) as zf:
                root = ElementTree.fromstring(zf.read("content.xml"))
        except (OSError, KeyError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
            raise ValueError(f"not a readable .ods: {exc}") from exc

        pages: list[Page] = []
        # Spreadsheet tables live under office:spreadsheet; drawing pages do
        # not — findall on the whole tree is fine and skips master pages.
        for table in root.iter(_TABLE):
            style = table.get("{urn:oasis:names:tc:opendocument:xmlns:style:1.0}family", "")
            if style and style != "table":
                continue
            if table.find(_PROTECTED) is not None:
                continue  # database-range artifacts have no content of interest
            name = table.get(
                "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}name", f"Sheet {len(pages) + 1}"
            )
            rows = _rows(table)
            if not rows:
                continue
            text = clean_text(_table_to_text(rows))
            if is_embeddable(text):
                pages.append(Page(number=len(pages) + 1, text=text, section=name))
        return pages

    def doc_properties(self, path: Path) -> dict:
        return doc_properties(path)


extractor = OdsExtractor()
register(extractor)
