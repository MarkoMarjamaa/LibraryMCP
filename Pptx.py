"""PPTX extraction via python-pptx.

One `Page` per slide (slide ordinal as page number, slide title in `section`).
Speaker notes are appended to the slide's text — they are usually the part
that actually answers "what was said". Tables are linearised row by row like
in the spreadsheet extractors.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation

from Extract import (
    Extractor,
    Page,
    clean_text,
    doc_properties,
    is_embeddable,
    register,
)


def _shape_text(shape) -> str:
    if shape.shape_type == 19 or getattr(shape, "has_table", False):  # MSO_SHAPE_TYPE.TABLE
        rows: list[str] = []
        for row in shape.table.rows:
            cells = [(c.text or "").strip() for c in row.cells]
            line = " | ".join(c for c in cells if c)
            if line:
                rows.append(line)
        return "\n".join(rows)
    if getattr(shape, "has_text_frame", False):
        return shape.text_frame.text.strip()
    return ""


class PptxExtractor(Extractor):
    suffixes = frozenset({".pptx"})
    kind = "pptx"

    def extract_pages(self, path: Path) -> list[Page]:
        try:
            prs = Presentation(str(path))
        except Exception as exc:
            raise ValueError(f"not a readable .pptx: {exc}") from exc

        pages: list[Page] = []
        for slide in prs.slides:
            title: str | None = None
            try:
                if slide.shapes.title is not None:
                    title = (slide.shapes.title.text or "").strip() or None
            except (AttributeError, NotImplementedError):
                title = None

            blocks: list[str] = []
            for shape in slide.shapes:
                # The title shape is emitted once, as `section` context;
                # include it in text too so the chunk itself is self-describing.
                text = _shape_text(shape)
                if text:
                    blocks.append(text)

            if slide.has_notes_slide:
                notes = (slide.notes_slide.notes_text_frame.text or "").strip()
                if notes:
                    blocks.append(notes)

            text = clean_text("\n\n".join(blocks))
            if is_embeddable(text):
                pages.append(
                    Page(number=len(pages) + 1, text=text, section=title)
                )
        return pages

    def doc_properties(self, path: Path) -> dict:
        return doc_properties(path)


extractor = PptxExtractor()
register(extractor)
