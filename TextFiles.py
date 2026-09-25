"""Plain text and Markdown extraction.

A "page" is a form-feed-delimited section (`\\f`, the classic typewriter page
break that plain text kept); files without form feeds are one synthetic page.
Markdown ATX headings (`#`, `##`) are stripped to bare text so the shared
heading heuristic in Extract.py recognises them as sections.
"""

from __future__ import annotations

import re
from pathlib import Path

from Extract import (
    Extractor,
    Page,
    clean_text,
    is_embeddable,
    register,
)

# Strip leading '#', '##', ... and trailing '#', keep the heading text.
_ATX = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# Fenced code blocks carry little embedding value and pollute the FTS index
# with syntax tokens; the fence markers themselves would index as noise.
_FENCE = re.compile(r"^\s*(```|~~~)")


def _decode(path: Path) -> str:
    raw = path.read_bytes()
    # utf-8-sig swallows a BOM; latin-1 never fails, so the fallback is total.
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", "replace")


class TextExtractor(Extractor):
    suffixes = frozenset({".txt", ".md", ".markdown"})
    kind = "text"

    def extract_pages(self, path: Path) -> list[Page]:
        if path.suffix.lower() in (".md", ".markdown"):
            return self._markdown_pages(path)
        return self._plain_pages(path)

    def _plain_pages(self, path: Path) -> list[Page]:
        pages: list[Page] = []
        for i, segment in enumerate(_decode(path).split("\f"), start=1):
            text = clean_text(segment)
            if is_embeddable(text):
                pages.append(Page(number=len(pages) + 1, text=text))
        return pages

    def _markdown_pages(self, path: Path) -> list[Page]:
        # Form feeds first, same as plain text; headings are not page breaks.
        pages: list[Page] = []
        for segment in _decode(path).split("\f"):
            lines: list[str] = []
            in_fence = False
            for line in segment.splitlines():
                if _FENCE.match(line):
                    in_fence = not in_fence
                    continue
                if in_fence:
                    continue
                m = _ATX.match(line)
                lines.append(m.group(2) if m else line)
            text = clean_text("\n".join(lines))
            if is_embeddable(text):
                pages.append(Page(number=len(pages) + 1, text=text))
        return pages


extractor = TextExtractor()
register(extractor)
