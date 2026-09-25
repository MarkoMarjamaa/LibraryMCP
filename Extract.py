"""Format-agnostic extraction contract and extractor registry.

`Page` and `Chunk` are deliberately generic: a "page" is any provenance unit
an extractor can name — a PDF page, a worksheet, a slide, a form-feed page of
a text file — and `section` carries whatever heading-like structure the format
provides. `page_start`/`page_end` are nullable in the schema precisely so
formats without pagination need no special casing downstream.

Chunking lives here rather than in Pdf.py because it only ever sees
`list[Page]`: every format inherits the same paragraph packing.

Chunk metadata convention (surfaced in `documents.extra`):
    {"kind": "pdf|docx|xlsx|ods|txt|md|pptx", "page_numbering": "real|breaks|none"}
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

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
# Headings: numbered sections, a short ALL-CAPS line, or a short Title Case
# line anchored on a digit ("Palaveri 1.2.2026", "Budetti 2017"). The digit
# anchor keeps ordinary prose sentences out of the Title Case arm.
_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+\S.{0,80}"
    r"|[A-Z][A-Z0-9 \-/]{3,60}"
    r"|[A-Z][\w\-ÔÄÖÅäöå]+(?:[ \-/][\w\-ÔÄÖÅäöå]+)*\s+\d[\d.]{1,14})$"
)


@dataclass(slots=True)
class Page:
    number: int          # 1-based ordinal within the document
    text: str
    section: str | None = None   # heading/sheet/slide title, if the format has one


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
    # marks. PDF generators scatter these through justified text; Office XML
    # adds them around smart quotes and autocorrect. They are invisible, they
    # carry no meaning for an embedding, and a tokenizer without byte fallback
    # throws on the ones it has no piece for.
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


# ---------------------------------------------------------------------------
# Extractor protocol and registry
# ---------------------------------------------------------------------------

class Extractor:
    """One extractor class per file family, registered by suffix."""

    suffixes: frozenset[str] = frozenset()
    kind: str = ""

    def extract_pages(self, path: Path) -> list[Page]: ...

    def page_count(self, path: Path) -> int | None:
        """Real page count when the format has one, else None."""
        return None

    def doc_properties(self, path: Path) -> dict[str, Any]:
        """Embedded title/created-date etc., for metadata resolution."""
        return {}


_REGISTRY: dict[str, Extractor] = {}


def register(extractor: Extractor) -> None:
    for suffix in extractor.suffixes:
        _REGISTRY[suffix] = extractor


def supported_suffixes() -> set[str]:
    return set(_REGISTRY)


def extract_for(path: Path) -> Extractor | None:
    return _REGISTRY.get(path.suffix.lower())


def is_legacy_office(path: Path) -> bool:
    """OLE Compound File (Word .doc / Excel .xls / PowerPoint .ppt).

    These need an external converter (LibreOffice, antiword); not implemented,
    but they must not be treated as unknown-garbage-and-fail-loudly either.
    """
    try:
        with path.open("rb") as fh:
            return fh.read(8) == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    except OSError:
        return False


# Shared helpers for the Office-XML extractors -------------------------------

# OOXML / ODF namespaces
_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_CP_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
_DC_NS = "http://purl.org/dc/elements/1.1/"
_DCTERMS_NS = "http://purl.org/dc/terms/"
_ODF_MANIFEST_NS = "urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"


def _core_properties(path: Path, part: str) -> dict[str, Any]:
    """Read dc:title / dcterms:created from a docProps/core.xml part."""
    out: dict[str, Any] = {}
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read(part)
    except (OSError, KeyError, zipfile.BadZipFile):
        return out
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError:
        return out
    title = root.findtext(f"{{{_DC_NS}}}title")
    if title and title.strip():
        out["title"] = title.strip()
    created = root.findtext(f"{{{_DCTERMS_NS}}}created")
    if created:
        try:
            out["created"] = dt.datetime.fromisoformat(
                created.strip().replace("Z", "+00:00")
            ).date()
        except ValueError:
            pass
    return out


def _mimetype(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.read("mimetype").decode("ascii", "replace").strip()
    except (OSError, KeyError, zipfile.BadZipFile):
        return ""


# ---------------------------------------------------------------------------
# Generic chunking (moved from Pdf.py; only ever sees list[Page])
# ---------------------------------------------------------------------------

def _page_section(page: Page) -> str | None:
    """Heading for a page: the extractor's own if it has one, else scan."""
    if page.section:
        return page.section
    first_line = page.text.split("\n", 1)[0].strip()
    if _HEADING.match(first_line) and len(first_line) < 90:
        return first_line
    return None


def _paragraphs(pages: list[Page]) -> list[tuple[str, int, str | None]]:
    """Flatten to (paragraph, page_number, current_section)."""
    out: list[tuple[str, int, str | None]] = []
    section: str | None = None
    for page in pages:
        page_heading = _page_section(page)
        if page_heading:
            section = page_heading
        paras = page.text.split("\n\n")
        for i, para in enumerate(paras):
            para = para.strip()
            if not para:
                continue
            # On pages without an extractor heading, fall back to the
            # original heuristic: a heading-styled first paragraph.
            if i == 0 and page_heading is None and _HEADING.match(para.split("\n", 1)[0].strip()):
                heading = para.split("\n", 1)[0].strip()
                if len(heading) < 90:
                    section = heading
            out.append((para, page.number, section))
    return out


def chunk_pages(pages: list[Page], config) -> list[Chunk]:
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


# ---------------------------------------------------------------------------
# Office document properties, available for every zipped format
# ---------------------------------------------------------------------------

def doc_properties(path: Path) -> dict[str, Any]:
    """Embedded title/created date for any Office-family file, best effort.

    OOXML keeps them in docProps/core.xml; ODF in meta.xml. Legacy .doc/.xls
    keep them in the OLE root entry, which we do not read.
    """
    suffix = path.suffix.lower()
    if suffix in (".docx", ".xlsx", ".pptx"):
        return _core_properties(path, "docProps/core.xml")
    if suffix == ".ods":
        return _odf_meta(path)
    return {}


def _odf_meta(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        with zipfile.ZipFile(path) as zf:
            root = ElementTree.fromstring(zf.read("meta.xml"))
    except (OSError, KeyError, zipfile.BadZipFile, ElementTree.ParseError):
        return out
    title = root.findtext(f"{{{_DC_NS}}}title")
    if title and title.strip():
        out["title"] = title.strip()
    created = root.findtext(f"{{{_DCTERMS_NS}}}date")
    if created:
        try:
            out["created"] = dt.datetime.strptime(
                created.strip()[:10], "%Y-%m-%d"
            ).date()
        except ValueError:
            pass
    return out
