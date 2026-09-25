"""Per-shelf-type metadata resolution.

Each shelf type answers the same questions differently: what is this document
called, when is it from, and what one-paragraph summary should be embedded for
document-level search.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import httpx

from Config import Shelf
import Extract as extractmod
import Pdf as pdfmod

log = logging.getLogger(__name__)

# 2503.01234, 2503.01234v2, or the pre-2007 form math.GT/0309136
ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})")
DOI_RE = re.compile(r"(10\.\d{4,9}[/_.][^\s]+)")

DATE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(\d{4})-(\d{2})-(\d{2})"), "ymd"),
    (re.compile(r"^(\d{4})_(\d{2})_(\d{2})"), "ymd"),
    (re.compile(r"^(\d{4})(\d{2})(\d{2})$"), "ymd"),
    (re.compile(r"^(\d{2})\.(\d{2})\.(\d{4})"), "dmy"),   # 14.03.2026, common in FI
    (re.compile(r"^(\d{4})[-_](\d{2})$"), "ym"),
    # 170630 -> 2017-06-30: yymmdd as a *prefix*, the way Finnish meeting
    # notes are often named (170630putkiremonttikatsaus).
    (re.compile(r"^(\d{2})(\d{2})(\d{2})(?=\D|$)"), "yymmdd"),
)


@dataclass(slots=True)
class DocumentMeta:
    title: str | None = None
    authors: list[str] | None = None
    doc_date: dt.date | None = None
    year: int | None = None
    summary: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def parse_dir_date(name: str) -> dt.date | None:
    """Directory or file name to date. 'YYYY-MM' returns the first of the month."""
    for pattern, order in DATE_PATTERNS:
        m = pattern.match(name.strip())
        if not m:
            continue
        try:
            if order == "ymd":
                return dt.date(int(m[1]), int(m[2]), int(m[3]))
            if order == "dmy":
                return dt.date(int(m[3]), int(m[2]), int(m[1]))
            if order == "ym":
                return dt.date(int(m[1]), int(m[2]), 1)
            if order == "yymmdd":
                # Two-digit years: 70-99 are 19xx, 00-69 are 20xx. A 19xx
                # meeting note is unlikely, but 991231 should not become 2099.
                yy = int(m[1])
                year = 1900 + yy if yy >= 70 else 2000 + yy
                return dt.date(year, int(m[2]), int(m[3]))
        except ValueError:
            return None
    return None


async def resolve(
    path: Path,
    shelf: Shelf,
    source_dir: str,
    pages: list[pdfmod.Page],
    client: httpx.AsyncClient | None,
) -> DocumentMeta:
    match shelf.type:
        case "manuals":
            return _resolve_manual(path, source_dir, pages)
        case "meetings":
            return _resolve_meeting(path, source_dir, pages)
        case "scientific":
            return await _resolve_paper(path, pages, client)
        case _:
            return DocumentMeta(title=path.stem)


def _first_text(pages: list[pdfmod.Page], limit: int = 1200) -> str:
    return " ".join(p.text for p in pages[:2])[:limit].strip()


def _resolve_manual(path: Path, source_dir: str, pages: list[pdfmod.Page]) -> DocumentMeta:
    # The directory is the device name and model; that is the useful title.
    # The filename usually only carries a revision.
    return DocumentMeta(
        title=source_dir,
        summary=f"{source_dir} device manual. {_first_text(pages, 800)}",
        extra={"revision": path.stem},
    )


def _resolve_meeting(path: Path, source_dir: str, pages: list[pdfmod.Page]) -> DocumentMeta:
    date = parse_dir_date(source_dir) or parse_dir_date(path.stem)
    # Office files carry their own title/created date; trust them over the
    # filename when the filename says nothing (no parseable date, generic stem).
    props = extractmod.doc_properties(path)
    title = _clean_stem(path.stem)
    if date is None and isinstance(props.get("created"), dt.date):
        date = props["created"]
    if date:
        title = f"{title} ({date.isoformat()})"
    if props.get("title") and not parse_dir_date(path.stem):
        title = props["title"]
    return DocumentMeta(
        title=title,
        doc_date=date,
        year=date.year if date else None,
        # Replace with an LLM-generated summary if you want better recall on
        # vague queries like "what did we decide about the budget".
        summary=_first_text(pages, 1200),
        extra={"doc_title": props["title"]} if props.get("title") else {},
    )


def _clean_stem(stem: str) -> str:
    return re.sub(r"[_\-]+", " ", stem).strip().title()


async def _resolve_paper(
    path: Path, pages: list[pdfmod.Page], client: httpx.AsyncClient | None
) -> DocumentMeta:
    """Papers are filed by ID, so look the metadata up rather than parse it.

    An authoritative title and abstract from arXiv or Crossref beats anything
    extracted from a preprint's first page, which is full of watermarks,
    submission banners and multi-line titles.
    """
    stem = path.stem
    meta: DocumentMeta | None = None

    if client is not None:
        if (m := DOI_RE.search(stem.replace("_", "/"))):
            meta = await _crossref(m.group(1), client)
        if meta is None and (m := ARXIV_RE.search(stem)):
            meta = await _arxiv(m.group(1), client)

    if meta is None:
        title = pdfmod.extract_title_by_font(path) or _clean_stem(stem)
        abstract = _extract_abstract(pages)
        meta = DocumentMeta(
            title=title,
            summary=abstract or _first_text(pages, 1200),
            extra={"metadata_source": "extracted"},
        )

    if meta.summary:
        meta.summary = f"{meta.title}. {meta.summary}"
    return meta


async def _arxiv(arxiv_id: str, client: httpx.AsyncClient) -> DocumentMeta | None:
    try:
        resp = await client.get(
            "https://export.arxiv.org/api/query",
            params={"id_list": arxiv_id, "max_results": 1},
        )
        resp.raise_for_status()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entry = ElementTree.fromstring(resp.text).find("a:entry", ns)
        if entry is None:
            return None
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        published = entry.findtext("a:published", default="", namespaces=ns) or ""
        authors = [
            (a.findtext("a:name", default="", namespaces=ns) or "").strip()
            for a in entry.findall("a:author", ns)
        ]
        date = None
        if published:
            date = dt.datetime.fromisoformat(published.replace("Z", "+00:00")).date()
        return DocumentMeta(
            title=re.sub(r"\s+", " ", title),
            authors=[a for a in authors if a] or None,
            doc_date=date,
            year=date.year if date else None,
            summary=re.sub(r"\s+", " ", summary),
            extra={"arxiv_id": arxiv_id, "metadata_source": "arxiv"},
        )
    except Exception as exc:
        log.warning("arXiv lookup failed for %s: %s", arxiv_id, exc)
        return None


async def _crossref(doi: str, client: httpx.AsyncClient) -> DocumentMeta | None:
    try:
        resp = await client.get(f"https://api.crossref.org/works/{doi}")
        resp.raise_for_status()
        item = resp.json()["message"]
        title = (item.get("title") or [""])[0]
        authors = [
            " ".join(filter(None, [a.get("given"), a.get("family")]))
            for a in item.get("author", [])
        ]
        parts = (item.get("issued", {}).get("date-parts") or [[None]])[0]
        year = parts[0] if parts and parts[0] else None
        date = None
        if year:
            try:
                date = dt.date(year, parts[1] if len(parts) > 1 else 1,
                               parts[2] if len(parts) > 2 else 1)
            except (ValueError, IndexError):
                date = dt.date(year, 1, 1)
        abstract = re.sub(r"<[^>]+>", "", item.get("abstract", "") or "").strip()
        return DocumentMeta(
            title=re.sub(r"\s+", " ", title) or None,
            authors=[a for a in authors if a] or None,
            doc_date=date,
            year=year,
            summary=abstract or None,
            extra={"doi": doi, "metadata_source": "crossref"},
        )
    except Exception as exc:
        log.warning("Crossref lookup failed for %s: %s", doi, exc)
        return None


def _extract_abstract(pages: list[pdfmod.Page]) -> str | None:
    if not pages:
        return None
    text = pages[0].text
    m = re.search(r"\bAbstract\b[:.\s]*(.{100,2500}?)(?:\n\s*\n|\bIntroduction\b|\b1\.?\s+Introduction\b)",
                  text, re.IGNORECASE | re.DOTALL)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else None

