"""MCP server exposing the library over streamable-http.

Tools are named per shelf type rather than exposed as one generic search, so
the assistant picks the right scope from the tool name alone and the
parameters that make sense for that scope (device vs. date vs. author) are the
only ones on offer.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import logging
import sys
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncIterator

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from Config import Config, load_config
from Db import Database, SearchFilters
from Embed import Embedder, query_text_for

log = logging.getLogger("library.server")

_config: Config
_db: Database
_embedder: Embedder


@asynccontextmanager
async def lifespan(_server: FastMCP) -> AsyncIterator[None]:
    await _db.connect()
    await _db.sync_shelves(_config.shelves)
    log.info("Connected; shelves: %s", ", ".join(s.name for s in _config.shelves))
    try:
        yield
    finally:
        await _embedder.close()
        await _db.close()


mcp = FastMCP("library", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Result shaping
# ---------------------------------------------------------------------------

def _citation(row: dict[str, Any]) -> dict[str, Any]:
    """Structured citation, not a pre-formatted string.

    The speech layer decides how to say it: "page 12 of the Shelly Plus 2PM
    manual" is spoken, while the filename stays available for a screen or a
    follow-up without being read aloud.
    """
    page_start, page_end = row.get("page_start"), row.get("page_end")
    if page_start and page_end and page_end != page_start:
        pages: str | None = f"{page_start}-{page_end}"
    elif page_start:
        pages = str(page_start)
    else:
        pages = None

    citation = {
        "document_id": row["document_id"],
        "shelf": row["shelf"],
        "title": row.get("title"),
        "filename": row.get("filename"),
        "pages": pages,
    }
    if row.get("section"):
        citation["section"] = row["section"]
    if row.get("doc_date"):
        citation["date"] = row["doc_date"].isoformat()
    if row.get("authors"):
        citation["authors"] = row["authors"][:5]
    return citation


def _hits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "results": [
            {
                "chunk_id": r["chunk_id"],
                "text": r["content"],
                "score": round(float(r["score"]), 5),
                "citation": _citation(r),
            }
            for r in rows
        ],
    }


def _parse_dates(values: list[str] | None) -> list[dt.date] | None:
    if not values:
        return None
    return [dt.date.fromisoformat(v) for v in values]


def _parse_date(value: str | None) -> dt.date | None:
    return dt.date.fromisoformat(value) if value else None


async def _search(query: str, filters: SearchFilters, limit: int) -> dict[str, Any]:
    vector = await _embedder.embed_one(
        query_text_for(query, _config.embedding.model)
    )
    rows = await _db.hybrid_search(query, vector, filters, limit=limit)
    return _hits(rows)


def _shelf_names(type_: str) -> list[str]:
    return [s.name for s in _config.shelves_of_type(type_)]


# ---------------------------------------------------------------------------
# Orientation tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_shelves() -> dict[str, Any]:
    """List every shelf, its type, and how many documents it holds.

    Call this when unsure which shelf a question belongs to.
    """
    return {"shelves": await _db.list_shelves()}


@mcp.tool()
async def list_documents(
    shelf: Annotated[str | None, Field(description="Shelf name, e.g. 'manuals' or 'work_meetings'")] = None,
    date_from: Annotated[str | None, Field(description="ISO date, inclusive lower bound")] = None,
    date_to: Annotated[str | None, Field(description="ISO date, inclusive upper bound")] = None,
    dates: Annotated[list[str] | None, Field(description="Exact ISO dates, e.g. ['2026-03-14','2026-04-27']")] = None,
    limit: int = 200,
) -> dict[str, Any]:
    """List documents on a shelf, so you can pick specific ones to search.

    Suited to small and medium shelves (manuals, meetings). For large shelves
    such as papers, use find_documents instead.
    """
    filters = SearchFilters(
        shelves=[shelf] if shelf else None,
        date_from=_parse_date(date_from),
        date_to=_parse_date(date_to),
        dates=_parse_dates(dates),
    )
    docs = await _db.list_documents(filters, limit=limit)
    return {
        "count": len(docs),
        "documents": [
            {
                "document_id": d["id"],
                "shelf": d["shelf"],
                "title": d["title"],
                "folder": d["source_dir"],
                "date": d["doc_date"].isoformat() if d["doc_date"] else None,
                "pages": d["page_count"],
            }
            for d in docs
        ],
    }


@mcp.tool()
async def find_documents(
    query: Annotated[str, Field(description="What the document is about")],
    shelf: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Find documents by topic, searching titles and summaries rather than body text.

    Use on large shelves where listing everything is impractical, then pass the
    returned document_ids to search_papers or search_meetings to search inside
    them.
    """
    vector = await _embedder.embed_one(query_text_for(query, _config.embedding.model))
    filters = SearchFilters(shelves=[shelf] if shelf else None)
    docs = await _db.find_documents(vector, filters, limit=limit)
    return {
        "count": len(docs),
        "documents": [
            {
                "document_id": d["id"],
                "shelf": d["shelf"],
                "title": d["title"],
                "authors": d["authors"],
                "year": d["year"],
                "date": d["doc_date"].isoformat() if d["doc_date"] else None,
                "summary": d["summary_preview"],
                "score": round(float(d["score"]), 4),
            }
            for d in docs
        ],
    }


# ---------------------------------------------------------------------------
# Retrieval tools, one per shelf type
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_manuals(
    query: Annotated[str, Field(description="What you need to do, e.g. 'factory reset'")],
    device: Annotated[str | None, Field(description="Device name or model to narrow to, e.g. 'Shelly Plus 2PM'")] = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Search device manuals for procedures, settings, error codes and specifications.

    Pass `device` whenever it is known — manuals for different devices describe
    near-identical procedures, and without the filter the wrong device's
    instructions can outrank the right one.
    """
    return await _search(
        query,
        SearchFilters(shelves=_shelf_names("manuals"), source_dir=device),
        limit,
    )


@mcp.tool()
async def search_meetings(
    query: Annotated[str, Field(description="Topic, decision or discussion to find")],
    shelf: Annotated[str | None, Field(description="Restrict to one meeting shelf, e.g. 'condo_meetings'")] = None,
    date_from: Annotated[str | None, Field(description="ISO date, inclusive lower bound")] = None,
    date_to: Annotated[str | None, Field(description="ISO date, inclusive upper bound")] = None,
    dates: Annotated[list[str] | None, Field(description="Exact meeting dates, ISO format")] = None,
    document_ids: Annotated[list[int] | None, Field(description="Restrict to documents from a previous list_documents call")] = None,
    limit: int = 6,
) -> dict[str, Any]:
    """Search meeting documents. Filter by date when the question mentions a time period.

    'March 2026' becomes date_from='2026-03-01' with date_to='2026-03-31'.
    Named specific meetings become `dates`.
    """
    shelves = [shelf] if shelf else _shelf_names("meetings")
    return await _search(
        query,
        SearchFilters(
            shelves=shelves,
            document_ids=document_ids,
            date_from=_parse_date(date_from),
            date_to=_parse_date(date_to),
            dates=_parse_dates(dates),
        ),
        limit,
    )


@mcp.tool()
async def search_papers(
    query: Annotated[str, Field(description="Topic, method or finding to search for")],
    document_ids: Annotated[list[int] | None, Field(description="Restrict to papers from a previous find_documents call")] = None,
    author: Annotated[str | None, Field(description="Author surname to filter by")] = None,
    year_from: int | None = None,
    year_to: int | None = None,
    limit: int = 6,
) -> dict[str, Any]:
    """Search the full text of scientific papers.

    For broad topic questions, call find_documents first to identify relevant
    papers by abstract, then pass their document_ids here.
    """
    return await _search(
        query,
        SearchFilters(
            shelves=_shelf_names("scientific"),
            document_ids=document_ids,
            author=author,
            year_from=year_from,
            year_to=year_to,
        ),
        limit,
    )


# ---------------------------------------------------------------------------
# Follow-up tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_context(
    chunk_id: Annotated[int, Field(description="chunk_id from a previous search result")],
    before: int = 1,
    after: int = 1,
) -> dict[str, Any]:
    """Fetch the passages surrounding a search result.

    Use when a retrieved procedure appears to start or stop mid-way.
    """
    rows = await _db.get_chunk_window(chunk_id, before=before, after=after)
    return {
        "passages": [
            {
                "chunk_id": r["chunk_id"],
                "text": r["content"],
                "section": r["section"],
                "pages": r["page_start"],
            }
            for r in rows
        ]
    }


@mcp.tool()
async def get_document_info(document_id: int) -> dict[str, Any]:
    """Full metadata for one document: title, authors, date, summary, page count."""
    doc = await _db.get_document(document_id)
    if doc is None:
        return {"error": f"No document with id {document_id}"}
    return {
        "document_id": doc["id"],
        "shelf": doc["shelf"],
        "title": doc["title"],
        "authors": doc["authors"],
        "date": doc["doc_date"].isoformat() if doc["doc_date"] else None,
        "year": doc["year"],
        "filename": doc["filename"],
        "folder": doc["source_dir"],
        "pages": doc["page_count"],
        "summary": doc["summary"],
        "indexed_at": doc["indexed_at"].isoformat() if doc["indexed_at"] else None,
    }


@mcp.tool()
async def reindex(
    shelf: Annotated[str | None, Field(description="Shelf to rescan; omit for all shelves")] = None,
) -> dict[str, Any]:
    """Ask the watcher process to rescan a shelf for new or changed files.

    Returns immediately; indexing happens in the background and may take a
    while for large documents.
    """
    if shelf and _config.shelf(shelf) is None:
        return {"error": f"Unknown shelf {shelf!r}",
                "available": [s.name for s in _config.shelves]}
    await _db.notify_reindex(shelf)
    return {"status": "requested", "shelf": shelf or "all"}


# ---------------------------------------------------------------------------

def main() -> int:
    global _config, _db, _embedder

    parser = argparse.ArgumentParser(description="Library MCP server")
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    _config = load_config(args.config)
    _db = Database(_config)
    _embedder = Embedder(_config.embedding)

    mcp.settings.host = _config.server.host
    mcp.settings.port = _config.server.port
    mcp.settings.streamable_http_path = _config.server.path

    log.info("Serving on http://%s:%d%s",
             _config.server.host, _config.server.port, _config.server.path)
    mcp.run(transport="streamable-http")
    return 0


if __name__ == "__main__":
    sys.exit(main())
