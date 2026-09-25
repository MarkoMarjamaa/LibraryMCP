"""Postgres access: pool management, upserts, and hybrid retrieval."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from typing import Any, Sequence

import asyncpg

from Config import Config, Shelf

REINDEX_CHANNEL = "library_reindex"

# Reciprocal Rank Fusion constant. 60 is the value from the original paper and
# works fine here; lower values weight the top of each list more heavily.
RRF_K = 60

# How many candidates each retrieval arm contributes before fusion.
ARM_LIMIT = 60


def to_pgvector(values: Sequence[float]) -> str:
    """asyncpg has no native vector codec; pgvector accepts this text form."""
    return "[" + ",".join(f"{v:.7g}" for v in values) + "]"


@dataclass(slots=True)
class SearchFilters:
    shelves: list[str] | None = None
    document_ids: list[int] | None = None
    source_dir: str | None = None       # trigram / ILIKE match
    date_from: dt.date | None = None
    date_to: dt.date | None = None
    dates: list[dt.date] | None = None
    year_from: int | None = None
    year_to: int | None = None
    author: str | None = None

    def build(self, params: list[Any]) -> str:
        """Append parameters to `params` and return a SQL WHERE fragment.

        The fragment is injected into both retrieval arms so filtering happens
        before ranking, not after.
        """
        clauses: list[str] = []

        def add(sql_tmpl: str, value: Any) -> None:
            params.append(value)
            clauses.append(sql_tmpl.format(n=len(params)))

        if self.shelves:
            add("d.shelf = ANY(${n}::text[])", self.shelves)
        if self.document_ids:
            add("d.id = ANY(${n}::bigint[])", self.document_ids)
        if self.source_dir:
            add("d.source_dir ILIKE '%' || ${n} || '%'", self.source_dir)
        if self.dates:
            add("d.doc_date = ANY(${n}::date[])", self.dates)
        if self.date_from:
            add("d.doc_date >= ${n}::date", self.date_from)
        if self.date_to:
            add("d.doc_date <= ${n}::date", self.date_to)
        if self.year_from:
            add("d.year >= ${n}::int", self.year_from)
        if self.year_to:
            add("d.year <= ${n}::int", self.year_to)
        if self.author:
            add(
                "EXISTS (SELECT 1 FROM unnest(d.authors) a "
                "WHERE a ILIKE '%' || ${n} || '%')",
                self.author,
            )

        return (" AND " + " AND ".join(clauses)) if clauses else ""


class Database:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self._config.database.dsn,
                min_size=self._config.database.min_pool,
                max_size=self._config.database.max_pool,
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def acquire_pool(self) -> asyncpg.Pool:
        """The pool, (re)created if a previous teardown closed it.

        Mirrors Embedder.client(): an mcp SDK that runs the lifespan per
        session closes the pool when one session ends; the next tool call in
        the still-running process then reconnects instead of failing.
        asyncpg pools cannot be reopened, so this builds a fresh one.
        """
        if self._pool is None:
            await self.connect()
        assert self._pool is not None
        return self._pool

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database.connect() has not been awaited")
        return self._pool

    # -- shelves -----------------------------------------------------------

    async def sync_shelves(self, shelves: Sequence[Shelf]) -> None:
        """Reconcile the shelves table with config. Never deletes: removing a
        shelf from config leaves its documents queryable until you drop them
        explicitly."""
        async with (await self.acquire_pool()).acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO shelves (name, type, description, dir_meaning, root_path)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (name) DO UPDATE
                   SET type        = EXCLUDED.type,
                       description = EXCLUDED.description,
                       dir_meaning = EXCLUDED.dir_meaning,
                       root_path   = EXCLUDED.root_path
                """,
                [
                    (s.name, s.type, s.description or s.name,
                     s.dir_meaning, str(s.path))
                    for s in shelves
                ],
            )

    async def list_shelves(self) -> list[dict[str, Any]]:
        rows = await (await self.acquire_pool()).fetch(
            """
            SELECT s.name, s.type, s.description,
                   count(d.id)                       AS document_count,
                   min(d.doc_date)                   AS earliest,
                   max(d.doc_date)                   AS latest
            FROM shelves s
            LEFT JOIN documents d ON d.shelf = s.name
            GROUP BY s.name, s.type, s.description
            ORDER BY s.name
            """
        )
        return [dict(r) for r in rows]

    # -- documents ---------------------------------------------------------

    async def get_document_state(self, shelf: str) -> dict[str, tuple[int, str | None]]:
        """rel_path -> (document_id, content_hash) for change detection."""
        rows = await (await self.acquire_pool()).fetch(
            "SELECT id, rel_path, content_hash FROM documents WHERE shelf = $1",
            shelf,
        )
        return {r["rel_path"]: (r["id"], r["content_hash"]) for r in rows}

    async def upsert_document(self, doc: dict[str, Any]) -> int:
        """Insert or replace a document and return its id.

        Chunks are deleted on update (ON DELETE CASCADE does not fire for an
        UPDATE, so this is explicit) because re-chunking invalidates them all.
        """
        async with (await self.acquire_pool()).acquire() as conn:
            async with conn.transaction():
                doc_id: int = await conn.fetchval(
                    """
                    INSERT INTO documents
                        (shelf, rel_path, filename, source_dir, title, authors,
                         doc_date, year, summary, summary_embedding,
                         content_hash, page_count, extra, indexed_at)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::vector,$11,$12,$13::jsonb, now())
                    ON CONFLICT (rel_path) DO UPDATE SET
                        shelf             = EXCLUDED.shelf,
                        filename          = EXCLUDED.filename,
                        source_dir        = EXCLUDED.source_dir,
                        title             = EXCLUDED.title,
                        authors           = EXCLUDED.authors,
                        doc_date          = EXCLUDED.doc_date,
                        year              = EXCLUDED.year,
                        summary           = EXCLUDED.summary,
                        summary_embedding = EXCLUDED.summary_embedding,
                        content_hash      = EXCLUDED.content_hash,
                        page_count        = EXCLUDED.page_count,
                        extra             = EXCLUDED.extra,
                        indexed_at        = now()
                    RETURNING id
                    """,
                    doc["shelf"], doc["rel_path"], doc["filename"], doc["source_dir"],
                    doc.get("title"), doc.get("authors"), doc.get("doc_date"),
                    doc.get("year"), doc.get("summary"),
                    to_pgvector(doc["summary_embedding"]) if doc.get("summary_embedding") else None,
                    doc.get("content_hash"), doc.get("page_count"),
                    json.dumps(doc.get("extra") or {}),
                )
                await conn.execute("DELETE FROM chunks WHERE document_id = $1", doc_id)
                return doc_id

    async def insert_chunks(self, document_id: int, chunks: Sequence[dict[str, Any]]) -> None:
        if not chunks:
            return
        await (await self.acquire_pool()).executemany(
            """
            INSERT INTO chunks
                (document_id, chunk_index, page_start, page_end, section,
                 content, token_count, embedding)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8::vector)
            """,
            [
                (document_id, c["chunk_index"], c.get("page_start"), c.get("page_end"),
                 c.get("section"), c["content"], c.get("token_count"),
                 to_pgvector(c["embedding"]))
                for c in chunks
            ],
        )

    async def delete_documents(self, document_ids: Sequence[int]) -> int:
        if not document_ids:
            return 0
        return int(
            await (await self.acquire_pool()).fetchval(
                "WITH d AS (DELETE FROM documents WHERE id = ANY($1::bigint[]) RETURNING 1) "
                "SELECT count(*) FROM d",
                list(document_ids),
            )
        )

    async def list_documents(
        self, filters: SearchFilters, limit: int = 200
    ) -> list[dict[str, Any]]:
        params: list[Any] = []
        where = filters.build(params)
        params.append(limit)
        rows = await (await self.acquire_pool()).fetch(
            f"""
            SELECT d.id, d.shelf, d.title, d.source_dir, d.filename,
                   d.doc_date, d.year, d.authors, d.page_count,
                   left(d.summary, 300) AS summary_preview
            FROM documents d
            WHERE true {where}
            ORDER BY d.doc_date DESC NULLS LAST, d.title
            LIMIT ${len(params)}
            """,
            *params,
        )
        return [dict(r) for r in rows]

    async def find_documents(
        self, embedding: Sequence[float], filters: SearchFilters, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Semantic search at document level, over title + summary embeddings.

        This is the entry point for large shelves where listing everything is
        impractical.
        """
        params: list[Any] = [to_pgvector(embedding)]
        where = filters.build(params)
        params.append(limit)
        rows = await (await self.acquire_pool()).fetch(
            f"""
            SELECT d.id, d.shelf, d.title, d.source_dir, d.filename,
                   d.doc_date, d.year, d.authors,
                   left(d.summary, 600) AS summary_preview,
                   1 - (d.summary_embedding <=> $1::vector) AS score
            FROM documents d
            WHERE d.summary_embedding IS NOT NULL {where}
            ORDER BY d.summary_embedding <=> $1::vector
            LIMIT ${len(params)}
            """,
            *params,
        )
        return [dict(r) for r in rows]

    async def get_document(self, document_id: int) -> dict[str, Any] | None:
        row = await (await self.acquire_pool()).fetchrow(
            "SELECT * FROM documents WHERE id = $1", document_id
        )
        return dict(row) if row else None

    async def get_document_chunks(self, document_id: int) -> list[dict[str, Any]]:
        """All chunks of a document, in reading order."""
        rows = await (await self.acquire_pool()).fetch(
            """
            SELECT id AS chunk_id, chunk_index, content, section,
                   page_start, page_end
            FROM chunks
            WHERE document_id = $1
            ORDER BY chunk_index
            """,
            document_id,
        )
        return [dict(r) for r in rows]

    # -- retrieval ---------------------------------------------------------

    async def hybrid_search(
        self,
        query_text: str,
        embedding: Sequence[float],
        filters: SearchFilters,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Vector + full-text retrieval fused with Reciprocal Rank Fusion.

        Both arms are needed: dense vectors miss exact tokens like 'E04' or
        'SNSW-102P16EU', and keyword search misses paraphrase.
        """
        params: list[Any] = [to_pgvector(embedding), query_text]
        where = filters.build(params)
        params.extend([ARM_LIMIT, limit])
        arm_param = len(params) - 1
        limit_param = len(params)

        rows = await (await self.acquire_pool()).fetch(
            f"""
            WITH vec AS (
                SELECT c.id AS chunk_id,
                       row_number() OVER (ORDER BY c.embedding <=> $1::vector) AS rank
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.embedding IS NOT NULL {where}
                ORDER BY c.embedding <=> $1::vector
                LIMIT ${arm_param}
            ),
            kw AS (
                SELECT c.id AS chunk_id,
                       row_number() OVER (
                           ORDER BY ts_rank_cd(c.tsv, websearch_to_tsquery('simple', $2)) DESC
                       ) AS rank
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.tsv @@ websearch_to_tsquery('simple', $2) {where}
                ORDER BY ts_rank_cd(c.tsv, websearch_to_tsquery('simple', $2)) DESC
                LIMIT ${arm_param}
            ),
            fused AS (
                SELECT COALESCE(vec.chunk_id, kw.chunk_id) AS chunk_id,
                       COALESCE(1.0 / ({RRF_K} + vec.rank), 0.0)
                     + COALESCE(1.0 / ({RRF_K} + kw.rank), 0.0) AS score,
                       vec.rank AS vector_rank,
                       kw.rank  AS keyword_rank
                FROM vec
                FULL OUTER JOIN kw ON kw.chunk_id = vec.chunk_id
            )
            SELECT cc.chunk_id, cc.content, cc.section,
                   cc.page_start, cc.page_end,
                   cc.document_id, cc.shelf, cc.title, cc.filename,
                   cc.source_dir, cc.doc_date, cc.authors, cc.year,
                   f.score, f.vector_rank, f.keyword_rank
            FROM fused f
            JOIN chunk_citations cc ON cc.chunk_id = f.chunk_id
            ORDER BY f.score DESC
            LIMIT ${limit_param}
            """,
            *params,
        )
        return [dict(r) for r in rows]

    async def get_chunk_window(
        self, chunk_id: int, before: int = 1, after: int = 1
    ) -> list[dict[str, Any]]:
        """Neighbouring chunks, for when a retrieved procedure is truncated."""
        rows = await (await self.acquire_pool()).fetch(
            """
            WITH target AS (
                SELECT document_id, chunk_index FROM chunks WHERE id = $1
            )
            SELECT c.id AS chunk_id, c.chunk_index, c.content, c.section,
                   c.page_start, c.page_end
            FROM chunks c, target t
            WHERE c.document_id = t.document_id
              AND c.chunk_index BETWEEN t.chunk_index - $2 AND t.chunk_index + $3
            ORDER BY c.chunk_index
            """,
            chunk_id, before, after,
        )
        return [dict(r) for r in rows]

    # -- cross-process signalling -----------------------------------------

    async def notify_reindex(self, shelf: str | None) -> None:
        """Ask the watcher to rescan. Postgres LISTEN/NOTIFY avoids needing a
        queue or an HTTP endpoint between the two processes."""
        await (await self.acquire_pool()).execute("SELECT pg_notify($1, $2)", REINDEX_CHANNEL, shelf or "*")
