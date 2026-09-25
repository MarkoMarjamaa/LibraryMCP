"""Ingest pipeline: scan a shelf's directory, index new or changed documents."""

from __future__ import annotations

import asyncio
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import httpx

from Config import Config, Shelf
from Db import Database
from Embed import Embedder, document_text_for
import Extract as extractmod

# Import order registers the extractors; Pdf last so the scientific helpers
# sit alongside the format that still uses them most.
import Docx  # noqa: F401  (.docx)
import Ods  # noqa: F401    (.ods)
import Pptx  # noqa: F401   (.pptx)
import TextFiles  # noqa: F401 (.txt, .md)
import Xlsx  # noqa: F401   (.xlsx, .xlsm)
import Metadata as metamod
import Pdf as pdfmod  # noqa: F401  (.pdf)

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ScanResult:
    shelf: str
    indexed: int = 0
    skipped: int = 0
    deleted: int = 0
    failed: int = 0

    def __str__(self) -> str:
        return (f"{self.shelf}: {self.indexed} indexed, {self.skipped} unchanged, "
                f"{self.deleted} deleted, {self.failed} failed")


class Indexer:
    def __init__(self, config: Config, db: Database, embedder: Embedder) -> None:
        self._config = config
        self._db = db
        self._embedder = embedder
        self._http: httpx.AsyncClient | None = None
        if config.watcher.network_metadata:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                headers={"User-Agent": config.watcher.user_agent},
                follow_redirects=True,
            )

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()

    async def scan_shelf(self, shelf: Shelf, force: bool = False) -> ScanResult:
        result = ScanResult(shelf=shelf.name)

        if not shelf.path.is_dir():
            log.error("Shelf %r path does not exist: %s", shelf.name, shelf.path)
            return result

        known = await self._db.get_document_state(shelf.name)
        seen: set[str] = set()

        for file in sorted(shelf.path.rglob("*")):
            if not file.is_file():
                continue
            extractor = extractmod.extract_for(file)
            if extractor is None:
                if extractmod.is_legacy_office(file):
                    # Legacy .doc/.xls need an external converter; skipped
                    # quietly rather than counted as a failure.
                    log.debug("Skipping legacy Office file: %s", file)
                continue
            if file.name.startswith("."):
                continue

            # Filenames with combining diacritics (NFD) hash and compare as
            # different strings from their NFC form; normalise before the path
            # goes anywhere near the database.
            rel_path = unicodedata.normalize(
                "NFC", str(file.relative_to(shelf.path.parent))
            )
            seen.add(rel_path)

            try:
                digest = await asyncio.to_thread(extractmod.file_hash, file)
            except OSError as exc:
                log.warning("Cannot read %s: %s", file, exc)
                result.failed += 1
                continue

            existing = known.get(rel_path)
            if existing and existing[1] == digest and not force:
                result.skipped += 1
                continue

            try:
                await self._index_file(file, shelf, rel_path, digest, extractor)
                result.indexed += 1
                log.info("Indexed %s", rel_path)
            except Exception as exc:
                result.failed += 1
                log.debug("Failed to index %s: %s", rel_path, exc)
                continue
                

        # Files that vanished from disk.
        stale = [doc_id for path, (doc_id, _) in known.items() if path not in seen]
        result.deleted = await self._db.delete_documents(stale)
        if stale:
            log.info("Removed %d deleted documents from %s", result.deleted, shelf.name)

        return result

    async def _index_file(
        self, file: Path, shelf: Shelf, rel_path: str, digest: str,
        extractor: extractmod.Extractor,
    ) -> None:
        pages = await asyncio.to_thread(extractor.extract_pages, file)
        if not pages:
            #log.warning("Failed to index %s: %s", rel_path, "no extractable text (scanned PDF? try OCR first)")
            #return
            raise ValueError("no extractable text (scanned PDF? try OCR first)")

        # For a flat shelf the parent directory is the shelf root itself, which
        # carries no meaning; that is what dir_meaning='none' encodes.
        source_dir = file.parent.name

        meta = await metamod.resolve(file, shelf, source_dir, pages, self._http)

        chunks = await asyncio.to_thread(
            extractmod.chunk_pages, pages, self._config.chunking
        )
        if not chunks:
            #log.warning("Failed to index %s: %s", rel_path, "produced no chunks")
            raise ValueError("produced no chunks")
            #return

        model = self._config.embedding.model
        chunk_vectors = await self._embedder.embed(
            [document_text_for(c.content, model) for c in chunks]
        )

        # Drop chunks the embedder could not represent rather than failing the
        # document: one unreadable page should not cost the other 200.
        pairs = [(c, v) for c, v in zip(chunks, chunk_vectors, strict=True)
                 if v is not None]
        if len(pairs) != len(chunks):
            log.warning("%s: %d of %d chunks had no embeddable content",
                        rel_path, len(chunks) - len(pairs), len(chunks))
        if not pairs:
            raise ValueError("no chunk produced an embedding")

        summary_vector = None
        if meta.summary:
            try:
                summary_vector = await self._embedder.embed_one(
                    document_text_for(f"{meta.title or ''}\n{meta.summary}", model)
                )
            except ValueError:
                log.debug("%s: summary has no embeddable content", rel_path)

        doc_id = await self._db.upsert_document(
            {
                "shelf": shelf.name,
                "rel_path": rel_path,
                "filename": file.name,
                "source_dir": source_dir,
                "title": meta.title,
                "authors": meta.authors,
                "doc_date": meta.doc_date,
                "year": meta.year,
                "summary": meta.summary,
                "summary_embedding": summary_vector,
                "content_hash": digest,
                "page_count": len(pages),
                "extra": {**meta.extra, "kind": extractor.kind},
            }
        )

        await self._db.insert_chunks(
            doc_id,
            [
                {
                    "chunk_index": i,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "section": c.section,
                    "content": c.content,
                    "token_count": c.token_count,
                    "embedding": vec,
                }
                for i, (c, vec) in enumerate(pairs)
            ],
        )

    async def scan_all(self, force: bool = False) -> list[ScanResult]:
        results = []
        for shelf in self._config.shelves:
            results.append(await self.scan_shelf(shelf, force=force))
        return results
