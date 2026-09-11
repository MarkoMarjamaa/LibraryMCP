"""Watcher daemon: keeps the index in step with the filesystem.

Runs as a separate process from the MCP server. Indexing is CPU-heavy and
bursty; queries are latency-sensitive. Keeping them apart means a 400-page
manual landing in a watched folder cannot stall a spoken "how do I reset the
Shelly".

This is also the only component that needs outbound internet access, for
arXiv and Crossref lookups on 'scientific' shelves.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys

import asyncpg

from Config import load_config
from Db import REINDEX_CHANNEL, Database
from Embed import Embedder
from Ingest import Indexer

log = logging.getLogger("library.watcher")


class Watcher:
    def __init__(self, config, db: Database, indexer: Indexer) -> None:
        self._config = config
        self._db = db
        self._indexer = indexer
        self._wake = asyncio.Event()
        self._requested: set[str] = set()
        self._stopping = asyncio.Event()
        self._lock = asyncio.Lock()

    def request(self, shelf: str) -> None:
        self._requested.add(shelf)
        self._wake.set()

    async def run(self) -> None:
        listener_conn = await asyncpg.connect(self._config.database.dsn)
        await listener_conn.add_listener(
            REINDEX_CHANNEL,
            lambda _conn, _pid, _chan, payload: self.request(payload or "*"),
        )
        log.info("Listening on %s for reindex requests", REINDEX_CHANNEL)

        try:
            await self._scan(set())  # initial pass
            while not self._stopping.is_set():
                try:
                    await asyncio.wait_for(
                        self._wake.wait(),
                        timeout=self._config.watcher.scan_interval_seconds,
                    )
                except asyncio.TimeoutError:
                    pass  # periodic scan

                if self._stopping.is_set():
                    break

                self._wake.clear()
                requested, self._requested = self._requested, set()
                await self._scan(requested)
        finally:
            with contextlib.suppress(Exception):
                await listener_conn.close()

    async def _scan(self, requested: set[str]) -> None:
        # A slow scan must not overlap with the next tick.
        if self._lock.locked():
            log.debug("Scan already running; skipping this tick")
            return

        async with self._lock:
            if requested and "*" not in requested:
                shelves = [s for s in self._config.shelves if s.name in requested]
            else:
                shelves = list(self._config.shelves)

            for shelf in shelves:
                try:
                    result = await self._indexer.scan_shelf(shelf)
                    if result.indexed or result.deleted or result.failed:
                        log.info("%s", result)
                except Exception:
                    log.exception("Scan failed for shelf %r", shelf.name)

    def stop(self) -> None:
        self._stopping.set()
        self._wake.set()


async def main_async(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    db = Database(config)
    await db.connect()
    await db.sync_shelves(config.shelves)

    embedder = Embedder(config.embedding)
    indexer = Indexer(config, db, embedder)

    try:
        if args.once:
            for result in await indexer.scan_all(force=args.force):
                log.info("%s", result)
            return 0

        watcher = Watcher(config, db, indexer)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, watcher.stop)
        await watcher.run()
        return 0
    finally:
        await indexer.close()
        await embedder.close()
        await db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Library index watcher")
    parser.add_argument("-c", "--config", default=None, help="Path to config.yaml")
    parser.add_argument("--once", action="store_true",
                        help="Scan every shelf once and exit")
    parser.add_argument("--force", action="store_true",
                        help="With --once, reindex even unchanged files")
    parser.add_argument("-v", "--verbose", action="store_true")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
