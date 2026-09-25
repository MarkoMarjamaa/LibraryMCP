"""Configuration loading and shelf definitions."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

ShelfType = Literal["manuals", "meetings", "scientific"]

# How each shelf type interprets the immediate parent directory of a file.
DIR_MEANING: dict[str, str] = {
    "manuals": "device",
    "meetings": "date",
    "scientific": "none",
}


@dataclass(frozen=True, slots=True)
class Shelf:
    name: str
    type: ShelfType
    path: Path
    description: str = ""

    @property
    def dir_meaning(self) -> str:
        return DIR_MEANING[self.type]


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    dsn: str
    min_pool: int = 2
    max_pool: int = 10


EmbeddingBackend = Literal["openai", "ollama"]


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    # "openai" covers llama.cpp's llama-server, TEI, vLLM and LM Studio.
    # "ollama" is Ollama's own /api/embed.
    backend: EmbeddingBackend = "openai"
    base_url: str = "http://localhost:8081"
    model: str = "bge-m3"
    dimensions: int = 1024
    batch_size: int = 8
    # llama.cpp folds every text of one /v1/embeddings request into a single
    # physical batch, so the SUM of the request's tokens must fit the
    # server's --ubatch-size, not just each chunk. The chunker's 4
    # chars/token estimate is optimistic for Finnish and other inflected
    # languages (real tokenisers land nearer 2), so cap requests well below
    # the typical 8192 ubatch.
    max_tokens_per_request: int = 6000
    api_key: str | None = None
    timeout: float = 120.0

    def __post_init__(self) -> None:
        if self.backend not in ("openai", "ollama"):
            raise ValueError(
                f"embedding.backend must be 'openai' or 'ollama', got {self.backend!r}"
            )


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_tokens: int = 400
    overlap_tokens: int = 60
    min_tokens: int = 40


@dataclass(frozen=True, slots=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    path: str = "/mcp"


@dataclass(frozen=True, slots=True)
class WatcherConfig:
    scan_interval_seconds: int = 300
    use_inotify: bool = False
    network_metadata: bool = True
    user_agent: str = "library-indexer/0.1"


@dataclass(frozen=True, slots=True)
class Config:
    database: DatabaseConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    server: ServerConfig
    watcher: WatcherConfig
    shelves: tuple[Shelf, ...] = field(default=())

    def shelf(self, name: str) -> Shelf | None:
        return next((s for s in self.shelves if s.name == name), None)

    def shelves_of_type(self, type_: ShelfType) -> tuple[Shelf, ...]:
        return tuple(s for s in self.shelves if s.type == type_)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load config.yaml. Path resolution order: argument, $LIBRARY_CONFIG, ./config.yaml."""
    candidate = path or os.environ.get("LIBRARY_CONFIG") or "config.yaml"
    cfg_path = Path(candidate).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    shelves: list[Shelf] = []
    seen: set[str] = set()
    for entry in raw.get("shelves", []):
        name = entry["name"]
        if name in seen:
            raise ValueError(f"Duplicate shelf name: {name}")
        seen.add(name)
        stype = entry["type"]
        if stype not in DIR_MEANING:
            raise ValueError(
                f"Shelf {name!r} has unknown type {stype!r}; "
                f"expected one of {sorted(DIR_MEANING)}"
            )
        shelves.append(
            Shelf(
                name=name,
                type=stype,
                path=Path(entry["path"]).expanduser(),
                description=entry.get("description", ""),
            )
        )

    if not shelves:
        raise ValueError("No shelves configured")

    db = raw.get("database", {})
    if "dsn" not in db:
        raise ValueError("database.dsn is required")

    return Config(
        database=DatabaseConfig(**db),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        chunking=ChunkingConfig(**raw.get("chunking", {})),
        server=ServerConfig(**raw.get("server", {})),
        watcher=WatcherConfig(**raw.get("watcher", {})),
        shelves=tuple(shelves),
    )

