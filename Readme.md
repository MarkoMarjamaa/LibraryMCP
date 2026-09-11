# Library

Local document RAG over Postgres + pgvector, exposed to a speech assistant via MCP.

Two processes, one codebase:

- **`library.watcher`** — scans configured directories, indexes new and changed PDFs. CPU-heavy, bursty, needs outbound internet for paper metadata.
- **`library.server`** — MCP server over streamable-http. Latency-sensitive, needs only Postgres and Ollama on localhost.

They talk through Postgres `LISTEN`/`NOTIFY`, so there is no queue or extra port between them.

## Setup

```bash
sudo apt install postgresql-17-pgvector
psql -U postgres            -f sql/01_bootstrap.sql
psql -U librarian -d library -f sql/02_schema.sql
psql -U librarian -d library -f sql/03_shelf_types.sql

python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp config.yaml.example config.yaml   # edit paths
```

### Embedding server

Any OpenAI-compatible `/v1/embeddings` endpoint works — llama.cpp, TEI, vLLM,
LM Studio — or Ollama via `backend: "ollama"`.

With llama.cpp, run a **second** `llama-server` on its own port, separate from
your chat model:

```bash
llama-server \
  -m models/bge-m3-F16.gguf \
  --embeddings \
  --pooling cls \
  -c 8192 -b 8192 -ub 8192 \
  --host 127.0.0.1 --port 8081
```
Or in ini-file when you can use current llama-server
```
[bge-m3]
model = /models/bge-m3-f16.gguf
load-on-startup = true
embeddings = on
pooling = cls
ctx-size = 8192
ubatch-size = 8192
batch-size = 8192
```

Then in `config.yaml`:

```yaml
embedding:
  backend: "openai"
  base_url: "http://localhost:8081"
  model: "bge-m3"
  dimensions: 1024
  batch_size: 8
```

Both processes call `health_check()` at startup, so a wrong port, a missing
`--embeddings`, or a dimension mismatch fails immediately rather than partway
through the first index.

First index, in the foreground so you can watch it:

```bash
.venv/bin/python -m library.watcher --once -v
```

Then run both:

```bash
.venv/bin/python -m library.watcher      # daemon
.venv/bin/python -m library.server       # http://0.0.0.0:8765/mcp
```

Or install the systemd units in `systemd/` (adjust `User=` and paths).

## Adding a shelf

Append to `shelves:` in `config.yaml` and restart the watcher. The `shelves`
table is reconciled on startup. Several shelves may share a type:

```yaml
  - name: condo_meetings
    type: meetings
    path: /srv/documents/meetings/condo
```

Shelf types and what they do at ingest:

| type | parent directory means | title from | network lookup |
|---|---|---|---|
| `manuals` | device name + model | directory name | no |
| `meetings` | meeting date | filename + date | no |
| `scientific` | nothing | arXiv/Crossref, else largest font on page 1 | yes |

## MCP tools

| tool | purpose |
|---|---|
| `list_shelves` | orientation: what exists, how big |
| `list_documents` | enumerate a small/medium shelf, optionally date-filtered |
| `find_documents` | semantic search over titles + abstracts, for large shelves |
| `search_manuals` | full-text + vector over manuals, `device` filter |
| `search_meetings` | same over meetings, with `date_from`/`date_to`/`dates` |
| `search_papers` | same over papers, with `author`/`year_from`/`year_to` |
| `get_context` | neighbouring passages when a result is truncated |
| `get_document_info` | full metadata for one document |
| `reindex` | signal the watcher to rescan |

## Notes

- Retrieval is hybrid: vector and full-text arms, fused with Reciprocal Rank
  Fusion. Both are needed — dense vectors miss `E04` and `SNSW-102P16EU`,
  keyword search misses paraphrase.
- Text search config is `simple`, not `english`, so model numbers survive and
  Finnish is not mangled by an English stemmer.
- Chunks are paragraph-packed to ~400 tokens with overlap, not split by page.
  `page_start`/`page_end` are recorded as provenance; `page_end > page_start`
  means the chunk crossed a break, and the citation reads "pages 6-7".
- Citations are returned as structured fields, not a formatted string, so the
  speech layer can say "page 12 of the Shelly Plus 2PM manual" without reading
  a filename aloud.
- Scanned PDFs with no text layer are skipped with an error. Run OCR
  (`ocrmypdf`) over them first.

