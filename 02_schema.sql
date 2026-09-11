-- 02_schema.sql
-- Run as librarian against the library database:
--     psql -U librarian -d library -f 02_schema.sql
--
-- Embedding dimension is 1024 (BAAI/bge-m3). Change every vector(1024) below
-- if you use a different model: nomic-embed-text = 768, bge-large-en = 1024,
-- qwen3-embedding:8b = 4096 (note: too wide for an HNSW index, max is 2000).

BEGIN;

-- ===========================================================================
-- Shelves
-- ===========================================================================
CREATE TABLE IF NOT EXISTS shelves (
    name        text PRIMARY KEY,
    description text NOT NULL,
    -- How source_dir should be interpreted for documents on this shelf.
    dir_meaning text NOT NULL CHECK (dir_meaning IN ('device', 'date', 'none')),
    created_at  timestamptz NOT NULL DEFAULT now()
);

COMMENT ON COLUMN shelves.dir_meaning IS
    'device = directory is device name+model; date = directory is meeting date; none = flat directory';

INSERT INTO shelves (name, description, dir_meaning) VALUES
    ('manuals',  'Home automation device manuals',      'device'),
    ('meetings', 'Work meeting documents, by date',     'date'),
    ('papers',   'Scientific papers, filed by paper ID', 'none')
ON CONFLICT (name) DO NOTHING;

-- ===========================================================================
-- Documents  (one row per source file)
-- ===========================================================================
CREATE TABLE IF NOT EXISTS documents (
    id                bigserial PRIMARY KEY,
    shelf             text NOT NULL REFERENCES shelves(name) ON UPDATE CASCADE,

    -- Location on disk
    rel_path          text NOT NULL,            -- 'manuals/Shelly Plus 2PM/manual_v3.pdf'
    filename          text NOT NULL,            -- 'manual_v3.pdf'
    source_dir        text NOT NULL,            -- 'Shelly Plus 2PM' | '2026-03-14' | 'papers'

    -- Human-facing identity. For manuals this is device + model; for meetings
    -- the meeting subject; for papers the real title resolved from arXiv/Crossref.
    title             text,
    authors           text[],                   -- papers; NULL elsewhere
    doc_date          date,                     -- meetings: parsed from source_dir
                                                -- papers: publication date
    year              int,

    -- Document-level semantics: abstract for papers, LLM summary for meetings,
    -- product blurb for manuals. Embedded so find_documents() can work.
    summary           text,
    summary_embedding vector(1024),

    -- Ingest bookkeeping
    content_hash      text,                     -- sha256 of the file; skip unchanged
    page_count        int,
    extra             jsonb NOT NULL DEFAULT '{}'::jsonb,  -- doi, arxiv_id, firmware ver, attendees...
    indexed_at        timestamptz,
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT documents_rel_path_key UNIQUE (rel_path)
);

-- ===========================================================================
-- Chunks  (retrieval units; NOT pages -- page span is provenance)
-- ===========================================================================
CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    document_id  bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  int NOT NULL,                  -- order within the document

    -- Citation target. page_end differs from page_start when a chunk
    -- straddles a page break.
    page_start   int,
    page_end     int,
    section      text,                          -- 'Troubleshooting', '4.2 Results'

    content      text NOT NULL,
    token_count  int,
    embedding    vector(1024),

    -- 'simple' rather than 'english': no stemming, so model numbers and error
    -- codes (E04, SNSW-102P16EU) survive intact, and Finnish text is not
    -- mangled by an English stemmer.
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,

    created_at   timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT chunks_doc_index_key UNIQUE (document_id, chunk_index),
    CONSTRAINT chunks_page_order CHECK (page_end IS NULL OR page_start IS NULL
                                        OR page_end >= page_start)
);

-- ===========================================================================
-- Indexes
-- ===========================================================================

-- Vector search over chunks. Cosine matches bge-m3's normalized output.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Keyword arm of hybrid search.
CREATE INDEX IF NOT EXISTS chunks_tsv_gin
    ON chunks USING gin (tsv);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx
    ON chunks (document_id);

-- Document-level semantic search (find_documents on the papers shelf).
CREATE INDEX IF NOT EXISTS documents_summary_hnsw
    ON documents USING hnsw (summary_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Shelf scoping and date filtering (date_from / date_to / dates params).
CREATE INDEX IF NOT EXISTS documents_shelf_idx        ON documents (shelf);
CREATE INDEX IF NOT EXISTS documents_shelf_date_idx   ON documents (shelf, doc_date);
CREATE INDEX IF NOT EXISTS documents_shelf_dir_idx    ON documents (shelf, source_dir);

-- Fuzzy matching on device names and paper titles: 'shelly 2pm' -> 'Shelly Plus 2PM'.
CREATE INDEX IF NOT EXISTS documents_title_trgm
    ON documents USING gin (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS documents_source_dir_trgm
    ON documents USING gin (source_dir gin_trgm_ops);

-- Author lookup on the papers shelf.
CREATE INDEX IF NOT EXISTS documents_authors_gin
    ON documents USING gin (authors);

CREATE INDEX IF NOT EXISTS documents_extra_gin
    ON documents USING gin (extra jsonb_path_ops);

-- ===========================================================================
-- updated_at trigger
-- ===========================================================================
CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS documents_touch_updated_at ON documents;
CREATE TRIGGER documents_touch_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

-- ===========================================================================
-- Convenience view: everything the MCP layer needs to build a citation,
-- without joining by hand in every query.
-- ===========================================================================
CREATE OR REPLACE VIEW chunk_citations AS
SELECT
    c.id            AS chunk_id,
    c.content,
    c.embedding,
    c.tsv,
    c.section,
    c.page_start,
    c.page_end,
    d.id            AS document_id,
    d.shelf,
    d.title,
    d.filename,
    d.source_dir,
    d.doc_date,
    d.authors,
    d.year
FROM chunks c
JOIN documents d ON d.id = c.document_id;

COMMIT;

-- ===========================================================================
-- Post-install notes (not executed)
-- ===========================================================================
-- Build HNSW indexes faster on bulk ingest by raising memory first:
--     SET maintenance_work_mem = '2GB';
--
-- Faster bulk load: drop the two HNSW indexes, insert everything, recreate.
--
-- Recall/speed tradeoff at query time (higher = better recall, slower):
--     SET hnsw.ef_search = 100;
--
-- Sanity check:
--     SELECT extname, extversion FROM pg_extension;
--     \d+ documents

