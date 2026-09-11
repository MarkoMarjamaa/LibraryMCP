-- 01_bootstrap.sql
-- Run as a superuser against the maintenance database:
--     psql -U postgres -f 01_bootstrap.sql
--
-- Prerequisite: pgvector must be installed on the host first, e.g.
--     sudo apt install postgresql-17-pgvector      # Debian/Ubuntu, match your PG major version
--     brew install pgvector                        # macOS
-- Docker users: the pgvector/pgvector:pg17 image already has it.

-- ---------------------------------------------------------------------------
-- Role
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'librarian') THEN
        CREATE ROLE librarian LOGIN PASSWORD 'librarian';
    END IF;
END
$$;

-- ---------------------------------------------------------------------------
-- Database
-- CREATE DATABASE cannot run inside a DO block or a transaction, so this is
-- plain and will error harmlessly if the database already exists.
-- ---------------------------------------------------------------------------
CREATE DATABASE library
    OWNER      librarian
    ENCODING   'UTF8'
    LC_COLLATE 'en_US.UTF-8'
    LC_CTYPE   'en_US.UTF-8'
    TEMPLATE   template0;

COMMENT ON DATABASE library IS 'Document library: manuals, meetings, papers';

-- ---------------------------------------------------------------------------
-- Extensions must be created inside the target database, not this one.
-- CREATE EXTENSION requires superuser for pgvector, so it stays here rather
-- than in the schema file.
-- ---------------------------------------------------------------------------
\connect library

CREATE EXTENSION IF NOT EXISTS vector;      -- embeddings + HNSW
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- fuzzy title / device-name matching
CREATE EXTENSION IF NOT EXISTS unaccent;    -- optional: accent-insensitive text search

-- Let librarian own the public schema so it can create tables there.
ALTER SCHEMA public OWNER TO librarian;
GRANT ALL ON SCHEMA public TO librarian;

\echo 'Bootstrap complete. Now run: psql -U librarian -d library -f 02_schema.sql'

