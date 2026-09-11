-- 03_shelf_types.sql
-- Adds shelf type + root path to the schema from 02_schema.sql, so several
-- shelves can share a type ('work_meetings' and 'condo_meetings' are both
-- type 'meetings' in different directories).
--
--     psql -U librarian -d library -f sql/03_shelf_types.sql

BEGIN;

ALTER TABLE shelves ADD COLUMN IF NOT EXISTS type      text;
ALTER TABLE shelves ADD COLUMN IF NOT EXISTS root_path text;

UPDATE shelves SET type = CASE
    WHEN dir_meaning = 'device' THEN 'manuals'
    WHEN dir_meaning = 'date'   THEN 'meetings'
    ELSE 'scientific'
END WHERE type IS NULL;

ALTER TABLE shelves ALTER COLUMN type SET NOT NULL;

ALTER TABLE shelves DROP CONSTRAINT IF EXISTS shelves_type_check;
ALTER TABLE shelves ADD CONSTRAINT shelves_type_check
    CHECK (type IN ('manuals', 'meetings', 'scientific'));

CREATE INDEX IF NOT EXISTS shelves_type_idx ON shelves (type);

COMMENT ON COLUMN shelves.type IS
    'Shelf type; several shelves may share one. Drives ingest behaviour and MCP tool routing.';

COMMIT;

