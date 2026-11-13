-- Era B (about 1965-2009, printed SCR with margin letters A-H) is parsed
-- from Phase 4. Widens the era check written inline in 001_schema.sql.
ALTER TABLE judgments DROP CONSTRAINT IF EXISTS judgments_era_check;
ALTER TABLE judgments ADD CONSTRAINT judgments_era_check
    CHECK (era IN ('b', 'c', 'd', 'out_of_scope'));
