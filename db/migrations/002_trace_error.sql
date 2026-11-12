-- Why a request ended with status `error` (an exception or the request
-- timeout). Phase 1 integration, 2026-11-13.
ALTER TABLE traces ADD COLUMN IF NOT EXISTS error text;
