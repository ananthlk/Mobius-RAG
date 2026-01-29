-- Delete document and all related records
-- Run this with: psql mobius_rag -f delete_document.sql
-- Or copy/paste into psql

BEGIN;

-- First, show what will be deleted
SELECT 'Documents to delete:' AS info;
SELECT id, filename, created_at FROM documents WHERE filename LIKE '%01-05-26-MFL%';

SELECT 'Chunking results to delete:' AS info;
SELECT COUNT(*) FROM chunking_results WHERE document_id IN (
    SELECT id FROM documents WHERE filename LIKE '%01-05-26-MFL%'
);

SELECT 'Document pages to delete:' AS info;
SELECT COUNT(*) FROM document_pages WHERE document_id IN (
    SELECT id FROM documents WHERE filename LIKE '%01-05-26-MFL%'
);

-- Delete in correct order (respecting foreign keys)
DELETE FROM chunking_results WHERE document_id IN (
    SELECT id FROM documents WHERE filename LIKE '%01-05-26-MFL%'
);

DELETE FROM document_pages WHERE document_id IN (
    SELECT id FROM documents WHERE filename LIKE '%01-05-26-MFL%'
);

DELETE FROM documents WHERE filename LIKE '%01-05-26-MFL%';

COMMIT;

SELECT '✅ Deletion complete!' AS result;
