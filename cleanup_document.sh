#!/bin/bash
# Cleanup script to delete document from PostgreSQL
# Run this when PostgreSQL is accessible

echo "Attempting to delete document from database..."

# Try using psql if available
if command -v psql &> /dev/null; then
    echo "Using psql..."
    psql mobius_rag <<EOF
BEGIN;
DELETE FROM chunking_results WHERE document_id IN (
    SELECT id FROM documents WHERE filename LIKE '%01-05-26-MFL%'
);
DELETE FROM document_pages WHERE document_id IN (
    SELECT id FROM documents WHERE filename LIKE '%01-05-26-MFL%'
);
DELETE FROM documents WHERE filename LIKE '%01-05-26-MFL%';
COMMIT;
SELECT 'Deleted documents matching pattern' AS result;
EOF
else
    echo "psql not found. Please run the SQL commands manually:"
    echo ""
    echo "BEGIN;"
    echo "DELETE FROM chunking_results WHERE document_id IN ("
    echo "    SELECT id FROM documents WHERE filename LIKE '%01-05-26-MFL%'"
    echo ");"
    echo "DELETE FROM document_pages WHERE document_id IN ("
    echo "    SELECT id FROM documents WHERE filename LIKE '%01-05-26-MFL%'"
    echo ");"
    echo "DELETE FROM documents WHERE filename LIKE '%01-05-26-MFL%';"
    echo "COMMIT;"
fi
