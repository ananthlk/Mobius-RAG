# How to Delete the Document from PostgreSQL

Since there are permission issues with automated scripts, here are manual steps:

## Option 1: Using psql (Recommended)

```bash
psql mobius_rag -f delete_document.sql
```

Or connect interactively:
```bash
psql mobius_rag
```

Then run:
```sql
BEGIN;
DELETE FROM chunking_results WHERE document_id IN (
    SELECT id FROM documents WHERE filename LIKE '%01-05-26-MFL%'
);
DELETE FROM document_pages WHERE document_id IN (
    SELECT id FROM documents WHERE filename LIKE '%01-05-26-MFL%'
);
DELETE FROM documents WHERE filename LIKE '%01-05-26-MFL%';
COMMIT;
```

## Option 2: Using the API (if server is accessible)

Once PostgreSQL permissions are fixed, you can use:
```bash
python delete_via_api.py
```

## Option 3: Direct SQL commands

If you have database access through another tool (pgAdmin, DBeaver, etc.):

1. Connect to `mobius_rag` database
2. Run these commands in order:

```sql
DELETE FROM chunking_results WHERE document_id IN (
    SELECT id FROM documents WHERE filename LIKE '%01-05-26-MFL%'
);

DELETE FROM document_pages WHERE document_id IN (
    SELECT id FROM documents WHERE filename LIKE '%01-05-26-MFL%'
);

DELETE FROM documents WHERE filename LIKE '%01-05-26-MFL%';
```

## Verify deletion

```sql
SELECT COUNT(*) FROM documents WHERE filename LIKE '%01-05-26-MFL%';
-- Should return 0
```
