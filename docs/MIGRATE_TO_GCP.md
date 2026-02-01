# Migrate Mobius RAG Data to Cloud SQL

Steps to migrate your local database (documents, chunks, facts, chunking events, etc.) to GCP Cloud SQL.

## Why are there no documents in production?

**Production (Cloud Run)** uses **Cloud SQL**: instance `mobius-platform-db`, database `mobius_rag`. That database is **separate** from:

- Your **local PostgreSQL** (e.g. `localhost:5432/mobius_rag`) where you may have uploaded documents
- Any **other** Cloud SQL instance or database

If you see empty document list in production, it usually means:

1. **Data was never migrated** – Documents exist only in local (or another DB). The app creates **schema** on first run (migrations), but **data** is not copied unless you run the migration below.
2. **Wrong database** – Production might be pointing at a different Cloud SQL instance/database than the one you loaded data into.

**Fix:** Run the migration steps below to copy data from your local (or a backup) into the **same** Cloud SQL database that production uses (`mobius-platform-db` / `mobius_rag`).

---

## Prerequisites

- Cloud SQL instance `mobius-platform-db` created
- Database `mobius_rag` exists
- Postgres password set: `gcloud sql users set-password postgres --instance=mobius-platform-db --password=YOUR_PASSWORD`

## Migration Steps

### 1. Create a dump from local (if you don't have one)

```bash
# Replace user/host if different
pg_dump -h localhost -U ananth -d mobius_rag --no-owner --no-acl > mobius_rag_backup.sql
```

### 2. Run migration via Cloud SQL Auth Proxy

```bash
CLOUD_SQL_PASSWORD=YOUR_PASSWORD ./scripts/migrate_to_gcp_psql.sh mobius_rag_backup.sql
```

This script:
- Prepares the dump (replaces owner with postgres, removes DROP statements)
- Starts Cloud SQL Auth Proxy on port 5433
- Imports via psql

### 3. Run embedding migration (pgvector + embedding tables)

```bash
# Start proxy in background (if not running)
./cloud-sql-proxy mobiusos-new:us-central1:mobius-platform-db --port 5433 &

# Run migration
DATABASE_URL="postgresql+asyncpg://postgres:YOUR_PASSWORD@127.0.0.1:5433/mobius_rag" \
  python -m app.migrations.add_embedding_tables
```

## DATABASE_URL for Production

For the VM or Cloud Run, use:

```
DATABASE_URL=postgresql+asyncpg://postgres:YOUR_PASSWORD@/cloudsql/mobiusos-new:us-central1:mobius-platform-db/mobius_rag
```

(With Cloud SQL Unix socket) or via Cloud SQL Auth Proxy:

```
DATABASE_URL=postgresql+asyncpg://postgres:YOUR_PASSWORD@127.0.0.1:5432/mobius_rag
```

**Important**: Store the password in Secret Manager or a secure .env file. Do not commit it.
