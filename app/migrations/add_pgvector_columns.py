"""Migration: add pgvector typed columns + HNSW indexes for Chroma → pgvector switch.

Step 3 of the Chroma → pgvector migration. After the ``mobius-chroma`` VM
went unreachable for 2h on 2026-04-27 and broke ingestion, we are moving
the durable vector store into the existing Cloud SQL Postgres so it
inherits the same HA and backup story as the rest of the app.

This migration is **read-path only**: it adds a typed ``vector(1536)``
column alongside the existing ``embedding JSONB`` column and backfills
from JSONB. Workers and ``publish_sync.py`` continue to write JSONB
(parallel-write comes in Step 5). The new column is what
``PgVectorStore.search()`` queries.

Properties:

* Idempotent — ``CREATE EXTENSION IF NOT EXISTS``, ``ADD COLUMN IF
  NOT EXISTS``, ``UPDATE ... WHERE embedding_vec IS NULL``,
  ``CREATE INDEX IF NOT EXISTS``. Safe to re-run.
* Per-row backfill loop — if a JSONB row is malformed (not a flat
  list of 1536 floats) we log the doc/row id and skip it, rather than
  aborting the whole batch and leaving the table half-backfilled.
* Prints row counts before/after each phase so the operator can
  sanity-check the result against the JSONB row count.
* Exits non-zero on any phase failure (asyncpg raises → CLI propagates).

Usage (run via Cloud Run Job or one-shot exec — DO NOT run on prod
without explicit operator approval):

    python -m app.migrations.add_pgvector_columns

The script connects via ``DATABASE_URL`` so the Cloud SQL connector
socket path is honoured the same way every other migration uses it.
"""
import asyncio
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg  # noqa: E402

from app.config import DATABASE_URL  # noqa: E402

# Embedding dimensionality. Must match EMBEDDING_DIMENSIONS in
# app/config.py and the OpenAI/Vertex 1536-d models we use today
# (text-embedding-3-small at 1536, gemini-embedding-001 at 1536).
VECTOR_DIM = 1536

TABLES = ("chunk_embeddings", "rag_published_embeddings")


async def _ensure_extension(conn: asyncpg.Connection) -> None:
    """Enable pgvector. No-op if already installed."""
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    print("  Ensured extension: vector")


async def _add_column(conn: asyncpg.Connection, table: str) -> None:
    """Add ``embedding_vec vector(1536)`` if missing."""
    await conn.execute(
        f"ALTER TABLE {table} "
        f"ADD COLUMN IF NOT EXISTS embedding_vec vector({VECTOR_DIM})"
    )
    print(f"  Ensured column: {table}.embedding_vec")


async def _row_counts(conn: asyncpg.Connection, table: str) -> tuple[int, int, int]:
    """(total, with_jsonb, with_vec)"""
    total = await conn.fetchval(f"SELECT count(*) FROM {table}")
    with_jsonb = await conn.fetchval(
        f"SELECT count(*) FROM {table} WHERE embedding IS NOT NULL"
    )
    with_vec = await conn.fetchval(
        f"SELECT count(*) FROM {table} WHERE embedding_vec IS NOT NULL"
    )
    return int(total or 0), int(with_jsonb or 0), int(with_vec or 0)


async def _backfill(conn: asyncpg.Connection, table: str) -> tuple[int, int]:
    """Backfill ``embedding_vec`` from ``embedding`` JSONB. Returns
    (filled, skipped). Skips rows whose JSONB is not a flat list of
    floats with the expected dimensionality — logs the offending id
    so the operator can investigate, rather than silently writing
    garbage."""
    rows = await conn.fetch(
        f"SELECT id, embedding FROM {table} "
        f"WHERE embedding_vec IS NULL AND embedding IS NOT NULL"
    )
    filled = 0
    skipped = 0
    for r in rows:
        row_id = r["id"]
        raw = r["embedding"]
        # asyncpg returns JSONB as str (we don't register a json codec
        # on this connection), so parse defensively.
        try:
            value = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            print(f"  SKIP {table} id={row_id}: embedding JSONB is not valid JSON")
            skipped += 1
            continue
        if not isinstance(value, list) or len(value) != VECTOR_DIM:
            shape = type(value).__name__
            length = len(value) if isinstance(value, list) else "n/a"
            print(
                f"  SKIP {table} id={row_id}: expected list[{VECTOR_DIM}], "
                f"got {shape} (len={length})"
            )
            skipped += 1
            continue
        if not all(isinstance(x, (int, float)) for x in value):
            print(f"  SKIP {table} id={row_id}: list contains non-numeric values")
            skipped += 1
            continue
        # pgvector accepts the canonical '[f1,f2,...]' text form.
        text_form = "[" + ",".join(repr(float(x)) for x in value) + "]"
        try:
            await conn.execute(
                f"UPDATE {table} SET embedding_vec = $1::vector WHERE id = $2",
                text_form,
                row_id,
            )
            filled += 1
        except asyncpg.PostgresError as e:
            print(f"  SKIP {table} id={row_id}: pgvector cast failed: {e}")
            skipped += 1
    return filled, skipped


async def _build_index(conn: asyncpg.Connection, table: str) -> None:
    """HNSW index over the new column. ``m=16, ef_construction=64`` are
    pgvector defaults — at our row counts (1k–10k) the build is seconds.
    Cosine ops because that's the metric the rest of the stack uses
    (Chroma is configured for cosine in ``ChromaVectorStore`` too)."""
    index_name = f"{table}_vec_hnsw"
    await conn.execute(
        f"CREATE INDEX IF NOT EXISTS {index_name} "
        f"ON {table} USING hnsw (embedding_vec vector_cosine_ops)"
    )
    print(f"  Ensured HNSW index: {index_name}")


async def migrate() -> None:
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        await _ensure_extension(conn)
        for table in TABLES:
            print(f"\n[{table}]")
            await _add_column(conn, table)
            total, with_jsonb, with_vec_before = await _row_counts(conn, table)
            print(
                f"  Pre-backfill: total={total} with_jsonb={with_jsonb} "
                f"with_vec={with_vec_before}"
            )
            filled, skipped = await _backfill(conn, table)
            print(f"  Backfill: filled={filled} skipped={skipped}")
            _, _, with_vec_after = await _row_counts(conn, table)
            print(f"  Post-backfill: with_vec={with_vec_after}")
            await _build_index(conn, table)
    finally:
        await conn.close()


def main() -> None:
    asyncio.run(migrate())
    print("\nMigration add_pgvector_columns completed.")


if __name__ == "__main__":
    main()
