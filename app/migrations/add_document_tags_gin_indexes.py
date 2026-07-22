"""Migration: GIN indexes on document_tags.{d,p,j}_tags (jsonb ? operator).

Found 2026-07-22 during the arm-a latency trace: the dtag arm's IDF query
`SELECT COUNT(DISTINCT document_id) FROM document_tags WHERE d_tags ? '<key>'`
was doing a Seq Scan on document_tags (9,529 rows) — 300-577ms per call, and
the dtag arm fires one per d-tag key + the fetch, so ~900ms of arm-a latency.
The jsonb ``?`` (key-exists) operator cannot use a btree; it needs a GIN index.

document_tags had only btree(id) / btree(document_id) — no GIN on the tag
columns. Adding GIN(d_tags/p_tags/j_tags) turns the ``?`` predicate into an
index scan: measured 300ms Seq Scan -> 1.1ms Index (268x). Tiny table, so the
indexes are small and build instantly.

This is orthogonal to the BM25/vector/dtag chunk-count gates (RAG's code) —
it's a pure indexing fix for every ``d_tags ? key`` / ``p_tags ? key`` /
``j_tags ? key`` query, of which the dtag-arm IDF count is the hot one.

Idempotent — CREATE INDEX IF NOT EXISTS.
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL


_DDL = [
    "CREATE INDEX IF NOT EXISTS ix_document_tags_d_tags_gin ON document_tags USING gin (d_tags);",
    "CREATE INDEX IF NOT EXISTS ix_document_tags_p_tags_gin ON document_tags USING gin (p_tags);",
    "CREATE INDEX IF NOT EXISTS ix_document_tags_j_tags_gin ON document_tags USING gin (j_tags);",
    "ANALYZE document_tags;",
]


async def migrate():
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        for ddl in _DDL:
            await conn.execute(ddl)
        print(f"  Applied {len(_DDL)} statements — GIN indexes on document_tags d/p/j_tags")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_document_tags_gin_indexes completed.")
