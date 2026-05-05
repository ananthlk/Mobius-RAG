"""Migration: add composite (document_id, paragraph_index, page_number) index
to rag_published_embeddings.

Without this index the sibling-fetch query (used by the reranker neighbor-text
pass) does a full sequential scan of 882K rows per seed chunk, causing 13-17s
rerank latency. The composite covers the exact join conditions in
_fetch_sibling_chunks_batch:

    m.document_id = r.doc_id::uuid
    AND m.paragraph_index BETWEEN r.lo  AND r.hi
    AND m.page_number     BETWEEN r.plo AND r.phi

CONCURRENTLY so the table stays readable during the build (~2-3 min on 882K rows).
Safe to re-run (CREATE INDEX IF NOT EXISTS).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg  # noqa: E402
from app.config import DATABASE_URL  # noqa: E402


async def migrate() -> None:
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        total = await conn.fetchval("SELECT count(*) FROM rag_published_embeddings")
        print(f"  rag_published_embeddings rows: {total:,}")

        # Check if index already exists
        exists = await conn.fetchval("""
            SELECT 1 FROM pg_indexes
            WHERE tablename = 'rag_published_embeddings'
              AND indexname  = 'idx_rpe_document_id_para_page'
        """)
        if exists:
            print("  idx_rpe_document_id_para_page already exists — skipping")
        else:
            print("  Creating idx_rpe_document_id_para_page CONCURRENTLY …")
            # CONCURRENTLY cannot run inside a transaction; asyncpg auto-commits
            # single execute() calls when not in an explicit transaction.
            await conn.execute("""
                CREATE INDEX CONCURRENTLY idx_rpe_document_id_para_page
                ON rag_published_embeddings (document_id, paragraph_index, page_number)
            """)
            print("  Created idx_rpe_document_id_para_page")

        # Verify
        rows = await conn.fetch("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'rag_published_embeddings'
            ORDER BY indexname
        """)
        print("  Indexes now:", [r["indexname"] for r in rows])
    finally:
        await conn.close()


def main() -> None:
    asyncio.run(migrate())
    print("Migration add_rpe_document_id_index completed.")


if __name__ == "__main__":
    main()
