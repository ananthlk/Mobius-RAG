"""
Migration: add chunk_d_tags / chunk_p_tags / chunk_j_tags (JSONB) to
rag_published_embeddings.

These carry per-chunk topic tags from policy_paragraphs so retrieval can
boost at chunk level (not just document level) without a query-time join.

Backfill: JOIN rag_published_embeddings → policy_paragraphs on
  (document_id, page_number, rpe.paragraph_index = pp.order_index)
with DISTINCT ON dedup — policy_paragraphs has duplicate rows from
repeated ingest runs that append instead of upsert (see Sunshine Provider
Manual: 14 rows for page 121, order_index=3). We take the row with the
latest created_at to get the freshest tags. Rows with no matching
policy_paragraph row stay NULL (scraped/fact chunks with no Path-B tags).
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL


async def migrate() -> None:
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        # 1 — add columns (idempotent)
        for col_name, col_def in [
            ("chunk_d_tags", "JSONB"),
            ("chunk_p_tags", "JSONB"),
            ("chunk_j_tags", "JSONB"),
        ]:
            exists = await conn.fetchval(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'rag_published_embeddings' AND column_name = $1",
                col_name,
            )
            if not exists:
                await conn.execute(
                    f"ALTER TABLE rag_published_embeddings ADD COLUMN {col_name} {col_def}"
                )
                print(f"  Added column rag_published_embeddings.{col_name}")
            else:
                print(f"  Column rag_published_embeddings.{col_name} already exists")

        # 2 — backfill from policy_paragraphs (idempotent — only NULL rows)
        # DISTINCT ON dedup: policy_paragraphs has duplicate (document_id,
        # page_number, order_index) rows from repeated ingest runs. Take the
        # latest by created_at so the JOIN is 1:1 and never fans out.
        result = await conn.execute(
            """
            UPDATE rag_published_embeddings rpe
            SET
                chunk_d_tags = pp_dedup.d_tags,
                chunk_p_tags = pp_dedup.p_tags,
                chunk_j_tags = pp_dedup.j_tags
            FROM (
                SELECT DISTINCT ON (document_id, page_number, order_index)
                    document_id, page_number, order_index, d_tags, p_tags, j_tags
                FROM policy_paragraphs
                ORDER BY document_id, page_number, order_index, created_at DESC
            ) pp_dedup
            WHERE rpe.document_id     = pp_dedup.document_id
              AND rpe.page_number     = pp_dedup.page_number
              AND rpe.paragraph_index = pp_dedup.order_index
              AND rpe.chunk_d_tags IS NULL
            """
        )
        print(f"  Backfill complete: {result}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_chunk_tags_to_published completed.")
