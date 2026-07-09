"""
Migration: add chunk_d_tags / chunk_p_tags / chunk_j_tags (JSONB) to
rag_published_embeddings.

DDL only — columns are added here. Backfill is done via the admin endpoint
POST /admin/backfill_chunk_tags, which runs per-document batches so large
table MVCC cost is spread across many small transactions instead of one
hours-long single UPDATE.
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
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_chunk_tags_to_published completed.")
