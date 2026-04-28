"""Migration: add full-text search column + GIN index to rag_published_embeddings.

Adds a GENERATED ALWAYS AS STORED tsvector column so BM25 queries can
use a GIN index rather than recomputing to_tsvector on every scan.

Safe to re-run (idempotent). Column and index are created with IF NOT EXISTS
guards; the generated column check inspects information_schema first.
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
        # Check if generated column already exists
        exists = await conn.fetchval("""
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'rag_published_embeddings'
              AND column_name = 'search_vec'
        """)
        if exists:
            print("  search_vec column already exists — skipping ADD COLUMN")
        else:
            # GENERATED ALWAYS AS STORED auto-updates when text changes.
            # 'english' config strips stopwords & applies stemming (matches
            # plainto_tsquery / websearch_to_tsquery defaults).
            await conn.execute("""
                ALTER TABLE rag_published_embeddings
                ADD COLUMN search_vec tsvector
                GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED
            """)
            print("  Added search_vec tsvector GENERATED column")

        # GIN index is safe to create after the column exists (even if it was
        # just added — Postgres auto-populates GENERATED columns on ALTER TABLE).
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS rag_published_fts_gin
            ON rag_published_embeddings USING GIN(search_vec)
        """)
        print("  Ensured GIN index: rag_published_fts_gin")

        # Verify row count for sanity
        total = await conn.fetchval("SELECT count(*) FROM rag_published_embeddings")
        indexed = await conn.fetchval(
            "SELECT count(*) FROM rag_published_embeddings WHERE search_vec IS NOT NULL"
        )
        print(f"  Rows: total={total}  search_vec_populated={indexed}")
    finally:
        await conn.close()


def main() -> None:
    asyncio.run(migrate())
    print("Migration add_rag_published_fts completed.")


if __name__ == "__main__":
    main()
