"""Migration: rebuild search_vec on rag_published_embeddings as a weighted
multi-field tsvector.

Why
---
The original ``add_rag_published_fts`` migration produced a single-field
tsvector over ``text`` only.  Filename, display name, summary, and
section/chapter paths were invisible to BM25 — so a query like
``"DME prior auth"`` could not match a document whose filename literally
contained "DME-Prior-Authorization-Policy.pdf" if the body text used
different terminology.

This migration drops and recreates ``search_vec`` as a weighted union:

  A: document_filename + document_display_name        (highest weight)
  B: summary
  C: section_path + chapter_path
  D: text                                              (lowest weight)

Postgres' ts_rank uses these letters as weight classes (defaults
{A:1.0, B:0.4, C:0.2, D:0.1}), so a hit in the filename outranks a hit
in the body — which is exactly the recall+precision behaviour we want.

Idempotent — the column drop uses IF EXISTS and the GIN index uses
IF NOT EXISTS.  The operator runs this via Cloud Run Job (same pattern as
``add_pgvector_columns``); we do NOT auto-run on prod.
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
        # CASCADE drops the dependent GIN index too — we'll recreate it.
        await conn.execute(
            """
            ALTER TABLE rag_published_embeddings
              DROP COLUMN IF EXISTS search_vec CASCADE
            """
        )
        print("  Dropped existing search_vec column (and dependent index)")

        await conn.execute(
            """
            ALTER TABLE rag_published_embeddings
              ADD COLUMN search_vec tsvector
              GENERATED ALWAYS AS (
                setweight(to_tsvector('english', coalesce(document_filename, '')),     'A') ||
                setweight(to_tsvector('english', coalesce(document_display_name, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(summary, '')),               'B') ||
                setweight(to_tsvector('english', coalesce(section_path, '')),          'C') ||
                setweight(to_tsvector('english', coalesce(chapter_path, '')),          'C') ||
                setweight(to_tsvector('english', coalesce(text, '')),                  'D')
              ) STORED
            """
        )
        print("  Added multi-field weighted search_vec (A=filename, B=summary, C=path, D=text)")

        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS rag_published_fts_gin
              ON rag_published_embeddings USING GIN(search_vec)
            """
        )
        print("  Ensured GIN index: rag_published_fts_gin")

        # ── Verify on one row that filename tokens make it into the vector ──
        sample = await conn.fetchrow(
            """
            SELECT id, document_filename, search_vec::text AS sv
            FROM rag_published_embeddings
            WHERE document_filename IS NOT NULL AND document_filename <> ''
            LIMIT 1
            """
        )
        if sample:
            print(f"  Sample row id={sample['id']}  filename={sample['document_filename']!r}")
            sv_preview = (sample["sv"] or "")[:200]
            print(f"  search_vec[:200]={sv_preview!r}")

        total = await conn.fetchval("SELECT count(*) FROM rag_published_embeddings")
        indexed = await conn.fetchval(
            "SELECT count(*) FROM rag_published_embeddings WHERE search_vec IS NOT NULL"
        )
        print(f"  Rows: total={total}  search_vec_populated={indexed}")
    finally:
        await conn.close()


def main() -> None:
    asyncio.run(migrate())
    print("Migration rebuild_rag_published_fts_multifield completed.")


if __name__ == "__main__":
    main()
