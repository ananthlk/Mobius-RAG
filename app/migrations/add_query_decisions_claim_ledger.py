"""Migration: add ``per_claim_ledger`` JSONB column to ``rag_query_decisions``.

Each element: {fact: str, status: "validated"|"unvalidated"|"contradicted",
               chunk_id: str|null, quote: str|null}

Owned by the EVAL agent — the rubric writer stamps evidence per fact.
Frontend reads this column to render the per-claim ledger in the chat
answer card and in the RAG admin UI drilldown.

Idempotent — safe to re-run.
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL


async def migrate():
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        exists = await conn.fetchval("""
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name   = 'rag_query_decisions'
              AND column_name  = 'per_claim_ledger'
        """)
        if exists:
            print("  Column rag_query_decisions.per_claim_ledger already exists — skipping")
        else:
            await conn.execute(
                "ALTER TABLE rag_query_decisions ADD COLUMN per_claim_ledger JSONB"
            )
            print("  Added column rag_query_decisions.per_claim_ledger (JSONB)")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_query_decisions_claim_ledger completed.")
