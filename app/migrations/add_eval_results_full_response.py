"""Migration: add ``full_response`` JSONB column to ``rag_eval_results``.

Stores the complete CorpusSearchAgentResponse for each eval row so the
frontend can render the full thinking pipeline (parser → partition →
pool → router → strategy execution → assembler) without needing per-
field columns. Idempotent.
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
        col_exists = await conn.fetchval("""
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name   = 'rag_eval_results'
              AND column_name  = 'full_response'
        """)
        if col_exists:
            print("  Column rag_eval_results.full_response already exists")
        else:
            await conn.execute(
                "ALTER TABLE rag_eval_results ADD COLUMN full_response JSONB"
            )
            print("  Added column rag_eval_results.full_response (JSONB)")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_eval_results_full_response completed.")
