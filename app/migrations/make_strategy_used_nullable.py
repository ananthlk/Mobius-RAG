"""Migration: make strategy_used column nullable.

The strategy_used column is a legacy artifact from the pre-July 23 architecture
(corpus_search_router). The new Router architecture (persist_decision) uses
strategy_chosen and strategy_sequence instead, and never writes to strategy_used.

The NOT NULL constraint on this dead column blocked persist_decision() from
inserting any rows. Making it nullable preserves existing rows while allowing
new rows to have NULL for this legacy field (which is the correct state given
the new architecture doesn't use it).

This is a schema cleanup to reconcile old (July 15 schema) with new (July 23+
architecture).
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL


_ALTER_SQL = """
ALTER TABLE public.rag_query_decisions
ALTER COLUMN strategy_used DROP NOT NULL;
"""


async def migrate():
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        # Check if column is already nullable
        nullable = await conn.fetchval("""
            SELECT is_nullable FROM information_schema.columns
            WHERE table_schema='public' AND table_name='rag_query_decisions'
            AND column_name='strategy_used'
        """)

        if nullable == "YES":
            print("  strategy_used is already nullable — skipping")
            return

        await conn.execute(_ALTER_SQL)
        print("  Made strategy_used nullable (legacy column, unused by new Router architecture)")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration make_strategy_used_nullable completed.")
