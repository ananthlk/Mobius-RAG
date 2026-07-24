"""Migration: add ``is_calibration`` column to rag_query_decisions.

Router's persist_decision() writes is_calibration to mark calibration-mode
rows (Eval's isolation runs) vs production rows. This column is required for
Eval's is_calibration-based dedup/filtering on the rag_query_decisions rows.

Schema addition (one column):
  - is_calibration BOOLEAN NOT NULL DEFAULT false

No backfill needed: the column has never existed, so no rows have ever been
written with it. This is a forward-only schema gap, not corrupted data.
Once the column exists, new persist_decision() calls (including Eval's
in-flight calibration run) will start writing it correctly.
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL


_ADD_COLUMN_SQL = """
ALTER TABLE public.rag_query_decisions
ADD COLUMN is_calibration BOOLEAN NOT NULL DEFAULT false;
"""


async def migrate():
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        # Check if column already exists
        exists = await conn.fetchval("""
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public' AND table_name='rag_query_decisions'
            AND column_name='is_calibration'
        """)
        if exists:
            print("  is_calibration column already exists — skipping")
            return

        await conn.execute(_ADD_COLUMN_SQL)
        print("  Added is_calibration BOOLEAN NOT NULL DEFAULT false to rag_query_decisions")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_rag_query_decisions_is_calibration completed.")
