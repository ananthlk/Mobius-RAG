"""Migration: add ``human_verdict`` columns to ``rag_eval_results``.

The LLM judge is a starting point but it makes mistakes — especially
on ambiguous queries where 'correct' vs 'partial' is judgment. The
human-override columns let an analyst inspect a result and overwrite
the judge's verdict; aggregates then prefer the human's call.

Effective verdict precedence (used by the API + dashboards):
    COALESCE(human_verdict, judge_verdict)

Idempotent.
"""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL


_COLUMNS = [
    ("human_verdict",     "TEXT"),                 # correct | partial | wrong | unable_to_verify
    ("human_reasoning",   "TEXT"),
    ("human_verdict_at",  "TIMESTAMPTZ"),
    ("human_verdict_by",  "TEXT"),
]


async def migrate():
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        for col_name, col_type in _COLUMNS:
            exists = await conn.fetchval("""
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name   = 'rag_eval_results'
                  AND column_name  = $1
            """, col_name)
            if exists:
                print(f"  Column rag_eval_results.{col_name} already exists")
            else:
                await conn.execute(
                    f"ALTER TABLE rag_eval_results ADD COLUMN {col_name} {col_type}"
                )
                print(f"  Added rag_eval_results.{col_name} ({col_type})")
        # Index on human_verdict so dashboards can quickly count overrides.
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_eval_results_human_verdict "
            "ON rag_eval_results (human_verdict) "
            "WHERE human_verdict IS NOT NULL"
        )
        print("  Ensured idx_eval_results_human_verdict")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_eval_results_human_verdict completed.")
