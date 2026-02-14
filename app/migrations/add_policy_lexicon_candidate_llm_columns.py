"""
Migration: add LLM triage columns to policy_lexicon_candidates.

New nullable columns:
  llm_verdict       TEXT     -- new_tag | alias | reject
  llm_confidence    FLOAT    -- 0.0 - 1.0
  llm_reason        TEXT     -- one-line explanation
  llm_suggested_parent TEXT  -- parent code suggestion
  llm_suggested_code   TEXT  -- full tag code suggestion
  llm_suggested_kind   TEXT  -- p | d | j
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
        table_exists = await conn.fetchval(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'policy_lexicon_candidates'"
        )
        if not table_exists:
            print("  policy_lexicon_candidates table does not exist; skipping")
            return

        new_columns = [
            ("llm_verdict", "TEXT"),
            ("llm_confidence", "DOUBLE PRECISION"),
            ("llm_reason", "TEXT"),
            ("llm_suggested_parent", "TEXT"),
            ("llm_suggested_code", "TEXT"),
            ("llm_suggested_kind", "TEXT"),
        ]

        for col, typ in new_columns:
            exists = await conn.fetchval(
                "SELECT 1 FROM information_schema.columns "
                f"WHERE table_schema = 'public' AND table_name = 'policy_lexicon_candidates' AND column_name = '{col}'"
            )
            if not exists:
                await conn.execute(f"ALTER TABLE policy_lexicon_candidates ADD COLUMN {col} {typ}")
                print(f"  + added column policy_lexicon_candidates.{col} ({typ})")
            else:
                print(f"  column policy_lexicon_candidates.{col} already exists")

        # Index on llm_verdict for fast filtering in the UI
        idx_name = "ix_plc_llm_verdict"
        idx_exists = await conn.fetchval(
            f"SELECT 1 FROM pg_indexes WHERE indexname = '{idx_name}'"
        )
        if not idx_exists:
            await conn.execute(
                f"CREATE INDEX {idx_name} ON policy_lexicon_candidates (llm_verdict) WHERE llm_verdict IS NOT NULL"
            )
            print(f"  + created index {idx_name}")

        print("  LLM triage columns migration complete")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
