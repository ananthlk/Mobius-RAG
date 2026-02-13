"""
Migration: add normalized_key, proposed_tag_key and unique constraint to policy_lexicon_candidate_catalog.
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
            "SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'policy_lexicon_candidate_catalog'"
        )
        if not table_exists:
            print("  policy_lexicon_candidate_catalog table does not exist; skipping")
            return
        for col, typ in (("normalized_key", "VARCHAR(300)"), ("proposed_tag_key", "VARCHAR(300)")):
            exists = await conn.fetchval(
                "SELECT 1 FROM information_schema.columns "
                f"WHERE table_schema = 'public' AND table_name = 'policy_lexicon_candidate_catalog' AND column_name = '{col}'"
            )
            if not exists:
                await conn.execute(f"ALTER TABLE policy_lexicon_candidate_catalog ADD COLUMN {col} {typ}")
                print(f"  Added policy_lexicon_candidate_catalog.{col}")

        # Unique constraint for upsert
        try:
            await conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_policy_lexicon_catalog_uniq "
                "ON policy_lexicon_candidate_catalog (candidate_type, normalized_key, proposed_tag_key)"
            )
            print("  Unique index on (candidate_type, normalized_key, proposed_tag_key) ready")
        except Exception as e:
            if "already exists" not in str(e).lower():
                print(f"  Note: {e}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_policy_lexicon_candidate_catalog completed.")
