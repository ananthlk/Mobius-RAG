"""
Migration: add occurrences (INTEGER) to policy_lexicon_candidates if missing.
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
        exists = await conn.fetchval(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'policy_lexicon_candidates' AND column_name = 'occurrences'"
        )
        if not exists:
            await conn.execute("ALTER TABLE policy_lexicon_candidates ADD COLUMN occurrences INTEGER")
            print("  Added policy_lexicon_candidates.occurrences")
        print("  policy_lexicon_candidates occurrences ready")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_policy_lexicon_candidate_occurrences completed.")
