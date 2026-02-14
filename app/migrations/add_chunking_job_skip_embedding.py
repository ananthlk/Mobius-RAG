"""
Migration: add skip_embedding (VARCHAR(10)) to chunking_jobs.
When 'true', the worker will NOT auto-enqueue an embedding job after completion.
Used by retag jobs so lexicon-only re-tagging does not trigger wasteful re-embedding.
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
        exists = await conn.fetchval(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'chunking_jobs' AND column_name = 'skip_embedding'"
        )
        if not exists:
            await conn.execute("ALTER TABLE chunking_jobs ADD COLUMN skip_embedding VARCHAR(10)")
            print("  Added column chunking_jobs.skip_embedding")
        else:
            print("  Column chunking_jobs.skip_embedding already exists")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_chunking_job_skip_embedding completed.")
