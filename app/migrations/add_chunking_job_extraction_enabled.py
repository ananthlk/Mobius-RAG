"""
Migration: add extraction_enabled (VARCHAR(10)) to chunking_jobs for hierarchical-only run mode.
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
            "WHERE table_name = 'chunking_jobs' AND column_name = 'extraction_enabled'"
        )
        if not exists:
            await conn.execute("ALTER TABLE chunking_jobs ADD COLUMN extraction_enabled VARCHAR(10)")
            await conn.execute(
                "UPDATE chunking_jobs SET extraction_enabled = 'true' WHERE extraction_enabled IS NULL"
            )
            print("  Added column chunking_jobs.extraction_enabled")
        else:
            print("  Column chunking_jobs.extraction_enabled already exists")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_chunking_job_extraction_enabled completed.")
