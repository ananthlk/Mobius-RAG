"""
Migration: add chunking_config_snapshot JSONB column to chunking_jobs.

Stores the resolved run configuration at job start so every job is self-describing.
Safe to run multiple times (checks column existence first).
"""
import logging

from sqlalchemy import text
from app.database import AsyncSessionLocal

logger = logging.getLogger(__name__)


async def migrate() -> None:
    async with AsyncSessionLocal() as db:
        # Check if column already exists
        check = await db.execute(text("""
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'chunking_jobs'
              AND column_name = 'chunking_config_snapshot'
        """))
        if check.scalar_one_or_none():
            logger.info("  Column chunking_jobs.chunking_config_snapshot already exists")
            return

        await db.execute(text("""
            ALTER TABLE chunking_jobs
            ADD COLUMN chunking_config_snapshot JSONB
        """))
        await db.commit()
        logger.info("  Added column chunking_jobs.chunking_config_snapshot")
