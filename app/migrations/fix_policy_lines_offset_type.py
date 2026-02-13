"""
Migration: fix policy_lines.offset_match_quality from varchar to double precision.

Idempotent — safe to re-run.
"""
import logging

from sqlalchemy import text

logger = logging.getLogger(__name__)

MIGRATION_SQL = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'policy_lines'
          AND column_name = 'offset_match_quality'
          AND data_type = 'character varying'
    ) THEN
        ALTER TABLE policy_lines
            ALTER COLUMN offset_match_quality TYPE double precision
            USING offset_match_quality::double precision;
        RAISE NOTICE 'policy_lines.offset_match_quality changed to double precision';
    END IF;
END $$;
"""


async def migrate():
    from app.database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        await session.execute(text(MIGRATION_SQL))
        await session.commit()
        logger.info("Migration fix_policy_lines_offset_type complete")
