"""
Migration: add ON DELETE CASCADE to all document child table FKs.

Several tables have FKs to documents(id) without CASCADE, causing FK
violation 500s when the delete_document endpoint is called on a published
doc (document_tags, hierarchical_chunks, chunking_jobs, etc.).

Approach: query information_schema for all FKs to documents that are NOT
already CASCADE, then ALTER each one to add CASCADE.  Running this once
fixes the gap regardless of which child tables are missing it.

Safe to run multiple times: only alters constraints that still lack CASCADE.
"""
import logging

from sqlalchemy import text
from app.database import AsyncSessionLocal

logger = logging.getLogger(__name__)

_FIND_NON_CASCADE = """
    SELECT
        tc.table_name,
        kcu.column_name,
        rc.constraint_name
    FROM information_schema.table_constraints tc
    JOIN information_schema.key_column_usage kcu
      ON tc.constraint_name = kcu.constraint_name
     AND tc.table_schema   = kcu.table_schema
    JOIN information_schema.referential_constraints rc
      ON tc.constraint_name = rc.constraint_name
    JOIN information_schema.constraint_column_usage ccu
      ON ccu.constraint_name = rc.unique_constraint_name
     AND ccu.table_schema   = tc.table_schema
    WHERE tc.constraint_type = 'FOREIGN KEY'
      AND ccu.table_name     = 'documents'
      AND ccu.column_name    = 'id'
      AND rc.delete_rule     != 'CASCADE'
      AND tc.table_schema    = 'public'
    ORDER BY tc.table_name
"""


async def migrate() -> None:
    async with AsyncSessionLocal() as db:
        rows = await db.execute(text(_FIND_NON_CASCADE))
        targets = rows.fetchall()

        if not targets:
            logger.info("  All document child FKs already have ON DELETE CASCADE — skipping")
            return

        for (table_name, col_name, constraint_name) in targets:
            logger.info("  Adding CASCADE to %s.%s (%s)", table_name, col_name, constraint_name)
            await db.execute(text(
                f'ALTER TABLE "{table_name}" '
                f'DROP CONSTRAINT IF EXISTS "{constraint_name}"'
            ))
            await db.execute(text(
                f'ALTER TABLE "{table_name}" '
                f'ADD CONSTRAINT "{constraint_name}" '
                f'FOREIGN KEY ("{col_name}") REFERENCES documents(id) ON DELETE CASCADE'
            ))

        await db.commit()
        logger.info("  Updated %d document child FKs to ON DELETE CASCADE", len(targets))
