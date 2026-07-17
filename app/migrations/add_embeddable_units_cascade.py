"""
Migration: add ON DELETE CASCADE to embeddable_units.document_id FK.

Without CASCADE, DELETE FROM documents fails with a FK violation whenever
embeddable_units rows exist for that document_id and the caller hasn't
explicitly deleted them first (e.g. the delete_document endpoint).  Adding
CASCADE means any document delete automatically cleans up its embeddable_units,
making the code-level deletes belt-and-suspenders rather than load-bearing.

Safe to run multiple times: checks whether CASCADE is already present first.
"""
import logging

from sqlalchemy import text
from app.database import AsyncSessionLocal

logger = logging.getLogger(__name__)

_CHECK_CASCADE = """
    SELECT 1
    FROM information_schema.referential_constraints rc
    JOIN information_schema.table_constraints tc
      ON rc.constraint_name = tc.constraint_name
     AND tc.table_name = 'embeddable_units'
    WHERE rc.delete_rule = 'CASCADE'
      AND tc.constraint_type = 'FOREIGN KEY'
"""


async def migrate() -> None:
    async with AsyncSessionLocal() as db:
        already = await db.execute(text(_CHECK_CASCADE))
        if already.scalar_one_or_none():
            logger.info("  embeddable_units FK already has ON DELETE CASCADE — skipping")
            return

        await db.execute(text(
            "ALTER TABLE embeddable_units "
            "DROP CONSTRAINT IF EXISTS embeddable_units_document_id_fkey"
        ))
        await db.execute(text(
            "ALTER TABLE embeddable_units "
            "ADD CONSTRAINT embeddable_units_document_id_fkey "
            "FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE"
        ))
        await db.commit()
        logger.info("  embeddable_units FK updated to ON DELETE CASCADE")
