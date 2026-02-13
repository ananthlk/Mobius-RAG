"""
Migration: create embeddable_units table.

Safe to run multiple times (checks table existence first).
"""
import logging

from sqlalchemy import text
from app.database import AsyncSessionLocal

logger = logging.getLogger(__name__)


async def migrate() -> None:
    async with AsyncSessionLocal() as db:
        check = await db.execute(text("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = 'embeddable_units'
        """))
        if check.scalar_one_or_none():
            logger.info("  Table embeddable_units already exists")
            return

        await db.execute(text("""
            CREATE TABLE embeddable_units (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id UUID NOT NULL REFERENCES documents(id),
                generator_id VARCHAR(10),
                source_type VARCHAR(30) NOT NULL,
                source_id UUID NOT NULL,
                text TEXT NOT NULL,
                page_number INTEGER,
                paragraph_index INTEGER,
                section_path VARCHAR(500),
                metadata JSONB DEFAULT '{}'::jsonb,
                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP NOT NULL DEFAULT now()
            )
        """))
        await db.execute(text("""
            CREATE INDEX ix_embeddable_units_document_id ON embeddable_units(document_id)
        """))
        await db.execute(text("""
            CREATE INDEX ix_embeddable_units_status ON embeddable_units(status)
        """))
        await db.commit()
        logger.info("  Created table embeddable_units with indexes")
