"""
Migration: create document_tags table.

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
              AND table_name = 'document_tags'
        """))
        if check.scalar_one_or_none():
            logger.info("  Table document_tags already exists")
            return

        await db.execute(text("""
            CREATE TABLE document_tags (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id UUID NOT NULL REFERENCES documents(id) UNIQUE,
                p_tags JSONB,
                d_tags JSONB,
                j_tags JSONB,
                created_at TIMESTAMP NOT NULL DEFAULT now(),
                updated_at TIMESTAMP NOT NULL DEFAULT now()
            )
        """))
        await db.execute(text("""
            CREATE INDEX ix_document_tags_document_id ON document_tags(document_id)
        """))
        await db.commit()
        logger.info("  Created table document_tags")
