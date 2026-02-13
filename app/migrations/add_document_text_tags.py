"""
Migration: create document_text_tags table for user-applied text-range tags.

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
              AND table_name = 'document_text_tags'
        """))
        if check.scalar_one_or_none():
            logger.info("  Table document_text_tags already exists")
            return

        await db.execute(text("""
            CREATE TABLE document_text_tags (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id UUID NOT NULL REFERENCES documents(id),
                page_number INTEGER NOT NULL,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                tagged_text TEXT NOT NULL,
                tag VARCHAR(100) NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT now()
            )
        """))
        await db.execute(text("""
            CREATE INDEX ix_document_text_tags_document_id ON document_text_tags(document_id)
        """))
        await db.execute(text("""
            CREATE INDEX ix_document_text_tags_doc_page ON document_text_tags(document_id, page_number)
        """))
        await db.commit()
        logger.info("  Created table document_text_tags")
