"""
Migration: create drive_connections table for Google Drive OAuth tokens.

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
              AND table_name = 'drive_connections'
        """))
        if check.scalar_one_or_none():
            logger.info("  Table drive_connections already exists")
            return

        await db.execute(text("""
            CREATE TABLE drive_connections (
                session_id VARCHAR(64) PRIMARY KEY,
                access_token TEXT NOT NULL,
                refresh_token TEXT,
                expires_at TIMESTAMP,
                email VARCHAR(255),
                created_at TIMESTAMP NOT NULL DEFAULT now()
            )
        """))
        await db.commit()
        logger.info("  Created table drive_connections")
