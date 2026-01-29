#!/usr/bin/env python3
"""Migration: Add text_markdown column to document_pages for structured reader display."""
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
        # Use public schema explicitly so we always check/alter the table the app uses
        exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'document_pages' AND column_name = 'text_markdown'
            )
        """)
        if not exists:
            await conn.execute("ALTER TABLE public.document_pages ADD COLUMN text_markdown TEXT NULL")
            print("✓ Added column document_pages.text_markdown")
        else:
            print("✓ Column document_pages.text_markdown already exists")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
