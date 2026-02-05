"""
Migration: add source_url (TEXT) to document_pages for scraped page original URL.
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
            "WHERE table_schema = 'public' AND table_name = 'document_pages' AND column_name = 'source_url'"
        )
        if not exists:
            await conn.execute("ALTER TABLE public.document_pages ADD COLUMN source_url TEXT")
            print("  Added column document_pages.source_url")
        else:
            print("  Column document_pages.source_url already exists")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_document_page_source_url completed.")
