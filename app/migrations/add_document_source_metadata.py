"""
Migration: add source_metadata (JSONB) to documents for scraped provenance.
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
            "WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = 'source_metadata'"
        )
        if not exists:
            await conn.execute("ALTER TABLE public.documents ADD COLUMN source_metadata JSONB")
            print("  Added column documents.source_metadata")
        else:
            print("  Column documents.source_metadata already exists")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_document_source_metadata completed.")
