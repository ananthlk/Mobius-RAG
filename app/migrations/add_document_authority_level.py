"""
Migration: add authority_level column to documents.
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
        exists = await conn.fetchval("""
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = 'authority_level'
        """)
        if not exists:
            await conn.execute("""
                ALTER TABLE public.documents ADD COLUMN authority_level VARCHAR(100)
            """)
            print("  Added column documents.authority_level")
        else:
            print("  Column documents.authority_level already exists")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_document_authority_level completed.")
