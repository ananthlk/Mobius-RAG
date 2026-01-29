"""
Migration: add display_name column to documents.
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
            "WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = 'display_name'"
        )
        if not exists:
            await conn.execute("""
                ALTER TABLE public.documents ADD COLUMN display_name VARCHAR(255)
            """)
            print("  Added column documents.display_name")
        else:
            print("  Column documents.display_name already exists")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_document_display_name completed.")
