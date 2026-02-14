"""
Migration: add lexicon_revision (BIGINT) and tagged_at (TIMESTAMP) to document_tags.
Tracks which lexicon revision was used to tag each document, enabling stale-document detection.
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
        for col_name, col_def in [
            ("lexicon_revision", "BIGINT"),
            ("tagged_at", "TIMESTAMP"),
        ]:
            exists = await conn.fetchval(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = 'document_tags' AND column_name = $1",
                col_name,
            )
            if not exists:
                await conn.execute(f"ALTER TABLE document_tags ADD COLUMN {col_name} {col_def}")
                print(f"  Added column document_tags.{col_name}")
            else:
                print(f"  Column document_tags.{col_name} already exists")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_document_tags_lexicon_revision completed.")
