#!/usr/bin/env python3
"""Migration: Add page_number, start_offset, end_offset to extracted_facts for reader-added facts and persistent highlights."""
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
        for col_name, col_type in [
            ("page_number", "INTEGER NULL"),
            ("start_offset", "INTEGER NULL"),
            ("end_offset", "INTEGER NULL"),
        ]:
            exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = 'extracted_facts' AND column_name = $1
                )
            """, col_name)
            if not exists:
                await conn.execute(f"ALTER TABLE public.extracted_facts ADD COLUMN {col_name} {col_type}")
                print(f"✓ Added column extracted_facts.{col_name}")
            else:
                print(f"✓ Column extracted_facts.{col_name} already exists")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
