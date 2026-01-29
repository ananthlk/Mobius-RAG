#!/usr/bin/env python3
"""Migration: Add start_offset_in_page to hierarchical_chunks for LLM source highlighting."""
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
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'hierarchical_chunks' AND column_name = 'start_offset_in_page'
            )
        """)
        if not exists:
            await conn.execute("ALTER TABLE public.hierarchical_chunks ADD COLUMN start_offset_in_page INTEGER NULL")
            print("✓ Added column hierarchical_chunks.start_offset_in_page")
        else:
            print("✓ Column hierarchical_chunks.start_offset_in_page already exists")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
