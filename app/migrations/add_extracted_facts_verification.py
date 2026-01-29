"""
Migration: add verification fields to extracted_facts (verified_by, verified_at, verification_status).
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
            ("verified_by", "VARCHAR(20)"),
            ("verified_at", "TIMESTAMP"),
            ("verification_status", "VARCHAR(20)"),
        ]:
            exists = await conn.fetchval("""
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'extracted_facts' AND column_name = $1
            """, col_name)
            if not exists:
                await conn.execute(f"""
                    ALTER TABLE public.extracted_facts ADD COLUMN {col_name} {col_def}
                """)
                print(f"  Added column extracted_facts.{col_name}")
            else:
                print(f"  Column extracted_facts.{col_name} already exists")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_extracted_facts_verification completed.")
