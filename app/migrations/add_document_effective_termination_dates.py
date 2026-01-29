"""
Migration: add effective_date and termination_date columns to documents.
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
            ("effective_date", "VARCHAR(20)"),
            ("termination_date", "VARCHAR(20)"),
        ]:
            exists = await conn.fetchval(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = $1",
                col_name,
            )
            if not exists:
                await conn.execute(f"""
                    ALTER TABLE public.documents ADD COLUMN {col_name} {col_def}
                """)
                print(f"  Added column documents.{col_name}")
            else:
                print(f"  Column documents.{col_name} already exists")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_document_effective_termination_dates completed.")
