"""
Migration: add verification_passed and verification_message to publish_events
for post-publish integrity check results.
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
        for col, col_type, comment in [
            ("verification_passed", "BOOLEAN", "true if integrity check passed after publish"),
            ("verification_message", "TEXT", "error or null if passed"),
        ]:
            exists = await conn.fetchval(
                "SELECT 1 FROM information_schema.columns "
                f"WHERE table_schema = 'public' AND table_name = 'publish_events' AND column_name = '{col}'"
            )
            if not exists:
                await conn.execute(f"""
                    ALTER TABLE public.publish_events ADD COLUMN {col} {col_type}
                """)
                print(f"  Added column publish_events.{col}")
            else:
                print(f"  Column publish_events.{col} already exists")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_publish_verification_columns completed.")
