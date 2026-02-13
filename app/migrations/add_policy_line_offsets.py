"""
Migration: add start_offset, end_offset, offset_match_quality to policy_lines if missing.
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
        table_exists = await conn.fetchval(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'policy_lines'"
        )
        if not table_exists:
            print("  policy_lines table does not exist; skipping")
            return
        for col in ("start_offset", "end_offset", "offset_match_quality"):
            exists = await conn.fetchval(
                "SELECT 1 FROM information_schema.columns "
                f"WHERE table_schema = 'public' AND table_name = 'policy_lines' AND column_name = '{col}'"
            )
            if not exists:
                if col == "offset_match_quality":
                    await conn.execute(f"ALTER TABLE policy_lines ADD COLUMN {col} DOUBLE PRECISION")
                else:
                    await conn.execute(f"ALTER TABLE policy_lines ADD COLUMN {col} INTEGER")
                print(f"  Added policy_lines.{col}")
        print("  policy_lines offsets ready")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_policy_line_offsets completed.")
