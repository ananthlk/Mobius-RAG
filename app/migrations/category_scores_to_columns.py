#!/usr/bin/env python3
"""Migration: Replace category_scores JSONB with individual score/direction columns per category."""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncpg
from app.config import DATABASE_URL
from app.models import CATEGORY_NAMES


async def migrate():
    url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)

    try:
        # Widen fact_type to VARCHAR(255) if needed
        try:
            await conn.execute("""
                ALTER TABLE extracted_facts
                ALTER COLUMN fact_type TYPE VARCHAR(255)
                USING fact_type::VARCHAR(255)
            """)
            print("✓ fact_type set to VARCHAR(255)")
        except Exception as e:
            print(f"  fact_type alter skipped or already 255: {e}")

        for cat in CATEGORY_NAMES:
            for suffix in ("_score", "_direction"):
                col_name = f"{cat}{suffix}"
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'extracted_facts' AND column_name = $1
                    )
                """, col_name)
                if not exists:
                    await conn.execute(f"""
                        ALTER TABLE extracted_facts ADD COLUMN {col_name} REAL NULL
                    """)
                    print(f"✓ Added column {col_name}")

        # Drop category_scores if it exists
        has_jsonb = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'extracted_facts' AND column_name = 'category_scores'
            )
        """)
        if has_jsonb:
            try:
                await conn.execute("ALTER TABLE extracted_facts DROP COLUMN category_scores")
                print("✓ Dropped column category_scores")
            except Exception as e:
                print(f"  Drop category_scores skipped: {e}")
        else:
            print("✓ Column category_scores already removed")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
