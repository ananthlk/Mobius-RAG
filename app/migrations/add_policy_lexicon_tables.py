"""
Migration: create policy_lexicon_meta and policy_lexicon_entries for in-DB lexicon (Path B).
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
        # policy_lexicon_meta: single row for revision and meta
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS policy_lexicon_meta (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                revision BIGINT NOT NULL DEFAULT 0,
                lexicon_version VARCHAR(50) NOT NULL DEFAULT 'v1',
                lexicon_meta JSONB,
                created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc'),
                updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc')
            )
        """)
        # Ensure one row
        r = await conn.fetchval("SELECT 1 FROM policy_lexicon_meta LIMIT 1")
        if not r:
            await conn.execute(
                "INSERT INTO policy_lexicon_meta (revision, lexicon_version, lexicon_meta) VALUES (0, 'v1', '{}')"
            )
        # policy_lexicon_entries: kind (p|d|j), code, parent_code, spec (JSONB with description, phrases, etc.), active
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS policy_lexicon_entries (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                kind VARCHAR(10) NOT NULL,
                code VARCHAR(500) NOT NULL,
                parent_code VARCHAR(500),
                spec JSONB NOT NULL DEFAULT '{}',
                active BOOLEAN NOT NULL DEFAULT true,
                created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc'),
                updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc'),
                UNIQUE(kind, code)
            )
        """)
        print("  policy_lexicon_meta and policy_lexicon_entries ready")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
    print("Migration add_policy_lexicon_tables completed.")
