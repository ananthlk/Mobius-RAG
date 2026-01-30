"""
Migration: create llm_configs table for storing LLM provider configs.

Table: llm_configs
  name (VARCHAR(100) PK), provider, model, version_label, options (JSONB),
  ollama (JSONB), vertex (JSONB), openai (JSONB), created_at, updated_at.

Usage:
    python -m app.migrations.add_llm_configs_table
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
        exists = await conn.fetchval("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'llm_configs'
        """)
        if exists:
            print("  Table llm_configs already exists")
            return
        await conn.execute("""
            CREATE TABLE llm_configs (
                name VARCHAR(100) PRIMARY KEY,
                provider VARCHAR(50) NOT NULL,
                model VARCHAR(200),
                version_label VARCHAR(100),
                options JSONB DEFAULT '{}',
                ollama JSONB DEFAULT '{}',
                vertex JSONB DEFAULT '{}',
                openai JSONB DEFAULT '{}',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("  Created table llm_configs")
    finally:
        await conn.close()


def main():
    asyncio.run(migrate())
    print("Migration add_llm_configs_table completed.")


if __name__ == "__main__":
    main()
