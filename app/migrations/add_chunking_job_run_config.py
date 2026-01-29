"""
Migration: add run-configured and run-mode columns to chunking_jobs.

Adds: prompt_versions (JSONB), llm_config_version (VARCHAR), critique_enabled (VARCHAR), max_retries (INTEGER).
Existing rows: prompt_versions=NULL, llm_config_version=NULL, critique_enabled='true', max_retries=2.

Usage:
    python -m app.migrations.add_chunking_job_run_config
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
            ("prompt_versions", "JSONB"),
            ("llm_config_version", "VARCHAR(100)"),
            ("critique_enabled", "VARCHAR(10)"),
            ("max_retries", "INTEGER"),
        ]:
            exists = await conn.fetchval("""
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'chunking_jobs' AND column_name = $1
            """, col_name)
            if not exists:
                await conn.execute(f"""
                    ALTER TABLE chunking_jobs ADD COLUMN {col_name} {col_def}
                """)
                print(f"  Added column chunking_jobs.{col_name}")
            else:
                print(f"  Column chunking_jobs.{col_name} already exists")

        # Set defaults for existing rows (critique_enabled and max_retries only; prompt/llm stay null for old jobs)
        await conn.execute("""
            UPDATE chunking_jobs
            SET critique_enabled = 'true', max_retries = 2
            WHERE critique_enabled IS NULL OR max_retries IS NULL
        """)
        print("  Updated existing rows with default critique_enabled and max_retries")
    finally:
        await conn.close()


def main():
    asyncio.run(migrate())
    print("Migration add_chunking_job_run_config completed.")


if __name__ == "__main__":
    main()
