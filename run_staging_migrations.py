#!/usr/bin/env python3
"""
Run full RAG DB migrations (create_all + all migration modules) against the configured DATABASE_URL.
Use for staging/prod before deploying RAG workers. Idempotent.
"""
import asyncio
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Ensure DATABASE_URL is set before config loads
if not os.environ.get("DATABASE_URL"):
    print("ERROR: Set DATABASE_URL (e.g. postgresql+asyncpg://mobius_app:PASSWORD@127.0.0.1:5433/mobius_rag)")
    sys.exit(1)

from sqlalchemy.ext.asyncio import create_async_engine
from app.database import Base
import app.models  # noqa: F401

# Migration modules in dependency order (matches app.main startup)
# Each module has async def migrate() or run_migration()
MIGRATIONS = [
    "add_category_scores_column",
    "category_scores_to_columns",
    "add_document_pages_text_markdown",
    "add_extracted_facts_reader_fields",
    "add_chunk_start_offset_in_page",
    "add_chunking_job_run_config",
    "add_extracted_facts_verification",
    "add_document_authority_level",
    "add_document_effective_termination_dates",
    "add_document_display_name",
    "add_publish_tables",
    "add_publish_verification_columns",
    "add_embedding_tables",
    "add_llm_configs_table",
    "add_is_pertinent_field",
    "add_error_tracking",
]


async def run_migrations():
    from app.config import DATABASE_URL

    engine = create_async_engine(DATABASE_URL, echo=True)
    print("Creating tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    print("Tables created.")

    for module_name in MIGRATIONS:
        try:
            mod = __import__(f"app.migrations.{module_name}", fromlist=["migrate", "run_migration"])
            mig = getattr(mod, "run_migration", None) or getattr(mod, "migrate")
            await mig()
            print(f"  OK {module_name}")
        except Exception as e:
            # Many migrations are idempotent (IF NOT EXISTS); log and continue
            print(f"  (skipped or already applied) {module_name}: {e}")

    print("RAG migrations completed.")


if __name__ == "__main__":
    asyncio.run(run_migrations())
