import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.ext.asyncio import create_async_engine
from app.database import Base
from app.config import DATABASE_URL
import app.models  # noqa: F401 — register ChunkingResult etc. with Base.metadata


async def init_db():
    engine = create_async_engine(DATABASE_URL, echo=True)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    print("Database tables created successfully!")
    
    # Run migrations to add any missing columns
    print("Running migrations...")
    from app.migrations.add_category_scores_column import migrate as migrate_category_scores
    await migrate_category_scores()
    from app.migrations.category_scores_to_columns import migrate as migrate_category_columns
    await migrate_category_columns()
    from app.migrations.add_document_pages_text_markdown import migrate as migrate_text_markdown
    await migrate_text_markdown()
    from app.migrations.add_extracted_facts_reader_fields import migrate as migrate_reader_fields
    await migrate_reader_fields()
    from app.migrations.add_chunk_start_offset_in_page import migrate as migrate_chunk_offset
    await migrate_chunk_offset()
    print("Migrations completed!")


if __name__ == "__main__":
    asyncio.run(init_db())
