#!/usr/bin/env python3
"""Delete a document from PostgreSQL database by filename pattern."""
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, delete
from app.config import DATABASE_URL
from app.models import Document, DocumentPage, ChunkingResult

async def delete_document_by_filename(filename_pattern: str):
    """Delete document(s) matching filename pattern from database."""
    engine = create_async_engine(DATABASE_URL, echo=False)
    AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with AsyncSessionLocal() as db:
        try:
            # Find documents matching the pattern
            result = await db.execute(
                select(Document).where(Document.filename.like(f"%{filename_pattern}%"))
            )
            documents = result.scalars().all()
            
            if not documents:
                print(f"No documents found matching pattern: {filename_pattern}")
                return
            
            print(f"Found {len(documents)} document(s) matching pattern:")
            for doc in documents:
                print(f"  - ID: {doc.id}, Filename: {doc.filename}")
            
            # Delete related records for each document
            for doc in documents:
                doc_id = doc.id
                print(f"\nDeleting document {doc.id} ({doc.filename})...")
                
                # Delete chunking results
                await db.execute(
                    delete(ChunkingResult).where(ChunkingResult.document_id == doc_id)
                )
                print("  ✓ Deleted chunking_results")
                
                # Delete pages
                pages_result = await db.execute(
                    delete(DocumentPage).where(DocumentPage.document_id == doc_id)
                )
                print(f"  ✓ Deleted document_pages")
                
                # Delete document
                await db.execute(
                    delete(Document).where(Document.id == doc_id)
                )
                print(f"  ✓ Deleted document")
            
            await db.commit()
            print(f"\n✅ Successfully deleted {len(documents)} document(s) from database!")
            
        except Exception as e:
            await db.rollback()
            print(f"❌ Error: {e}", file=sys.stderr)
            raise
        finally:
            await engine.dispose()

if __name__ == "__main__":
    # Default pattern matches the file mentioned
    pattern = "01-05-26-MFL-Medicaid-Provider-Handbook"
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
    
    print(f"Deleting documents matching: {pattern}")
    asyncio.run(delete_document_by_filename(pattern))
