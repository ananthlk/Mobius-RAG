#!/usr/bin/env python3
"""Force delete using app's database connection - run when PostgreSQL is accessible."""
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.database import AsyncSessionLocal
from sqlalchemy import select, delete
from app.models import Document, DocumentPage, ChunkingResult

async def force_delete():
    pattern = "01-05-26-MFL-Medicaid-Provider-Handbook"
    
    async with AsyncSessionLocal() as db:
        try:
            # Find documents
            result = await db.execute(
                select(Document).where(Document.filename.like(f"%{pattern}%"))
            )
            docs = result.scalars().all()
            
            if not docs:
                print(f"No documents found matching: {pattern}")
                return
            
            print(f"Found {len(docs)} document(s):")
            for doc in docs:
                print(f"  - {doc.id}: {doc.filename}")
            
            # Delete
            for doc in docs:
                await db.execute(delete(ChunkingResult).where(ChunkingResult.document_id == doc.id))
                await db.execute(delete(DocumentPage).where(DocumentPage.document_id == doc.id))
                await db.execute(delete(Document).where(Document.id == doc.id))
            
            await db.commit()
            print(f"✅ Deleted {len(docs)} document(s)!")
            
        except Exception as e:
            await db.rollback()
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(force_delete())
