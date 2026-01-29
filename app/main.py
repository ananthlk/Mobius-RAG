import hashlib
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func
from google.cloud import storage
import json
from datetime import datetime
from collections import deque
from asyncio import Lock
import asyncio
from app.config import GCS_BUCKET, ENV, CRITIQUE_RETRY_THRESHOLD
from app.database import get_db, Base
from app.models import Document, DocumentPage, ChunkingResult, HierarchicalChunk, ExtractedFact, ProcessingError, ChunkingJob, ChunkingEvent, fact_to_category_scores_dict
from app.services.error_tracker import log_error, classify_error
from app.services.extract_text import extract_text_from_gcs
from app.services.chunking import split_paragraphs, split_paragraphs_from_markdown
from app.services.extraction import stream_extract_facts
from app.services.critique import stream_critique, critique_extraction, normalize_critique_result
from app.services.utils import parse_json_response

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Set specific loggers to appropriate levels
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)  # Reduce SQLAlchemy verbosity
logging.getLogger('uvicorn').setLevel(logging.INFO)
logging.getLogger('fastapi').setLevel(logging.INFO)

_chunking_cancel: set[str] = set()
_chunking_running: set[str] = set()

# Event buffer: document_id -> deque of events
_chunking_events: dict[str, deque] = {}
_chunking_events_lock: dict[str, Lock] = {}

app = FastAPI(title="Mobius RAG", version="0.1.0")


@app.on_event("startup")
async def run_startup_migrations():
    """Run database migrations on startup to ensure schema is up to date."""
    from sqlalchemy import text
    from app.database import AsyncSessionLocal, engine

    # Ensure all ORM tables exist first (documents, document_pages, processing_errors, etc.)
    # so raw SQL below that references them does not fail on a fresh database.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as db:
        try:
            logger.info("Running startup migrations...")

            # Create processing_errors table (if not already created by create_all)
            await db.execute(text("""
                CREATE TABLE IF NOT EXISTS processing_errors (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    paragraph_id VARCHAR(100),
                    error_type VARCHAR(50) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    error_message TEXT NOT NULL,
                    error_details JSONB,
                    stage VARCHAR(50) NOT NULL,
                    resolved VARCHAR(10) NOT NULL DEFAULT 'false',
                    resolution VARCHAR(20),
                    resolved_by VARCHAR(255),
                    resolved_at TIMESTAMP,
                    resolution_notes TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Create indexes (using DO block to check existence)
            await db.execute(text("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_indexes 
                        WHERE indexname = 'idx_processing_errors_document_id'
                    ) THEN
                        CREATE INDEX idx_processing_errors_document_id 
                        ON processing_errors(document_id);
                    END IF;
                END $$;
            """))
            
            await db.execute(text("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_indexes 
                        WHERE indexname = 'idx_processing_errors_resolved'
                    ) THEN
                        CREATE INDEX idx_processing_errors_resolved 
                        ON processing_errors(resolved);
                    END IF;
                END $$;
            """))
            
            await db.execute(text("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_indexes 
                        WHERE indexname = 'idx_processing_errors_severity'
                    ) THEN
                        CREATE INDEX idx_processing_errors_severity 
                        ON processing_errors(severity);
                    END IF;
                END $$;
            """))
            
            # Create chunking_jobs table
            await db.execute(text("""
                CREATE TABLE IF NOT EXISTS chunking_jobs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    threshold VARCHAR(10) NOT NULL,
                    worker_id VARCHAR(100),
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Create index on chunking_jobs status
            await db.execute(text("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_indexes 
                        WHERE indexname = 'idx_chunking_jobs_status'
                    ) THEN
                        CREATE INDEX idx_chunking_jobs_status 
                        ON chunking_jobs(status);
                    END IF;
                END $$;
            """))
            
            # Create chunking_events table
            await db.execute(text("""
                CREATE TABLE IF NOT EXISTS chunking_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    event_type VARCHAR(50) NOT NULL,
                    event_data JSONB NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """))
            
            # Create index on chunking_events document_id and created_at
            await db.execute(text("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_indexes 
                        WHERE indexname = 'idx_chunking_events_document_id'
                    ) THEN
                        CREATE INDEX idx_chunking_events_document_id 
                        ON chunking_events(document_id, created_at);
                    END IF;
                END $$;
            """))
            
            # Add columns to documents table if they don't exist
            for col_name, col_def in [
                ("has_errors", "VARCHAR(10) NOT NULL DEFAULT 'false'"),
                ("error_count", "INTEGER NOT NULL DEFAULT 0"),
                ("critical_error_count", "INTEGER NOT NULL DEFAULT 0"),
                ("review_status", "VARCHAR(20) NOT NULL DEFAULT 'pending'")
            ]:
                await db.execute(text(f"""
                    DO $$ 
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name = 'documents' AND column_name = '{col_name}'
                        ) THEN
                            ALTER TABLE documents ADD COLUMN {col_name} {col_def};
                        END IF;
                    END $$;
                """))
            
            await db.commit()

            # Run Python migrations (add columns to existing tables)
            try:
                from app.migrations.add_document_pages_text_markdown import migrate as migrate_text_markdown
                await migrate_text_markdown()
            except Exception as migrate_err:
                logger.warning(f"Startup migration (text_markdown) skipped: {migrate_err}")
            try:
                from app.migrations.add_extracted_facts_reader_fields import migrate as migrate_reader_fields
                await migrate_reader_fields()
            except Exception as migrate_err:
                logger.warning(f"Startup migration (extracted_facts reader fields) skipped: {migrate_err}")
            try:
                from app.migrations.add_chunk_start_offset_in_page import migrate as migrate_chunk_offset
                await migrate_chunk_offset()
            except Exception as migrate_err:
                logger.warning(f"Startup migration (chunk start_offset_in_page) skipped: {migrate_err}")

            logger.info("✓ Startup migrations completed successfully")
        except Exception as e:
            await db.rollback()
            logger.error(f"Startup migration failed: {e}", exc_info=True)
            # Don't raise - allow server to start even if migration fails
            # (might be a temporary DB issue, or tables might already exist)


async def _get_or_create_manual_chunk(db: AsyncSession, document_id) -> HierarchicalChunk:
    """Get or create the single 'manual' chunk per document for reader-added facts (page_number=0, paragraph_index=0)."""
    from uuid import UUID
    doc_uuid = document_id if isinstance(document_id, UUID) else UUID(str(document_id))
    result = await db.execute(
        select(HierarchicalChunk).where(
            HierarchicalChunk.document_id == doc_uuid,
            HierarchicalChunk.page_number == 0,
            HierarchicalChunk.paragraph_index == 0,
        )
    )
    chunk = result.scalar_one_or_none()
    if chunk is not None:
        return chunk
    manual_text = "[Reader-added facts]"
    chunk = HierarchicalChunk(
        document_id=doc_uuid,
        page_number=0,
        paragraph_index=0,
        text=manual_text,
        text_length=len(manual_text),
        extraction_status="manual",
        critique_status="passed",
    )
    db.add(chunk)
    await db.flush()
    return chunk


async def _create_event_buffer_callback(document_id: str, db: Optional[AsyncSession] = None):
    """Create an event callback that writes events to the in-memory buffer and optionally to the database."""
    from uuid import UUID
    
    doc_uuid = UUID(document_id)
    
    if document_id not in _chunking_events:
        _chunking_events[document_id] = deque(maxlen=50000)  # Keep last 50000 events to minimize rotation issues
        _chunking_events_lock[document_id] = Lock()
    
    async def event_callback(event_type: str, data: dict):
        event = {
            "event": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        # Add to buffer immediately
        async with _chunking_events_lock[document_id]:
            _chunking_events[document_id].append(event)
            # logger.debug(f"Added event {event_type} to buffer for document {document_id}, buffer size: {len(_chunking_events[document_id])}")  # Reduced logging
        
        # Also write to database if db session is provided (for SSE live updates)
        if db is not None:
            try:
                db_event = ChunkingEvent(
                    document_id=doc_uuid,
                    event_type=event_type,
                    event_data=data
                )
                db.add(db_event)
                # Commit each event so SSE endpoint can pick it up immediately
                await db.commit()
            except Exception as e:
                logger.error(f"Failed to write event {event_type} to database: {e}", exc_info=True)
                await db.rollback()
        
        # Yield SSE format for compatibility (but this is consumed and discarded in background task)
        yield f"data: {json.dumps(event)}\n\n"
    
    return event_callback

# CORS - allow frontend in dev
cors_origins = ["http://localhost:5173"] if ENV == "dev" else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/test/gcs")
async def test_gcs():
    """Test GCS connectivity."""
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        # Try to list blobs (just check access)
        list(bucket.list_blobs(max_results=1))
        return {"status": "ok", "message": f"GCS bucket {GCS_BUCKET} is accessible"}
    except Exception as e:
        logger.error(f"GCS test failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.get("/admin/create-database")
@app.post("/admin/create-database")
async def create_database_endpoint():
    """Create the database if it doesn't exist."""
    import asyncpg
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    from app.config import DATABASE_URL
    from app.database import Base
    
    try:
        # Parse DATABASE_URL
        url_parts = DATABASE_URL.replace("postgresql+asyncpg://", "").split("/")
        if len(url_parts) < 2:
            return {"status": "error", "message": "Invalid DATABASE_URL format"}
        
        auth_part = url_parts[0]  # user@host:port
        target_db = url_parts[1]
        
        # Parse connection details
        if "@" in auth_part:
            user, host_port = auth_part.split("@")
            if ":" in host_port:
                host, port = host_port.split(":")
            else:
                host = host_port
                port = "5432"
        else:
            user = auth_part
            host = "localhost"
            port = "5432"
        
        # Connect to 'postgres' database using asyncpg directly
        conn = await asyncpg.connect(
            user=user,
            host=host,
            port=int(port),
            database='postgres'
        )
        
        try:
            # Check if database exists
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", target_db
            )
            
            if exists:
                await conn.close()
                # Database exists, just initialize tables
                engine = create_async_engine(DATABASE_URL, echo=False)
                async with engine.begin() as db_conn:
                    await db_conn.run_sync(Base.metadata.create_all)
                await engine.dispose()
                
                # Run migrations to add any missing columns
                from app.migrations.add_category_scores_column import migrate as migrate_category_scores
                await migrate_category_scores()
                from app.migrations.category_scores_to_columns import migrate as migrate_category_columns
                await migrate_category_columns()
                from app.migrations.add_document_pages_text_markdown import migrate as migrate_text_markdown
                await migrate_text_markdown()
                return {"status": "ok", "message": f"Database '{target_db}' already exists, tables initialized and migrations run"}
            
            # Create database (autocommit mode)
            await conn.execute(f'CREATE DATABASE "{target_db}"')
            await conn.close()
            
            # Now initialize tables
            engine = create_async_engine(DATABASE_URL, echo=False)
            async with engine.begin() as db_conn:
                await db_conn.run_sync(Base.metadata.create_all)
            await engine.dispose()
            
            # Run migrations to add any missing columns
            from app.migrations.add_category_scores_column import migrate as migrate_category_scores
            await migrate_category_scores()
            from app.migrations.category_scores_to_columns import migrate as migrate_category_columns
            await migrate_category_columns()
            from app.migrations.add_document_pages_text_markdown import migrate as migrate_text_markdown
            await migrate_text_markdown()
            return {"status": "ok", "message": f"Database '{target_db}' created and tables initialized with migrations"}
            
        except Exception as db_error:
            await conn.close()
            raise db_error
        
    except Exception as e:
        logger.error(f"Database creation failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.get("/documents")
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """List all documents with extraction and chunking status."""
    result = await db.execute(
        select(Document)
        .order_by(Document.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    documents = result.scalars().all()
    
    # Get chunking status for each document
    document_list = []
    for doc in documents:
        # First check for active jobs (pending or processing)
        job_result = await db.execute(
            select(ChunkingJob).where(
                ChunkingJob.document_id == doc.id,
                ChunkingJob.status.in_(["pending", "processing"])
            ).order_by(ChunkingJob.created_at.desc())
        )
        active_job = job_result.scalar_one_or_none()
        
        chunking_status = None
        if active_job:
            # Job exists - map job status to chunking_status
            if active_job.status == "pending":
                chunking_status = "queued"
            elif active_job.status == "processing":
                chunking_status = "in_progress"
        else:
            # No active job - check chunking results
            chunking_result = await db.execute(
                select(ChunkingResult).where(ChunkingResult.document_id == doc.id)
            )
            chunking = chunking_result.scalar_one_or_none()
            
            if chunking and chunking.metadata_:
                chunking_status = chunking.metadata_.get("status", "idle")
            else:
                chunking_status = "idle"
        
        document_list.append({
            "id": str(doc.id),
            "filename": doc.filename,
            "extraction_status": doc.status,  # uploaded, extracting, completed, failed, completed_with_errors
            "chunking_status": chunking_status,  # idle, queued, in_progress, completed, stopped, failed
            "created_at": doc.created_at.isoformat(),
            "gcs_path": doc.file_path,
            "has_errors": doc.has_errors or "false",
            "error_count": doc.error_count or 0,
            "critical_error_count": doc.critical_error_count or 0,
            "review_status": doc.review_status or "pending",
        })
    
    return {
        "total": len(document_list),
        "documents": document_list
    }


@app.get("/documents/{document_id}/pages")
async def get_document_pages(
    document_id: str,
    page_number: int = None,
    db: AsyncSession = Depends(get_db)
):
    """Get extracted pages for a document. Optionally filter by page number."""
    from uuid import UUID
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    
    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Build query
    query = select(DocumentPage).where(DocumentPage.document_id == doc_uuid)
    if page_number:
        query = query.where(DocumentPage.page_number == page_number)
    query = query.order_by(DocumentPage.page_number)
    
    pages_result = await db.execute(query)
    pages = pages_result.scalars().all()
    
    return {
        "document_id": str(document.id),
        "filename": document.filename,
        "total_pages": len(pages),
        "pages": [
            {
                "page_number": p.page_number,
                "text": p.text,
                "text_markdown": getattr(p, "text_markdown", None),
                "text_length": p.text_length,
                "extraction_status": p.extraction_status,
                "extraction_error": p.extraction_error,
            }
            for p in pages
        ]
    }


@app.get("/documents/{document_id}/status")
async def get_document_status(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get processing status of a document."""
    from uuid import UUID
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    
    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get pages with details
    pages_result = await db.execute(
        select(DocumentPage).where(DocumentPage.document_id == doc_uuid)
        .order_by(DocumentPage.page_number)
    )
    pages = pages_result.scalars().all()
    
    # Calculate statistics
    total_pages = len(pages)
    successful_pages = sum(1 for p in pages if p.extraction_status == "success")
    failed_pages = sum(1 for p in pages if p.extraction_status == "failed")
    empty_pages = sum(1 for p in pages if p.extraction_status == "empty")
    
    # Get problematic pages
    problematic_pages = [
        {
            "page_number": p.page_number,
            "status": p.extraction_status,
            "error": p.extraction_error,
            "text_length": p.text_length,
        }
        for p in pages
        if p.extraction_status != "success"
    ]
    
    return {
        "document_id": str(document.id),
        "filename": document.filename,
        "status": document.status,
        "pages_extracted": total_pages,
        "pages_summary": {
            "total": total_pages,
            "successful": successful_pages,
            "failed": failed_pages,
            "empty": empty_pages,
        },
        "problematic_pages": problematic_pages,
        "created_at": document.created_at.isoformat(),
    }


@app.post("/documents/{document_id}/extract/restart")
async def restart_extraction(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Restart text extraction for a document that failed or was stopped."""
    from uuid import UUID
    import asyncio
    
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    
    # Get document
    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check if extraction is already in progress
    if document.status == "extracting":
        raise HTTPException(status_code=409, detail="Extraction already in progress")
    
    # Reset status and start extraction in background
    document.status = "extracting"
    await db.commit()
    
    # Start extraction in background task
    async def extract_task():
        db_session = None
        try:
            from app.database import AsyncSessionLocal
            db_session = AsyncSessionLocal()
            
            # Re-fetch document in new session
            doc_result = await db_session.execute(select(Document).where(Document.id == doc_uuid))
            doc = doc_result.scalar_one_or_none()
            if not doc:
                return
            
            try:
                # Extract text from PDF
                pages = await extract_text_from_gcs(doc.file_path)
                
                # Delete existing pages
                await db_session.execute(
                    delete(DocumentPage).where(DocumentPage.document_id == doc_uuid)
                )
                
                # Insert new pages (raw + markdown for reader)
                from app.services.page_to_markdown import raw_page_to_markdown
                for page_data in pages:
                    raw_text = page_data.get("text") or ""
                    page = DocumentPage(
                        document_id=doc_uuid,
                        page_number=page_data["page_number"],
                        text=raw_text,
                        text_markdown=raw_page_to_markdown(raw_text) if raw_text else None,
                        text_length=page_data.get("text_length", 0),
                        extraction_status=page_data.get("extraction_status", "failed"),
                        extraction_error=page_data.get("extraction_error"),
                    )
                    db_session.add(page)
                
                # Update document status
                doc.status = "completed"
                await db_session.commit()
                logger.info(f"Extraction restarted and completed for document {document_id}")
            except Exception as e:
                logger.error(f"Extraction error: {e}", exc_info=True)
                doc.status = "failed"
                await db_session.commit()
        except Exception as e:
            logger.error(f"Background extraction task error: {e}", exc_info=True)
        finally:
            if db_session:
                await db_session.close()
    
    # Spawn background task
    asyncio.create_task(extract_task())
    
    return {"status": "restarted", "document_id": document_id}


@app.get("/gcs/files")
async def list_gcs_files():
    """List all files in GCS bucket."""
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        
        # List all blobs (excluding the hashes/ directory)
        blobs = bucket.list_blobs()
        
        files = []
        for blob in blobs:
            # Skip hash index files
            if blob.name.startswith("hashes/"):
                continue
            files.append({
                "filename": blob.name,
                "size": blob.size,
                "content_type": blob.content_type,
                "created": blob.time_created.isoformat() if blob.time_created else None,
                "gcs_path": f"gs://{GCS_BUCKET}/{blob.name}",
            })
        
        return {
            "bucket": GCS_BUCKET,
            "total_files": len(files),
            "files": files,
        }
    except Exception as e:
        logger.error(f"Failed to list GCS files: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/gcs/files/{filename:path}")
async def delete_gcs_file(
    filename: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a file from GCS and its database records."""
    from uuid import UUID
    from sqlalchemy import delete
    
    try:
        # URL decode filename
        import urllib.parse
        filename = urllib.parse.unquote(filename)
        
        # Find document in database
        result = await db.execute(
            select(Document).where(Document.file_path.like(f"%{filename}"))
        )
        document = result.scalar_one_or_none()
        
        # Delete from GCS
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(filename)
        
        deleted_from_gcs = False
        if blob.exists():
            blob.delete()
            deleted_from_gcs = True
            logger.info(f"Deleted file from GCS: {filename}")
        
        # Delete hash index if exists
        if document:
            hash_index = bucket.blob(f"hashes/{document.file_hash}")
            if hash_index.exists():
                hash_index.delete()
        
        # Delete from database if document exists (cascade delete)
        if document:
            from sqlalchemy import text
            doc_uuid = document.id
            
            # Delete in correct order to respect foreign key constraints
            # 1. Delete extracted_facts (references both documents and hierarchical_chunks)
            await db.execute(
                text("DELETE FROM extracted_facts WHERE document_id = :doc_id"),
                {"doc_id": doc_uuid}
            )
            
            # 2. Delete hierarchical_chunks (references documents)
            await db.execute(
                text("DELETE FROM hierarchical_chunks WHERE document_id = :doc_id"),
                {"doc_id": doc_uuid}
            )
            
            # 3. Delete chunking_results (references documents)
            await db.execute(
                text("DELETE FROM chunking_results WHERE document_id = :doc_id"),
                {"doc_id": doc_uuid}
            )
            
            # 4. Delete document_pages (references documents)
            await db.execute(
                text("DELETE FROM document_pages WHERE document_id = :doc_id"),
                {"doc_id": doc_uuid}
            )
            
            # 5. Delete document (master table)
            await db.execute(
                text("DELETE FROM documents WHERE id = :doc_id"),
                {"doc_id": doc_uuid}
            )
            
            await db.commit()
            logger.info(f"Deleted document and all related records from database: {document.id}")
        
        return {
            "status": "ok",
            "message": f"File '{filename}' deleted",
            "deleted_from_gcs": deleted_from_gcs,
            "deleted_from_db": document is not None,
        }
        
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to delete file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a document and all its related records from the database."""
    from uuid import UUID
    from sqlalchemy import delete
    
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    
    # Get document
    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Delete related records first (foreign key constraints)
        # Delete chunking results
        await db.execute(
            delete(ChunkingResult).where(ChunkingResult.document_id == doc_uuid)
        )
        # Delete pages
        await db.execute(
            delete(DocumentPage).where(DocumentPage.document_id == doc_uuid)
        )
        # Delete document
        await db.execute(
            delete(Document).where(Document.id == doc_uuid)
        )
        await db.commit()
        
        logger.info(f"Deleted document {document_id} from database")
        
        return {
            "status": "ok",
            "message": f"Document '{document.filename}' deleted from database",
            "document_id": document_id
        }
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to delete document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/check/{filename:path}")
async def check_file(filename: str):
    """Check if a file exists in GCS bucket."""
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(filename)
        
        exists = blob.exists()
        
        if exists:
            blob.reload()
            return {
                "exists": True,
                "filename": filename,
                "size": blob.size,
                "content_type": blob.content_type,
                "gcs_path": f"gs://{GCS_BUCKET}/{filename}",
            }
        else:
            return {
                "exists": False,
                "filename": filename,
                "gcs_path": f"gs://{GCS_BUCKET}/{filename}",
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(
    file: UploadFile,
    payer: str = None,
    state: str = None,
    program: str = None,
    db: AsyncSession = Depends(get_db)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    try:
        logger.info(f"Starting upload for file: {file.filename}")
        
        # Read file contents
        contents = await file.read()
        logger.info(f"File read, size: {len(contents)} bytes")
        
        # Compute SHA-256 hash
        file_hash = hashlib.sha256(contents).hexdigest()
        logger.info(f"File hash computed: {file_hash[:16]}...")
        
        # Check for duplicate in database
        result = await db.execute(select(Document).where(Document.file_hash == file_hash))
        existing_doc = result.scalar_one_or_none()
        
        if existing_doc:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "duplicate_file",
                    "message": "This file has already been uploaded.",
                    "original_filename": existing_doc.filename,
                    "help": "If you need to upload a different version, please rename the file first."
                }
            )
        
        # Upload file to GCS
        logger.info(f"Uploading to GCS bucket: {GCS_BUCKET}")
        try:
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob(file.filename)
            blob.upload_from_string(contents, content_type=file.content_type)
            logger.info(f"File uploaded to GCS successfully")
        except Exception as gcs_error:
            logger.error(f"GCS upload failed: {str(gcs_error)}")
            raise
        
        gcs_path = f"gs://{GCS_BUCKET}/{file.filename}"
        
        # Save to database with status "uploaded"
        document = Document(
            filename=file.filename,
            file_hash=file_hash,
            file_path=gcs_path,
            payer=payer,
            state=state,
            program=program,
            status="uploaded",
        )
        db.add(document)
        await db.commit()
        await db.refresh(document)

        # Start text extraction (async in background)
        # For now, do it synchronously but could be moved to background task
        try:
            # Update status to extracting
            document.status = "extracting"
            await db.commit()
            
            # Extract text from PDF
            pages = await extract_text_from_gcs(gcs_path)
            
            # Save pages to database with error tracking (raw + markdown for reader)
            from app.services.page_to_markdown import raw_page_to_markdown
            for page_data in pages:
                raw_text = page_data.get("text") or ""
                page = DocumentPage(
                    document_id=document.id,
                    page_number=page_data["page_number"],
                    text=raw_text,
                    text_markdown=raw_page_to_markdown(raw_text) if raw_text else None,
                    extraction_status=page_data.get("extraction_status", "failed"),
                    extraction_error=page_data.get("extraction_error"),
                    text_length=page_data.get("text_length", 0),
                )
                db.add(page)
            
            # Check if any pages failed
            failed_pages = [p for p in pages if p.get("extraction_status") == "failed"]
            empty_pages = [p for p in pages if p.get("extraction_status") == "empty"]
            
            # Update status to completed (even if some pages had issues)
            document.status = "completed"
            await db.commit()
            
        except Exception as e:
            # Mark as failed
            document.status = "failed"
            await db.commit()
            # Don't raise - return success but with failed status
            # Frontend can check status endpoint

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(contents),
            "hash": file_hash,
            "gcs_path": gcs_path,
            "document_id": str(document.id),
            "status": document.status,
        }
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


def _critique_score(c: dict) -> float:
    s = c.get("score")
    if s is None:
        return 0.5
    try:
        return max(0.0, min(1.0, float(s)))
    except (TypeError, ValueError):
        return 0.5


async def _persist_paragraph_to_db(
    db: AsyncSession,
    doc_uuid,
    para_id: str,  # Format: "page_number_para_idx"
    paragraph_text: str,
    extraction: dict,
    critique: dict,
    final_status: str,
    retry_count: int,
    needs_human_review: bool
):
    """
    Persist a paragraph and its extracted facts to normalized PostgreSQL tables.
    Creates or updates HierarchicalChunk and creates ExtractedFact records.
    """
    from uuid import UUID

    # Ensure paragraph_text is a string (split_paragraphs returns dicts; caller may pass dict by mistake)
    if isinstance(paragraph_text, dict):
        paragraph_text = paragraph_text.get("text", "") or ""
    paragraph_text = str(paragraph_text) if paragraph_text is not None else ""

    facts_list = extraction.get('facts', [])
    # logger.info(f"[_persist] Starting persistence for {para_id}, facts count: {len(facts_list)}")  # Reduced logging
    # if facts_list:
    #     logger.info(f"[_persist] First fact sample for {para_id}: {facts_list[0] if facts_list else 'N/A'}")  # Reduced logging
    try:
        # Parse para_id to get page_number and para_idx
        parts = para_id.split('_')
        if len(parts) < 2:
            logger.error(f"[_persist] Invalid para_id format: {para_id}")
            return False
        page_number = int(parts[0])
        paragraph_index = int(parts[1])
        # logger.debug(f"[_persist] Parsed para_id: page={page_number}, para_idx={paragraph_index}")  # Reduced logging
        
        # Check if document still exists before persisting
        # logger.debug(f"[_persist] Checking if document {doc_uuid} exists")  # Reduced logging
        doc_check = await db.execute(select(Document).where(Document.id == doc_uuid))
        if doc_check.scalar_one_or_none() is None:
            logger.warning(f"[_persist] Document {doc_uuid} no longer exists, skipping persistence for paragraph {para_id}")
            return False
        # logger.debug(f"[_persist] Document exists, proceeding with persistence")  # Reduced logging
        
        # Get or create HierarchicalChunk
        chunk_result = await db.execute(
            select(HierarchicalChunk).where(
                HierarchicalChunk.document_id == doc_uuid,
                HierarchicalChunk.page_number == page_number,
                HierarchicalChunk.paragraph_index == paragraph_index
            )
        )
        chunk = chunk_result.scalar_one_or_none()
        
        # Determine eligibility-related status from facts
        facts = extraction.get("facts", [])
        is_eligibility_related = None
        if facts:
            # Check if any fact is pertinent to claims or members (new field name) or eligibility-related (backward compatibility)
            any_elig = any(f.get("is_pertinent_to_claims_or_members") is True or f.get("is_eligibility_related") is True for f in facts)
            all_elig_false = all((f.get("is_pertinent_to_claims_or_members") is False or f.get("is_pertinent_to_claims_or_members") is None) and (f.get("is_eligibility_related") is False or f.get("is_eligibility_related") is None) for f in facts)
            if any_elig:
                is_eligibility_related = "true"
            elif all_elig_false:
                is_eligibility_related = "false"
        
        # Determine critique status
        critique_status = "passed" if final_status == "passed" else "failed"
        if retry_count > 0:
            critique_status = "retrying" if needs_human_review else critique_status
        
        # Determine extraction status
        extraction_status = "extracted" if extraction.get("summary") or facts else "failed"
        
        if chunk:
            # Update existing chunk
            # logger.debug(f"[_persist] Updating existing chunk {chunk.id} for {para_id}")  # Reduced logging
            chunk.text = paragraph_text
            chunk.text_length = len(paragraph_text)
            chunk.summary = extraction.get("summary")
            chunk.is_eligibility_related = is_eligibility_related
            chunk.extraction_status = extraction_status
            chunk.critique_status = critique_status
            chunk.critique_feedback = critique.get("feedback")
            chunk.retry_count = retry_count
            
            # Delete old facts for this chunk (if retry or update)
            try:
                delete_result = await db.execute(
                    delete(ExtractedFact).where(ExtractedFact.hierarchical_chunk_id == chunk.id)
                )
                deleted_count = delete_result.rowcount if hasattr(delete_result, 'rowcount') else 0
                # logger.debug(f"[_persist] Deleted {deleted_count} old facts for chunk {chunk.id}")  # Reduced logging
            except Exception as delete_err:
                logger.error(f"[_persist] Error deleting old facts for chunk {chunk.id}: {delete_err}", exc_info=True)
                # Continue anyway - we'll try to add new facts
        else:
            # Create new chunk
            # logger.debug(f"[_persist] Creating new chunk for {para_id}")  # Reduced logging
            chunk = HierarchicalChunk(
                document_id=doc_uuid,
                page_number=page_number,
                paragraph_index=paragraph_index,
                text=paragraph_text,
                text_length=len(paragraph_text),
                summary=extraction.get("summary"),
                is_eligibility_related=is_eligibility_related,
                extraction_status=extraction_status,
                critique_status=critique_status,
                critique_feedback=critique.get("feedback"),
                retry_count=retry_count
            )
            db.add(chunk)
            try:
                await db.flush()  # Flush to get the chunk.id
                # logger.info(f"[_persist] Created chunk {chunk.id} for {para_id} (page={page_number}, para_idx={paragraph_index})")  # Reduced logging
            except Exception as flush_err:
                logger.error(f"[_persist] Error flushing chunk for {para_id}: {flush_err}", exc_info=True)
                await db.rollback()
                return False
        
        if not chunk.id:
            logger.error(f"[_persist] CRITICAL: chunk.id is None for {para_id} after flush!")
            return False
        
        # Create ExtractedFact records for each fact
        facts_created = 0
        facts_failed = 0
        # logger.info(f"[_persist] Creating {len(facts)} ExtractedFact records for {para_id}")  # Reduced logging
        for idx, fact_data in enumerate(facts):
            try:
                # Validate required fields
                fact_text = fact_data.get("fact_text", "")
                if not fact_text or not fact_text.strip():
                    logger.warning(f"[_persist] Fact {idx+1} for {para_id} has empty fact_text, skipping")
                    facts_failed += 1
                    continue
                
                # Convert boolean values to strings for database storage
                is_verified_val = fact_data.get("is_verified")
                if isinstance(is_verified_val, bool):
                    is_verified_val = "true" if is_verified_val else "false"
                elif is_verified_val is not None:
                    is_verified_val = str(is_verified_val).lower()
                
                # Get is_pertinent_to_claims_or_members (new field)
                is_pertinent_val = fact_data.get("is_pertinent_to_claims_or_members")
                if isinstance(is_pertinent_val, bool):
                    is_pertinent_val = "true" if is_pertinent_val else "false"
                elif is_pertinent_val is not None:
                    is_pertinent_val = str(is_pertinent_val).lower()
                
                # Get is_eligibility_related (separate field)
                is_eligibility_related_val = fact_data.get("is_eligibility_related")
                if isinstance(is_eligibility_related_val, bool):
                    is_eligibility_related_val = "true" if is_eligibility_related_val else "false"
                elif is_eligibility_related_val is not None:
                    is_eligibility_related_val = str(is_eligibility_related_val).lower()
                
                confidence_val = fact_data.get("confidence")
                if confidence_val is not None and not isinstance(confidence_val, str):
                    confidence_val = str(confidence_val)
                
                # Validate chunk.id exists
                if not chunk.id:
                    logger.error(f"[_persist] CRITICAL: chunk.id is None when creating fact {idx+1} for {para_id}")
                    facts_failed += 1
                    continue
                
                from app.models import category_scores_dict_to_columns
                cat_cols = category_scores_dict_to_columns(fact_data.get("category_scores"))
                fact = ExtractedFact(
                    hierarchical_chunk_id=chunk.id,
                    document_id=doc_uuid,
                    fact_text=fact_text,
                    fact_type=fact_data.get("fact_type"),
                    who_eligible=fact_data.get("who_eligible"),
                    how_verified=fact_data.get("how_verified"),
                    conflict_resolution=fact_data.get("conflict_resolution"),
                    when_applies=fact_data.get("when_applies"),
                    limitations=fact_data.get("limitations"),
                    is_verified=is_verified_val,
                    is_eligibility_related=is_eligibility_related_val,
                    is_pertinent_to_claims_or_members=is_pertinent_val,
                    confidence=confidence_val,
                    **cat_cols
                )
                db.add(fact)
                facts_created += 1
                # logger.debug(f"[_persist] Added fact {idx+1}/{len(facts)} for {para_id}: fact_text='{fact.fact_text[:50]}...', chunk_id={chunk.id}")  # Reduced logging
            except Exception as fact_err:
                error_type = type(fact_err).__name__
                error_msg = str(fact_err)
                logger.error(f"[_persist] Error creating fact {idx+1} for {para_id}: {error_type}: {error_msg}", exc_info=True)
                facts_failed += 1
                # Continue with other facts
        
        # logger.info(f"[_persist] Added {facts_created} facts to session for {para_id} (failed: {facts_failed}, total: {len(facts)}), committing to database...")  # Reduced logging
        try:
            # Flush before commit to catch any constraint violations early
            await db.flush()
            # logger.debug(f"[_persist] Flush successful, now committing...")  # Reduced logging
            await db.commit()
            # Only log if there were facts created or if there were failures
            if facts_created > 0 or facts_failed > 0:
                logger.info(f"[_persist] Committed {para_id}: {facts_created} facts saved, {facts_failed} failed")
        except Exception as commit_err:
            error_type = type(commit_err).__name__
            error_msg = str(commit_err)
            logger.error(f"[_persist] Commit failed for {para_id}: {error_type}: {error_msg}", exc_info=True)
            # Log database commit error as critical
            try:
                import traceback
                error_details = {
                    "exception_type": error_type,
                    "exception_message": error_msg,
                    "paragraph_id": para_id,
                    "chunk_id": str(chunk.id) if chunk.id else None,
                    "facts_count": facts_created
                }
                if hasattr(commit_err, 'orig'):
                    error_details["original_error"] = str(commit_err.orig)
                if hasattr(commit_err, 'statement'):
                    error_details["sql_statement"] = str(commit_err.statement)
                error_details["stack_trace"] = traceback.format_exc()
                
                await log_error(
                    db=db,
                    document_id=str(doc_uuid),
                    error_type="database_error",
                    error_message=f"Database commit failed for paragraph {para_id}: {error_msg}",
                    severity="critical",
                    stage="persistence",
                    paragraph_id=para_id,
                    error_details=error_details
                )
            except Exception as log_err:
                logger.error(f"Failed to log commit error: {log_err}", exc_info=True)
            # Log more details about the error
            if hasattr(commit_err, 'orig'):
                logger.error(f"[_persist] Original error: {commit_err.orig}")
            if hasattr(commit_err, 'statement'):
                logger.error(f"[_persist] Failed SQL statement: {commit_err.statement}")
            try:
                await db.rollback()
                # logger.debug(f"[_persist] Rollback successful")  # Reduced logging
            except Exception as rollback_err:
                logger.error(f"[_persist] Rollback also failed: {rollback_err}", exc_info=True)
            return False
        
        # Verify facts were actually saved
        try:
            # Use a fresh query to ensure we're reading committed data
            verify_result = await db.execute(
                select(ExtractedFact).where(ExtractedFact.hierarchical_chunk_id == chunk.id)
            )
            verified_facts = verify_result.scalars().all()
            # Only log verification if there's a problem
            if facts_created > 0 and len(verified_facts) == 0:
                logger.error(f"[_persist] WARNING: {facts_created} facts were added but 0 found in DB after commit for {para_id}!")
            # logger.info(f"[_persist] Verification: {len(verified_facts)} facts found in DB for chunk {chunk.id} after commit")  # Reduced logging
                logger.error(f"[_persist] This indicates a silent failure - facts were not persisted despite successful commit")
                # Try to get more info about what went wrong
                try:
                    # Check if chunk exists
                    chunk_check = await db.execute(
                        select(HierarchicalChunk).where(HierarchicalChunk.id == chunk.id)
                    )
                    chunk_exists = chunk_check.scalar_one_or_none()
                    if not chunk_exists:
                        logger.error(f"[_persist] CRITICAL: Chunk {chunk.id} also doesn't exist in DB after commit!")
                    else:
                        logger.error(f"[_persist] Chunk exists but facts are missing - possible foreign key or constraint issue")
                except Exception as check_err:
                    logger.error(f"[_persist] Error checking chunk existence: {check_err}")
        except Exception as verify_err:
            logger.error(f"[_persist] Error verifying facts for {para_id}: {verify_err}", exc_info=True)
        
        return True
        
    except Exception as e:
        await db.rollback()
        logger.error(f"[_persist] Failed to persist paragraph {para_id}: {e}", exc_info=True)
        logger.error(f"[_persist] Exception type: {type(e).__name__}, message: {str(e)}")
        # Don't raise - allow chunking to continue even if persistence fails
        return False


async def _run_chunking_loop(
    document_id: str,
    doc_uuid,
    pages: list,
    threshold: float,
    db: AsyncSession,
    event_callback=None,  # Optional async generator callback: async def callback(event_type, data) -> yields SSE strings
):
    """
    Shared chunking loop logic. Runs the full paragraph processing loop.
    - If event_callback is provided, it should be an async generator that yields SSE event strings
    - Otherwise, just processes and persists results (for background task)
    - Checks _chunking_cancel at the start of each paragraph iteration
    """
    results_paragraphs: dict = {}
    
    try:
        from app.services.page_to_markdown import raw_page_to_markdown
        total_paragraphs = 0
        # Calculate total pages (pages with markdown; canonical source)
        def _page_md(p):
            md = getattr(p, "text_markdown", None) and (p.text_markdown or "").strip()
            return md or raw_page_to_markdown(p.text or "")
        total_pages = len([p for p in pages if _page_md(p)])
        for page in pages:
            page_md = _page_md(page)
            if not page_md:
                continue
            total_paragraphs += len(split_paragraphs_from_markdown(page_md))

        async def _upsert(status: str = "in_progress"):
            try:
                # logger.debug(f"[_upsert] Starting upsert with status: {status}, completed: {len(results_paragraphs)}")  # Reduced logging
                # Check if document still exists before upserting
                try:
                    doc_check = await db.execute(select(Document).where(Document.id == doc_uuid))
                    if doc_check.scalar_one_or_none() is None:
                        logger.warning(f"[_upsert] Document {doc_uuid} no longer exists, cannot update chunking results")
                        return False
                except Exception as check_err:
                    logger.error(f"[_upsert] Error checking document existence: {check_err}", exc_info=True)
                    # Continue anyway - might be a transient error
                
                # Get error counts for this document
                error_counts = {"critical": 0, "warning": 0, "info": 0}
                error_summary = {}
                try:
                    error_result = await db.execute(
                        select(ProcessingError).where(ProcessingError.document_id == doc_uuid)
                    )
                    errors = error_result.scalars().all()
                    for err in errors:
                        error_counts[err.severity] = error_counts.get(err.severity, 0) + 1
                        error_type = err.error_type
                        if error_type not in error_summary:
                            error_summary[error_type] = {"critical": 0, "warning": 0, "info": 0}
                        error_summary[error_type][err.severity] = error_summary[error_type].get(err.severity, 0) + 1
                except Exception as err_count_err:
                    logger.error(f"Failed to get error counts: {err_count_err}", exc_info=True)
                
                # Update document error tracking
                try:
                    doc_result = await db.execute(select(Document).where(Document.id == doc_uuid))
                    doc = doc_result.scalar_one_or_none()
                    if doc:
                        doc.error_count = sum(error_counts.values())
                        doc.critical_error_count = error_counts["critical"]
                        if error_counts["critical"] > 0:
                            doc.has_errors = "true"
                            # Set status to completed_with_errors if we're completing and have critical errors
                            if status == "completed" and error_counts["critical"] > 0:
                                status = "completed_with_errors"
                        elif error_counts["warning"] > 0 or error_counts["info"] > 0:
                            doc.has_errors = "true"
                except Exception as doc_update_err:
                    logger.error(f"Failed to update document error counts: {doc_update_err}", exc_info=True)
                
                meta = {
                    "total_paragraphs": total_paragraphs,
                    "retry_threshold": threshold,
                    "completed_count": len(results_paragraphs),
                    "status": status,
                    "updated_at": datetime.utcnow().isoformat(),
                    "error_counts": error_counts,
                    "error_summary": error_summary
                }
                res = {"paragraphs": dict(results_paragraphs)}
                q = await db.execute(select(ChunkingResult).where(ChunkingResult.document_id == doc_uuid))
                row = q.scalar_one_or_none()
                if row:
                    row.metadata_ = meta
                    row.results = res
                    row.updated_at = datetime.utcnow()
                else:
                    db.add(ChunkingResult(document_id=doc_uuid, metadata_=meta, results=res))
                try:
                    logger.debug(f"[_upsert] Committing to database")
                    await db.commit()
                    logger.debug(f"[_upsert] Successfully committed")
                    return True
                except Exception as e:
                    await db.rollback()
                    # Check if it's a foreign key violation (document deleted)
                    if "ForeignKeyViolationError" in str(type(e)) or "foreign key constraint" in str(e).lower():
                        logger.warning(f"[_upsert] Document {doc_uuid} was deleted, cannot update chunking results")
                        return False
                    # Log but don't raise - allow processing to continue
                    logger.error(f"[_upsert] Failed to commit: {e}", exc_info=True)
                    logger.error(f"[_upsert] Exception type: {type(e).__name__}, message: {str(e)}")
                    return False
            except Exception as e:
                # Catch any other database errors and log but don't stop processing
                logger.error(f"[_upsert] Unexpected error: {e}", exc_info=True)
                logger.error(f"[_upsert] Exception type: {type(e).__name__}, message: {str(e)}")
                try:
                    await db.rollback()
                except Exception as rollback_err:
                    logger.error(f"[_upsert] Error during rollback: {rollback_err}", exc_info=True)
                return False

        # Initialize status in DB immediately
        try:
            if not await _upsert("in_progress"):
                logger.warning(f"Document {document_id} was deleted before chunking could start")
                return
        except Exception as e:
            logger.error(f"Failed to initialize chunking status in DB: {e}", exc_info=True)
            # Continue anyway - we can still process and update later
        logger.info(f"Chunking started for document {document_id}: {total_paragraphs} paragraphs to process")

        # Load existing results to determine where to resume from
        existing_results = {}
        try:
            q = await db.execute(select(ChunkingResult).where(ChunkingResult.document_id == doc_uuid))
            row = q.scalar_one_or_none()
            if row and row.results and isinstance(row.results, dict):
                existing_results = row.results.get("paragraphs", {})
                # Merge existing results into current results_paragraphs
                results_paragraphs.update(existing_results)
                logger.info(f"Resuming chunking for {document_id}: {len(existing_results)} paragraphs already completed")
        except Exception as e:
            logger.warning(f"Could not load existing results for resume: {e}")

        for page in pages:
            page_md = _page_md(page)
            if not page_md:
                continue
            # Check for cancel at start of page processing
            if document_id in _chunking_cancel:
                _chunking_cancel.discard(document_id)
                await _upsert("stopped")
                return
            # Split into paragraphs (markdown as canonical source)
            paragraphs = split_paragraphs_from_markdown(page_md)
            
            for para_idx, para_data in enumerate(paragraphs):
                # Check for cancel at start of each paragraph iteration
                if document_id in _chunking_cancel:
                    _chunking_cancel.discard(document_id)
                    await _upsert("stopped")
                    return
                
                para_id = f"{page.page_number}_{para_idx}"
                
                # Skip if this paragraph was already successfully completed (resume logic)
                # Only skip if it has a successful final_status (not failed or pending)
                if para_id in existing_results:
                    existing_para = existing_results[para_id]
                    # Skip if it was successfully completed (passed or has facts)
                    if isinstance(existing_para, dict):
                        final_status = existing_para.get("final_status")
                        facts = existing_para.get("facts", [])
                        # Skip if it passed or has facts (successful completion)
                        if final_status == "passed" or (facts and len(facts) > 0):
                            logger.info(f"Skipping {para_id} - already completed successfully")
                            continue
                        # If it failed, retry it
                        elif final_status == "failed":
                            logger.info(f"Retrying {para_id} - previous attempt failed")
                        # Otherwise continue processing
                
                paragraph_text = para_data["text"]
                section_path = para_data.get("section_path")
                # Extract page and paragraph numbers from para_id
                page_num = page.page_number
                para_idx_num = para_idx
                
                # Calculate current progress
                completed = len(results_paragraphs)
                progress_pct = (completed / total_paragraphs * 100) if total_paragraphs > 0 else 0
                
                # logger.debug(f"[{para_id}] Starting paragraph processing (text length: {len(paragraph_text)})")  # Reduced logging
                
                # Wrap each paragraph processing in try/except to continue on errors
                try:
                    if event_callback:
                        try:
                            async for line in event_callback("paragraph_start", {
                                "paragraph_id": para_id,
                                "paragraph_text": paragraph_text,
                                "page_number": page_num,
                                "paragraph_index": para_idx_num,
                                "total_pages": total_pages,
                                "total_paragraphs": total_paragraphs,
                                "completed_paragraphs": completed,
                                "progress_percent": progress_pct
                            }):
                                yield line
                        except Exception as event_err:
                            logger.error(f"Error sending paragraph_start event: {event_err}", exc_info=True)
                    
                    # Stage 1: Stream raw LLM output for extraction
                    # logger.debug(f"[{para_id}] Starting extraction streaming")  # Reduced logging
                    raw_extraction_output = ""
                    try:
                        async for chunk in stream_extract_facts(paragraph_text, section_path=section_path):
                            # Check for cancellation during streaming
                            if document_id in _chunking_cancel:
                                _chunking_cancel.discard(document_id)
                                logger.info(f"Stop requested during extraction for {para_id}, stopping...")
                                await _upsert("stopped")
                                return
                            
                            raw_extraction_output += chunk
                            if event_callback:
                                try:
                                    async for line in event_callback("llm_stream", {
                                        "paragraph_id": para_id,
                                        "chunk": chunk
                                    }):
                                        yield line
                                except Exception as event_err:
                                    logger.error(f"Error sending llm_stream event: {event_err}", exc_info=True)
                    except Exception as stream_err:
                        logger.error(f"[{para_id}] Error streaming extraction: {stream_err}", exc_info=True)
                        # Log as critical error
                        try:
                            severity, stage = classify_error("llm_failure", stream_err, recovered=False)
                            await log_error(
                                db=db,
                                document_id=document_id,
                                error_type="llm_failure",
                                error_message=f"LLM streaming failed: {str(stream_err)}",
                                severity=severity,
                                stage=stage,
                                paragraph_id=para_id,
                                error_details={
                                    "exception_type": type(stream_err).__name__,
                                    "exception_message": str(stream_err),
                                    "paragraph_id": para_id
                                }
                            )
                        except Exception as log_err:
                            logger.error(f"Failed to log error: {log_err}", exc_info=True)
                        raw_extraction_output = ""  # Set empty so parsing will create error extraction
                    
                    # Parse extraction result
                    # logger.debug(f"[{para_id}] Parsing extraction result (output length: {len(raw_extraction_output)})")  # Reduced logging
                    try:
                        extraction = parse_json_response(raw_extraction_output)
                        if "summary" not in extraction:
                            extraction["summary"] = ""
                        if "facts" not in extraction:
                            extraction["facts"] = []
                    except Exception as e:
                        logger.error(f"[{para_id}] Failed to parse extraction JSON: {e}", exc_info=True)
                        # Check if parse_json_response recovered (it tries partial recovery)
                        recovered = False
                        try:
                            # Try to see if we got any partial data
                            if raw_extraction_output and "{" in raw_extraction_output:
                                recovered = True
                        except:
                            pass
                        # Log as warning if recovered, critical if not
                        try:
                            severity, stage = classify_error("json_parse_error", e, recovered=recovered)
                            await log_error(
                                db=db,
                                document_id=document_id,
                                error_type="json_parse_error",
                                error_message=f"Failed to parse extraction JSON: {str(e)}",
                                severity=severity,
                                stage=stage,
                                paragraph_id=para_id,
                                error_details={
                                    "exception_type": type(e).__name__,
                                    "exception_message": str(e),
                                    "recovered": recovered,
                                    "raw_output_preview": raw_extraction_output[:500] if raw_extraction_output else None,
                                    "paragraph_id": para_id
                                }
                            )
                        except Exception as log_err:
                            logger.error(f"Failed to log error: {log_err}", exc_info=True)
                        # logger.debug(f"[{para_id}] Raw extraction output (first 500 chars): {raw_extraction_output[:500]}")  # Reduced logging
                        extraction = {"summary": "", "facts": [], "error": str(e)}
                    
                    # Stage 2: Send parsed extraction results
                    if event_callback:
                        try:
                            async for line in event_callback("extraction_complete", {
                                "paragraph_id": para_id,
                                "summary": extraction.get("summary", ""),
                                "facts": extraction.get("facts", [])
                            }):
                                yield line
                        except Exception as event_err:
                            logger.error(f"Error sending extraction_complete event: {event_err}", exc_info=True)
                    
                    # Stage 3: Critique agent
                    if event_callback:
                        try:
                            async for line in event_callback("critique_start", {"paragraph_id": para_id}):
                                yield line
                        except Exception as event_err:
                            logger.error(f"Error sending critique_start event: {event_err}", exc_info=True)
                    
                    # Stream critique
                    # logger.debug(f"[{para_id}] Starting critique streaming (extraction has {len(extraction.get('facts', []))} facts)")  # Reduced logging
                    raw_critique_output = ""
                    try:
                        async for chunk in stream_critique(paragraph_text, extraction):
                            # Check for cancellation during streaming
                            if document_id in _chunking_cancel:
                                _chunking_cancel.discard(document_id)
                                logger.info(f"Stop requested during critique for {para_id}, stopping...")
                                await _upsert("stopped")
                                return
                            
                            raw_critique_output += chunk
                            if event_callback:
                                try:
                                    async for line in event_callback("llm_stream", {
                                        "paragraph_id": para_id,
                                        "chunk": chunk
                                    }):
                                        yield line
                                except Exception as event_err:
                                    logger.error(f"Error sending llm_stream event during critique: {event_err}", exc_info=True)
                    except Exception as stream_err:
                        logger.error(f"[{para_id}] Error streaming critique: {stream_err}", exc_info=True)
                        raw_critique_output = ""  # Set empty so parsing will create error critique
                    
                    # Parse critique result
                    # logger.debug(f"[{para_id}] Parsing critique result (output length: {len(raw_critique_output)})")  # Reduced logging
                    try:
                        critique = normalize_critique_result(parse_json_response(raw_critique_output))
                    except Exception as e:
                        logger.error(f"Failed to parse critique: {e}")
                        # Log critique parse error
                        try:
                            recovered = False
                            if raw_critique_output and "{" in raw_critique_output:
                                recovered = True
                            severity, stage = classify_error("json_parse_error", e, recovered=recovered)
                            await log_error(
                                db=db,
                                document_id=document_id,
                                error_type="json_parse_error",
                                error_message=f"Failed to parse critique JSON: {str(e)}",
                                severity=severity,
                                stage="critique",
                                paragraph_id=para_id,
                                error_details={
                                    "exception_type": type(e).__name__,
                                    "exception_message": str(e),
                                    "recovered": recovered,
                                    "raw_output_preview": raw_critique_output[:500] if raw_critique_output else None,
                                    "paragraph_id": para_id
                                }
                            )
                        except Exception as log_err:
                            logger.error(f"Failed to log error: {log_err}", exc_info=True)
                        critique = {
                            "pass": False,
                            "score": 0.0,
                            "category_assessment": {},
                            "feedback": f"Failed to parse critique: {str(e)}",
                            "issues": []
                        }
                    
                    # Send critique results
                    if event_callback:
                        try:
                            async for line in event_callback("critique_complete", {
                                "paragraph_id": para_id,
                                "pass": critique.get("pass", False),
                                "score": critique.get("score", 0.5),
                                "category_assessment": critique.get("category_assessment", {}),
                                "feedback": critique.get("feedback"),
                                "issues": critique.get("issues", [])
                            }):
                                yield line
                        except Exception as event_err:
                            logger.error(f"Error sending critique_complete event: {event_err}", exc_info=True)
                    
                    # Stage 4: Retry if needed (score < threshold)
                    retry_count = 0
                    max_retries = 2
                    current_extraction = extraction
                    current_critique = critique
                    consecutive_errors = 0
                    max_consecutive_errors = 3  # Break retry loop if too many errors
                    
                    critique_score = _critique_score(current_critique)
                    # logger.debug(f"[{para_id}] Initial critique score: {critique_score}, threshold: {threshold}, needs retry: {critique_score < threshold}")  # Reduced logging
                    
                    while _critique_score(current_critique) < threshold and retry_count < max_retries and consecutive_errors < max_consecutive_errors:
                        retry_count += 1
                        logger.info(f"[{para_id}] Starting retry attempt {retry_count}/{max_retries}")  # Keep retry attempts visible
                        
                        if event_callback:
                            try:
                                async for line in event_callback("retry_start", {
                                    "paragraph_id": para_id,
                                    "retry_count": retry_count,
                                    "feedback": current_critique.get("feedback")
                                }):
                                    yield line
                            except Exception as event_err:
                                logger.error(f"Error sending retry_start event: {event_err}", exc_info=True)
                        
                        # Retry extraction with streaming
                        issues_list = [issue.get("description", "") for issue in current_critique.get("issues", [])]
                        # logger.debug(f"[{para_id}] Retry {retry_count}: Streaming extraction with {len(issues_list)} issues")  # Reduced logging
                        retry_raw = ""
                        try:
                            async for chunk in stream_extract_facts(
                                paragraph_text,
                                critique_feedback=current_critique.get("feedback"),
                                issues=issues_list,
                                section_path=section_path
                            ):
                                # Check for cancellation during retry streaming
                                if document_id in _chunking_cancel:
                                    _chunking_cancel.discard(document_id)
                                    logger.info(f"Stop requested during retry extraction for {para_id}, stopping...")
                                    await _upsert("stopped")
                                    return
                                
                                retry_raw += chunk
                                if event_callback:
                                    try:
                                        async for line in event_callback("llm_stream", {
                                            "paragraph_id": para_id,
                                            "chunk": chunk
                                        }):
                                            yield line
                                    except Exception as event_err:
                                        logger.error(f"Error sending llm_stream event during retry: {event_err}", exc_info=True)
                        except Exception as stream_err:
                            logger.error(f"[{para_id}] Retry {retry_count}: Error streaming retry extraction: {stream_err}", exc_info=True)
                            retry_raw = ""  # Set empty so parsing will create error extraction
                            consecutive_errors += 1
                        
                        # Parse retry extraction
                        # logger.debug(f"[{para_id}] Retry {retry_count}: Parsing retry extraction (output length: {len(retry_raw)})")  # Reduced logging
                        try:
                            retry_extraction = parse_json_response(retry_raw) if retry_raw else {"summary": "", "facts": [], "error": "Stream failed"}
                            if "summary" not in retry_extraction:
                                retry_extraction["summary"] = ""
                            if "facts" not in retry_extraction:
                                retry_extraction["facts"] = []
                            # Reset error counter if parsing succeeds
                            if retry_raw:  # Only reset if we had data to parse
                                consecutive_errors = 0
                        except Exception as e:
                            logger.error(f"[{para_id}] Retry {retry_count}: Failed to parse retry extraction JSON: {e}", exc_info=True)
                            # Log retry parse error
                            try:
                                recovered = False
                                if retry_raw and "{" in retry_raw:
                                    recovered = True
                                severity, stage = classify_error("json_parse_error", e, recovered=recovered)
                                await log_error(
                                    db=db,
                                    document_id=document_id,
                                    error_type="json_parse_error",
                                    error_message=f"Failed to parse retry extraction JSON (retry {retry_count}): {str(e)}",
                                    severity=severity,
                                    stage="extraction",
                                    paragraph_id=para_id,
                                    error_details={
                                        "exception_type": type(e).__name__,
                                        "exception_message": str(e),
                                        "recovered": recovered,
                                        "retry_count": retry_count,
                                        "raw_output_preview": retry_raw[:500] if retry_raw else None,
                                        "paragraph_id": para_id
                                    }
                                )
                            except Exception as log_err:
                                logger.error(f"Failed to log error: {log_err}", exc_info=True)
                            # logger.debug(f"[{para_id}] Retry {retry_count}: Raw retry output (first 500 chars): {retry_raw[:500]}")  # Reduced logging
                            retry_extraction = {"summary": "", "facts": [], "error": str(e)}
                            consecutive_errors += 1
                        
                        if event_callback:
                            try:
                                async for line in event_callback("retry_extraction_complete", {
                                    "paragraph_id": para_id,
                                    "summary": retry_extraction.get("summary", ""),
                                    "facts": retry_extraction.get("facts", [])
                                }):
                                    yield line
                            except Exception as event_err:
                                logger.error(f"Error sending retry_extraction_complete event: {event_err}", exc_info=True)
                        
                        # Re-run critique
                        # logger.debug(f"[{para_id}] Retry {retry_count}: Running critique on retry extraction")  # Reduced logging
                        try:
                            current_critique = await critique_extraction(paragraph_text, retry_extraction)
                            current_extraction = retry_extraction
                            consecutive_errors = 0  # Reset error counter on success
                            retry_score = _critique_score(current_critique)
                            logger.info(f"[{para_id}] Retry {retry_count}: Critique score after retry: {retry_score}")  # Keep retry scores visible
                        except Exception as critique_err:
                            logger.error(f"[{para_id}] Retry {retry_count}: Error in critique_extraction: {critique_err}", exc_info=True)
                            consecutive_errors += 1
                            # Use a default failed critique
                            current_critique = {
                                "pass": False,
                                "score": 0.0,
                                "category_assessment": {},
                                "feedback": f"Critique error: {str(critique_err)}",
                                "issues": []
                            }
                            current_extraction = retry_extraction
                            # If too many consecutive errors, break out of retry loop
                            if consecutive_errors >= max_consecutive_errors:
                                logger.warning(f"Too many consecutive errors in retry loop for paragraph {para_id}, stopping retries")
                                break
                        
                        if event_callback:
                            try:
                                async for line in event_callback("critique_complete", {
                                    "paragraph_id": para_id,
                                    "pass": current_critique.get("pass", False),
                                    "score": _critique_score(current_critique),
                                    "category_assessment": current_critique.get("category_assessment", {}),
                                    "feedback": current_critique.get("feedback"),
                                    "issues": current_critique.get("issues", [])
                                }):
                                    yield line
                            except Exception as event_err:
                                logger.error(f"Error sending critique_complete event during retry: {event_err}", exc_info=True)
                    
                    # Final status - flag for human review if failed after retries
                    final_score = _critique_score(current_critique)
                    needs_human_review = final_score < threshold and retry_count >= max_retries
                    final_status = "passed" if final_score >= threshold else "failed"
                    logger.info(f"[{para_id}] Final status: {final_status}, score: {final_score}, retries: {retry_count}")  # Keep final status visible
                    
                    if event_callback:
                        try:
                            async for line in event_callback("paragraph_complete", {
                                "paragraph_id": para_id,
                                "status": final_status,
                                "needs_human_review": needs_human_review,
                                "retry_count": retry_count
                            }):
                                yield line
                        except Exception as event_err:
                            logger.error(f"Error sending paragraph_complete event: {event_err}", exc_info=True)
                        
                    # Persist to normalized tables (HierarchicalChunk and ExtractedFact)
                    facts_count = len(current_extraction.get('facts', []))
                    # logger.info(f"[{para_id}] About to persist to database - extraction has {facts_count} facts")  # Reduced logging
                    # if facts_count > 0:
                    #     logger.info(f"[{para_id}] Sample fact from extraction: {current_extraction.get('facts', [])[0] if current_extraction.get('facts') else 'N/A'}")  # Reduced logging
                    persistence_success = False
                    try:
                        persistence_success = await _persist_paragraph_to_db(
                            db=db,
                            doc_uuid=doc_uuid,
                            para_id=para_id,
                            paragraph_text=paragraph_text,
                            extraction=current_extraction,
                            critique=current_critique,
                            final_status=final_status,
                            retry_count=retry_count,
                            needs_human_review=needs_human_review
                        )
                        if not persistence_success:
                            logger.warning(f"[{para_id}] Persistence reported failure")
                            # Log persistence failure as critical error
                            try:
                                await log_error(
                                    db=db,
                                    document_id=document_id,
                                    error_type="persistence_error",
                                    error_message=f"Failed to persist paragraph {para_id} to database",
                                    severity="critical",
                                    stage="persistence",
                                    paragraph_id=para_id,
                                    error_details={
                                        "paragraph_id": para_id,
                                        "facts_count": len(current_extraction.get("facts", [])),
                                        "has_summary": bool(current_extraction.get("summary"))
                                    }
                                )
                            except Exception as log_err:
                                logger.error(f"Failed to log persistence error: {log_err}", exc_info=True)
                        # logger.info(f"[{para_id}] Persistence reported success")  # Reduced logging
                        # Send event to notify frontend that data has been persisted to DB
                        if persistence_success and event_callback:
                            try:
                                async for line in event_callback("paragraph_persisted", {
                                    "paragraph_id": para_id,
                                    "summary": current_extraction.get("summary", ""),
                                    "facts_count": len(current_extraction.get("facts", []))
                                }):
                                    yield line
                            except Exception as event_err:
                                logger.error(f"Error sending paragraph_persisted event: {event_err}", exc_info=True)
                    except Exception as e:
                        logger.error(f"[{para_id}] Error persisting to database: {e}", exc_info=True)
                        # Continue even if persistence fails - don't let DB errors stop processing

                    blob = {
                        "paragraph_id": para_id,
                        "paragraph_text": paragraph_text,
                        "summary": current_extraction.get("summary", ""),
                        "facts": current_extraction.get("facts", []),
                        "critique_result": {
                            "pass": current_critique.get("pass", False),
                            "score": current_critique.get("score", 0.5),
                            "category_assessment": current_critique.get("category_assessment", {}),
                            "feedback": current_critique.get("feedback"),
                            "issues": current_critique.get("issues", []),
                        },
                        "retries": [],
                        "final_status": final_status,
                        "needs_human_review": needs_human_review,
                    }
                    results_paragraphs[para_id] = blob
                    
                    # Send progress update after paragraph is added to results
                    if event_callback:
                        try:
                            completed_after = len(results_paragraphs)
                            progress_pct_after = (completed_after / total_paragraphs * 100) if total_paragraphs > 0 else 0
                            
                            async for line in event_callback("progress_update", {
                                "current_paragraph": para_id,
                                "current_page": page_num,
                                "total_pages": total_pages,
                                "total_paragraphs": total_paragraphs,
                                "completed_paragraphs": completed_after,
                                "progress_percent": progress_pct_after
                            }):
                                yield line
                        except Exception as event_err:
                            logger.error(f"Error sending progress_update event: {event_err}", exc_info=True)
                    
                    # Update status in DB, but don't let DB errors stop processing
                    # logger.debug(f"[{para_id}] Updating chunking status in DB")  # Reduced logging
                    try:
                        await _upsert("in_progress")
                        # logger.debug(f"[{para_id}] Successfully updated chunking status")  # Reduced logging
                    except Exception as upsert_err:
                        logger.error(f"[{para_id}] Failed to upsert chunking status: {upsert_err}", exc_info=True)
                        # Log database upsert error as warning
                        try:
                            await log_error(
                                db=db,
                                document_id=document_id,
                                error_type="database_error",
                                error_message=f"Failed to update chunking status in database: {str(upsert_err)}",
                                severity="warning",
                                stage="persistence",
                                paragraph_id=para_id,
                                error_details={
                                    "exception_type": type(upsert_err).__name__,
                                    "exception_message": str(upsert_err),
                                    "paragraph_id": para_id
                                }
                            )
                        except Exception as log_err:
                            logger.error(f"Failed to log upsert error: {log_err}", exc_info=True)
                        # Continue processing even if DB update fails
                    
                    # logger.debug(f"[{para_id}] Paragraph processing completed successfully")  # Reduced logging
                
                except Exception as para_error:
                    # Log error for this specific paragraph but continue with next one
                    logger.error(f"[{para_id}] ERROR: Exception in paragraph processing: {para_error}", exc_info=True)
                    logger.error(f"[{para_id}] Exception type: {type(para_error).__name__}, message: {str(para_error)}")
                    # Log as critical error
                    try:
                        import traceback
                        await log_error(
                            db=db,
                            document_id=document_id,
                            error_type="other",
                            error_message=f"Exception in paragraph processing: {str(para_error)}",
                            severity="critical",
                            stage="other",
                            paragraph_id=para_id,
                            error_details={
                                "exception_type": type(para_error).__name__,
                                "exception_message": str(para_error),
                                "stack_trace": traceback.format_exc(),
                                "paragraph_id": para_id
                            }
                        )
                    except Exception as log_err:
                        logger.error(f"Failed to log paragraph error: {log_err}", exc_info=True)
                    # Mark this paragraph as failed in results
                    results_paragraphs[para_id] = {
                        "paragraph_id": para_id,
                        "paragraph_text": paragraph_text[:100] + "..." if len(paragraph_text) > 100 else paragraph_text,
                        "summary": "",
                        "facts": [],
                        "critique_result": None,
                        "retries": [],
                        "final_status": "failed",
                        "needs_human_review": True,
                        "error": str(para_error)
                    }
                    # Send error event
                    if event_callback:
                        async for line in event_callback("paragraph_error", {
                            "paragraph_id": para_id,
                            "error": str(para_error)
                        }):
                            yield line
                        
                        # Send progress update even for failed paragraphs
                        try:
                            completed_after = len(results_paragraphs)
                            progress_pct_after = (completed_after / total_paragraphs * 100) if total_paragraphs > 0 else 0
                            
                            async for line in event_callback("progress_update", {
                                "current_paragraph": para_id,
                                "current_page": page_num,
                                "total_pages": total_pages,
                                "total_paragraphs": total_paragraphs,
                                "completed_paragraphs": completed_after,
                                "progress_percent": progress_pct_after
                            }):
                                yield line
                        except Exception as event_err:
                            logger.error(f"Error sending progress_update event: {event_err}", exc_info=True)
                    
                    # Continue to next paragraph
                    continue

        # Mark as completed, but don't let DB errors stop the process
        try:
            await _upsert("completed")
        except Exception as e:
            logger.error(f"Failed to mark chunking as completed in DB: {e}", exc_info=True)
            # Continue anyway - send completion event
        if event_callback:
            try:
                async for line in event_callback("chunking_complete", {"total_paragraphs": total_paragraphs}):
                    yield line
            except Exception as event_err:
                logger.error(f"Error sending chunking_complete event: {event_err}", exc_info=True)
            
    except Exception as e:
        logger.error(f"Chunking loop error: {e}", exc_info=True)
        try:
            await _upsert("failed")
        except Exception as upsert_err:
            logger.error(f"Failed to update status to failed: {upsert_err}", exc_info=True)
        if event_callback:
            try:
                async for line in event_callback("error", {"error": str(e)}):
                    yield line
            except Exception as event_err:
                logger.error(f"Error sending error event: {event_err}", exc_info=True)
        # Don't raise - log the error but allow the background task to complete gracefully
        # The error has been logged and status updated, but we don't want to crash the entire process
        logger.error(f"Chunking loop encountered an error but continuing: {e}")


@app.get("/documents/{document_id}/chunking/results")
async def get_chunking_results(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return persisted chunking/extraction results from PostgreSQL (metadata + results JSONB).
    Also merges facts from normalized ExtractedFact table if available."""
    from uuid import UUID
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    
    # Get JSONB results
    r = await db.execute(select(ChunkingResult).where(ChunkingResult.document_id == doc_uuid))
    row = r.scalar_one_or_none()
    
    if not row:
        # Even if no JSONB results, try to load from normalized tables
        chunks_result = await db.execute(
            select(HierarchicalChunk).where(HierarchicalChunk.document_id == doc_uuid)
            .order_by(HierarchicalChunk.page_number, HierarchicalChunk.paragraph_index)
        )
        chunks = chunks_result.scalars().all()
        
        if not chunks:
            return {"document_id": document_id, "metadata": {}, "results": {}}
        
        # Build results from normalized tables
        results_paragraphs = {}
        for chunk in chunks:
            para_id = f"{chunk.page_number}_{chunk.paragraph_index}"
            
            # Get facts for this chunk
            facts_result = await db.execute(
                select(ExtractedFact).where(ExtractedFact.hierarchical_chunk_id == chunk.id)
            )
            facts = facts_result.scalars().all()
            
            # Convert facts to dict format
            facts_list = []
            for fact in facts:
                facts_list.append({
                    "fact_text": fact.fact_text,
                    "fact_type": fact.fact_type,
                    "who_eligible": fact.who_eligible,
                    "how_verified": fact.how_verified,
                    "conflict_resolution": fact.conflict_resolution,
                    "when_applies": fact.when_applies,
                    "limitations": fact.limitations,
                    "is_verified": fact.is_verified,
                    "is_eligibility_related": fact.is_eligibility_related,
                    "is_pertinent_to_claims_or_members": fact.is_pertinent_to_claims_or_members,
                    "confidence": fact.confidence,
                    "category_scores": fact_to_category_scores_dict(fact),
                })
            
            results_paragraphs[para_id] = {
                "paragraph_id": para_id,
                "paragraph_text": chunk.text,
                "summary": chunk.summary or "",
                "facts": facts_list,
                "critique_result": {
                    "pass": chunk.critique_status == "passed",
                    "score": 0.5,  # Default if not available
                    "category_assessment": {},
                    "feedback": chunk.critique_feedback,
                    "issues": [],
                } if chunk.critique_feedback else None,
                "retries": [],
                "final_status": chunk.critique_status if chunk.critique_status in ["passed", "failed"] else "pending",
                "needs_human_review": chunk.critique_status == "failed" and chunk.retry_count >= 2,
            }
        
        return {
            "document_id": document_id,
            "metadata": {"status": "completed", "total_paragraphs": len(results_paragraphs)},
            "results": {"paragraphs": results_paragraphs},
        }
    
    # Merge facts from normalized tables into JSONB results
    results = row.results or {}
    paragraphs = results.get("paragraphs", {})
    
    # Get all chunks for this document
    chunks_result = await db.execute(
        select(HierarchicalChunk).where(HierarchicalChunk.document_id == doc_uuid)
        .order_by(HierarchicalChunk.page_number, HierarchicalChunk.paragraph_index)
    )
    chunks = chunks_result.scalars().all()
    
    # Build a map of para_id -> chunk
    chunks_by_para_id = {}
    for chunk in chunks:
        para_id = f"{chunk.page_number}_{chunk.paragraph_index}"
        chunks_by_para_id[para_id] = chunk
    
    # For each paragraph in JSONB, merge facts from normalized tables if available
    for para_id, para_data in paragraphs.items():
        chunk = chunks_by_para_id.get(para_id)
        if chunk:
            # Get facts from normalized table
            facts_result = await db.execute(
                select(ExtractedFact).where(ExtractedFact.hierarchical_chunk_id == chunk.id)
            )
            normalized_facts = facts_result.scalars().all()
            
            # If we have facts in normalized table but not in JSONB, or if JSONB facts are empty, use normalized
            jsonb_facts = para_data.get("facts", [])
            if normalized_facts and (not jsonb_facts or len(jsonb_facts) == 0):
                logger.info(f"[get_chunking_results] Merging {len(normalized_facts)} facts from normalized table for {para_id}")
                facts_list = []
                for fact in normalized_facts:
                    facts_list.append({
                        "fact_text": fact.fact_text,
                        "fact_type": fact.fact_type,
                        "who_eligible": fact.who_eligible,
                        "how_verified": fact.how_verified,
                        "conflict_resolution": fact.conflict_resolution,
                        "when_applies": fact.when_applies,
                        "limitations": fact.limitations,
                        "is_verified": fact.is_verified,
                        "is_eligibility_related": fact.is_eligibility_related,
                        "is_pertinent_to_claims_or_members": fact.is_pertinent_to_claims_or_members,
                        "confidence": fact.confidence,
                        "category_scores": fact_to_category_scores_dict(fact),
                    })
                para_data["facts"] = facts_list
    
    return {
        "document_id": document_id,
        "metadata": row.metadata_ or {},
        "results": results,
    }


@app.get("/documents/{document_id}/facts")
async def get_document_facts(
    document_id: str,
    page_number: int = None,
    db: AsyncSession = Depends(get_db),
):
    """Get extracted facts from normalized PostgreSQL tables (HierarchicalChunk and ExtractedFact)."""
    from uuid import UUID
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    
    # Verify document exists
    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Build query for chunks
    chunk_query = select(HierarchicalChunk).where(HierarchicalChunk.document_id == doc_uuid)
    if page_number is not None:
        chunk_query = chunk_query.where(HierarchicalChunk.page_number == page_number)
    chunk_query = chunk_query.order_by(HierarchicalChunk.page_number, HierarchicalChunk.paragraph_index)
    
    chunks_result = await db.execute(chunk_query)
    chunks = chunks_result.scalars().all()
    
    # Get facts for all chunks
    chunk_ids = [chunk.id for chunk in chunks]
    facts_result = await db.execute(
        select(ExtractedFact).where(ExtractedFact.document_id == doc_uuid)
        .order_by(ExtractedFact.created_at)
    )
    all_facts = facts_result.scalars().all()
    
    # Group facts by chunk
    facts_by_chunk = {}
    for fact in all_facts:
        if fact.hierarchical_chunk_id in chunk_ids:
            if fact.hierarchical_chunk_id not in facts_by_chunk:
                facts_by_chunk[fact.hierarchical_chunk_id] = []
            facts_by_chunk[fact.hierarchical_chunk_id].append({
                "id": str(fact.id),
                "fact_text": fact.fact_text,
                "fact_type": fact.fact_type,
                "who_eligible": fact.who_eligible,
                "how_verified": fact.how_verified,
                "conflict_resolution": fact.conflict_resolution,
                "when_applies": fact.when_applies,
                "limitations": fact.limitations,
                "is_verified": fact.is_verified,
                "is_eligibility_related": fact.is_eligibility_related,
                "is_pertinent_to_claims_or_members": fact.is_pertinent_to_claims_or_members,
                "confidence": fact.confidence,
                "category_scores": fact_to_category_scores_dict(fact),
                "page_number": fact.page_number,
                "start_offset": fact.start_offset,
                "end_offset": fact.end_offset,
            })
    
    # Build response
    chunks_data = []
    for chunk in chunks:
        chunks_data.append({
            "id": str(chunk.id),
            "page_number": chunk.page_number,
            "paragraph_index": chunk.paragraph_index,
            "text": chunk.text,
            "text_length": chunk.text_length,
            "summary": chunk.summary,
            "is_eligibility_related": chunk.is_eligibility_related,
            "extraction_status": chunk.extraction_status,
            "critique_status": chunk.critique_status,
            "critique_feedback": chunk.critique_feedback,
            "retry_count": chunk.retry_count,
            "created_at": chunk.created_at.isoformat(),
            "facts": facts_by_chunk.get(chunk.id, []),
        })
    
    return {
        "document_id": document_id,
        "filename": document.filename,
        "total_chunks": len(chunks_data),
        "total_facts": len(all_facts),
        "chunks": chunks_data,
    }


@app.post("/documents/{document_id}/reader-facts")
async def create_reader_fact(
    document_id: str,
    body: dict = Body(...),
    db: AsyncSession = Depends(get_db),
):
    """Create a fact from the reader (user-selected text). Uses a per-document manual chunk; stores page_number and start_offset/end_offset for persistent highlights."""
    from uuid import UUID
    from app.models import category_scores_dict_to_columns

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    fact_text = (body.get("fact_text") or "").strip()
    if not fact_text:
        raise HTTPException(status_code=400, detail="fact_text is required and cannot be empty")

    page_number = body.get("page_number")
    if page_number is not None:
        try:
            page_number = int(page_number)
        except (TypeError, ValueError):
            page_number = None
    start_offset = body.get("start_offset")
    end_offset = body.get("end_offset")
    if start_offset is not None:
        try:
            start_offset = int(start_offset)
        except (TypeError, ValueError):
            start_offset = None
    if end_offset is not None:
        try:
            end_offset = int(end_offset)
        except (TypeError, ValueError):
            end_offset = None
    if start_offset is not None and end_offset is not None and start_offset >= end_offset:
        raise HTTPException(status_code=400, detail="start_offset must be less than end_offset")

    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    manual_chunk = await _get_or_create_manual_chunk(db, doc_uuid)

    is_pertinent = body.get("is_pertinent_to_claims_or_members")
    if is_pertinent is None:
        is_pertinent = True
    is_pertinent_val = "true" if is_pertinent else "false"

    cat_scores = body.get("category_scores") or {}
    cat_cols = category_scores_dict_to_columns(cat_scores)

    fact = ExtractedFact(
        hierarchical_chunk_id=manual_chunk.id,
        document_id=doc_uuid,
        fact_text=fact_text,
        fact_type="other",
        who_eligible=None,
        how_verified=None,
        conflict_resolution=None,
        when_applies=None,
        limitations=None,
        is_verified="false",
        is_eligibility_related="false",
        is_pertinent_to_claims_or_members=is_pertinent_val,
        confidence=None,
        page_number=page_number,
        start_offset=start_offset,
        end_offset=end_offset,
        **cat_cols,
    )
    db.add(fact)
    await db.commit()
    await db.refresh(fact)

    return {
        "id": str(fact.id),
        "fact_text": fact.fact_text,
        "page_number": fact.page_number,
        "start_offset": fact.start_offset,
        "end_offset": fact.end_offset,
        "category_scores": fact_to_category_scores_dict(fact),
    }


@app.delete("/documents/{document_id}/facts/{fact_id}")
async def delete_document_fact(
    document_id: str,
    fact_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a fact. Fact must belong to the given document."""
    from uuid import UUID
    from sqlalchemy import delete as sql_delete
    try:
        doc_uuid = UUID(document_id)
        fact_uuid = UUID(fact_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document or fact ID")
    result = await db.execute(
        select(ExtractedFact).where(
            ExtractedFact.id == fact_uuid,
            ExtractedFact.document_id == doc_uuid,
        )
    )
    fact = result.scalar_one_or_none()
    if not fact:
        raise HTTPException(status_code=404, detail="Fact not found")
    await db.execute(sql_delete(ExtractedFact).where(ExtractedFact.id == fact_uuid))
    await db.commit()
    return {"status": "deleted", "id": fact_id}


@app.patch("/documents/{document_id}/facts/{fact_id}")
async def update_document_fact(
    document_id: str,
    fact_id: str,
    body: dict = Body(...),
    db: AsyncSession = Depends(get_db),
):
    """Update a fact (fact_text, category_scores, is_pertinent_to_claims_or_members). Fact must belong to the document."""
    from uuid import UUID
    from app.models import category_scores_dict_to_columns
    try:
        doc_uuid = UUID(document_id)
        fact_uuid = UUID(fact_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document or fact ID")
    result = await db.execute(
        select(ExtractedFact).where(
            ExtractedFact.id == fact_uuid,
            ExtractedFact.document_id == doc_uuid,
        )
    )
    fact = result.scalar_one_or_none()
    if not fact:
        raise HTTPException(status_code=404, detail="Fact not found")
    if "fact_text" in body:
        val = (body.get("fact_text") or "").strip()
        if val:
            fact.fact_text = val
    if "is_pertinent_to_claims_or_members" in body:
        is_pertinent = body["is_pertinent_to_claims_or_members"]
        fact.is_pertinent_to_claims_or_members = "true" if is_pertinent else "false"
    if "category_scores" in body:
        cat_scores = body.get("category_scores") or {}
        cat_cols = category_scores_dict_to_columns(cat_scores)
        for k, v in cat_cols.items():
            setattr(fact, k, v)
    await db.commit()
    await db.refresh(fact)
    return {
        "id": str(fact.id),
        "fact_text": fact.fact_text,
        "page_number": fact.page_number,
        "start_offset": fact.start_offset,
        "end_offset": fact.end_offset,
        "category_scores": fact_to_category_scores_dict(fact),
    }


@app.post("/documents/{document_id}/chunking/stop")
async def stop_chunking(document_id: str):
    """Signal the chunking stream to stop after the current paragraph."""
    from uuid import UUID
    from app.database import AsyncSessionLocal
    
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    
    # Add to cancel set
    _chunking_cancel.add(document_id)
    logger.info(f"Stop requested for document {document_id}")
    
    # Immediately update status in database to "stopped"
    try:
        db_session = AsyncSessionLocal()
        try:
            q = await db_session.execute(select(ChunkingResult).where(ChunkingResult.document_id == doc_uuid))
            row = q.scalar_one_or_none()
            if row:
                meta = row.metadata_ or {}
                meta["status"] = "stopped"
                meta["updated_at"] = datetime.utcnow().isoformat()
                row.metadata_ = meta
                row.updated_at = datetime.utcnow()
                await db_session.commit()
                logger.info(f"Updated chunking status to 'stopped' for document {document_id}")
        finally:
            await db_session.close()
    except Exception as e:
        logger.error(f"Failed to update stop status in DB: {e}", exc_info=True)
        # Continue anyway - the cancel flag is set
    
    return {"status": "ok", "message": "Stop requested"}


@app.post("/documents/{document_id}/chunking/start")
async def start_chunking(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    threshold: float | None = None,
):
    """Queue a chunking job. Returns immediately. Worker process will pick it up."""
    from uuid import UUID

    th = threshold if threshold is not None else CRITIQUE_RETRY_THRESHOLD
    th = max(0.0, min(1.0, th))
    
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    
    # Check if document exists
    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check if there's already a pending or processing job for this document
    existing_job = await db.execute(
        select(ChunkingJob).where(
            ChunkingJob.document_id == doc_uuid,
            ChunkingJob.status.in_(["pending", "processing"])
        )
    )
    if existing_job.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Chunking job already queued or in progress for this document")
    
    # Create new job
    job = ChunkingJob(
        document_id=doc_uuid,
        status="pending",
        threshold=str(th)
    )
    db.add(job)
    await db.commit()
    
    logger.info(f"Queued chunking job {job.id} for document {document_id} with threshold {th}")
    
    return {"status": "queued", "document_id": document_id, "job_id": str(job.id)}


@app.post("/documents/{document_id}/chunking/restart")
async def restart_chunking(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    threshold: float | None = None,
):
    """Restart chunking for a document that failed or was stopped."""
    from uuid import UUID
    import asyncio
    
    th = threshold if threshold is not None else CRITIQUE_RETRY_THRESHOLD
    th = max(0.0, min(1.0, th))
    
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    
    # Check if already running
    if document_id in _chunking_running:
        raise HTTPException(status_code=409, detail="Chunking already in progress for this document")
    
    # Get document
    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check if extraction is complete
    if document.status != "completed":
        raise HTTPException(status_code=400, detail="Document extraction must be completed before chunking")
    
    # Get pages
    pages_result = await db.execute(
        select(DocumentPage).where(DocumentPage.document_id == doc_uuid)
        .order_by(DocumentPage.page_number)
    )
    pages = pages_result.scalars().all()
    
    if not pages:
        raise HTTPException(status_code=400, detail="No pages found for document")
    
    # Create background task (same as start_chunking)
    async def background_task():
        from app.database import AsyncSessionLocal
        db_session = None
        try:
            _chunking_running.add(document_id)
            logger.info(f"Restarting background chunking task for document {document_id} with threshold {th}")
            db_session = AsyncSessionLocal()
            try:
                event_callback = await _create_event_buffer_callback(document_id, db_session)
                async for _ in _run_chunking_loop(document_id, doc_uuid, pages, th, db_session, event_callback=event_callback):
                    pass
            finally:
                await db_session.close()
        except Exception as e:
            logger.error(f"Background chunking task error: {e}", exc_info=True)
            # Update status to failed
            try:
                if db_session is None:
                    db_session = AsyncSessionLocal()
                q = await db_session.execute(select(ChunkingResult).where(ChunkingResult.document_id == doc_uuid))
                row = q.scalar_one_or_none()
                if row:
                    meta = row.metadata_ or {}
                    meta["status"] = "failed"
                    meta["error"] = str(e)
                    row.metadata_ = meta
                    row.updated_at = datetime.utcnow()
                    await db_session.commit()
            except Exception as db_err:
                logger.error(f"Failed to update error status: {db_err}")
            finally:
                if db_session:
                    await db_session.close()
        finally:
            _chunking_running.discard(document_id)
            # Clean up event buffer
            await asyncio.sleep(5)
            if document_id in _chunking_events:
                async with _chunking_events_lock.get(document_id, Lock()):
                    _chunking_events[document_id].clear()
                if document_id in _chunking_events:
                    del _chunking_events[document_id]
                if document_id in _chunking_events_lock:
                    del _chunking_events_lock[document_id]
    
    # Spawn background task
    logger.info(f"Spawning restart chunking task for document {document_id}")
    asyncio.create_task(background_task())
    
    return {"status": "restarted", "document_id": document_id}


@app.get("/documents/{document_id}/chunking/stream")
async def stream_chunking_process(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    threshold: float | None = None,
):
    """Stream chunking and extraction process in real-time. Optional ?threshold= (0–1): retry when critique score < threshold."""
    from uuid import UUID

    th = threshold if threshold is not None else CRITIQUE_RETRY_THRESHOLD
    th = max(0.0, min(1.0, th))
    
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    
    # Get document
    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get pages
    pages_result = await db.execute(
        select(DocumentPage).where(DocumentPage.document_id == doc_uuid)
        .order_by(DocumentPage.page_number)
    )
    pages = pages_result.scalars().all()

    async def event_callback(event_type: str, data: dict):
        """Async generator that yields SSE event strings"""
        yield f"data: {json.dumps({'event': event_type, 'data': data})}\n\n"
    
    async def event_generator():
        # Use shared loop with event callback
        async for event_line in _run_chunking_loop(document_id, doc_uuid, pages, th, db, event_callback):
            yield event_line
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/documents/{document_id}/chunking/live")
async def stream_chunking_live(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Stream live chunking events via SSE. Reads from database events table."""
    from uuid import UUID
    
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    
    # Verify document exists
    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    async def event_generator():
        last_event_id = None
        consecutive_no_events = 0
        max_no_events = 120  # 120 polls * 0.5s = 60 seconds of no events before closing
        initial_events_sent = False
        
        while True:
            try:
                # Query for new events from database
                query = select(ChunkingEvent).where(
                    ChunkingEvent.document_id == doc_uuid
                ).order_by(ChunkingEvent.created_at)
                
                if last_event_id:
                    # Only get events after the last one we sent
                    query = query.where(ChunkingEvent.id > last_event_id)
                
                result = await db.execute(query)
                events = result.scalars().all()
                
                if events:
                    if not initial_events_sent and last_event_id is None:
                        logger.info(f"SSE: Sending {len(events)} existing events for document {document_id} on initial connection")
                        initial_events_sent = True
                    
                    for event in events:
                        # Format as SSE event
                        event_dict = {
                            "event": event.event_type,
                            "data": event.event_data,
                            "timestamp": event.created_at.isoformat()
                        }
                        yield f"data: {json.dumps(event_dict)}\n\n"
                        last_event_id = event.id
                    
                    consecutive_no_events = 0
                elif not initial_events_sent and last_event_id is None:
                    # No events found on initial connection
                    logger.info(f"SSE: No existing events found for document {document_id}")
                    initial_events_sent = True
                else:
                    consecutive_no_events += 1
                    
                    # Check if job is still active
                    job_result = await db.execute(
                        select(ChunkingJob).where(
                            ChunkingJob.document_id == doc_uuid,
                            ChunkingJob.status.in_(["pending", "processing"])
                        )
                    )
                    active_job = job_result.scalar_one_or_none()
                    
                    # If no active job and no events for a while, check if chunking is complete
                    if not active_job and consecutive_no_events >= 10:
                        # Check chunking result status
                        chunking_result = await db.execute(
                            select(ChunkingResult).where(ChunkingResult.document_id == doc_uuid)
                        )
                        chunking = chunking_result.scalar_one_or_none()
                        
                        if chunking and chunking.metadata_:
                            status = chunking.metadata_.get("status")
                            if status in ["completed", "failed", "stopped"]:
                                # Send completion event and close
                                yield f"data: {json.dumps({'event': 'stream_end', 'data': {'reason': 'chunking_complete', 'status': status}})}\n\n"
                                break
                    
                    # Close if no events for too long (likely connection issue or job stopped)
                    if consecutive_no_events >= max_no_events:
                        logger.warning(f"SSE: No events for {max_no_events * 0.5}s, closing stream for document {document_id}")
                        yield f"data: {json.dumps({'event': 'stream_end', 'data': {'reason': 'timeout'}})}\n\n"
                        break
                
                # Wait before next poll
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in SSE stream generator: {e}", exc_info=True)
                yield f"data: {json.dumps({'event': 'error', 'data': {'message': str(e)}})}\n\n"
                await asyncio.sleep(1)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/documents/{document_id}/chunking/events")
async def get_chunking_events(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    limit: int = 1000,
):
    """Get existing chunking events for a document. Useful for loading past events when opening Live Updates tab."""
    from uuid import UUID
    
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    
    # Verify document exists
    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Query events
    query = select(ChunkingEvent).where(
        ChunkingEvent.document_id == doc_uuid
    ).order_by(ChunkingEvent.created_at).limit(limit)
    
    result = await db.execute(query)
    events = result.scalars().all()
    
    # Format events
    formatted_events = []
    for event in events:
        formatted_events.append({
            "event": event.event_type,
            "data": event.event_data,
            "timestamp": event.created_at.isoformat(),
            "id": str(event.id)
        })
    
    return {
        "document_id": document_id,
        "events": formatted_events,
        "count": len(formatted_events)
    }


# ============================================================================
# Database Admin Endpoints
# ============================================================================

@app.get("/admin/db/tables")
async def list_tables(db: AsyncSession = Depends(get_db)):
    """List all database tables."""
    from sqlalchemy import inspect
    from sqlalchemy import text
    
    try:
        # Get all tables from metadata
        tables = []
        for table_name in Base.metadata.tables.keys():
            # Get row count
            result = await db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            count = result.scalar()
            
            tables.append({
                "name": table_name,
                "row_count": count
            })
        
        return {"tables": sorted(tables, key=lambda x: x["name"])}
    except Exception as e:
        logger.error(f"Error listing tables: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/db/tables/{table_name}/schema")
async def get_table_schema(table_name: str, db: AsyncSession = Depends(get_db)):
    """Get schema information for a table."""
    from sqlalchemy import inspect, text
    from sqlalchemy.engine import reflection
    
    try:
        # Validate table exists
        if table_name not in Base.metadata.tables:
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
        
        table = Base.metadata.tables[table_name]
        columns = []
        
        for column in table.columns:
            col_info = {
                "name": column.name,
                "type": str(column.type),
                "nullable": column.nullable,
                "primary_key": column.primary_key,
                "foreign_key": None,
                "default": str(column.default) if column.default else None
            }
            
            # Check for foreign keys
            for fk in column.foreign_keys:
                col_info["foreign_key"] = {
                    "table": fk.column.table.name,
                    "column": fk.column.name
                }
            
            columns.append(col_info)
        
        return {
            "table_name": table_name,
            "columns": columns
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting table schema: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Error Review Endpoints
@app.get("/documents/{document_id}/errors")
async def get_document_errors(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    resolved: bool | None = None,
    severity: str | None = None
):
    """Get all errors for a specific document."""
    from uuid import UUID
    
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    
    try:
        query = select(ProcessingError).where(ProcessingError.document_id == doc_uuid)
        
        if resolved is not None:
            query = query.where(ProcessingError.resolved == ("true" if resolved else "false"))
        if severity:
            query = query.where(ProcessingError.severity == severity)
        
        query = query.order_by(ProcessingError.created_at.desc())
        
        result = await db.execute(query)
        errors = result.scalars().all()
        
        return {
            "document_id": document_id,
            "errors": [{
                "id": str(err.id),
                "paragraph_id": err.paragraph_id,
                "error_type": err.error_type,
                "severity": err.severity,
                "error_message": err.error_message,
                "error_details": err.error_details,
                "stage": err.stage,
                "resolved": err.resolved == "true",
                "resolution": err.resolution,
                "resolved_by": err.resolved_by,
                "resolved_at": err.resolved_at.isoformat() if err.resolved_at else None,
                "resolution_notes": err.resolution_notes,
                "created_at": err.created_at.isoformat()
            } for err in errors]
        }
    except Exception as e:
        logger.error(f"Error fetching document errors: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/errors")
async def list_errors(
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    document_id: str | None = None,
    error_type: str | None = None,
    severity: str | None = None,
    resolved: bool | None = None,
    review_status: str | None = None
):
    """List all processing errors with filters."""
    try:
        query = select(ProcessingError)
        
        if document_id:
            try:
                doc_uuid = UUID(document_id)
                query = query.where(ProcessingError.document_id == doc_uuid)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid document_id format")
        
        if error_type:
            query = query.where(ProcessingError.error_type == error_type)
        if severity:
            query = query.where(ProcessingError.severity == severity)
        if resolved is not None:
            query = query.where(ProcessingError.resolved == ("true" if resolved else "false"))
        
        # If review_status is provided, filter by document review_status
        if review_status:
            doc_query = select(Document.id).where(Document.review_status == review_status)
            doc_result = await db.execute(doc_query)
            doc_ids = [str(doc_id) for doc_id in doc_result.scalars().all()]
            if doc_ids:
                from uuid import UUID
                doc_uuids = [UUID(doc_id) for doc_id in doc_ids]
                query = query.where(ProcessingError.document_id.in_(doc_uuids))
            else:
                # No documents match, return empty
                return {"total": 0, "errors": []}
        
        query = query.order_by(ProcessingError.created_at.desc()).offset(skip).limit(limit)
        
        result = await db.execute(query)
        errors = result.scalars().all()
        
        # Get total count
        count_query = select(ProcessingError)
        if document_id:
            try:
                doc_uuid = UUID(document_id)
                count_query = count_query.where(ProcessingError.document_id == doc_uuid)
            except ValueError:
                pass
        if error_type:
            count_query = count_query.where(ProcessingError.error_type == error_type)
        if severity:
            count_query = count_query.where(ProcessingError.severity == severity)
        if resolved is not None:
            count_query = count_query.where(ProcessingError.resolved == ("true" if resolved else "false"))
        
        count_result = await db.execute(select(func.count()).select_from(count_query.subquery()))
        total = count_result.scalar() or 0
        
        return {
            "total": total,
            "errors": [{
                "id": str(err.id),
                "document_id": str(err.document_id),
                "paragraph_id": err.paragraph_id,
                "error_type": err.error_type,
                "severity": err.severity,
                "error_message": err.error_message,
                "error_details": err.error_details,
                "stage": err.stage,
                "resolved": err.resolved == "true",
                "resolution": err.resolution,
                "resolved_by": err.resolved_by,
                "resolved_at": err.resolved_at.isoformat() if err.resolved_at else None,
                "resolution_notes": err.resolution_notes,
                "created_at": err.created_at.isoformat()
            } for err in errors]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing errors: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/errors/{error_id}/resolve")
async def resolve_error(
    error_id: str,
    resolution: dict,
    db: AsyncSession = Depends(get_db)
):
    """Resolve an error (approve, reject, or mark for reprocess)."""
    from uuid import UUID
    from sqlalchemy import func
    
    try:
        err_uuid = UUID(error_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid error ID")
    
    resolution_action = resolution.get("resolution")  # 'approved', 'rejected', 'reprocess'
    resolution_notes = resolution.get("notes", "")
    resolved_by = resolution.get("resolved_by", "system")
    
    if resolution_action not in ["approved", "rejected", "reprocess"]:
        raise HTTPException(status_code=400, detail="resolution must be 'approved', 'rejected', or 'reprocess'")
    
    try:
        result = await db.execute(select(ProcessingError).where(ProcessingError.id == err_uuid))
        error = result.scalar_one_or_none()
        
        if not error:
            raise HTTPException(status_code=404, detail="Error not found")
        
        error.resolved = "true"
        error.resolution = resolution_action
        error.resolved_by = resolved_by
        error.resolved_at = datetime.utcnow()
        error.resolution_notes = resolution_notes
        
        await db.commit()
        await db.refresh(error)
        
        # If reprocess, update document review_status
        if resolution_action == "reprocess":
            doc_result = await db.execute(select(Document).where(Document.id == error.document_id))
            doc = doc_result.scalar_one_or_none()
            if doc:
                doc.review_status = "reprocessing"
                await db.commit()
        
        return {
            "status": "ok",
            "error_id": error_id,
            "resolution": resolution_action
        }
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error resolving error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/{document_id}/errors/resolve-all")
async def resolve_all_document_errors(
    document_id: str,
    resolution: dict,
    db: AsyncSession = Depends(get_db)
):
    """Bulk resolve all unresolved errors for a document."""
    from uuid import UUID
    
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    
    resolution_action = resolution.get("resolution")
    resolution_notes = resolution.get("notes", "")
    resolved_by = resolution.get("resolved_by", "system")
    
    if resolution_action not in ["approved", "rejected", "reprocess"]:
        raise HTTPException(status_code=400, detail="resolution must be 'approved', 'rejected', or 'reprocess'")
    
    try:
        result = await db.execute(
            select(ProcessingError).where(
                ProcessingError.document_id == doc_uuid,
                ProcessingError.resolved == "false"
            )
        )
        errors = result.scalars().all()
        
        for err in errors:
            err.resolved = "true"
            err.resolution = resolution_action
            err.resolved_by = resolved_by
            err.resolved_at = datetime.utcnow()
            err.resolution_notes = resolution_notes
        
        await db.commit()
        
        # Update document review_status
        doc_result = await db.execute(select(Document).where(Document.id == doc_uuid))
        doc = doc_result.scalar_one_or_none()
        if doc:
            if resolution_action == "reprocess":
                doc.review_status = "reprocessing"
            elif resolution_action == "approved":
                doc.review_status = "approved"
            elif resolution_action == "rejected":
                doc.review_status = "rejected"
            await db.commit()
        
        return {
            "status": "ok",
            "document_id": document_id,
            "resolved_count": len(errors),
            "resolution": resolution_action
        }
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error bulk resolving errors: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/errors/stats")
async def get_error_stats(db: AsyncSession = Depends(get_db)):
    """Get error statistics (counts by type, severity, status)."""
    from sqlalchemy import func
    
    try:
        # Count by severity
        severity_counts = await db.execute(
            select(ProcessingError.severity, func.count(ProcessingError.id))
            .group_by(ProcessingError.severity)
        )
        severity_stats = {row[0]: row[1] for row in severity_counts.all()}
        
        # Count by error_type
        type_counts = await db.execute(
            select(ProcessingError.error_type, func.count(ProcessingError.id))
            .group_by(ProcessingError.error_type)
        )
        type_stats = {row[0]: row[1] for row in type_counts.all()}
        
        # Count by resolution status
        resolution_counts = await db.execute(
            select(ProcessingError.resolution, func.count(ProcessingError.id))
            .where(ProcessingError.resolved == "true")
            .group_by(ProcessingError.resolution)
        )
        resolution_stats = {row[0]: row[1] for row in resolution_counts.all()}
        
        # Total counts
        total_result = await db.execute(select(func.count(ProcessingError.id)))
        total = total_result.scalar() or 0
        
        unresolved_result = await db.execute(
            select(func.count(ProcessingError.id)).where(ProcessingError.resolved == "false")
        )
        unresolved = unresolved_result.scalar() or 0
        
        return {
            "total": total,
            "unresolved": unresolved,
            "resolved": total - unresolved,
            "by_severity": severity_stats,
            "by_type": type_stats,
            "by_resolution": resolution_stats
        }
    except Exception as e:
        logger.error(f"Error getting error stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/db/tables/{table_name}/records")
async def get_table_records(
    table_name: str,
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """Get records from a table with pagination."""
    from sqlalchemy import text, select
    
    try:
        # Validate table exists
        if table_name not in Base.metadata.tables:
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
        
        table = Base.metadata.tables[table_name]
        
        # Get total count
        count_result = await db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        total = count_result.scalar()
        
        # Get records
        query = select(table).offset(skip).limit(limit)
        result = await db.execute(query)
        rows = result.fetchall()
        
        # Convert to dict
        from app.models import CATEGORY_NAMES
        records = []
        for row in rows:
            record = {}
            for col in table.columns:
                value = getattr(row, col.name, None)
                # Convert UUID, datetime, etc. to strings
                if value is not None:
                    if hasattr(value, 'isoformat'):  # datetime
                        value = value.isoformat()
                    elif hasattr(value, '__str__'):
                        value = str(value)
                record[col.name] = value
            if table_name == "extracted_facts":
                # Use row (not record) so score/direction stay numeric in JSON; record stringifies all values
                record["category_scores"] = {
                    cat: {"score": getattr(row, f"{cat}_score", None), "direction": getattr(row, f"{cat}_direction", None)}
                    for cat in CATEGORY_NAMES
                }
            records.append(record)
        
        return {
            "table_name": table_name,
            "total": total,
            "skip": skip,
            "limit": limit,
            "records": records
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting records: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/db/tables/{table_name}/records/{record_id}")
async def get_table_record(
    table_name: str,
    record_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get a single record from a table by ID."""
    from sqlalchemy import text, select
    from uuid import UUID
    
    try:
        # Validate table exists
        if table_name not in Base.metadata.tables:
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
        
        table = Base.metadata.tables[table_name]
        
        # Find primary key column
        pk_col = None
        for col in table.columns:
            if col.primary_key:
                pk_col = col
                break
        
        if not pk_col:
            raise HTTPException(status_code=400, detail="Table has no primary key")
        
        # Convert record_id to appropriate type
        pk_value = record_id
        if isinstance(pk_col.type, UUID):
            try:
                pk_value = UUID(record_id)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid UUID format: {record_id}")
        
        # Get record
        query = select(table).where(pk_col == pk_value)
        result = await db.execute(query)
        row = result.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Record not found")
        
        # Convert to dict
        from app.models import CATEGORY_NAMES
        record = {}
        for col in table.columns:
            value = getattr(row, col.name, None)
            # Convert UUID, datetime, etc. to strings
            if value is not None:
                if hasattr(value, 'isoformat'):  # datetime
                    value = value.isoformat()
                elif hasattr(value, '__str__'):
                    value = str(value)
            record[col.name] = value
        if table_name == "extracted_facts":
            record["category_scores"] = {
                cat: {
                    "score": getattr(row, f"{cat}_score", None),
                    "direction": getattr(row, f"{cat}_direction", None),
                }
                for cat in CATEGORY_NAMES
            }
        
        return {
            "table_name": table_name,
            "record": record
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting record: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/db/tables/{table_name}/records")
async def create_record(
    table_name: str,
    record: dict,
    db: AsyncSession = Depends(get_db)
):
    """Create a new record in a table."""
    from sqlalchemy import text, inspect
    from uuid import UUID
    from datetime import datetime
    
    try:
        # Validate table exists
        if table_name not in Base.metadata.tables:
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
        
        table = Base.metadata.tables[table_name]
        
        # Prepare values - convert types appropriately
        values = {}
        for col_name, col in table.columns.items():
            if col_name in record:
                value = record[col_name]
                # Convert string UUIDs to UUID objects
                if isinstance(col.type, UUID) and isinstance(value, str):
                    try:
                        value = UUID(value)
                    except ValueError:
                        pass
                # Convert string datetimes
                elif hasattr(col.type, 'python_type') and col.type.python_type == datetime and isinstance(value, str):
                    try:
                        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except:
                        pass
                values[col_name] = value
        
        # Build INSERT statement
        columns = list(values.keys())
        placeholders = [f":{col}" for col in columns]
        insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)}) RETURNING *"
        
        result = await db.execute(text(insert_sql), values)
        await db.commit()
        
        # Get the inserted row
        row = result.fetchone()
        if row:
            inserted = {}
            for col in table.columns:
                value = getattr(row, col.name, None)
                if value is not None and hasattr(value, 'isoformat'):
                    value = value.isoformat()
                elif value is not None:
                    value = str(value)
                inserted[col.name] = value
            return {"status": "created", "record": inserted}
        
        return {"status": "created"}
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error creating record: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/admin/db/tables/{table_name}/records/{record_id}")
async def update_record(
    table_name: str,
    record_id: str,
    updates: dict,
    db: AsyncSession = Depends(get_db)
):
    """Update a record in a table."""
    from sqlalchemy import text
    from uuid import UUID
    from datetime import datetime
    
    try:
        # Validate table exists
        if table_name not in Base.metadata.tables:
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
        
        table = Base.metadata.tables[table_name]
        
        # Find primary key column
        pk_col = None
        for col in table.columns:
            if col.primary_key:
                pk_col = col
                break
        
        if not pk_col:
            raise HTTPException(status_code=400, detail="Table has no primary key")
        
        # Prepare update values
        set_clauses = []
        values = {}
        for col_name, col in table.columns.items():
            if col_name in updates and col_name != pk_col.name:
                value = updates[col_name]
                # Type conversions
                if isinstance(col.type, UUID) and isinstance(value, str):
                    try:
                        value = UUID(value)
                    except ValueError:
                        pass
                elif hasattr(col.type, 'python_type') and col.type.python_type == datetime and isinstance(value, str):
                    try:
                        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except:
                        pass
                set_clauses.append(f"{col_name} = :{col_name}")
                values[col_name] = value
        
        if not set_clauses:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        # Convert record_id to appropriate type
        pk_value = record_id
        if isinstance(pk_col.type, UUID):
            try:
                pk_value = UUID(record_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid record ID format")
        
        values[pk_col.name] = pk_value
        
        # Build UPDATE statement
        update_sql = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE {pk_col.name} = :{pk_col.name} RETURNING *"
        
        result = await db.execute(text(update_sql), values)
        await db.commit()
        
        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Record not found")
        
        updated = {}
        for col in table.columns:
            value = getattr(row, col.name, None)
            if value is not None and hasattr(value, 'isoformat'):
                value = value.isoformat()
            elif value is not None:
                value = str(value)
            updated[col.name] = value
        
        return {"status": "updated", "record": updated}
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error updating record: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/admin/db/tables/{table_name}/records/{record_id}")
async def delete_record(
    table_name: str,
    record_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a record from a table."""
    from sqlalchemy import text
    from uuid import UUID
    
    try:
        # Validate table exists
        if table_name not in Base.metadata.tables:
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
        
        table = Base.metadata.tables[table_name]
        
        # Find primary key column
        pk_col = None
        for col in table.columns:
            if col.primary_key:
                pk_col = col
                break
        
        if not pk_col:
            raise HTTPException(status_code=400, detail="Table has no primary key")
        
        # Convert record_id to appropriate type
        pk_value = record_id
        if isinstance(pk_col.type, UUID):
            try:
                pk_value = UUID(record_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid record ID format")
        
        # Build DELETE statement
        delete_sql = f"DELETE FROM {table_name} WHERE {pk_col.name} = :pk_value RETURNING *"
        
        result = await db.execute(text(delete_sql), {"pk_value": pk_value})
        await db.commit()
        
        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Record not found")
        
        deleted = {}
        for col in table.columns:
            value = getattr(row, col.name, None)
            if value is not None and hasattr(value, 'isoformat'):
                value = value.isoformat()
            elif value is not None:
                value = str(value)
            deleted[col.name] = value
        
        return {"status": "deleted", "record": deleted}
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error deleting record: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/db/execute")
async def execute_sql(
    request: dict,
    db: AsyncSession = Depends(get_db)
):
    """Execute raw SQL query (read-only for safety, or allow writes with caution)."""
    from sqlalchemy import text
    
    try:
        sql = request.get("sql", "")
        if not sql:
            raise HTTPException(status_code=400, detail="SQL query is required")
        
        # Basic safety check - prevent dangerous operations
        sql_upper = sql.strip().upper()
        dangerous_keywords = ['DROP', 'TRUNCATE', 'ALTER', 'CREATE', 'GRANT', 'REVOKE']
        
        # Allow SELECT, INSERT, UPDATE, DELETE
        if any(keyword in sql_upper for keyword in dangerous_keywords):
            raise HTTPException(
                status_code=400,
                detail="Dangerous SQL operations are not allowed. Use the CRUD endpoints instead."
            )
        
        result = await db.execute(text(sql))
        
        # For SELECT queries, return results
        if sql_upper.startswith('SELECT'):
            rows = result.fetchall()
            if rows:
                # Get column names
                columns = list(result.keys())
                records = []
                for row in rows:
                    record = {}
                    for i, col in enumerate(columns):
                        value = row[i]
                        if value is not None and hasattr(value, 'isoformat'):
                            value = value.isoformat()
                        elif value is not None:
                            value = str(value)
                        record[col] = value
                    records.append(record)
                return {"status": "success", "columns": columns, "records": records, "count": len(records)}
            return {"status": "success", "columns": [], "records": [], "count": 0}
        else:
            # For INSERT/UPDATE/DELETE, commit and return affected rows
            await db.commit()
            return {"status": "success", "affected_rows": result.rowcount}
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        error_msg = str(e)
        # Provide helpful error messages for foreign key violations
        if "foreign key constraint" in error_msg.lower() or "ForeignKeyViolationError" in error_msg:
            error_msg = f"Foreign key constraint violation: Cannot delete this record because it is referenced by other tables. Delete related records first.\n\nFor documents, use the 'Delete (Cascade)' button in the UI or the /admin/db/documents/{{id}}/delete-cascade endpoint.\n\nDetails: {error_msg}"
        logger.error(f"Error executing SQL: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/admin/db/documents/{document_id}/delete-cascade")
async def delete_document_cascade(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a document and all its related records in the correct order."""
    from uuid import UUID
    from sqlalchemy import text
    
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    
    # Verify document exists
    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Delete in correct order to respect foreign key constraints
        # 1. Delete extracted_facts (references both documents and hierarchical_chunks)
        await db.execute(
            text("DELETE FROM extracted_facts WHERE document_id = :doc_id"),
            {"doc_id": doc_uuid}
        )
        
        # 2. Delete hierarchical_chunks (references documents)
        await db.execute(
            text("DELETE FROM hierarchical_chunks WHERE document_id = :doc_id"),
            {"doc_id": doc_uuid}
        )
        
        # 3. Delete chunking_results (references documents)
        await db.execute(
            text("DELETE FROM chunking_results WHERE document_id = :doc_id"),
            {"doc_id": doc_uuid}
        )
        
        # 4. Delete document_pages (references documents)
        await db.execute(
            text("DELETE FROM document_pages WHERE document_id = :doc_id"),
            {"doc_id": doc_uuid}
        )
        
        # 5. Delete document (master table)
        await db.execute(
            text("DELETE FROM documents WHERE id = :doc_id"),
            {"doc_id": doc_uuid}
        )
        
        await db.commit()
        
        return {
            "status": "success",
            "message": f"Document '{document.filename}' and all related records deleted successfully",
            "document_id": document_id
        }
    except Exception as e:
        await db.rollback()
        logger.error(f"Error deleting document cascade: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
