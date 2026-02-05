import hashlib
import logging
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, UploadFile, HTTPException, Depends, Body, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func, text, bindparam, and_, or_
from google.cloud import storage
import json
from datetime import datetime
from collections import deque
from asyncio import Lock
import asyncio
from app.config import GCS_BUCKET, ENV, CRITIQUE_RETRY_THRESHOLD
from app.database import get_db, Base
from app.models import Document, DocumentPage, ChunkingResult, HierarchicalChunk, ExtractedFact, ProcessingError, ChunkingJob, ChunkingEvent, LlmConfig, EmbeddingJob, ChunkEmbedding, PublishEvent, RagPublishedEmbedding, fact_to_category_scores_dict
from app.services.error_tracker import log_error, classify_error
from app.services.extract_text import extract_text_from_gcs, html_to_plain_text
from app.services.chunking import split_paragraphs, split_paragraphs_from_markdown
from app.services.extraction import stream_extract_facts
from app.services.critique import stream_critique, critique_extraction, normalize_critique_result
from app.services.utils import parse_json_response, default_termination_date
from app.services.publish import publish_document, PublishResult

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
    """Schedule migrations in background so the server binds to PORT immediately (Cloud Run)."""
    asyncio.create_task(_run_startup_migrations_background())


async def _run_startup_migrations_background():
    """Run database migrations in background after server has bound to PORT."""
    from sqlalchemy import text
    from app.database import AsyncSessionLocal, engine

    # pgvector extension must exist before create_all (chunk_embeddings uses vector type)
    async with engine.begin() as conn:
        try:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception as ext_err:
            logger.warning("Could not enable pgvector extension: %s. Install pgvector: https://github.com/pgvector/pgvector", ext_err)

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
            try:
                from app.migrations.add_chunking_job_run_config import migrate as migrate_chunking_job_run_config
                await migrate_chunking_job_run_config()
            except Exception as migrate_err:
                logger.warning(f"Startup migration (chunking_job run_config) skipped: {migrate_err}")
            try:
                from app.migrations.add_extracted_facts_verification import migrate as migrate_facts_verification
                await migrate_facts_verification()
            except Exception as migrate_err:
                logger.warning(f"Startup migration (extracted_facts verification) skipped: {migrate_err}")
            try:
                from app.migrations.add_document_authority_level import migrate as migrate_document_authority_level
                await migrate_document_authority_level()
            except Exception as migrate_err:
                logger.warning(f"Startup migration (document authority_level) skipped: {migrate_err}")
            try:
                from app.migrations.add_document_effective_termination_dates import migrate as migrate_document_effective_termination_dates
                await migrate_document_effective_termination_dates()
            except Exception as migrate_err:
                logger.warning(f"Startup migration (document effective_date/termination_date) skipped: {migrate_err}")
            try:
                from app.migrations.add_document_display_name import migrate as migrate_document_display_name
                await migrate_document_display_name()
            except Exception as migrate_err:
                logger.warning(f"Startup migration (document display_name) skipped: {migrate_err}")
            try:
                from app.migrations.add_publish_tables import migrate as migrate_publish_tables
                await migrate_publish_tables()
            except Exception as migrate_err:
                logger.warning(f"Startup migration (publish tables) skipped: {migrate_err}")
            try:
                from app.migrations.add_publish_verification_columns import migrate as migrate_publish_verification
                await migrate_publish_verification()
            except Exception as migrate_err:
                logger.warning(f"Startup migration (publish verification columns) skipped: {migrate_err}")
            try:
                from app.migrations.add_document_source_metadata import migrate as migrate_document_source_metadata
                await migrate_document_source_metadata()
            except Exception as migrate_err:
                logger.warning(f"Startup migration (document source_metadata) skipped: {migrate_err}")
            try:
                from app.migrations.add_document_page_source_url import migrate as migrate_document_page_source_url
                await migrate_document_page_source_url()
            except Exception as migrate_err:
                logger.warning(f"Startup migration (document_pages source_url) skipped: {migrate_err}")
            try:
                from app.migrations.add_chunking_job_extraction_enabled import migrate as migrate_chunking_job_extraction_enabled
                await migrate_chunking_job_extraction_enabled()
            except Exception as migrate_err:
                logger.warning(f"Startup migration (chunking_job extraction_enabled) skipped: {migrate_err}")

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

# CORS - in dev allow any origin so Vite (any port) and localhost/127.0.0.1 all work
cors_origins = ["*"] if ENV == "dev" else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,  # must be False when allow_origins=["*"]
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


def _chunking_status_from_latest_event(event_type: str, event_data: dict) -> tuple[str | None, str | None]:
    """Derive chunking_status and processing_stage from latest chunking event. Returns (chunking_status, processing_stage).
    Returns (None, None) for embedding events so caller falls back to ChunkingJob/ChunkingResult."""
    if event_type and event_type.startswith("embedding_"):
        return (None, None)
    if event_type == "chunking_complete":
        return ("completed", None)
    if event_type in (
        "paragraph_start", "llm_stream", "extraction_complete",
        "critique_start", "critique_complete", "retry_start", "retry_extraction_complete",
        "paragraph_complete", "paragraph_persisted", "progress_update", "status_message", "paragraph_error", "error",
    ):
        stage = "idle"
        if event_type in ("paragraph_start", "llm_stream", "extraction_complete"):
            stage = "extracting"
        elif event_type in ("critique_start", "critique_complete"):
            stage = "critiquing"
        elif event_type in ("retry_start", "retry_extraction_complete"):
            stage = "retrying"
        elif event_type == "paragraph_persisted":
            stage = "persisting"
        return ("in_progress", stage)
    return ("in_progress", "idle")


@app.get("/documents")
async def list_documents(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """List all documents with extraction and chunking status. Chunking status is derived from the latest chunking_event when present to avoid fluctuation."""
    result = await db.execute(
        select(Document)
        .order_by(Document.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    documents = result.scalars().all()
    if not documents:
        return {"total": 0, "documents": []}

    # Latest chunking event per document (by created_at desc, id desc) – single source of truth for status
    doc_ids = [doc.id for doc in documents]
    if doc_ids:
        latest_events_result = await db.execute(
            text("""
                SELECT DISTINCT ON (document_id) document_id, event_type, event_data
                FROM chunking_events
                WHERE document_id IN :doc_ids
                ORDER BY document_id, created_at DESC, id DESC
            """).bindparams(bindparam("doc_ids", expanding=True)),
            {"doc_ids": doc_ids},
        )
        latest_events = {str(row.document_id): {"event_type": row.event_type, "event_data": row.event_data or {}} for row in latest_events_result}
    else:
        latest_events = {}

    # Latest embedding job per document – for embedding_status
    if doc_ids:
        latest_emb_result = await db.execute(
            text("""
                SELECT DISTINCT ON (document_id) document_id, status
                FROM embedding_jobs
                WHERE document_id IN :doc_ids
                ORDER BY document_id, created_at DESC, id DESC
            """).bindparams(bindparam("doc_ids", expanding=True)),
            {"doc_ids": doc_ids},
        )
        latest_embedding = {str(row.document_id): row.status for row in latest_emb_result}
    else:
        latest_embedding = {}

    # Latest publish_event per document – for publish_status (incl. verification)
    if doc_ids:
        latest_pub_result = await db.execute(
            text("""
                SELECT DISTINCT ON (document_id) document_id, published_at, rows_written,
                       verification_passed, verification_message
                FROM publish_events
                WHERE document_id IN :doc_ids
                ORDER BY document_id, published_at DESC, id DESC
            """).bindparams(bindparam("doc_ids", expanding=True)),
            {"doc_ids": doc_ids},
        )
        latest_publish = {}
        for row in latest_pub_result:
            latest_publish[str(row.document_id)] = {
                "published_at": row.published_at,
                "rows_written": row.rows_written,
                "verification_passed": getattr(row, "verification_passed", None),
                "verification_message": getattr(row, "verification_message", None),
            }
        # Publish count per document (1 = Published, 2+ = Republished)
        count_pub_result = await db.execute(
            text("""
                SELECT document_id, COUNT(*) AS publish_count
                FROM publish_events
                WHERE document_id IN :doc_ids
                GROUP BY document_id
            """).bindparams(bindparam("doc_ids", expanding=True)),
            {"doc_ids": doc_ids},
        )
        publish_count_by_doc = {str(row.document_id): (row.publish_count or 0) for row in count_pub_result}
    else:
        latest_publish = {}
        publish_count_by_doc = {}

    document_list = []
    for doc in documents:
        doc_id_str = str(doc.id)
        latest = latest_events.get(doc_id_str)
        chunking_meta = None  # ChunkingResult.metadata_ when used in fallback (for progress)

        if latest:
            chunking_status, processing_stage = _chunking_status_from_latest_event(
                latest["event_type"], latest["event_data"]
            )
        else:
            chunking_status = None
            processing_stage = None

        if chunking_status is None:
            # No events: fall back to ChunkingJob + ChunkingResult
            job_result = await db.execute(
                select(ChunkingJob).where(
                    ChunkingJob.document_id == doc.id,
                    ChunkingJob.status.in_(["pending", "processing"])
                ).order_by(ChunkingJob.created_at.desc())
            )
            active_job = job_result.scalar_one_or_none()
            if active_job:
                chunking_status = "queued" if active_job.status == "pending" else "in_progress"
                job_id = str(active_job.id)
                job_critique_enabled = active_job.critique_enabled is None or str(active_job.critique_enabled).lower() == "true"
                job_max_retries = active_job.max_retries if active_job.max_retries is not None else 2
            else:
                chunking_result = await db.execute(
                    select(ChunkingResult).where(ChunkingResult.document_id == doc.id)
                )
                chunking = chunking_result.scalar_one_or_none()
                if chunking and chunking.metadata_:
                    chunking_meta = chunking.metadata_
                    chunking_status = chunking_meta.get("status", "idle")
                else:
                    chunking_status = "idle"
                job_id = None
                job_critique_enabled = None
                job_max_retries = None
        else:
            job_result = await db.execute(
                select(ChunkingJob).where(
                    ChunkingJob.document_id == doc.id,
                    ChunkingJob.status.in_(["pending", "processing"])
                ).order_by(ChunkingJob.created_at.desc())
            )
            active_job = job_result.scalar_one_or_none()
            job_id = str(active_job.id) if active_job else None
            job_critique_enabled = (active_job.critique_enabled is None or str(active_job.critique_enabled).lower() == "true") if active_job else None
            job_max_retries = (active_job.max_retries if active_job.max_retries is not None else 2) if active_job else None

        # Prefer job/result when they say "completed" (e.g. chunking_complete event was never written)
        if chunking_status in ("in_progress", "queued"):
            latest_job_result = await db.execute(
                select(ChunkingJob).where(ChunkingJob.document_id == doc.id).order_by(ChunkingJob.created_at.desc()).limit(1)
            )
            latest_job = latest_job_result.scalar_one_or_none()
            if latest_job and latest_job.status == "completed":
                chunking_status = "completed"
            else:
                cr_check = await db.execute(select(ChunkingResult).where(ChunkingResult.document_id == doc.id))
                cr_row = cr_check.scalar_one_or_none()
                if cr_row and (cr_row.metadata_ or {}).get("status") == "completed":
                    chunking_status = "completed"

        emb_status = latest_embedding.get(doc_id_str)
        embedding_status = emb_status if emb_status else "idle"

        doc_item = {
            "id": doc_id_str,
            "filename": doc.filename,
            "display_name": getattr(doc, "display_name", None),
            "extraction_status": doc.status,
            "chunking_status": chunking_status,
            "embedding_status": embedding_status,
            "created_at": doc.created_at.isoformat(),
            "gcs_path": doc.file_path,
            "has_errors": doc.has_errors or "false",
            "error_count": doc.error_count or 0,
            "critical_error_count": doc.critical_error_count or 0,
            "review_status": doc.review_status or "pending",
            "payer": doc.payer,
            "state": doc.state,
            "program": doc.program,
            "authority_level": getattr(doc, "authority_level", None),
            "effective_date": getattr(doc, "effective_date", None),
            "termination_date": getattr(doc, "termination_date", None),
        }
        if processing_stage is not None:
            doc_item["chunking_processing_stage"] = processing_stage
        if job_id is not None:
            doc_item["chunking_job_id"] = job_id
            doc_item["critique_enabled"] = job_critique_enabled
            doc_item["max_retries"] = job_max_retries
        # Progress for Live Updates: from latest event event_data or ChunkingResult metadata (no per-doc fetch needed)
        if latest and isinstance(latest.get("event_data"), dict):
            ed = latest["event_data"]
            if ed.get("total_paragraphs") is not None:
                doc_item["chunking_total_paragraphs"] = ed["total_paragraphs"]
            if ed.get("completed_paragraphs") is not None:
                doc_item["chunking_completed_paragraphs"] = ed["completed_paragraphs"]
            if ed.get("current_paragraph") is not None:
                doc_item["chunking_current_paragraph"] = ed["current_paragraph"]
            elif ed.get("paragraph_id") is not None:
                doc_item["chunking_current_paragraph"] = ed["paragraph_id"]
            if ed.get("page_number") is not None:
                doc_item["chunking_current_page"] = ed["page_number"]
            if ed.get("total_pages") is not None:
                doc_item["chunking_total_pages"] = ed["total_pages"]
        elif chunking_meta:
            if chunking_meta.get("total_paragraphs") is not None:
                doc_item["chunking_total_paragraphs"] = chunking_meta["total_paragraphs"]
            if chunking_meta.get("completed_count") is not None:
                doc_item["chunking_completed_paragraphs"] = chunking_meta["completed_count"]
        # Publish status for list (Store / Chunk / Embed / Publish / Errors)
        pub = latest_publish.get(doc_id_str)
        publish_count = publish_count_by_doc.get(doc_id_str, 0)
        if pub:
            doc_item["published_at"] = pub["published_at"].isoformat()
            doc_item["published_rows"] = pub["rows_written"]
            doc_item["publish_verification_passed"] = pub.get("verification_passed")
            doc_item["publish_verification_message"] = pub.get("verification_message")
            doc_item["publish_count"] = publish_count
        else:
            doc_item["published_at"] = None
            doc_item["published_rows"] = None
            doc_item["publish_verification_passed"] = None
            doc_item["publish_verification_message"] = None
            doc_item["publish_count"] = 0
        document_list.append(doc_item)

    return {
        "total": len(document_list),
        "documents": document_list
    }


@app.get("/facts")
async def list_facts(
    db: AsyncSession = Depends(get_db),
    document_id: Optional[List[str]] = Query(None, description="Filter by document ID(s)"),
    page_number: Optional[int] = Query(None, description="Filter by page number"),
    section_path: Optional[str] = Query(None, description="Filter by section (contains)"),
    search: Optional[str] = Query(None, description="Search in fact text"),
    payer: Optional[List[str]] = Query(None, description="Filter by payer"),
    state: Optional[List[str]] = Query(None, description="Filter by state"),
    program: Optional[List[str]] = Query(None, description="Filter by program"),
    fact_type: Optional[List[str]] = Query(None, description="Filter by fact type"),
    is_pertinent: Optional[str] = Query(None, description="Filter by pertinent: all, yes, no"),
    is_eligibility: Optional[str] = Query(None, description="Filter by eligibility: all, yes, no"),
    verification_status: Optional[str] = Query(None, description="Filter by status: all, pending, approved, rejected"),
    category_min_scores: Optional[str] = Query(None, description="JSON object of category -> min score, e.g. {\"prior_authorization_required\":0.5}"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    sort: str = Query("created_at", description="Sort by: created_at, document, page"),
):
    """List extracted facts with server-side filtering and pagination."""
    from uuid import UUID
    from app.models import CATEGORY_NAMES

    # Use raw SQL for flexibility with joins and filters
    conditions = ["1=1"]
    params = {"skip": skip, "limit": limit}

    # Document filter
    if document_id and len(document_id) > 0:
        try:
            uuids = [str(UUID(d)) for d in document_id]
            placeholders = ", ".join([f":doc_id_{i}" for i in range(len(uuids))])
            conditions.append(f"ef.document_id IN ({placeholders})")
            for i, u in enumerate(uuids):
                params[f"doc_id_{i}"] = u
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid document_id")

    # Page number filter
    if page_number is not None:
        conditions.append("(ef.page_number = :page_number OR (ef.page_number IS NULL AND hc.page_number = :page_number))")
        params["page_number"] = page_number

    # Section filter (from hierarchical_chunks)
    if section_path and section_path.strip():
        conditions.append("hc.section_path ILIKE :section_pattern")
        params["section_pattern"] = f"%{section_path.strip()}%"

    # Search in fact text
    if search and search.strip():
        conditions.append("ef.fact_text ILIKE :search_pattern")
        params["search_pattern"] = f"%{search.strip()}%"

    # Payer, state, program (from documents)
    def _in_clause(param_name: str, values: list, col: str, params_dict: dict, conditions_list: list) -> None:
        if not values:
            return
        placeholders = ", ".join([f":{param_name}_{i}" for i in range(len(values))])
        conditions_list.append(f"{col} IN ({placeholders})")
        for i, v in enumerate(values):
            params_dict[f"{param_name}_{i}"] = v

    if payer and len(payer) > 0:
        _in_clause("payer", payer, "d.payer", params, conditions)
    if state and len(state) > 0:
        _in_clause("state", state, "d.state", params, conditions)
    if program and len(program) > 0:
        _in_clause("program", program, "d.program", params, conditions)
    if fact_type and len(fact_type) > 0:
        _in_clause("fact_type", fact_type, "ef.fact_type", params, conditions)

    # is_pertinent filter
    if is_pertinent and is_pertinent not in ("all", ""):
        if is_pertinent == "yes":
            conditions.append("ef.is_pertinent_to_claims_or_members = 'true'")
        elif is_pertinent == "no":
            conditions.append("(ef.is_pertinent_to_claims_or_members IS NULL OR ef.is_pertinent_to_claims_or_members != 'true')")

    # is_eligibility filter
    if is_eligibility and is_eligibility not in ("all", ""):
        if is_eligibility == "yes":
            conditions.append("ef.is_eligibility_related = 'true'")
        elif is_eligibility == "no":
            conditions.append("(ef.is_eligibility_related IS NULL OR ef.is_eligibility_related != 'true')")

    # verification_status filter
    if verification_status and verification_status not in ("all", ""):
        if verification_status == "pending":
            conditions.append("(ef.verification_status IS NULL OR ef.verification_status = 'pending')")
        else:
            conditions.append("ef.verification_status = :ver_status")
            params["ver_status"] = verification_status

    # Category min scores (JSON: {"category_name": 0.5, ...})
    if category_min_scores and category_min_scores.strip():
        try:
            cat_mins = json.loads(category_min_scores)
            if isinstance(cat_mins, dict):
                for cat, min_val in cat_mins.items():
                    if cat in CATEGORY_NAMES and isinstance(min_val, (int, float)) and min_val > 0:
                        col = f"{cat}_score"
                        if hasattr(ExtractedFact, col):
                            conditions.append(f"ef.{col} >= :min_{cat}")
                            params[f"min_{cat}"] = float(min_val)
        except json.JSONDecodeError:
            pass

    where_clause = " AND ".join(conditions)

    # Sort
    sort_col = "ef.created_at"
    if sort == "document":
        sort_col = "d.display_name, d.filename, ef.created_at"
    elif sort == "page":
        sort_col = "COALESCE(ef.page_number, hc.page_number), ef.created_at"

    # Count query
    count_sql = text(f"""
        SELECT COUNT(*) as cnt
        FROM extracted_facts ef
        JOIN hierarchical_chunks hc ON ef.hierarchical_chunk_id = hc.id
        JOIN documents d ON ef.document_id = d.id
        WHERE {where_clause}
    """)
    count_result = await db.execute(count_sql, params)
    total = count_result.scalar() or 0

    # Data query with limit/offset (include category score columns)
    fact_cols = [f"{c}_score" for c in CATEGORY_NAMES] + [f"{c}_direction" for c in CATEGORY_NAMES]
    cols_str = ", ".join([f"ef.{c}" for c in fact_cols])
    data_sql2 = text(f"""
        SELECT ef.id, ef.fact_text, ef.fact_type, ef.who_eligible, ef.how_verified, ef.conflict_resolution,
               ef.when_applies, ef.limitations, ef.is_pertinent_to_claims_or_members, ef.is_eligibility_related,
               ef.is_verified, ef.confidence, ef.document_id, ef.page_number, ef.verification_status, ef.verified_by, ef.verified_at, ef.created_at,
               hc.section_path, hc.page_number as chunk_page,
               d.filename, d.display_name, d.payer, d.state, d.program, d.effective_date, d.termination_date,
               {cols_str}
        FROM extracted_facts ef
        JOIN hierarchical_chunks hc ON ef.hierarchical_chunk_id = hc.id
        JOIN documents d ON ef.document_id = d.id
        WHERE {where_clause}
        ORDER BY {sort_col}
        OFFSET :skip LIMIT :limit
    """)
    data_result2 = await db.execute(data_sql2, params)
    rows2 = data_result2.fetchall()
    col_keys = [c for c in data_result2.keys()]

    records = []
    for row in rows2:
        r = dict(zip(col_keys, row))
        category_scores = {}
        for c in CATEGORY_NAMES:
            score = r.get(f"{c}_score")
            direction = r.get(f"{c}_direction")
            category_scores[c] = {"score": score, "direction": direction}
        page_num = r.get("page_number")
        if page_num is None:
            page_num = r.get("chunk_page")
        records.append({
            "id": str(r["id"]),
            "fact_text": r["fact_text"],
            "fact_type": r["fact_type"],
            "who_eligible": r["who_eligible"],
            "how_verified": r["how_verified"],
            "conflict_resolution": r["conflict_resolution"],
            "when_applies": r["when_applies"],
            "limitations": r["limitations"],
            "is_pertinent_to_claims_or_members": r["is_pertinent_to_claims_or_members"],
            "is_eligibility_related": r["is_eligibility_related"],
            "is_verified": r["is_verified"],
            "confidence": r["confidence"],
            "document_id": str(r["document_id"]),
            "document_filename": r["filename"],
            "document_display_name": r["display_name"],
            "payer": r["payer"],
            "state": r["state"],
            "program": r["program"],
            "effective_date": r["effective_date"],
            "termination_date": r["termination_date"],
            "page_number": page_num,
            "section_path": r.get("section_path"),
            "verification_status": r.get("verification_status"),
            "verified_by": r.get("verified_by"),
            "verified_at": r["verified_at"].isoformat() if r.get("verified_at") else None,
            "category_scores": category_scores,
        })

    # Get documents list for filter dropdowns
    docs_result = await db.execute(
        select(Document).order_by(Document.display_name, Document.filename)
    )
    all_docs = docs_result.scalars().all()
    documents = [
        {
            "id": str(d.id),
            "filename": d.filename,
            "display_name": d.display_name,
            "payer": d.payer,
            "state": d.state,
            "program": d.program,
            "effective_date": d.effective_date,
            "termination_date": d.termination_date,
        }
        for d in all_docs
    ]

    # Get distinct filter options (payers, states, programs, fact_types) from filtered facts
    filter_options = {"payers": [], "states": [], "programs": [], "fact_types": []}
    if total > 0:
        opts_sql = text(f"""
            SELECT DISTINCT d.payer, d.state, d.program, ef.fact_type
            FROM extracted_facts ef
            JOIN hierarchical_chunks hc ON ef.hierarchical_chunk_id = hc.id
            JOIN documents d ON ef.document_id = d.id
            WHERE {where_clause}
        """)
        opts_result = await db.execute(opts_sql, params)
        opts_rows = opts_result.fetchall()
        payers = sorted({r[0] for r in opts_rows if r[0]})
        states = sorted({r[1] for r in opts_rows if r[1]})
        programs = sorted({r[2] for r in opts_rows if r[2]})
        fact_types = sorted({r[3] for r in opts_rows if r[3]})
        filter_options = {"payers": payers, "states": states, "programs": programs, "fact_types": fact_types}

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "records": records,
        "documents": documents,
        "filter_options": filter_options,
    }


@app.get("/facts/sections")
async def list_fact_sections(
    db: AsyncSession = Depends(get_db),
    document_id: Optional[List[str]] = Query(None, description="Filter by document ID(s)"),
):
    """List distinct section_path values for filter dropdown."""
    from uuid import UUID

    if not document_id or len(document_id) == 0:
        # All sections across all documents
        result = await db.execute(
            text("""
                SELECT DISTINCT hc.section_path
                FROM hierarchical_chunks hc
                WHERE hc.section_path IS NOT NULL AND hc.section_path != ''
                ORDER BY hc.section_path
            """)
        )
    else:
        try:
            uuids = [str(UUID(d)) for d in document_id]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid document_id")
        placeholders = ", ".join([f":doc_{i}" for i in range(len(uuids))])
        result = await db.execute(
            text(f"""
                SELECT DISTINCT hc.section_path
                FROM hierarchical_chunks hc
                WHERE hc.document_id IN ({placeholders})
                  AND hc.section_path IS NOT NULL AND hc.section_path != ''
                ORDER BY hc.section_path
            """),
            {f"doc_{i}": u for i, u in enumerate(uuids)}
        )
    rows = result.fetchall()
    sections = [r[0] for r in rows]
    return {"sections": sections}


@app.patch("/documents/{document_id}")
async def update_document_metadata(
    document_id: str,
    body: dict = Body(...),
    db: AsyncSession = Depends(get_db),
):
    """Update document metadata (display_name, payer, state, program, authority_level, effective_date, termination_date, status)."""
    from uuid import UUID
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    for key in ("display_name", "payer", "state", "program", "authority_level", "effective_date", "termination_date"):
        if key in body:
            val = body[key]
            if val is None or (isinstance(val, str) and val.strip() == ""):
                setattr(doc, key, None)
            else:
                setattr(doc, key, val if isinstance(val, str) else str(val))
    # Allow setting status to "completed" for uploaded docs (e.g. scraped pages) so chunking can start
    if body.get("status") == "completed" and doc.status == "uploaded":
        pages_result = await db.execute(select(DocumentPage).where(DocumentPage.document_id == doc_uuid))
        if pages_result.scalars().first() is not None:
            doc.status = "completed"
        # else: leave status as uploaded if no pages
    await db.commit()
    await db.refresh(doc)
    # Build same shape as list_documents: derive chunking_status from latest chunking_event when present
    latest_ev = await db.execute(
        select(ChunkingEvent).where(ChunkingEvent.document_id == doc.id)
        .order_by(ChunkingEvent.created_at.desc(), ChunkingEvent.id.desc())
        .limit(1)
    )
    latest_event = latest_ev.scalar_one_or_none()
    if latest_event:
        chunking_status, processing_stage = _chunking_status_from_latest_event(
            latest_event.event_type, latest_event.event_data or {}
        )
    else:
        processing_stage = None
        chunking_status = None
    # Fallback when no events or when latest event yields None (e.g. embedding_complete)
    if chunking_status is None:
        job_result = await db.execute(
            select(ChunkingJob).where(
                ChunkingJob.document_id == doc.id,
                ChunkingJob.status.in_(["pending", "processing"])
            ).order_by(ChunkingJob.created_at.desc())
        )
        active_job = job_result.scalar_one_or_none()
        if active_job:
            chunking_status = "queued" if active_job.status == "pending" else "in_progress"
        else:
            cr_result = await db.execute(select(ChunkingResult).where(ChunkingResult.document_id == doc.id))
            chunking_row = cr_result.scalar_one_or_none()
            chunking_status = chunking_row.metadata_.get("status", "idle") if (chunking_row and chunking_row.metadata_) else "idle"
    # Prefer job/result when they say "completed" (same as list_documents / get_document_detail)
    if chunking_status in ("in_progress", "queued"):
        latest_job_res = await db.execute(
            select(ChunkingJob).where(ChunkingJob.document_id == doc.id).order_by(ChunkingJob.created_at.desc()).limit(1)
        )
        latest_job = latest_job_res.scalar_one_or_none()
        if latest_job and latest_job.status == "completed":
            chunking_status = "completed"
        else:
            cr_check = await db.execute(select(ChunkingResult).where(ChunkingResult.document_id == doc.id))
            cr_row = cr_check.scalar_one_or_none()
            if cr_row and (cr_row.metadata_ or {}).get("status") == "completed":
                chunking_status = "completed"
    # embedding_status from latest EmbeddingJob
    emb_job = await db.execute(
        select(EmbeddingJob).where(EmbeddingJob.document_id == doc.id).order_by(EmbeddingJob.created_at.desc()).limit(1)
    )
    emb_row = emb_job.scalar_one_or_none()
    embedding_status = emb_row.status if emb_row else "idle"
    out = {
        "id": str(doc.id),
        "filename": doc.filename,
        "display_name": getattr(doc, "display_name", None),
        "extraction_status": doc.status,
        "chunking_status": chunking_status,
        "embedding_status": embedding_status,
        "created_at": doc.created_at.isoformat(),
        "gcs_path": doc.file_path,
        "has_errors": doc.has_errors or "false",
        "error_count": doc.error_count or 0,
        "critical_error_count": doc.critical_error_count or 0,
        "review_status": doc.review_status or "pending",
        "payer": doc.payer,
        "state": doc.state,
        "program": doc.program,
        "authority_level": getattr(doc, "authority_level", None),
        "effective_date": getattr(doc, "effective_date", None),
        "termination_date": getattr(doc, "termination_date", None),
    }
    if processing_stage is not None:
        out["chunking_processing_stage"] = processing_stage
    return out


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
            await db.execute(
                text("DELETE FROM chunking_events WHERE document_id = :doc_id"),
                {"doc_id": doc_uuid}
            )
            await db.execute(
                text("DELETE FROM chunking_jobs WHERE document_id = :doc_id"),
                {"doc_id": doc_uuid}
            )
            await db.execute(
                text("DELETE FROM processing_errors WHERE document_id = :doc_id"),
                {"doc_id": doc_uuid}
            )
            await db.execute(
                text("DELETE FROM extracted_facts WHERE document_id = :doc_id"),
                {"doc_id": doc_uuid}
            )
            await db.execute(
                text("DELETE FROM hierarchical_chunks WHERE document_id = :doc_id"),
                {"doc_id": doc_uuid}
            )
            await db.execute(
                text("DELETE FROM chunking_results WHERE document_id = :doc_id"),
                {"doc_id": doc_uuid}
            )
            await db.execute(
                text("DELETE FROM document_pages WHERE document_id = :doc_id"),
                {"doc_id": doc_uuid}
            )
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
        # Delete related records first (foreign key constraints), in dependency order
        await db.execute(
            delete(ChunkingEvent).where(ChunkingEvent.document_id == doc_uuid)
        )
        await db.execute(
            delete(ChunkingJob).where(ChunkingJob.document_id == doc_uuid)
        )
        await db.execute(
            delete(ProcessingError).where(ProcessingError.document_id == doc_uuid)
        )
        await db.execute(
            delete(ExtractedFact).where(ExtractedFact.document_id == doc_uuid)
        )
        await db.execute(
            delete(HierarchicalChunk).where(HierarchicalChunk.document_id == doc_uuid)
        )
        await db.execute(
            delete(ChunkingResult).where(ChunkingResult.document_id == doc_uuid)
        )
        await db.execute(
            delete(DocumentPage).where(DocumentPage.document_id == doc_uuid)
        )
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
            termination_date=default_termination_date(),
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


class ImportFromGcsRequest(BaseModel):
    """Request to import a document from GCS (e.g. from web scraper)."""
    gcs_path: str
    filename: Optional[str] = None


@app.post("/documents/import-from-gcs")
async def import_document_from_gcs(
    body: ImportFromGcsRequest = Body(...),
    payer: str = None,
    state: str = None,
    program: str = None,
    db: AsyncSession = Depends(get_db)
):
    """Import a document from GCS path (e.g. from web scraper). PDF only for extraction."""
    gcs_path = body.gcs_path.strip()
    if not gcs_path.startswith("gs://"):
        raise HTTPException(status_code=400, detail="gcs_path must start with gs://")
    filename = body.filename or gcs_path.split("/")[-1]
    if not filename:
        raise HTTPException(status_code=400, detail="Could not derive filename from gcs_path")

    try:
        # Compute hash from GCS object
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        prefix = f"gs://{GCS_BUCKET}/"
        blob_path = gcs_path[len(prefix):].lstrip("/") if gcs_path.startswith(prefix) else gcs_path.split("/")[-1]
        blob = bucket.blob(blob_path)
        if not blob.exists():
            raise HTTPException(status_code=404, detail=f"Object not found in GCS: {gcs_path}")
        content = blob.download_as_bytes()
        file_hash = hashlib.sha256(content).hexdigest()

        result = await db.execute(select(Document).where(Document.file_hash == file_hash))
        existing_doc = result.scalar_one_or_none()
        if existing_doc:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "duplicate_file",
                    "message": "This file has already been imported.",
                    "original_filename": existing_doc.filename,
                    "document_id": str(existing_doc.id),
                }
            )

        document = Document(
            filename=filename,
            file_hash=file_hash,
            file_path=gcs_path,
            payer=payer,
            state=state,
            program=program,
            termination_date=default_termination_date(),
            status="uploaded",
        )
        db.add(document)
        await db.commit()
        await db.refresh(document)

        try:
            document.status = "extracting"
            await db.commit()
            from app.services.page_to_markdown import raw_page_to_markdown
            pages = await extract_text_from_gcs(gcs_path)
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
            document.status = "completed"
            await db.commit()
        except Exception as e:
            document.status = "failed"
            await db.commit()
            logger.warning("Extraction failed for import-from-gcs %s: %s", gcs_path, e)

        return {
            "filename": filename,
            "gcs_path": gcs_path,
            "document_id": str(document.id),
            "status": document.status,
        }
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Import from GCS error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _scraped_url_to_display_name(url: str) -> str:
    """Derive a short display name from a scraped page URL: drop scheme and trailing .html / index.html."""
    if not url or not url.strip():
        return "scraped"
    s = url.strip()
    for prefix in ("https://", "http://"):
        if s.lower().startswith(prefix):
            s = s[len(prefix) :].lstrip("/")
            break
    s = s.rstrip("/")
    for suffix in ("/index.html", "/index.htm", ".html", ".htm"):
        if s.lower().endswith(suffix):
            s = s[: -len(suffix)].rstrip("/")
            break
    if len(s) > 80:
        s = s[:77] + "..."
    return s or "scraped"


class ScrapedPageItem(BaseModel):
    """Single page from scraper: url and optional text/html."""
    url: str
    text: Optional[str] = None
    html: Optional[str] = None


class ImportScrapedPagesRequest(BaseModel):
    """Request to import scraped pages as one RAG document."""
    pages: List[ScrapedPageItem]
    display_name: Optional[str] = None
    authority_level: Optional[str] = None
    effective_date: Optional[str] = None
    termination_date: Optional[str] = None
    payer: Optional[str] = None
    state: Optional[str] = None
    program: Optional[str] = None


@app.post("/documents/import-scraped-pages")
async def import_scraped_pages(
    body: ImportScrapedPagesRequest = Body(...),
    db: AsyncSession = Depends(get_db),
):
    """Import scraped HTML/text pages as one RAG document. Dedup by content identity; 409 if duplicate."""
    if not body.pages:
        raise HTTPException(status_code=400, detail="pages must not be empty")

    from app.services.page_to_markdown import raw_page_to_markdown

    # Build text per page: use text if present else derive from html
    page_texts: List[tuple[str, str]] = []  # (url, plain_text)
    for p in body.pages:
        if p.text and p.text.strip():
            text = p.text.strip()
        elif p.html and p.html.strip():
            text = html_to_plain_text(p.html)
        else:
            text = ""
        page_texts.append((p.url, text))

    # Reject if no page has content so we don't create empty docs that won't chunk
    if not any(t for _, t in page_texts):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "no_content",
                "message": "No text or HTML content was provided for any of the selected pages. "
                "Ensure the scraper captured page content (e.g. wait for scrape to finish and use 'Add to RAG' again, or use content mode Text or Both).",
            },
        )

    # Synthetic identity for dedup: hash of first URL + sorted page URLs
    urls = [p.url for p in body.pages]
    first_url = urls[0] if urls else ""
    identity_str = first_url + "\n" + "\n".join(sorted(urls))
    file_hash = hashlib.sha256(identity_str.encode()).hexdigest()

    result = await db.execute(select(Document).where(Document.file_hash == file_hash))
    existing_doc = result.scalar_one_or_none()
    if existing_doc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "duplicate_scraped",
                "message": "A document with the same scraped pages already exists.",
                "document_id": str(existing_doc.id),
            },
        )

    file_path = ("scraped:" + first_url)[:500]
    # Auto-derive display name: drop https:// and trailing .html / index.html when user didn't set one
    _derived_display = _scraped_url_to_display_name(first_url)
    seed_display = body.display_name or _derived_display
    filename = seed_display.replace("/", "_").replace("?", "_")[:255] or "scraped"

    source_metadata = {
        "source_type": "scraped",
        "scraped_seed_url": first_url,
        "scraped_page_count": len(body.pages),
        "scraped_page_urls": urls,
    }
    term_date = body.termination_date if body.termination_date else default_termination_date()

    # Scraped pages already have text/text_markdown per page — no separate extraction job. Use "completed" so chunking can start.
    document = Document(
        filename=filename,
        file_hash=file_hash,
        file_path=file_path,
        display_name=body.display_name or _derived_display,
        payer=body.payer,
        state=body.state,
        program=body.program,
        authority_level=body.authority_level,
        effective_date=body.effective_date,
        termination_date=term_date,
        source_metadata=source_metadata,
        status="completed",
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    for i, (url, text) in enumerate(page_texts):
        text_markdown = raw_page_to_markdown(text) if text else ""
        extraction_status = "success" if text else "empty"
        page = DocumentPage(
            document_id=document.id,
            page_number=i + 1,
            text=text or None,
            text_markdown=text_markdown or None,
            extraction_status=extraction_status,
            text_length=len(text),
            source_url=url,
        )
        db.add(page)

    await db.commit()

    return {
        "document_id": str(document.id),
        "filename": filename,
        "pages_count": len(body.pages),
        "status": "completed",
    }


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
    critique_enabled: bool = True,
    max_retries: int = 2,
    extraction_enabled: bool = True,
):
    """
    Shared chunking loop logic. Runs the full paragraph processing loop.
    - If event_callback is provided, it should be an async generator that yields SSE event strings
    - Otherwise, just processes and persists results (for background task)
    - Checks _chunking_cancel at the start of each paragraph iteration
    - critique_enabled: if False, skip critique and retries (extraction only)
    - max_retries: when critique enabled, max extraction retries on critique fail
    - extraction_enabled: if False, only build hierarchical chunks (no LLM extraction/critique)
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
                    
                    # When extraction_enabled is False: only create hierarchical chunk, skip LLM
                    if not extraction_enabled:
                        para_start = para_data.get("start_offset") if isinstance(para_data, dict) else None
                        chunk_query = await db.execute(
                            select(HierarchicalChunk).where(
                                HierarchicalChunk.document_id == doc_uuid,
                                HierarchicalChunk.page_number == page.page_number,
                                HierarchicalChunk.paragraph_index == para_idx
                            )
                        )
                        chunk = chunk_query.scalar_one_or_none()
                        if not chunk:
                            chunk = HierarchicalChunk(
                                document_id=doc_uuid,
                                page_number=page.page_number,
                                paragraph_index=para_idx,
                                section_path=section_path,
                                text=paragraph_text,
                                text_length=len(paragraph_text),
                                start_offset_in_page=para_start,
                                extraction_status="skipped",
                                critique_status="skipped",
                            )
                            db.add(chunk)
                            await db.flush()
                        results_paragraphs[para_id] = {
                            "paragraph_id": para_id,
                            "final_status": "skipped",
                            "facts": [],
                        }
                        await _upsert("in_progress")
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
                            except Exception as progress_err:
                                logger.error(f"Error sending progress_update event: {progress_err}", exc_info=True)
                        continue
                    
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
                            if raw_extraction_output and "{" in raw_extraction_output:
                                recovered = True
                        except Exception:
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
                    
                    # Stage 3: Critique agent (skip if critique_enabled is False)
                    critique = {"pass": True, "score": 1.0, "category_assessment": {}, "feedback": None, "issues": []}
                    if critique_enabled:
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
                    
                    # Stage 4: Retry if needed (score < threshold) - only when critique enabled
                    if critique_enabled:
                        retry_count = 0
                        current_extraction = extraction
                        current_critique = critique
                    else:
                        current_extraction = extraction
                        current_critique = {"pass": True, "score": 1.0, "category_assessment": {}, "feedback": None, "issues": []}
                        retry_count = 0
                    consecutive_errors = 0
                    max_consecutive_errors = 3  # Break retry loop if too many errors
                    
                    critique_score = _critique_score(current_critique)
                    # logger.debug(f"[{para_id}] Initial critique score: {critique_score}, threshold: {threshold}, needs retry: {critique_score < threshold}")  # Reduced logging
                    
                    while critique_enabled and _critique_score(current_critique) < threshold and retry_count < max_retries and consecutive_errors < max_consecutive_errors:
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
                "verified_by": getattr(fact, "verified_by", None),
                "verified_at": fact.verified_at.isoformat() if getattr(fact, "verified_at", None) else None,
                "verification_status": getattr(fact, "verification_status", None),
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
        "document_metadata": {
            "display_name": getattr(document, "display_name", None),
            "payer": document.payer,
            "state": document.state,
            "program": document.program,
            "authority_level": getattr(document, "authority_level", None),
            "effective_date": getattr(document, "effective_date", None),
            "termination_date": getattr(document, "termination_date", None),
        },
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
    # Verification (facts sheet: approve, reject, delete)
    if "verification_status" in body:
        status = body.get("verification_status")
        if status in ("pending", "approved", "rejected", "deleted"):
            fact.verification_status = status
            if status == "approved" and "verified_by" not in body:
                fact.verified_by = "human"
                fact.verified_at = datetime.utcnow()
            elif status in ("rejected", "deleted") and body.get("verified_by"):
                fact.verified_by = body.get("verified_by")
                fact.verified_at = datetime.utcnow()
    if "verified_by" in body and body.get("verified_by") in ("ai", "human"):
        fact.verified_by = body["verified_by"]
    if "verified_at" in body:
        # Client can send ISO string; we leave as None and set above when approving
        pass
    await db.commit()
    await db.refresh(fact)
    out = {
        "id": str(fact.id),
        "fact_text": fact.fact_text,
        "page_number": fact.page_number,
        "start_offset": fact.start_offset,
        "end_offset": fact.end_offset,
        "category_scores": fact_to_category_scores_dict(fact),
    }
    if hasattr(fact, "verified_by"):
        out["verified_by"] = fact.verified_by
        out["verified_at"] = fact.verified_at.isoformat() if fact.verified_at else None
        out["verification_status"] = fact.verification_status
    return out


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


@app.get("/config/prompts")
async def list_prompt_versions():
    """List available prompt names and their versions (for UI / run-configured refs). Default set: extraction v1, extraction_retry v1, critique v1."""
    from app.services.prompt_registry import list_names, list_versions
    names = list_names() or ["extraction", "extraction_retry", "critique"]
    result = {}
    for name in names:
        versions = list_versions(name)
        result[name] = versions or []
    return {
        "prompts": result,
        "names": names,
        "default": {"extraction": "v1", "extraction_retry": "v1", "critique": "v1"},
    }


def _safe_version(version: str) -> bool:
    """Allow alphanumeric, hyphen, underscore for prompt version (no path traversal)."""
    if not version or len(version) > 80:
        return False
    return all(c.isalnum() or c in "-_" for c in version)


@app.get("/config/prompts/{name}/{version}")
async def get_prompt_version(name: str, version: str):
    """Get one prompt by name and version (body, variables, description)."""
    if not _safe_config_name(name) or not _safe_version(version):
        raise HTTPException(status_code=400, detail="Invalid name or version")
    from app.services.prompt_registry import get_prompt_with_meta
    data = get_prompt_with_meta(name, version)
    if data is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return data


class PromptCreateUpdateBody(BaseModel):
    body: str = ""
    description: str = ""
    variables: Optional[list[str]] = None


@app.post("/config/prompts/{name}/{version}")
async def create_prompt_version(name: str, version: str, body: PromptCreateUpdateBody = Body(...)):
    """Create a new prompt version (file)."""
    if not _safe_config_name(name) or not _safe_version(version):
        raise HTTPException(status_code=400, detail="Invalid name or version")
    from app.services.prompt_registry import save_prompt, list_versions, create_prompt_name
    existing = list_versions(name)
    if version in existing:
        raise HTTPException(status_code=409, detail="Version already exists; use PUT to update")
    create_prompt_name(name)
    ok = save_prompt(
        name, version,
        body.body,
        body.description or "",
        body.variables,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save prompt")
    return {"name": name, "version": version, "status": "created"}


@app.put("/config/prompts/{name}/{version}")
async def update_prompt_version(name: str, version: str, body: PromptCreateUpdateBody = Body(...)):
    """Update an existing prompt version."""
    if not _safe_config_name(name) or not _safe_version(version):
        raise HTTPException(status_code=400, detail="Invalid name or version")
    from app.services.prompt_registry import save_prompt, list_versions
    if version not in list_versions(name):
        raise HTTPException(status_code=404, detail="Prompt not found")
    ok = save_prompt(
        name, version,
        body.body,
        body.description or "",
        body.variables,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save prompt")
    return {"name": name, "version": version, "status": "updated"}


@app.delete("/config/prompts/{name}/{version}")
async def delete_prompt_version(name: str, version: str):
    """Delete a prompt version (file)."""
    if not _safe_config_name(name) or not _safe_version(version):
        raise HTTPException(status_code=400, detail="Invalid name or version")
    from app.services.prompt_registry import delete_prompt
    ok = delete_prompt(name, version)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to delete prompt")
    return {"name": name, "version": version, "status": "deleted"}


@app.post("/config/prompts/names")
async def create_prompt_name_api(body: dict = Body(...)):
    """Create a new prompt name (folder). Expects JSON: {\"name\": \"my_prompt\"}."""
    prompt_name = (body.get("name") or "").strip()
    if not prompt_name or not _safe_config_name(prompt_name):
        raise HTTPException(status_code=400, detail="Invalid or missing name")
    from app.services.prompt_registry import create_prompt_name
    ok = create_prompt_name(prompt_name)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to create prompt name")
    return {"name": prompt_name, "status": "created"}


@app.get("/config/llm")
async def list_llm_configs():
    """List available LLM config names (for UI / run-configured refs). Default: default."""
    from pathlib import Path
    configs_dir = Path(__file__).resolve().parent / "llm_configs"
    names = []
    if configs_dir.is_dir():
        for p in configs_dir.iterdir():
            if p.suffix == ".yaml" and p.stem:
                names.append(p.stem)
    return {
        "configs": sorted(names) if names else ["default"],
        "default": "default",
    }


def _safe_config_name(name: str) -> bool:
    """Allow only alphanumeric, hyphen, underscore (no path traversal)."""
    if not name or len(name) > 100:
        return False
    return all(c.isalnum() or c in "-_" for c in name)


_SECRET_KEYS = frozenset({"api_key", "apikey", "secret"})


def _sanitize_secrets_for_storage(obj: dict) -> dict:
    """Replace real secret values with placeholder so we never persist secrets to PostgreSQL."""
    out = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            out[k] = _sanitize_secrets_for_storage(v)
        elif k in _SECRET_KEYS and isinstance(v, str) and v.strip() and v.strip() != "***":
            out[k] = "***"
        else:
            out[k] = v
    return out


def _redact_secrets(obj: dict) -> dict:
    """Return a copy with secret-like values replaced by a placeholder."""
    out = {}
    for k, v in obj.items():
        key_lower = k.lower().replace("-", "").replace("_", "")
        if key_lower in ("apikey", "apisecret", "secret") and v:
            out[k] = "***"
        elif isinstance(v, dict):
            out[k] = _redact_secrets(v)
        else:
            out[k] = v
    return out


@app.get("/config/llm/providers")
async def list_llm_providers():
    """List registered LLM provider names (ollama, vertex, openai, etc.)."""
    from app.services.llm_provider import list_providers
    return {"providers": list_providers()}


@app.get("/config/llm/{version}")
async def get_llm_config_version(version: str, db: AsyncSession = Depends(get_db)):
    """Get one LLM config by version/name (DB first, then YAML). Secrets are redacted for display."""
    if not _safe_config_name(version):
        raise HTTPException(status_code=400, detail="Invalid config name")
    from app.services.llm_config import get_llm_config_resolved
    cfg = await get_llm_config_resolved(version, db)
    if cfg is None:
        raise HTTPException(status_code=404, detail="Config not found")
    return _redact_secrets(cfg)


@app.post("/config/llm/{version}/test")
async def test_llm_config(version: str, db: AsyncSession = Depends(get_db)):
    """Build the LLM provider for this config and run a short generate to verify credentials and connectivity."""
    if not _safe_config_name(version):
        raise HTTPException(status_code=400, detail="Invalid config name")
    from app.services.llm_config import get_llm_config_resolved, get_llm_provider_from_config
    cfg = await get_llm_config_resolved(version, db)
    if cfg is None:
        raise HTTPException(status_code=404, detail="Config not found")
    try:
        llm = get_llm_provider_from_config(cfg)
        reply = await llm.generate("Say hi in one word.", temperature=0)
        reply_text = (reply or "").strip()[:200]
        return {"ok": True, "message": reply_text or "Connected.", "reply": reply_text}
    except ImportError as e:
        err = str(e).strip()
        if "vertex" in err.lower() or "aiplatform" in err.lower() or "vertexai" in err.lower():
            return {
                "ok": False,
                "error": "Vertex AI SDK not installed. Run: pip install -e \".[vertex]\" (with venv active), then restart the backend.",
            }
        return {"ok": False, "error": err}
    except ValueError as e:
        err = str(e).strip()
        if "project_id" in err.lower() or "vertex" in err.lower():
            return {
                "ok": False,
                "error": f"{err} Set VERTEX_PROJECT_ID in .env and restart the backend.",
            }
        return {"ok": False, "error": err}
    except Exception as e:
        logger.exception("LLM config test failed for %s", version)
        err = str(e).strip()
        if "api has not been used" in err.lower() or "is disabled" in err.lower() or "service_disabled" in err.lower() or "activationurl" in err.lower():
            return {
                "ok": False,
                "error": "Vertex AI API is not enabled for this project. Enable it: https://console.cloud.google.com/apis/library/aiplatform.googleapis.com (select your project). Wait a few minutes after enabling, then try again.",
            }
        if "not found" in err.lower() and ("model" in err.lower() or "publisher" in err.lower()):
            return {
                "ok": False,
                "error": "Model not found. Use a valid Vertex AI model ID, e.g. gemini-1.5-pro, gemini-1.5-flash, or gemini-1.0-pro. Change the Model field in this config and try again.",
            }
        if "credentials" in err.lower() or "auth" in err.lower():
            return {
                "ok": False,
                "error": f"{err} Ensure .env has GOOGLE_APPLICATION_CREDENTIALS (full path to JSON key) and restart the backend.",
            }
        return {"ok": False, "error": err}


class LLMConfigUpdate(BaseModel):
    """Body for updating an LLM config (partial or full)."""
    provider: Optional[str] = None
    model: Optional[str] = None
    version: Optional[str] = None
    options: Optional[dict] = None
    ollama: Optional[dict] = None
    vertex: Optional[dict] = None
    openai: Optional[dict] = None


@app.put("/config/llm/{version}")
async def update_llm_config(version: str, body: LLMConfigUpdate, db: AsyncSession = Depends(get_db)):
    """Update an LLM config (saved to DB). Preserves existing secret values if new value is '***' or missing."""
    if not _safe_config_name(version):
        raise HTTPException(status_code=400, detail="Invalid config name")
    from app.services.llm_config import get_llm_config_resolved, save_llm_config

    existing = await get_llm_config_resolved(version, db) or {}
    merged = dict(existing)

    if body.provider is not None:
        merged["provider"] = body.provider
    if body.model is not None:
        merged["model"] = body.model
    if body.version is not None:
        merged["version"] = body.version
    if body.options is not None:
        merged["options"] = {**(merged.get("options") or {}), **body.options}
    for key in ("ollama", "vertex", "openai"):
        block = getattr(body, key, None)
        if block is None:
            continue
        current = merged.get(key) or {}
        updated = dict(current)
        for k, v in block.items():
            if v is None:
                continue
            if k in _SECRET_KEYS and (v == "***" or v == ""):
                if k in current and current[k]:
                    updated[k] = current[k]
                else:
                    updated[k] = v
            else:
                updated[k] = v
        merged[key] = updated

    # Never store real secrets in PostgreSQL; resolve at runtime from env or secret manager
    merged = _sanitize_secrets_for_storage(merged)
    saved = await save_llm_config(db, version, merged)
    await db.commit()
    return _redact_secrets(saved)


class ChunkingStartBody(BaseModel):
    """Optional body for chunking start: run mode and run-configured refs."""
    threshold: Optional[float] = None
    critique_enabled: Optional[bool] = None
    max_retries: Optional[int] = None
    extraction_enabled: Optional[bool] = None
    prompt_versions: Optional[dict] = None
    llm_config_version: Optional[str] = None


@app.post("/documents/{document_id}/chunking/start")
async def start_chunking(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    threshold: float | None = None,
    critique_enabled: bool | None = None,
    max_retries: int | None = None,
    extraction_enabled: bool | None = None,
    llm_config_version: str | None = None,
    body: ChunkingStartBody | None = Body(None),
):
    """Queue a chunking job. Returns immediately. Worker process will pick it up. Optional body can include prompt_versions, llm_config_version, run mode."""
    from uuid import UUID

    # Merge body with query params (query takes precedence)
    prompt_versions = None
    if body:
        if threshold is None and body.threshold is not None:
            threshold = body.threshold
        if critique_enabled is None and body.critique_enabled is not None:
            critique_enabled = body.critique_enabled
        if max_retries is None and body.max_retries is not None:
            max_retries = body.max_retries
        if extraction_enabled is None and body.extraction_enabled is not None:
            extraction_enabled = body.extraction_enabled
        if llm_config_version is None and body.llm_config_version is not None:
            llm_config_version = body.llm_config_version
        prompt_versions = body.prompt_versions

    th = threshold if threshold is not None else CRITIQUE_RETRY_THRESHOLD
    th = max(0.0, min(1.0, th))

    # Run mode defaults: critique on, max_retries 2, extraction on
    ce = critique_enabled if critique_enabled is not None else True
    mr = max_retries if max_retries is not None else 2
    mr = max(0, mr)
    ee = extraction_enabled if extraction_enabled is not None else True

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

    # Create new job (run-configured: prompt_versions, llm_config_version optional)
    job = ChunkingJob(
        document_id=doc_uuid,
        status="pending",
        threshold=str(th),
        critique_enabled="true" if ce else "false",
        max_retries=mr,
        extraction_enabled="true" if ee else "false",
        prompt_versions=prompt_versions,
        llm_config_version=llm_config_version,
    )
    db.add(job)
    await db.commit()

    logger.info(f"Queued chunking job {job.id} for document {document_id} with threshold {th}, critique_enabled={ce}, max_retries={mr}")

    out = {
        "status": "queued",
        "document_id": document_id,
        "job_id": str(job.id),
        "critique_enabled": ce,
        "max_retries": mr,
    }
    if prompt_versions is not None:
        out["prompt_versions"] = prompt_versions
    if llm_config_version is not None:
        out["llm_config_version"] = llm_config_version
    return out


class ChunkingRestartBody(BaseModel):
    """Optional body for chunking restart: run mode."""
    threshold: Optional[float] = None
    critique_enabled: Optional[bool] = None
    max_retries: Optional[int] = None
    extraction_enabled: Optional[bool] = None


class ChunkingStatusUpdateBody(BaseModel):
    """Body for PATCH chunking status (e.g. mark stuck in_progress as completed)."""
    status: str  # "completed", "idle", "in_progress"


@app.post("/documents/{document_id}/chunking/restart")
async def restart_chunking(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    threshold: float | None = None,
    body: ChunkingRestartBody | None = Body(None),
):
    """Restart chunking for a document that failed or was stopped."""
    from uuid import UUID
    import asyncio
    
    th = threshold if threshold is not None else CRITIQUE_RETRY_THRESHOLD
    if body and body.threshold is not None:
        th = body.threshold
    th = max(0.0, min(1.0, th))
    ce = body.critique_enabled if body and body.critique_enabled is not None else True
    mr = body.max_retries if body and body.max_retries is not None else 2
    mr = max(0, mr)
    ee = body.extraction_enabled if body and body.extraction_enabled is not None else True
    
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
            logger.info(f"Restarting background chunking task for document {document_id} with threshold {th}, critique_enabled={ce}, extraction_enabled={ee}, max_retries={mr}")
            db_session = AsyncSessionLocal()
            try:
                event_callback = await _create_event_buffer_callback(document_id, db_session)
                async for _ in _run_chunking_loop(document_id, doc_uuid, pages, th, db_session, event_callback=event_callback, critique_enabled=ce, extraction_enabled=ee, max_retries=mr):
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


@app.patch("/documents/{document_id}/chunking/status")
async def update_chunking_status(
    document_id: str,
    body: ChunkingStatusUpdateBody,
    db: AsyncSession = Depends(get_db),
):
    """Set chunking status for a document (e.g. mark stuck in_progress as completed so publish is allowed)."""
    from uuid import UUID

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    status = (body.status or "").strip().lower()
    if status not in ("completed", "idle", "in_progress"):
        raise HTTPException(status_code=400, detail="status must be one of: completed, idle, in_progress")

    doc_result = await db.execute(select(Document).where(Document.id == doc_uuid))
    if not doc_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Document not found")

    cr_result = await db.execute(select(ChunkingResult).where(ChunkingResult.document_id == doc_uuid))
    chunking_result = cr_result.scalar_one_or_none()
    if not chunking_result:
        chunking_result = ChunkingResult(document_id=doc_uuid, metadata_={"status": status}, results={})
        db.add(chunking_result)
    else:
        meta = chunking_result.metadata_ or {}
        meta["status"] = status
        chunking_result.metadata_ = meta
        chunking_result.updated_at = datetime.utcnow()

    if status == "completed":
        latest_job_result = await db.execute(
            select(ChunkingJob).where(ChunkingJob.document_id == doc_uuid).order_by(ChunkingJob.created_at.desc()).limit(1)
        )
        latest_job = latest_job_result.scalar_one_or_none()
        if latest_job and latest_job.status == "processing":
            latest_job.status = "completed"
            latest_job.completed_at = datetime.utcnow()

    await db.commit()
    return {"document_id": document_id, "chunking_status": status}


@app.post("/documents/{document_id}/embedding/start")
async def start_embedding(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Queue an embedding job for a document. Embedding worker will pick it up. Use after chunking completes or to re-embed."""
    from uuid import UUID

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Document not found")

    existing = await db.execute(
        select(EmbeddingJob).where(
            EmbeddingJob.document_id == doc_uuid,
            EmbeddingJob.status == "pending",
        ).limit(1)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Embedding job already queued for this document")

    job = EmbeddingJob(document_id=doc_uuid, status="pending")
    db.add(job)
    await db.commit()

    return {"status": "queued", "document_id": document_id, "job_id": str(job.id)}


@app.post("/documents/{document_id}/embedding/reset")
async def reset_embedding(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Reset stuck embedding jobs (pending/processing). Clears partial embeddings and sets job to pending so worker can retry. Use when worker was killed mid-run."""
    from uuid import UUID
    from app.services.vector_store import get_vector_store

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Document not found")

    jobs_result = await db.execute(
        select(EmbeddingJob).where(
            EmbeddingJob.document_id == doc_uuid,
            EmbeddingJob.status.in_(["pending", "processing"]),
        ).order_by(EmbeddingJob.created_at.desc())
    )
    jobs = jobs_result.scalars().all()
    if not jobs:
        raise HTTPException(status_code=404, detail="No pending or processing embedding job found for this document")

    await db.execute(delete(ChunkEmbedding).where(ChunkEmbedding.document_id == doc_uuid))
    vector_store = get_vector_store()
    vector_store.delete_by_document(document_id)

    for job in jobs:
        job.status = "pending"
        job.worker_id = None
        job.started_at = None
        job.completed_at = None
        job.error_message = None

    await db.commit()
    return {"status": "reset", "document_id": document_id, "jobs_reset": len(jobs)}


class PublishBody(BaseModel):
    """Optional body for POST /documents/{id}/publish (audit)."""
    published_by: Optional[str] = None


@app.post("/documents/{document_id}/publish")
async def publish_document_endpoint(
    document_id: str,
    body: Optional[PublishBody] = Body(None),
    db: AsyncSession = Depends(get_db),
):
    """Publish entire document to rag_published_embeddings (dbt contract). Writes all embeddings for this document; replaces any existing published rows for this document."""
    from uuid import UUID

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    try:
        result = await publish_document(doc_uuid, db)
    except ValueError as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        if "no chunk embeddings" in msg.lower():
            raise HTTPException(status_code=400, detail=msg)
        raise HTTPException(status_code=400, detail=msg)

    assert isinstance(result, PublishResult)
    event = PublishEvent(
        document_id=doc_uuid,
        published_by=(body.published_by if body else None) or None,
        rows_written=result.rows_written,
        verification_passed=result.verification_passed,
        verification_message=result.verification_message,
    )
    db.add(event)
    await db.commit()

    return {
        "status": "ok",
        "document_id": document_id,
        "rows_written": result.rows_written,
        "verification_passed": result.verification_passed,
        "verification_message": result.verification_message,
    }


@app.get("/documents/{document_id}/publish-status")
async def get_publish_status(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return whether this document has been published (latest publish_events row) and when."""
    from uuid import UUID

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    result = await db.execute(
        select(PublishEvent)
        .where(PublishEvent.document_id == doc_uuid)
        .order_by(PublishEvent.published_at.desc())
        .limit(1)
    )
    event = result.scalar_one_or_none()
    if not event:
        return {"published": False, "document_id": document_id}
    return {
        "published": True,
        "document_id": document_id,
        "published_at": event.published_at.isoformat(),
        "published_by": event.published_by,
        "rows_written": event.rows_written,
        "verification_passed": getattr(event, "verification_passed", None),
        "verification_message": getattr(event, "verification_message", None),
    }


@app.get("/documents/{document_id}/detail")
async def get_document_detail(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Aggregated detail for document detail / publish-readiness page: metadata, errors, fact counts, chunking/embedding status, last publish."""
    from uuid import UUID

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    doc_result = await db.execute(select(Document).where(Document.id == doc_uuid))
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Chunking status: latest event or fallback to ChunkingJob/ChunkingResult
    latest_ev = await db.execute(
        text("""
            SELECT event_type, event_data FROM chunking_events
            WHERE document_id = :doc_id
            ORDER BY created_at DESC, id DESC LIMIT 1
        """),
        {"doc_id": doc_uuid},
    )
    row = latest_ev.first()
    if row:
        chunking_status, _ = _chunking_status_from_latest_event(row[0], row[1] or {})
    else:
        chunking_status = None
    # Fallback when no events or when latest event yields None (e.g. embedding_complete)
    if chunking_status is None:
        job_result = await db.execute(
            select(ChunkingJob).where(ChunkingJob.document_id == doc_uuid).order_by(ChunkingJob.created_at.desc()).limit(1)
        )
        job = job_result.scalar_one_or_none()
        if job and job.status in ("pending", "processing"):
            chunking_status = "in_progress" if job.status == "processing" else "pending"
        else:
            cr_result = await db.execute(select(ChunkingResult).where(ChunkingResult.document_id == doc_uuid))
            cr = cr_result.scalar_one_or_none()
            chunking_status = (cr.metadata_ or {}).get("status", "idle") if cr and cr.metadata_ else "idle"

    # Prefer job/result when they say "completed" (same as list_documents)
    if chunking_status in ("in_progress", "queued"):
        latest_job_result = await db.execute(
            select(ChunkingJob).where(ChunkingJob.document_id == doc_uuid).order_by(ChunkingJob.created_at.desc()).limit(1)
        )
        latest_job = latest_job_result.scalar_one_or_none()
        if latest_job and latest_job.status == "completed":
            chunking_status = "completed"
        else:
            cr_check = await db.execute(select(ChunkingResult).where(ChunkingResult.document_id == doc_uuid))
            cr_row = cr_check.scalar_one_or_none()
            if cr_row and (cr_row.metadata_ or {}).get("status") == "completed":
                chunking_status = "completed"

    # Embedding status: latest EmbeddingJob
    emb_result = await db.execute(
        select(EmbeddingJob).where(EmbeddingJob.document_id == doc_uuid).order_by(EmbeddingJob.created_at.desc()).limit(1)
    )
    emb_job = emb_result.scalar_one_or_none()
    embedding_status = emb_job.status if emb_job else "idle"

    # Error counts
    err_total = await db.execute(select(func.count(ProcessingError.id)).where(ProcessingError.document_id == doc_uuid))
    err_critical = await db.execute(
        select(func.count(ProcessingError.id)).where(
            ProcessingError.document_id == doc_uuid,
            ProcessingError.severity == "critical",
        )
    )
    err_unresolved = await db.execute(
        select(func.count(ProcessingError.id)).where(
            ProcessingError.document_id == doc_uuid,
            ProcessingError.resolved == "false",
        )
    )
    errors_total = err_total.scalar() or 0
    errors_critical = err_critical.scalar() or 0
    errors_unresolved = err_unresolved.scalar() or 0

    # Fact counts by verification_status
    fact_total = await db.execute(select(func.count(ExtractedFact.id)).where(ExtractedFact.document_id == doc_uuid))
    fact_approved = await db.execute(
        select(func.count(ExtractedFact.id)).where(
            ExtractedFact.document_id == doc_uuid,
            ExtractedFact.verification_status == "approved",
        )
    )
    fact_pending = await db.execute(
        select(func.count(ExtractedFact.id)).where(
            ExtractedFact.document_id == doc_uuid,
            or_(ExtractedFact.verification_status.is_(None), ExtractedFact.verification_status == "pending"),
        )
    )
    fact_rejected = await db.execute(
        select(func.count(ExtractedFact.id)).where(
            ExtractedFact.document_id == doc_uuid,
            ExtractedFact.verification_status == "rejected",
        )
    )
    facts_total = fact_total.scalar() or 0
    facts_approved = fact_approved.scalar() or 0
    facts_pending = fact_pending.scalar() or 0
    facts_rejected = fact_rejected.scalar() or 0

    # Last publish
    pub_result = await db.execute(
        select(PublishEvent).where(PublishEvent.document_id == doc_uuid).order_by(PublishEvent.published_at.desc()).limit(1)
    )
    pub_ev = pub_result.scalar_one_or_none()

    return {
        "document": {
            "id": str(doc.id),
            "filename": doc.filename,
            "display_name": doc.display_name or "",
            "payer": doc.payer or "",
            "state": doc.state or "",
            "program": doc.program or "",
            "authority_level": getattr(doc, "authority_level", None) or "",
            "effective_date": getattr(doc, "effective_date", None) or "",
            "termination_date": getattr(doc, "termination_date", None) or "",
            "status": doc.status,
            "review_status": doc.review_status,
            "created_at": doc.created_at.isoformat(),
            "has_errors": doc.has_errors or "false",
            "error_count": doc.error_count or 0,
            "critical_error_count": doc.critical_error_count or 0,
        },
        "chunking_status": chunking_status,
        "embedding_status": embedding_status,
        "errors": {"total": errors_total, "critical": errors_critical, "unresolved": errors_unresolved},
        "facts": {"total": facts_total, "approved": facts_approved, "pending": facts_pending, "rejected": facts_rejected},
        "last_publish": {
            "published_at": pub_ev.published_at.isoformat(),
            "published_by": pub_ev.published_by,
            "rows_written": pub_ev.rows_written,
            "verification_passed": getattr(pub_ev, "verification_passed", None),
            "verification_message": getattr(pub_ev, "verification_message", None),
        } if pub_ev else None,
        "readiness": {
            "chunking_done": chunking_status in ("idle", "completed"),
            "embedding_done": embedding_status == "completed",
            "no_critical_errors": (doc.critical_error_count or 0) == 0,
            "ready": (
                chunking_status in ("idle", "completed")
                and embedding_status == "completed"
                and (doc.critical_error_count or 0) == 0
            ),
        },
    }


class QueryRequest(BaseModel):
    """Request body for semantic query (embed + search + resolve)."""
    query: str
    k: int = 10


class ChunkOut(BaseModel):
    """One retrieved chunk with text and citation info."""
    text: str
    source_type: str  # 'hierarchical' | 'fact'
    source_id: str
    document_id: str
    document_name: Optional[str] = None
    page_number: Optional[int] = None


class QueryResponse(BaseModel):
    """Response: top-k chunks with text for RAG context."""
    chunks: List[ChunkOut]


@app.post("/api/query", response_model=QueryResponse)
async def query_rag(
    body: QueryRequest = Body(...),
    db: AsyncSession = Depends(get_db),
):
    """Embed query, search vector store (top k), resolve source_id to text. Returns chunks for RAG context (no fact/chunk differentiation)."""
    from uuid import UUID
    from app.services.embedding_provider import get_embedding_provider, embed_async
    from app.services.vector_store import get_vector_store

    if not (body.query and body.query.strip()):
        raise HTTPException(status_code=400, detail="query is required")
    k = max(1, min(100, body.k))

    # 1. Embed query
    provider = get_embedding_provider()
    query_embeddings = await embed_async([body.query.strip()], provider)
    if not query_embeddings:
        return QueryResponse(chunks=[])
    query_embedding = query_embeddings[0]

    # 2. Search vector store (Chroma returns id + metadata: document_id, source_type, source_id)
    vector_store = get_vector_store()
    results = vector_store.search(query_embedding, k=k)
    if not results:
        return QueryResponse(chunks=[])

    # 3. Resolve each result to text (hierarchical chunk or fact)
    chunks_out: List[ChunkOut] = []
    for r in results:
        doc_id = r.get("document_id")
        source_type = r.get("source_type") or "hierarchical"
        source_id_raw = r.get("source_id")
        if not source_id_raw:
            continue
        try:
            source_uuid = UUID(str(source_id_raw))
        except ValueError:
            continue
        text = None
        page_number = None
        document_name = None
        if source_type == "hierarchical":
            chunk_result = await db.execute(select(HierarchicalChunk).where(HierarchicalChunk.id == source_uuid))
            chunk = chunk_result.scalar_one_or_none()
            if chunk:
                text = chunk.text
                page_number = chunk.page_number
        elif source_type == "fact":
            fact_result = await db.execute(select(ExtractedFact).where(ExtractedFact.id == source_uuid))
            fact = fact_result.scalar_one_or_none()
            if fact:
                text = fact.fact_text
                page_number = getattr(fact, "page_number", None)
                if page_number is None and fact.hierarchical_chunk_id:
                    ch_result = await db.execute(select(HierarchicalChunk).where(HierarchicalChunk.id == fact.hierarchical_chunk_id))
                    ch = ch_result.scalar_one_or_none()
                    if ch:
                        page_number = ch.page_number
        if not text:
            continue
        if doc_id:
            doc_result = await db.execute(select(Document).where(Document.id == UUID(str(doc_id))))
            doc = doc_result.scalar_one_or_none()
            if doc:
                document_name = doc.display_name or doc.filename
        chunks_out.append(ChunkOut(
            text=text,
            source_type=source_type,
            source_id=str(source_uuid),
            document_id=str(doc_id) if doc_id else "",
            document_name=document_name,
            page_number=page_number,
        ))
    return QueryResponse(chunks=chunks_out)


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


@app.get("/documents/{document_id}/chunking/events")
async def get_chunking_events(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    limit: int = 1000,
    after_id: Optional[str] = None,
):
    """Get chunking events for a document. Worker writes events to PostgreSQL; frontend polls this endpoint.
    - No after_id: returns the most recent `limit` events (desc order). Frontend should reverse for display.
    - after_id: returns events after that id (asc order), for polling new events only."""
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
    
    if after_id:
        try:
            after_uuid = UUID(after_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid after_id")
        # Polling: return only new events after this id, in chronological order (asc)
        query = (
            select(ChunkingEvent)
            .where(
                ChunkingEvent.document_id == doc_uuid,
                ChunkingEvent.id > after_uuid,
            )
            .order_by(ChunkingEvent.created_at, ChunkingEvent.id)
            .limit(min(limit, 500))
        )
    else:
        # Initial load: return most recent events first (desc) so client gets latest activity
        query = (
            select(ChunkingEvent)
            .where(ChunkingEvent.document_id == doc_uuid)
            .order_by(ChunkingEvent.created_at.desc(), ChunkingEvent.id.desc())
            .limit(limit)
        )
    
    result = await db.execute(query)
    events = result.scalars().all()
    
    formatted_events = []
    for event in events:
        formatted_events.append({
            "event": event.event_type,
            "data": event.event_data,
            "timestamp": event.created_at.isoformat(),
            "id": str(event.id),
        })
    
    return {
        "document_id": document_id,
        "events": formatted_events,
        "count": len(formatted_events),
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
    from sqlalchemy.dialects.postgresql import UUID as PG_UUID
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
        
        # Convert record_id to appropriate type (PK may be PostgreSQL UUID)
        pk_value = record_id
        if isinstance(pk_col.type, PG_UUID):
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
    from sqlalchemy.dialects.postgresql import UUID as PG_UUID
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
                # Convert string UUIDs to UUID objects (PostgreSQL UUID columns)
                if isinstance(col.type, PG_UUID) and isinstance(value, str):
                    try:
                        value = UUID(value)
                    except ValueError:
                        pass
                # Convert string datetimes
                elif hasattr(col.type, 'python_type') and col.type.python_type == datetime and isinstance(value, str):
                    try:
                        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except Exception:
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
    from sqlalchemy.dialects.postgresql import UUID as PG_UUID
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
                # Type conversions (PostgreSQL UUID columns)
                if isinstance(col.type, PG_UUID) and isinstance(value, str):
                    try:
                        value = UUID(value)
                    except ValueError:
                        pass
                elif hasattr(col.type, 'python_type') and col.type.python_type == datetime and isinstance(value, str):
                    try:
                        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except Exception:
                        pass
                set_clauses.append(f"{col_name} = :{col_name}")
                values[col_name] = value
        
        if not set_clauses:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        # Convert record_id to appropriate type (PK may be PostgreSQL UUID)
        pk_value = record_id
        if isinstance(pk_col.type, PG_UUID):
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
    from sqlalchemy.dialects.postgresql import UUID as PG_UUID
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
        
        # Convert record_id to appropriate type (PK may be PostgreSQL UUID)
        pk_value = record_id
        if isinstance(pk_col.type, PG_UUID):
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
        await db.execute(
            text("DELETE FROM chunking_events WHERE document_id = :doc_id"),
            {"doc_id": doc_uuid}
        )
        await db.execute(
            text("DELETE FROM chunking_jobs WHERE document_id = :doc_id"),
            {"doc_id": doc_uuid}
        )
        await db.execute(
            text("DELETE FROM processing_errors WHERE document_id = :doc_id"),
            {"doc_id": doc_uuid}
        )
        await db.execute(
            text("DELETE FROM extracted_facts WHERE document_id = :doc_id"),
            {"doc_id": doc_uuid}
        )
        await db.execute(
            text("DELETE FROM hierarchical_chunks WHERE document_id = :doc_id"),
            {"doc_id": doc_uuid}
        )
        await db.execute(
            text("DELETE FROM chunking_results WHERE document_id = :doc_id"),
            {"doc_id": doc_uuid}
        )
        await db.execute(
            text("DELETE FROM document_pages WHERE document_id = :doc_id"),
            {"doc_id": doc_uuid}
        )
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


# Serve frontend static files (for Cloud Run / single-container deployment)
# Must be last so API routes take precedence
_frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="static")
