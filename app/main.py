import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Any
from fastapi import FastAPI, UploadFile, HTTPException, Depends, Body, Query, Request, Header
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, update, func, text, bindparam, and_, or_, cast, Text as SAText
from sqlalchemy.orm import defer
from google.cloud import storage
import json
from datetime import datetime
from collections import deque
from asyncio import Lock
import asyncio
from app.config import (
    GCS_BUCKET,
    ENV,
    CRITIQUE_RETRY_THRESHOLD,
    DRIVE_API_ENABLED,
    GOOGLE_DRIVE_CLIENT_ID,
    GOOGLE_DRIVE_CLIENT_SECRET,
    GOOGLE_DRIVE_REDIRECT_URI,
    RAG_FRONTEND_URL,
)
from app.database import get_db, Base
from app.models import Document, DocumentPage, ChunkingResult, HierarchicalChunk, ExtractedFact, ProcessingError, ChunkingJob, ChunkingEvent, LlmConfig, EmbeddingJob, ChunkEmbedding, PublishEvent, RagPublishedEmbedding, fact_to_category_scores_dict, PolicyLine, PolicyParagraph, PolicyLexiconCandidate, DocumentTextTag, DriveConnection
from app.services.error_tracker import log_error, classify_error
from app.services.extract_text import extract_text_from_gcs, html_to_plain_text
from app.services.chunking import split_paragraphs, split_paragraphs_from_markdown
from app.services.extraction import stream_extract_facts
from app.services.critique import stream_critique, critique_extraction, normalize_critique_result
from app.services.utils import parse_json_response, default_termination_date, sanitize_text_for_db
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

# OAuth state cache: state -> (session_id, expiry_time). Cleaned on access.
_drive_oauth_state: dict[str, tuple[str, float]] = {}

app = FastAPI(title="Mobius RAG", version="0.1.0")


@app.on_event("startup")
async def run_startup_migrations():
    """Schedule migrations in background so the server binds to PORT immediately (Cloud Run)."""
    asyncio.create_task(_run_startup_migrations_background())


async def _run_startup_migrations_background():
    """Run database migrations in background after server has bound to PORT."""
    from sqlalchemy import text
    from app.database import AsyncSessionLocal, engine

    # Note: pgvector extension removed - embeddings are stored in Vertex AI Vector Search
    # The embedding columns in PostgreSQL are kept for schema compatibility but unused.

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
            try:
                from app.migrations.add_policy_lexicon_tables import migrate as migrate_policy_lexicon_tables
                await migrate_policy_lexicon_tables()
            except Exception as migrate_err:
                logger.warning(f"Startup migration (policy_lexicon tables) skipped: {migrate_err}")
            try:
                from app.migrations.add_policy_line_offsets import migrate as migrate_policy_line_offsets
                await migrate_policy_line_offsets()
            except Exception as migrate_err:
                logger.warning(f"Startup migration (policy_lines offsets) skipped: {migrate_err}")
            try:
                from app.migrations.add_policy_lexicon_candidate_occurrences import migrate as migrate_policy_lexcand_occ
                await migrate_policy_lexcand_occ()
            except Exception as migrate_err:
                logger.warning(f"Startup migration (policy_lexicon_candidates occurrences) skipped: {migrate_err}")
            try:
                from app.migrations.add_policy_lexicon_candidate_catalog import migrate as migrate_policy_lexcand_catalog
                await migrate_policy_lexcand_catalog()
            except Exception as migrate_err:
                logger.warning(f"Startup migration (policy_lexicon_candidate_catalog) skipped: {migrate_err}")
            try:
                from app.migrations.add_document_text_tags import migrate as migrate_document_text_tags
                await migrate_document_text_tags()
            except Exception as migrate_err:
                logger.warning(f"Startup migration (document_text_tags) skipped: {migrate_err}")
            try:
                from app.migrations.add_drive_connections import migrate as migrate_drive_connections
                await migrate_drive_connections()
            except Exception as migrate_err:
                logger.warning(f"Startup migration (drive_connections) skipped: {migrate_err}")

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

# CORS - in dev allow localhost origins with credentials (for Drive OAuth cookies)
# 3999 = Module Hub / Master landing (RAG UI may be loaded there)
_dev_origins = [
    "http://localhost:8001", "http://localhost:5173", "http://localhost:3999",
    "http://127.0.0.1:8001", "http://127.0.0.1:5173", "http://127.0.0.1:3999",
]
cors_origins = _dev_origins if ENV == "dev" else []
if not cors_origins:
    cors_origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=(cors_origins != ["*"]),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/admin/cleanup-stale-jobs")
async def cleanup_stale_jobs(
    timeout_minutes: float = Query(30.0, ge=1, le=1440, description="Mark jobs stuck in processing for this many minutes as failed"),
    db: AsyncSession = Depends(get_db),
):
    """
    Mark chunking and embedding jobs stuck in 'processing' as failed.
    Use when PostgreSQL connection slots are exhausted: fail stuck jobs,
    then stop chunking/embedding workers to free connections.
    If the API cannot connect, run scripts/cleanup_stale_jobs.sql as superuser.
    """
    from app.worker.db import fail_stale_jobs_for_cleanup

    chunking_failed, embedding_failed = await fail_stale_jobs_for_cleanup(db, timeout_minutes=timeout_minutes)
    total = chunking_failed + embedding_failed
    return {
        "status": "ok",
        "chunking_jobs_failed": chunking_failed,
        "embedding_jobs_failed": embedding_failed,
        "total_failed": total,
        "message": f"Marked {total} stuck job(s) as failed. Stop chunking/embedding workers to free DB connections if slots are exhausted.",
    }


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

    # Batch-fetch chunking job/result data (one connection, no per-row queries)
    active_job_per_doc: dict[str, ChunkingJob] = {}
    latest_job_per_doc: dict[str, ChunkingJob] = {}
    latest_result_per_doc: dict[str, ChunkingResult] = {}
    if doc_ids:
        # Active jobs (pending/processing) – latest per document
        active_result = await db.execute(
            text("""
                SELECT id FROM (
                    SELECT id, document_id, status, critique_enabled, max_retries,
                           ROW_NUMBER() OVER (PARTITION BY document_id ORDER BY created_at DESC, id DESC) AS rn
                    FROM chunking_jobs
                    WHERE document_id IN :doc_ids AND status IN ('pending', 'processing')
                ) sub WHERE rn = 1
            """).bindparams(bindparam("doc_ids", expanding=True)),
            {"doc_ids": doc_ids},
        )
        active_ids = [row.id for row in active_result]
        if active_ids:
            active_jobs_result = await db.execute(select(ChunkingJob).where(ChunkingJob.id.in_(active_ids)))
            for job in active_jobs_result.scalars().all():
                active_job_per_doc[str(job.document_id)] = job
        # Latest ChunkingJob per document (any status) – for completed override
        latest_job_result = await db.execute(
            text("""
                SELECT id FROM (
                    SELECT id, document_id, status,
                           ROW_NUMBER() OVER (PARTITION BY document_id ORDER BY created_at DESC, id DESC) AS rn
                    FROM chunking_jobs
                    WHERE document_id IN :doc_ids
                ) sub WHERE rn = 1
            """).bindparams(bindparam("doc_ids", expanding=True)),
            {"doc_ids": doc_ids},
        )
        latest_job_ids = [row.id for row in latest_job_result]
        if latest_job_ids:
            latest_jobs_rows = await db.execute(select(ChunkingJob).where(ChunkingJob.id.in_(latest_job_ids)))
            for job in latest_jobs_rows.scalars().all():
                latest_job_per_doc[str(job.document_id)] = job
        # Latest ChunkingResult per document
        latest_cr_result = await db.execute(
            text("""
                SELECT id FROM (
                    SELECT id, document_id,
                           ROW_NUMBER() OVER (PARTITION BY document_id ORDER BY id DESC) AS rn
                    FROM chunking_results
                    WHERE document_id IN :doc_ids
                ) sub WHERE rn = 1
            """).bindparams(bindparam("doc_ids", expanding=True)),
            {"doc_ids": doc_ids},
        )
        latest_cr_ids = [row.id for row in latest_cr_result]
        if latest_cr_ids:
            latest_cr_rows = await db.execute(select(ChunkingResult).where(ChunkingResult.id.in_(latest_cr_ids)))
            for cr in latest_cr_rows.scalars().all():
                latest_result_per_doc[str(cr.document_id)] = cr

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

        active_job = active_job_per_doc.get(doc_id_str)
        latest_job = latest_job_per_doc.get(doc_id_str)
        latest_result = latest_result_per_doc.get(doc_id_str)

        if chunking_status is None:
            # No events: fall back to ChunkingJob + ChunkingResult (from batch)
            if active_job:
                chunking_status = "queued" if active_job.status == "pending" else "in_progress"
                job_id = str(active_job.id)
                job_critique_enabled = active_job.critique_enabled is None or str(active_job.critique_enabled).lower() == "true"
                job_max_retries = active_job.max_retries if active_job.max_retries is not None else 2
            else:
                if latest_result and latest_result.metadata_:
                    chunking_meta = latest_result.metadata_
                    chunking_status = chunking_meta.get("status", "idle")
                else:
                    chunking_status = "idle"
                job_id = None
                job_critique_enabled = None
                job_max_retries = None
        else:
            job_id = str(active_job.id) if active_job else None
            job_critique_enabled = (active_job.critique_enabled is None or str(active_job.critique_enabled).lower() == "true") if active_job else None
            job_max_retries = (active_job.max_retries if active_job.max_retries is not None else 2) if active_job else None

        # Prefer job/result when they say "completed" (e.g. chunking_complete event was never written)
        if chunking_status in ("in_progress", "queued"):
            if latest_job and latest_job.status == "completed":
                chunking_status = "completed"
            elif latest_result and (latest_result.metadata_ or {}).get("status") == "completed":
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
            "source_metadata": getattr(doc, "source_metadata", None),
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
    sort_dir: str = Query("asc", description="Sort direction: asc, desc"),
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
    dir_norm = (sort_dir or "asc").strip().lower()
    if dir_norm not in ("asc", "desc"):
        dir_norm = "asc"
    dir_sql = dir_norm.upper()

    # Default stable tiebreakers: created_at then id (so pagination is stable)
    if sort == "document":
        sort_clause = f"d.display_name ASC, d.filename ASC, ef.created_at DESC, ef.id DESC"
    elif sort == "page":
        sort_clause = f"COALESCE(ef.page_number, hc.page_number) {dir_sql}, ef.created_at DESC, ef.id DESC"
    else:
        # created_at
        sort_clause = f"ef.created_at {dir_sql}, ef.id DESC"

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
        ORDER BY {sort_clause}
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


# ── Retag endpoints (MUST come before /documents/{document_id} routes) ─────
# Re-apply lexicon tags to already-processed documents by queueing Path B jobs.
# Designed to be called by the Lexicon module after publishing updated tags.

class RetagBulkBody(BaseModel):
    """Body for bulk retag. If document_ids is omitted or empty, ALL completed documents are retagged."""
    document_ids: Optional[List[str]] = None


async def _enqueue_retag_job(db: AsyncSession, doc_uuid, document_id_str: str) -> dict | None:
    """Enqueue a single Path B retag job.  Returns job summary dict or None if skipped."""
    from uuid import UUID as _UUID

    # Skip if a Path B job is already pending/processing
    existing = await db.execute(
        select(ChunkingJob).where(
            ChunkingJob.document_id == doc_uuid,
            ChunkingJob.generator_id == "B",
            ChunkingJob.status.in_(("pending", "processing")),
        )
    )
    if existing.scalar_one_or_none():
        return None  # already queued

    job = ChunkingJob(
        document_id=doc_uuid,
        generator_id="B",
        threshold="0.6",
        status="pending",
        extraction_enabled="false",
        critique_enabled="false",
        max_retries=0,
        skip_embedding="true",
    )
    db.add(job)
    return {"document_id": document_id_str, "job_id": str(job.id)}


@app.post("/documents/retag")
async def retag_documents_bulk(
    db: AsyncSession = Depends(get_db),
    body: RetagBulkBody | None = Body(None),
):
    """
    Re-apply the latest lexicon tags to multiple documents.
    - If body.document_ids is provided, retag only those documents.
    - If body is omitted or document_ids is empty, retag ALL documents that have completed extraction.

    Returns a summary of how many jobs were queued vs skipped.
    Intended to be called by the Lexicon module after a publish.
    """
    from uuid import UUID

    specific_ids: list[str] | None = body.document_ids if body else None

    if specific_ids:
        # Validate and fetch specific documents
        doc_uuids = []
        for did in specific_ids:
            try:
                doc_uuids.append(UUID(did))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid document ID: {did}")

        docs_result = await db.execute(
            select(Document).where(
                Document.id.in_(doc_uuids),
                Document.status == "completed",
            )
        )
        documents = docs_result.scalars().all()
    else:
        # All completed documents
        docs_result = await db.execute(
            select(Document).where(Document.status == "completed")
        )
        documents = docs_result.scalars().all()

    queued = []
    skipped = 0
    for doc in documents:
        job_info = await _enqueue_retag_job(db, doc.id, str(doc.id))
        if job_info:
            queued.append(job_info)
        else:
            skipped += 1

    if queued:
        await db.commit()

    logger.info("Bulk retag: %d queued, %d skipped (already pending)", len(queued), skipped)
    return {
        "status": "ok",
        "queued": len(queued),
        "skipped": skipped,
        "total_documents": len(documents),
        "jobs": queued,
    }


@app.get("/documents/retag/status")
async def retag_status(db: AsyncSession = Depends(get_db)):
    """
    Compare each document's lexicon_revision (in document_tags) against the
    current lexicon revision (in policy_lexicon_meta).  Returns a summary of
    which documents are stale and need retagging.
    """
    from sqlalchemy import text as _text

    # Current lexicon revision
    meta_row = await db.execute(
        _text("SELECT revision FROM policy_lexicon_meta ORDER BY updated_at DESC NULLS LAST LIMIT 1")
    )
    meta = meta_row.fetchone()
    current_revision = int(meta[0]) if meta else 0

    # Per-document revision info
    rows = await db.execute(
        _text("""
            SELECT d.id, d.filename, d.display_name, d.status,
                   dt.lexicon_revision, dt.tagged_at
            FROM documents d
            LEFT JOIN document_tags dt ON dt.document_id = d.id
            WHERE d.status = 'completed'
            ORDER BY d.filename
        """)
    )
    docs = rows.fetchall()

    stale = []
    current = []
    untagged = []
    for row in docs:
        doc_id, filename, display_name, status, lex_rev, tagged_at = row
        info = {
            "document_id": str(doc_id),
            "filename": filename,
            "display_name": display_name,
            "lexicon_revision": int(lex_rev) if lex_rev is not None else None,
            "tagged_at": tagged_at.isoformat() if tagged_at else None,
        }
        if lex_rev is None:
            untagged.append(info)
        elif int(lex_rev) < current_revision:
            stale.append(info)
        else:
            current.append(info)

    return {
        "current_lexicon_revision": current_revision,
        "total_documents": len(docs),
        "stale_count": len(stale),
        "current_count": len(current),
        "untagged_count": len(untagged),
        "stale": stale,
        "current": current,
        "untagged": untagged,
    }


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
                "source_url": getattr(p, "source_url", None),
            }
            for p in pages
        ]
    }


# ---------------------------------------------------------------------------
# Document file download endpoints
# ---------------------------------------------------------------------------

@app.get("/documents/{document_id}/file")
async def get_document_file(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Stream the original document file from GCS.

    Only works for documents whose ``file_path`` is a GCS path (starts with
    ``gs://``).  Returns the raw bytes with the correct ``Content-Type`` and
    a ``Content-Disposition: attachment`` header so the browser downloads it.
    """
    from uuid import UUID as _UUID

    try:
        doc_uuid = _UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = document.file_path or ""
    if not file_path.startswith("gs://"):
        raise HTTPException(
            status_code=404,
            detail="No original file available for this document (scraped or text-only)",
        )

    # Parse gs://<bucket>/<blob_name>
    gcs_parts = file_path.replace("gs://", "").split("/", 1)
    if len(gcs_parts) != 2:
        raise HTTPException(status_code=500, detail="Malformed GCS path")
    bucket_name, blob_name = gcs_parts

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        if not blob.exists():
            raise HTTPException(status_code=404, detail="File not found in GCS")

        content_type = blob.content_type or "application/octet-stream"
        safe_filename = document.filename or blob_name.split("/")[-1]

        def _stream():
            with blob.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 256)  # 256 KB chunks
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(
            _stream(),
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{safe_filename}"',
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to stream file from GCS: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file: {e}")


@app.get("/documents/{document_id}/download/markdown")
async def download_document_markdown(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Generate a ``.md`` file from all pages' ``text_markdown`` for download.

    Useful for scraped / text-only documents that have no original binary file.
    """
    from uuid import UUID as _UUID

    try:
        doc_uuid = _UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    pages_result = await db.execute(
        select(DocumentPage)
        .where(DocumentPage.document_id == doc_uuid)
        .order_by(DocumentPage.page_number)
    )
    pages = pages_result.scalars().all()

    if not pages:
        raise HTTPException(status_code=404, detail="No pages found for this document")

    parts: list[str] = []
    for p in pages:
        text = getattr(p, "text_markdown", None) or p.text or ""
        if text.strip():
            parts.append(f"<!-- Page {p.page_number} -->\n\n{text.strip()}")

    markdown_content = "\n\n---\n\n".join(parts)
    safe_name = (document.display_name or document.filename or "document").replace(" ", "_")
    if not safe_name.endswith(".md"):
        safe_name += ".md"

    return StreamingResponse(
        iter([markdown_content.encode("utf-8")]),
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}"',
        },
    )


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
                    raw_text = sanitize_text_for_db(page_data.get("text") or "") or ""
                    page = DocumentPage(
                        document_id=doc_uuid,
                        page_number=page_data["page_number"],
                        text=raw_text,
                        text_markdown=sanitize_text_for_db(raw_page_to_markdown(raw_text) if raw_text else None),
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
            delete(EmbeddingJob).where(EmbeddingJob.document_id == doc_uuid)
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
            delete(ChunkEmbedding).where(ChunkEmbedding.document_id == doc_uuid)
        )
        await db.execute(
            delete(RagPublishedEmbedding).where(RagPublishedEmbedding.document_id == doc_uuid)
        )
        await db.execute(
            delete(PublishEvent).where(PublishEvent.document_id == doc_uuid)
        )
        await db.execute(
            delete(DocumentPage).where(DocumentPage.document_id == doc_uuid)
        )
        await db.execute(
            delete(Document).where(Document.id == doc_uuid)
        )
        await db.commit()
        try:
            from app.services.vector_store import get_vector_store
            get_vector_store().delete_by_document(document_id)
        except Exception as vec_err:
            logger.warning("Vector store delete_by_document failed (non-fatal): %s", vec_err)
        
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
                raw_text = sanitize_text_for_db(page_data.get("text") or "") or ""
                md = raw_page_to_markdown(raw_text) if raw_text else None
                page = DocumentPage(
                    document_id=document.id,
                    page_number=page_data["page_number"],
                    text=raw_text,
                    text_markdown=sanitize_text_for_db(md),
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
    payer: Optional[str] = None
    state: Optional[str] = None
    program: Optional[str] = None
    authority_level: Optional[str] = None


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
    # Prefer body over query params for metadata
    payer_val = body.payer if body.payer is not None else payer
    state_val = body.state if body.state is not None else state
    program_val = body.program if body.program is not None else program

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
            payer=payer_val,
            state=state_val,
            program=program_val,
            authority_level=body.authority_level,
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
                raw_text = sanitize_text_for_db(page_data.get("text") or "") or ""
                md = raw_page_to_markdown(raw_text) if raw_text else None
                page = DocumentPage(
                    document_id=document.id,
                    page_number=page_data["page_number"],
                    text=raw_text,
                    text_markdown=sanitize_text_for_db(md),
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

        # Auto-chunk when extraction succeeded: queue Path B so chunk → embed run automatically
        if document.status == "completed":
            try:
                where_gen = ChunkingJob.generator_id == "B"
                existing = await db.execute(
                    select(ChunkingJob).where(
                        ChunkingJob.document_id == document.id,
                        where_gen,
                        ChunkingJob.status.in_(["pending", "processing"]),
                    ).limit(1)
                )
                if existing.scalar_one_or_none() is None:
                    job = ChunkingJob(
                        document_id=document.id,
                        generator_id="B",
                        status="pending",
                        threshold="0.6",
                        critique_enabled="false",
                        max_retries=0,
                        extraction_enabled="false",
                    )
                    db.add(job)
                    await db.commit()
                    logger.info("Auto-queued Path B chunking job %s for import-from-gcs document %s", job.id, document.id)
            except Exception as chunk_err:
                logger.warning("Auto-chunk after import-from-gcs failed (non-fatal): %s", chunk_err)
                await db.rollback()

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


# ---------------------------------------------------------------------------
# Google Drive OAuth and import
# ---------------------------------------------------------------------------

def _get_session_id(request: Request, x_rag_session: str | None = Header(None)) -> str:
    """Get or create session ID from cookie or X-RAG-Session header."""
    import uuid
    sid = x_rag_session or (request.cookies.get("rag_session") if hasattr(request, "cookies") else None)
    if sid:
        return sid
    return str(uuid.uuid4())


def _drive_require_enabled():
    """Raise 503 if Drive API is not configured."""
    if not DRIVE_API_ENABLED or not GOOGLE_DRIVE_CLIENT_ID or not GOOGLE_DRIVE_CLIENT_SECRET or not GOOGLE_DRIVE_REDIRECT_URI:
        raise HTTPException(
            status_code=503,
            detail="Google Drive import is not configured. Set DRIVE_API_ENABLED=true, GOOGLE_DRIVE_CLIENT_ID, GOOGLE_DRIVE_CLIENT_SECRET, GOOGLE_DRIVE_REDIRECT_URI in .env",
        )


@app.get("/drive/auth-url")
async def drive_auth_url(
    request: Request,
    x_rag_session: str | None = Header(None, alias="X-RAG-Session"),
    db: AsyncSession = Depends(get_db),
):
    """Return OAuth URL for Google Drive. Frontend redirects user there."""
    _drive_require_enabled()
    import secrets

    session_id = _get_session_id(request, x_rag_session)
    state = secrets.token_urlsafe(32)
    import time
    _drive_oauth_state[state] = (session_id, time.time() + 600)

    from google_auth_oauthlib.flow import Flow
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_DRIVE_CLIENT_ID,
                "client_secret": GOOGLE_DRIVE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [GOOGLE_DRIVE_REDIRECT_URI],
            }
        },
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
        redirect_uri=GOOGLE_DRIVE_REDIRECT_URI,
    )
    url, _ = flow.authorization_url(access_type="offline", prompt="consent", state=state)

    response = JSONResponse({"url": url, "session_id": session_id})
    response.set_cookie(key="rag_session", value=session_id, max_age=86400 * 7, samesite="lax", httponly=True)
    return response


@app.get("/drive/callback")
async def drive_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """OAuth callback: exchange code for tokens, store, redirect to frontend."""
    _drive_require_enabled()
    import time

    if error:
        logger.warning("Drive OAuth error: %s", error)
        return RedirectResponse(url=f"{RAG_FRONTEND_URL}/#/drive?error={error}")

    if not code or not state:
        return RedirectResponse(url=f"{RAG_FRONTEND_URL}/#/drive?error=missing_params")

    entry = _drive_oauth_state.pop(state, None)
    if not entry:
        return RedirectResponse(url=f"{RAG_FRONTEND_URL}/#/drive?error=invalid_state")

    session_id, expiry = entry
    if time.time() > expiry:
        return RedirectResponse(url=f"{RAG_FRONTEND_URL}/#/drive?error=state_expired")

    from google_auth_oauthlib.flow import Flow
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_DRIVE_CLIENT_ID,
                "client_secret": GOOGLE_DRIVE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [GOOGLE_DRIVE_REDIRECT_URI],
            }
        },
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
        redirect_uri=GOOGLE_DRIVE_REDIRECT_URI,
    )
    flow.fetch_token(code=code)

    creds = flow.credentials
    email = None
    try:
        from googleapiclient.discovery import build
        drive = build("drive", "v3", credentials=creds)
        about = drive.about().get(fields="user(emailAddress)").execute()
        email = about.get("user", {}).get("emailAddress")
    except Exception as e:
        logger.warning("Could not fetch Drive user email: %s", e)

    await db.execute(delete(DriveConnection).where(DriveConnection.session_id == session_id))
    conn = DriveConnection(
        session_id=session_id,
        access_token=creds.token,
        refresh_token=creds.refresh_token,
        expires_at=creds.expiry,
        email=email,
    )
    db.add(conn)
    await db.commit()

    return RedirectResponse(url=f"{RAG_FRONTEND_URL}/#/drive?connected=1")


@app.get("/drive/status")
async def drive_status(
    request: Request,
    x_rag_session: str | None = Header(None, alias="X-RAG-Session"),
    db: AsyncSession = Depends(get_db),
):
    """Return {connected: bool, email?: str} for current session."""
    if not DRIVE_API_ENABLED:
        return {"connected": False, "enabled": False}

    session_id = _get_session_id(request, x_rag_session)
    result = await db.execute(
        select(DriveConnection).where(DriveConnection.session_id == session_id)
    )
    conn = result.scalar_one_or_none()
    if not conn:
        return {"connected": False, "enabled": True}
    return {"connected": True, "enabled": True, "email": conn.email}


@app.delete("/drive/disconnect")
async def drive_disconnect(
    request: Request,
    x_rag_session: str | None = Header(None, alias="X-RAG-Session"),
    db: AsyncSession = Depends(get_db),
):
    """Remove Drive connection for current session."""
    session_id = _get_session_id(request, x_rag_session)
    await db.execute(delete(DriveConnection).where(DriveConnection.session_id == session_id))
    await db.commit()
    return {"status": "ok"}


@app.get("/drive/folders/{folder_id}/files")
async def drive_list_folder_files(
    folder_id: str,
    request: Request,
    x_rag_session: str | None = Header(None, alias="X-RAG-Session"),
    db: AsyncSession = Depends(get_db),
):
    """List folders and PDF/Google Doc files in a Drive folder. Use 'root' for My Drive."""
    _drive_require_enabled()
    from app.services.drive_sync import get_credentials, list_folder_contents, parse_folder_id

    fid = parse_folder_id(folder_id) if folder_id else "root"
    if fid is None:
        raise HTTPException(status_code=400, detail="Invalid folder link or ID")

    session_id = _get_session_id(request, x_rag_session)
    creds = await get_credentials(session_id, db)
    if not creds:
        raise HTTPException(status_code=401, detail="Connect Google Drive first")

    contents = list_folder_contents(creds, fid)
    return contents


class ImportFromDriveRequest(BaseModel):
    folder_id: str
    file_ids: Optional[List[str]] = None
    filename_overrides: Optional[dict[str, str]] = None
    payer: Optional[str] = None
    state: Optional[str] = None
    program: Optional[str] = None
    authority_level: Optional[str] = None


@app.post("/documents/import-from-drive")
async def import_from_drive(
    request: Request,
    body: ImportFromDriveRequest = Body(...),
    x_rag_session: str | None = Header(default=None, alias="X-RAG-Session"),
    db: AsyncSession = Depends(get_db),
):
    """Import selected files from a Drive folder. Downloads, uploads to GCS, creates documents."""
    _drive_require_enabled()
    from app.services.drive_sync import get_credentials, list_folder_files, download_file, parse_folder_id

    fid = parse_folder_id(body.folder_id)
    if not fid:
        raise HTTPException(status_code=400, detail="Invalid folder link or ID")

    session_id = _get_session_id(request, x_rag_session)
    creds = await get_credentials(session_id, db)
    if not creds:
        raise HTTPException(status_code=401, detail="Connect Google Drive first")

    file_list = list_folder_files(creds, fid)
    file_ids = body.file_ids or [f["id"] for f in file_list]
    overrides = body.filename_overrides or {}

    results = []
    for f in file_list:
        if f["id"] not in file_ids:
            continue
        file_id = f["id"]
        name = overrides.get(file_id, f.get("name", "unknown"))
        mime = f.get("mimeType", "")
        if mime not in ("application/pdf", "application/vnd.google-apps.document"):
            continue

        try:
            content = download_file(creds, file_id, mime)
        except Exception as e:
            logger.warning("Drive download failed for %s: %s", file_id, e)
            results.append({"file_id": file_id, "filename": name, "status": "failed", "error": str(e)})
            continue

        file_hash = hashlib.sha256(content).hexdigest()
        result = await db.execute(select(Document).where(Document.file_hash == file_hash))
        if result.scalar_one_or_none():
            results.append({"file_id": file_id, "filename": name, "status": "duplicate"})
            continue

        ext = ".pdf" if mime == "application/vnd.google-apps.document" else ("" if name.lower().endswith(".pdf") else "")
        if ext and not name.lower().endswith(".pdf"):
            name = name + ".pdf"
        blob_path = f"drive/{fid}/{name}"
        gcs_path = f"gs://{GCS_BUCKET}/{blob_path}"

        try:
            client = storage.Client()
            bucket = client.bucket(GCS_BUCKET)
            blob = bucket.blob(blob_path)
            blob.upload_from_string(content, content_type="application/pdf")
        except Exception as e:
            logger.error("GCS upload failed for %s: %s", name, e)
            results.append({"file_id": file_id, "filename": name, "status": "failed", "error": str(e)})
            continue

        doc = Document(
            filename=name,
            file_hash=file_hash,
            file_path=gcs_path,
            payer=body.payer,
            state=body.state,
            program=body.program,
            authority_level=body.authority_level,
            termination_date=default_termination_date(),
            status="uploaded",
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)

        try:
            doc.status = "extracting"
            await db.commit()
            from app.services.page_to_markdown import raw_page_to_markdown
            pages = await extract_text_from_gcs(gcs_path)
            for page_data in pages:
                raw_text = sanitize_text_for_db(page_data.get("text") or "") or ""
                md = raw_page_to_markdown(raw_text) if raw_text else None
                page = DocumentPage(
                    document_id=doc.id,
                    page_number=page_data["page_number"],
                    text=raw_text,
                    text_markdown=sanitize_text_for_db(md),
                    extraction_status=page_data.get("extraction_status", "failed"),
                    extraction_error=page_data.get("extraction_error"),
                    text_length=page_data.get("text_length", 0),
                )
                db.add(page)
            doc.status = "completed"
            await db.commit()
        except Exception as e:
            doc.status = "failed"
            await db.commit()
            logger.warning("Extraction failed for Drive import %s: %s", name, e)

        if doc.status == "completed":
            try:
                existing = await db.execute(
                    select(ChunkingJob).where(
                        ChunkingJob.document_id == doc.id,
                        ChunkingJob.generator_id == "B",
                        ChunkingJob.status.in_(["pending", "processing"]),
                    ).limit(1)
                )
                if existing.scalar_one_or_none() is None:
                    job = ChunkingJob(
                        document_id=doc.id,
                        generator_id="B",
                        status="pending",
                        threshold="0.6",
                        critique_enabled="false",
                        max_retries=0,
                        extraction_enabled="false",
                    )
                    db.add(job)
                    await db.commit()
                    logger.info("Auto-queued Path B for Drive import %s", doc.id)
            except Exception as chunk_err:
                logger.warning("Auto-chunk after Drive import failed: %s", chunk_err)
                await db.rollback()

        results.append({
            "file_id": file_id,
            "filename": name,
            "document_id": str(doc.id),
            "status": doc.status,
        })

    return {"results": results}


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
        raw_text = sanitize_text_for_db(text) if text else None
        text_markdown = raw_page_to_markdown(raw_text) if raw_text else ""
        extraction_status = "success" if raw_text else "empty"
        page = DocumentPage(
            document_id=document.id,
            page_number=i + 1,
            text=raw_text,
            text_markdown=sanitize_text_for_db(text_markdown or None),
            extraction_status=extraction_status,
            text_length=len(raw_text) if raw_text else 0,
            source_url=url,
        )
        db.add(page)

    await db.commit()

    # Auto-chunk: queue Path B chunking job so worker runs chunk → (embed auto-enqueued)
    try:
        where_gen = ChunkingJob.generator_id == "B"
        existing = await db.execute(
            select(ChunkingJob).where(
                ChunkingJob.document_id == document.id,
                where_gen,
                ChunkingJob.status.in_(["pending", "processing"]),
            ).limit(1)
        )
        if existing.scalar_one_or_none() is None:
            job = ChunkingJob(
                document_id=document.id,
                generator_id="B",
                status="pending",
                threshold="0.6",
                critique_enabled="false",
                max_retries=0,
                extraction_enabled="false",
            )
            db.add(job)
            await db.commit()
            logger.info("Auto-queued Path B chunking job %s for imported scraped document %s", job.id, document.id)
    except Exception as e:
        logger.warning("Auto-chunk after import-scraped-pages failed (non-fatal): %s", e)
        await db.rollback()

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


def _fact_text_value_for_db(v):
    """Normalize fact text field for DB: dict/list -> JSON string (LLM may return e.g. when_applies as object)."""
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return json.dumps(v, default=str)
    return v if isinstance(v, str) else str(v)


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
                # Validate required fields; normalize text fields (LLM may return dict/list e.g. when_applies)
                fact_text = _fact_text_value_for_db(fact_data.get("fact_text")) or ""
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
                    fact_type=_fact_text_value_for_db(fact_data.get("fact_type")),
                    who_eligible=_fact_text_value_for_db(fact_data.get("who_eligible")),
                    how_verified=_fact_text_value_for_db(fact_data.get("how_verified")),
                    conflict_resolution=_fact_text_value_for_db(fact_data.get("conflict_resolution")),
                    when_applies=_fact_text_value_for_db(fact_data.get("when_applies")),
                    limitations=_fact_text_value_for_db(fact_data.get("limitations")),
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


@app.get("/documents/{document_id}/policy/summary")
async def get_document_policy_summary(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get Path B understanding summary for a document (rollups + progress metadata)."""
    from uuid import UUID

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    # Verify document exists
    doc_result = await db.execute(select(Document).where(Document.id == doc_uuid))
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    cr_result = await db.execute(select(ChunkingResult).where(ChunkingResult.document_id == doc_uuid))
    cr = cr_result.scalar_one_or_none()
    meta = (cr.metadata_ or {}) if cr else {}

    return {
        "document_id": document_id,
        "document_metadata": {
            "filename": doc.filename,
            "display_name": getattr(doc, "display_name", None),
            "payer": doc.payer,
            "state": doc.state,
            "program": doc.program,
            "authority_level": getattr(doc, "authority_level", None),
            "effective_date": getattr(doc, "effective_date", None),
            "termination_date": getattr(doc, "termination_date", None),
        },
        "status": meta.get("status") or ("not_started" if not cr else None),
        "mode": meta.get("mode"),
        "lexicon_version": meta.get("lexicon_version"),
        "last_updated": meta.get("last_updated"),
        "total_pages": meta.get("total_pages"),
        "total_lines": meta.get("total_lines"),
        "completed_lines": meta.get("completed_lines"),
        "document_rollup": meta.get("document_rollup") or {},
        "section_rollups": meta.get("section_rollups") or {},
    }


@app.get("/documents/{document_id}/policy/paragraphs")
async def list_document_policy_paragraphs(
    document_id: str,
    page_number: Optional[int] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
    db: AsyncSession = Depends(get_db),
):
    """List Path B paragraphs for a document (optionally filtered by page)."""
    from uuid import UUID

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    # Verify document exists
    doc_result = await db.execute(select(Document.id).where(Document.id == doc_uuid))
    if not doc_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Document not found")

    q = select(PolicyParagraph).where(PolicyParagraph.document_id == doc_uuid)
    if page_number is not None:
        q = q.where(PolicyParagraph.page_number == int(page_number))

    total_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(total_q)).scalar_one()

    q = q.order_by(PolicyParagraph.page_number, PolicyParagraph.order_index).offset(skip).limit(limit)
    rows = (await db.execute(q)).scalars().all()

    paragraphs = []
    for p in rows:
        paragraphs.append(
            {
                "id": str(p.id),
                "document_id": str(p.document_id),
                "page_number": p.page_number,
                "order_index": p.order_index,
                "paragraph_type": p.paragraph_type,
                "heading_path": p.heading_path,
                "text": p.text,
                "p_tags": p.p_tags,
                "d_tags": p.d_tags,
                "j_tags": p.j_tags,
                "inferred_d_tags": getattr(p, "inferred_d_tags", None),
                "inferred_j_tags": getattr(p, "inferred_j_tags", None),
                "conflict_flags": p.conflict_flags,
                "created_at": p.created_at.isoformat() if getattr(p, "created_at", None) else None,
            }
        )

    return {"total": int(total), "paragraphs": paragraphs}


@app.get("/documents/{document_id}/policy/lines")
async def list_document_policy_lines(
    document_id: str,
    page_number: Optional[int] = Query(None),
    atomic_only: Optional[bool] = Query(None, description="If true, only return is_atomic lines"),
    skip: int = Query(0, ge=0),
    limit: int = Query(1000, ge=1, le=10000),
    db: AsyncSession = Depends(get_db),
):
    """List Path B atomic lines for a document (optionally filtered by page)."""
    from uuid import UUID

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    # Verify document exists
    doc_result = await db.execute(select(Document.id).where(Document.id == doc_uuid))
    if not doc_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Document not found")

    # Join paragraphs only for ordering; select PolicyLine only to avoid asyncpg type 1043 (varchar) on joined columns
    base = (
        select(PolicyLine)
        .join(PolicyParagraph, PolicyParagraph.id == PolicyLine.paragraph_id)
        .where(PolicyLine.document_id == doc_uuid)
    )
    if page_number is not None:
        base = base.where(PolicyLine.page_number == int(page_number))
    if atomic_only is True:
        base = base.where(PolicyLine.is_atomic.is_(True))

    # Count with same filters
    total_q = (
        select(func.count(PolicyLine.id))
        .select_from(PolicyLine)
        .join(PolicyParagraph, PolicyParagraph.id == PolicyLine.paragraph_id)
        .where(PolicyLine.document_id == doc_uuid)
    )
    if page_number is not None:
        total_q = total_q.where(PolicyLine.page_number == int(page_number))
    if atomic_only is True:
        total_q = total_q.where(PolicyLine.is_atomic.is_(True))
    total = (await db.execute(total_q)).scalar_one() or 0

    q = (
        base.order_by(PolicyLine.page_number, PolicyParagraph.order_index, PolicyLine.order_index)
        .options(defer(PolicyLine.offset_match_quality))
        .offset(skip)
        .limit(limit)
    )
    rows = (await db.execute(q)).scalars().all()

    lines = []
    for ln in rows:
        created_at = getattr(ln, "created_at", None)
        lines.append(
            {
                "id": str(ln.id),
                "document_id": str(ln.document_id),
                "page_number": ln.page_number,
                "paragraph_id": str(ln.paragraph_id),
                "paragraph_order_index": None,
                "order_index": ln.order_index,
                "parent_line_id": str(ln.parent_line_id) if getattr(ln, "parent_line_id", None) else None,
                "heading_path": ln.heading_path,
                "line_type": ln.line_type,
                "text": ln.text,
                "is_atomic": ln.is_atomic,
                "non_atomic_reason": ln.non_atomic_reason,
                "p_tags": ln.p_tags,
                "d_tags": ln.d_tags,
                "j_tags": ln.j_tags,
                "inferred_d_tags": getattr(ln, "inferred_d_tags", None),
                "inferred_j_tags": getattr(ln, "inferred_j_tags", None),
                "conflict_flags": ln.conflict_flags,
                "extracted_fields": ln.extracted_fields,
                "start_offset": getattr(ln, "start_offset", None),
                "end_offset": getattr(ln, "end_offset", None),
                "offset_match_quality": None,
                "created_at": created_at.isoformat() if created_at else None,
            }
        )

    return {"total": int(total), "lines": lines}


@app.get("/documents/{document_id}/policy/candidates")
async def list_document_policy_candidates(
    document_id: str,
    state: Optional[str] = Query("proposed", description="Filter by candidate state: proposed|approved|rejected|flagged|all"),
    candidate_type: Optional[List[str]] = Query(None, description="Filter by candidate_type (p|d|j|alias)"),
    skip: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=2000),
    db: AsyncSession = Depends(get_db),
):
    """List Path B lexicon candidates for a document (for human approval workflow)."""
    from uuid import UUID

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    q = select(PolicyLexiconCandidate).where(PolicyLexiconCandidate.document_id == doc_uuid)
    st = (state or "proposed").strip().lower()
    if st and st != "all":
        q = q.where(PolicyLexiconCandidate.state == st)

    if candidate_type:
        types = [str(t).strip().lower() for t in candidate_type if str(t).strip()]
        if types:
            q = q.where(PolicyLexiconCandidate.candidate_type.in_(types))

    total_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(total_q)).scalar_one()
    q = q.order_by(PolicyLexiconCandidate.created_at.desc()).offset(skip).limit(limit)
    rows = (await db.execute(q)).scalars().all()

    out = []
    for c in rows:
        out.append(
            {
                "id": str(c.id),
                "document_id": str(c.document_id),
                "run_id": str(c.run_id) if getattr(c, "run_id", None) else None,
                "candidate_type": c.candidate_type,
                "normalized": c.normalized,
                "proposed_tag": c.proposed_tag,
                "confidence": c.confidence,
                "examples": c.examples,
                "source": c.source,
                "occurrences": getattr(c, "occurrences", None),
                "state": c.state,
                "reviewer": c.reviewer,
                "reviewer_notes": c.reviewer_notes,
                "created_at": c.created_at.isoformat() if getattr(c, "created_at", None) else None,
            }
        )

    return {"total": int(total), "candidates": out}


class CandidateReviewBody(BaseModel):
    state: str  # proposed|approved|rejected|flagged
    reviewer: Optional[str] = None
    reviewer_notes: Optional[str] = None
    # Back-compat: older clients send target_tag
    target_tag: Optional[str] = None  # when approving: tag code to create/alias
    # New (Reader "edit" modal)
    candidate_type_override: Optional[str] = None  # p|d
    tag_code: Optional[str] = None  # target tag code/path to create/update
    tag_spec: Optional[dict] = None  # edited spec fields to write into YAML


@app.post("/policy/candidates/{candidate_id}/review")
async def review_policy_candidate(
    candidate_id: str,
    body: CandidateReviewBody = Body(...),
    db: AsyncSession = Depends(get_db),
):
    """Approve/reject a candidate. Approving p/d will also update the lexicon DB."""
    from uuid import UUID

    try:
        cand_uuid = UUID(candidate_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid candidate ID")

    q = await db.execute(select(PolicyLexiconCandidate).where(PolicyLexiconCandidate.id == cand_uuid))
    c = q.scalar_one_or_none()
    if not c:
        raise HTTPException(status_code=404, detail="Candidate not found")

    async def _upsert_catalog(*, kind: str, normalized: str, proposed_tag: str | None, state: str):
        """Upsert global catalog row for suppression and analytics."""
        try:
            from app.models import PolicyLexiconCandidateCatalog
            from datetime import datetime
            k = (kind or "").strip().lower() or "alias"
            norm = (normalized or "").strip()
            norm_key = norm.lower()[:200]
            prop = (proposed_tag or "").strip()
            prop_key = prop.lower()[:300] if prop else ""
            # Find existing
            existing = (
                await db.execute(
                    select(PolicyLexiconCandidateCatalog).where(
                        PolicyLexiconCandidateCatalog.candidate_type == k,
                        PolicyLexiconCandidateCatalog.normalized_key == norm_key,
                        PolicyLexiconCandidateCatalog.proposed_tag_key == prop_key,
                    )
                )
            ).scalar_one_or_none()
            if existing is None:
                db.add(
                    PolicyLexiconCandidateCatalog(
                        candidate_type=k,
                        normalized_key=norm_key,
                        normalized=norm[:200],
                        proposed_tag_key=prop_key,
                        proposed_tag=(prop[:300] if prop else None),
                        state=state,
                        reviewer=(body.reviewer or "").strip() or None,
                        reviewer_notes=body.reviewer_notes,
                        decided_at=datetime.utcnow(),
                    )
                )
            else:
                existing.state = state
                existing.normalized = norm[:200] or existing.normalized
                existing.proposed_tag = (prop[:300] if prop else existing.proposed_tag)
                existing.reviewer = (body.reviewer or "").strip() or existing.reviewer
                existing.reviewer_notes = body.reviewer_notes
                existing.decided_at = datetime.utcnow()
        except Exception:
            # best-effort; catalog should not block reviews
            return

    next_state = (body.state or "").strip().lower()
    if next_state not in ("proposed", "approved", "rejected", "flagged"):
        raise HTTPException(status_code=400, detail="Invalid state (expected proposed|approved|rejected|flagged)")

    # Optional override for misclassifications (ONLY allowed when approving).
    override_kind = (body.candidate_type_override or "").strip().lower()
    if override_kind and override_kind not in ("p", "d", "j"):
        raise HTTPException(status_code=400, detail="Invalid candidate_type_override (expected p|d|j)")
    if override_kind and next_state != "approved":
        raise HTTPException(status_code=400, detail="candidate_type_override is only allowed when state=approved")
    kind = override_kind or (c.candidate_type or "").strip().lower()
    if override_kind:
        c.candidate_type = override_kind

    updated_lexicon = None
    if next_state == "approved" and kind in ("p", "d", "j"):
        try:
            from app.services.policy_lexicon_repo import approve_phrase_to_db, bump_revision
            tag_code = (body.tag_code or body.target_tag or "").strip() or None
            res = await approve_phrase_to_db(
                db,
                kind=kind,
                normalized=str(c.normalized or ""),
                target_code=tag_code,
                tag_spec=(body.tag_spec if isinstance(body.tag_spec, dict) else None),
            )
            # Keep proposed_tag in sync with what was actually written to DB
            c.proposed_tag = str(res.get("code") or c.proposed_tag or "").strip() or c.proposed_tag
            revision = await bump_revision(db)
            updated_lexicon = {"path": None, "kind": res.get("kind"), "tag_code": res.get("code"), "action": res.get("action"), "revision": revision}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update lexicon DB: {e}")

    c.state = next_state
    c.reviewer = (body.reviewer or "").strip() or c.reviewer
    c.reviewer_notes = body.reviewer_notes
    # Maintain global catalog so rejected candidates don't resurface (and restore is possible).
    if next_state in ("rejected", "approved", "proposed"):
        await _upsert_catalog(kind=c.candidate_type, normalized=c.normalized, proposed_tag=c.proposed_tag, state=next_state)
    # Auto-backfill: approving a global lexicon change should trigger reprocessing for all impacted docs.
    backfill_enqueued = 0
    if next_state == "approved" and kind in ("p", "d", "j"):
        try:
            from uuid import UUID
            norm_key = (c.normalized or "").strip().lower()
            if norm_key:
                doc_rows = (
                    await db.execute(
                        text(
                            "SELECT DISTINCT document_id FROM policy_lexicon_candidates WHERE lower(normalized) = :n"
                        ),
                        {"n": norm_key},
                    )
                ).all()
                doc_ids = [UUID(str(r[0])) for r in doc_rows if r and r[0]]
                if doc_ids:
                    # Avoid duplicate pending/processing jobs.
                    existing = (
                        await db.execute(
                            select(ChunkingJob.document_id).where(
                                ChunkingJob.document_id.in_(doc_ids),
                                ChunkingJob.generator_id == "B",
                                ChunkingJob.status.in_(("pending", "processing")),
                            )
                        )
                    ).all()
                    existing_set = {r[0] for r in existing}
                    for did in doc_ids:
                        if did in existing_set:
                            continue
                        db.add(ChunkingJob(document_id=did, generator_id="B", threshold="0.6", status="pending", extraction_enabled="false", critique_enabled="false", max_retries=0))
                        backfill_enqueued += 1
        except Exception:
            # best-effort
            backfill_enqueued = backfill_enqueued

    await db.commit()

    # Best-effort export to YAML for backward compatibility/tools.
    if next_state == "approved" and kind in ("p", "d", "j"):
        try:
            from app.services.policy_lexicon_repo import export_yaml_from_db

            yaml_path = await export_yaml_from_db(db)
            if isinstance(updated_lexicon, dict):
                updated_lexicon["path"] = yaml_path
        except Exception:
            pass

    return {
        "status": "ok",
        "candidate": {
            "id": str(c.id),
            "document_id": str(c.document_id),
            "candidate_type": c.candidate_type,
            "normalized": c.normalized,
            "proposed_tag": c.proposed_tag,
            "confidence": c.confidence,
            "examples": c.examples,
            "source": c.source,
            "state": c.state,
            "reviewer": c.reviewer,
            "reviewer_notes": c.reviewer_notes,
        },
        "lexicon_update": updated_lexicon,
        "backfill_enqueued": backfill_enqueued,
    }


class CandidateBulkReviewBody(BaseModel):
    candidate_ids: List[str]
    state: str  # proposed|approved|rejected|flagged
    reviewer: Optional[str] = None
    reviewer_notes: Optional[str] = None
    candidate_type_override: Optional[str] = None  # p|d (applies to all)


@app.post("/policy/candidates/review-bulk")
async def review_policy_candidates_bulk(
    body: CandidateBulkReviewBody = Body(...),
    db: AsyncSession = Depends(get_db),
):
    """Bulk approve/reject candidates. Bulk approve uses each candidate's proposed_tag when possible."""
    raise HTTPException(
        status_code=410,
        detail="Lexicon maintenance has moved to the QA service. Use the QA Lexicon Maintenance API to bulk-review candidates, then publish into RAG.",
    )
    from uuid import UUID

    next_state = (body.state or "").strip().lower()
    if next_state not in ("proposed", "approved", "rejected", "flagged"):
        raise HTTPException(status_code=400, detail="Invalid state (expected proposed|approved|rejected|flagged)")

    override_kind = (body.candidate_type_override or "").strip().lower()
    if override_kind and override_kind not in ("p", "d", "j"):
        raise HTTPException(status_code=400, detail="Invalid candidate_type_override (expected p|d|j)")
    if override_kind and next_state != "approved":
        raise HTTPException(status_code=400, detail="candidate_type_override is only allowed when state=approved")

    ids: list[UUID] = []
    for s in body.candidate_ids or []:
        try:
            ids.append(UUID(str(s)))
        except Exception:
            continue
    if not ids:
        raise HTTPException(status_code=400, detail="candidate_ids is required")

    q = await db.execute(select(PolicyLexiconCandidate).where(PolicyLexiconCandidate.id.in_(ids)))
    rows = q.scalars().all()
    found = {str(c.id): c for c in rows}

    updated: list[dict] = []
    errors: list[dict] = []

    # Best-effort: maintain catalog for approved/rejected.
    async def _upsert_catalog(*, kind: str, normalized: str, proposed_tag: str | None, state: str):
        try:
            from app.models import PolicyLexiconCandidateCatalog
            from datetime import datetime
            k = (kind or "").strip().lower() or "alias"
            norm = (normalized or "").strip()
            norm_key = norm.lower()[:200]
            prop = (proposed_tag or "").strip()
            prop_key = prop.lower()[:300] if prop else ""
            existing = (
                await db.execute(
                    select(PolicyLexiconCandidateCatalog).where(
                        PolicyLexiconCandidateCatalog.candidate_type == k,
                        PolicyLexiconCandidateCatalog.normalized_key == norm_key,
                        PolicyLexiconCandidateCatalog.proposed_tag_key == prop_key,
                    )
                )
            ).scalar_one_or_none()
            if existing is None:
                db.add(
                    PolicyLexiconCandidateCatalog(
                        candidate_type=k,
                        normalized_key=norm_key,
                        normalized=norm[:200],
                        proposed_tag_key=prop_key,
                        proposed_tag=(prop[:300] if prop else None),
                        state=state,
                        reviewer=(body.reviewer or "").strip() or None,
                        reviewer_notes=body.reviewer_notes,
                        decided_at=datetime.utcnow(),
                    )
                )
            else:
                existing.state = state
                existing.normalized = norm[:200] or existing.normalized
                existing.proposed_tag = (prop[:300] if prop else existing.proposed_tag)
                existing.reviewer = (body.reviewer or "").strip() or existing.reviewer
                existing.reviewer_notes = body.reviewer_notes
                existing.decided_at = datetime.utcnow()
        except Exception:
            return

    approved_norms: set[str] = set()
    for cid in body.candidate_ids:
        c = found.get(str(cid))
        if not c:
            errors.append({"id": str(cid), "error": "not_found"})
            continue
        try:
            if override_kind:
                c.candidate_type = override_kind
            kind = override_kind or (c.candidate_type or "").strip().lower()

            updated_lexicon = None
            if next_state == "approved" and kind in ("p", "d", "j"):
                from app.services.policy_lexicon_repo import approve_phrase_to_db

                # Prefer approving into the candidate's proposed_tag, else create a new tag code.
                res = await approve_phrase_to_db(
                    db,
                    kind=kind,
                    normalized=str(c.normalized or ""),
                    target_code=(str(c.proposed_tag).strip() if c.proposed_tag else None),
                    tag_spec=None,
                )
                updated_lexicon = {"path": None, "kind": res.get("kind"), "tag_code": res.get("code"), "action": res.get("action")}
                c.proposed_tag = str(res.get("code") or c.proposed_tag or "").strip() or c.proposed_tag
                if (c.normalized or "").strip():
                    approved_norms.add((c.normalized or "").strip().lower())

            c.state = next_state
            c.reviewer = (body.reviewer or "").strip() or c.reviewer
            c.reviewer_notes = body.reviewer_notes
            if next_state in ("approved", "rejected", "proposed"):
                await _upsert_catalog(kind=c.candidate_type, normalized=c.normalized, proposed_tag=c.proposed_tag, state=next_state)
            updated.append({"id": str(c.id), "state": c.state, "candidate_type": c.candidate_type, "proposed_tag": c.proposed_tag, "lexicon_update": updated_lexicon})
        except Exception as e:
            errors.append({"id": str(c.id), "error": str(e)})

    backfill_enqueued = 0
    revision = None
    if next_state == "approved" and approved_norms:
        try:
            from app.services.policy_lexicon_repo import bump_revision

            revision = await bump_revision(db)
            # Auto-backfill: enqueue generator B reprocessing for docs that contain these candidates.
            from uuid import UUID

            doc_ids: set[UUID] = set()
            for n in approved_norms:
                rows2 = (
                    await db.execute(
                        text("SELECT DISTINCT document_id FROM policy_lexicon_candidates WHERE lower(normalized) = :n"),
                        {"n": n},
                    )
                ).all()
                for r in rows2:
                    if r and r[0]:
                        doc_ids.add(UUID(str(r[0])))
            if doc_ids:
                existing = (
                    await db.execute(
                        select(ChunkingJob.document_id).where(
                            ChunkingJob.document_id.in_(list(doc_ids)),
                            ChunkingJob.generator_id == "B",
                            ChunkingJob.status.in_(("pending", "processing")),
                        )
                    )
                ).all()
                existing_set = {r[0] for r in existing}
                for did in doc_ids:
                    if did in existing_set:
                        continue
                    db.add(ChunkingJob(document_id=did, generator_id="B", threshold="0.6", status="pending", extraction_enabled="false", critique_enabled="false", max_retries=0))
                    backfill_enqueued += 1
        except Exception:
            pass

    await db.commit()
    yaml_path = None
    if next_state == "approved" and approved_norms:
        try:
            from app.services.policy_lexicon_repo import export_yaml_from_db

            yaml_path = await export_yaml_from_db(db)
        except Exception:
            yaml_path = None

    return {"status": "ok", "updated": updated, "errors": errors, "lexicon_revision": revision, "lexicon_yaml_path": yaml_path, "backfill_enqueued": backfill_enqueued}


@app.get("/policy/lexicon")
async def get_policy_lexicon_snapshot(db: AsyncSession = Depends(get_db)):
    """Return the active policy lexicon snapshot (DB source-of-truth) in a UI-friendly shape."""
    from app.services.policy_lexicon_repo import load_lexicon_snapshot_db

    lex = await load_lexicon_snapshot_db(db)

    def flatten_tags(kind: str, root: dict) -> list[dict]:
        out: list[dict] = []

        def rec(code: str, spec: Any, parent: str | None):
            if not isinstance(spec, dict):
                out.append({"kind": kind, "code": code, "spec": {"value": spec}, "parent": parent, "has_children": False})
                return
            children = spec.get("children")
            spec_no_children = dict(spec)
            spec_no_children.pop("children", None)
            out.append(
                {
                    "kind": kind,
                    "code": code,
                    "spec": spec_no_children,
                    "parent": parent,
                    "has_children": bool(children) if isinstance(children, dict) else False,
                }
            )
            if isinstance(children, dict):
                for child_code, child_spec in children.items():
                    rec(str(child_code), child_spec, code)

        for code, spec in (root or {}).items():
            rec(str(code), spec, None)
        return out

    return {
        "lexicon_version": lex.version,
        "lexicon_meta": lex.meta,
        "tags": flatten_tags("p", lex.p_tags) + flatten_tags("d", lex.d_tags) + flatten_tags("j", lex.j_tags),
    }


@app.get("/policy/lexicon/overview")
async def get_policy_lexicon_overview(
    kind: str = Query("all", description="Kind filter: all|p|d|j"),
    status: str = Query("all", description="Status filter: all|approved|proposed|rejected"),
    search: Optional[str] = Query(None, description="Search (tag code/description or candidate phrase)"),
    min_score: float = Query(0.6, ge=0.0, le=1.0),
    limit: int = Query(500, ge=1, le=2000),
    top_docs: int = Query(5, ge=0, le=20),
    db: AsyncSession = Depends(get_db),
):
    """Merged Lexicon view for UI: approved tags + candidate groups with status chips."""
    from app.services.policy_lexicon_repo import load_lexicon_snapshot_db

    kind_norm = (kind or "all").strip().lower()
    if kind_norm not in ("all", "p", "d", "j"):
        raise HTTPException(status_code=400, detail="kind must be all|p|d|j")
    status_norm = (status or "all").strip().lower()
    if status_norm not in ("all", "approved", "proposed", "rejected"):
        raise HTTPException(status_code=400, detail="status must be all|approved|proposed|rejected")
    q = (search or "").strip().lower()

    lex = await load_lexicon_snapshot_db(db)
    # Pull revision if present
    revision = None
    if isinstance(lex.meta, dict):
        try:
            revision = int(lex.meta.get("revision")) if lex.meta.get("revision") is not None else None
        except Exception:
            revision = None

    # Candidate groups
    cand_rows: list[dict] = []
    if status_norm in ("all", "proposed", "rejected", "approved"):
        cand_state = "all" if status_norm == "all" else status_norm
        cand_types = None if kind_norm == "all" else [kind_norm]
        cand_payload = await list_policy_candidates_aggregate(
            state=cand_state,
            candidate_type=cand_types,
            search=(q if q else None),
            limit=min(limit, 2000),
            sort="occurrences",
            sort_dir="desc",
            top_docs=top_docs,
            db=db,
        )
        for c in (cand_payload.get("candidates") or []):
            cand_rows.append(
                {
                    "id": f"cand:{c.get('key')}",
                    "row_type": "candidate",
                    "kind": (
                        kind_norm
                        if kind_norm in ("p", "d", "j")
                        else ("p" if "p" in (c.get("candidate_types") or []) else "d" if "d" in (c.get("candidate_types") or []) else "j" if "j" in (c.get("candidate_types") or []) else "d")
                    ),
                    "status": c.get("state") or ("mixed" if cand_state == "all" else cand_state),
                    "key": c.get("key"),
                    "normalized": c.get("normalized"),
                    "doc_count": c.get("doc_count"),
                    "total_occurrences": c.get("total_occurrences"),
                    "max_confidence": c.get("max_confidence"),
                    "candidate_types": c.get("candidate_types"),
                    "proposed_tags": c.get("proposed_tags"),
                    "examples": c.get("examples"),
                    "top_documents": c.get("top_documents"),
                }
            )

    # Tag stats (approved tags)
    tag_rows: list[dict] = []
    if status_norm in ("all", "approved"):
        # Build D-tag lookup (flatten hierarchy) for category/description/search.
        d_lookup: dict[str, dict] = {}
        try:
            def _walk(code: str, spec_any: Any):
                if isinstance(spec_any, dict):
                    spec_no_children = dict(spec_any)
                    children = spec_no_children.pop("children", None)
                    d_lookup[str(code)] = spec_no_children
                    if isinstance(children, dict):
                        for cc, cs in children.items():
                            _walk(str(cc), cs)
            for root_code, root_spec in (lex.d_tags or {}).items():
                _walk(str(root_code), root_spec)
        except Exception:
            d_lookup = {}

        kinds = ("p", "d", "j") if kind_norm == "all" else (kind_norm,)
        for k in kinds:
            tag_expr = (
                "policy_lines.p_tags"
                if k == "p"
                else "COALESCE(policy_lines.inferred_d_tags, policy_lines.d_tags)"
                if k == "d"
                else "policy_lines.j_tags"
            )
            stats_sql = f"""
                WITH base AS (
                    SELECT policy_lines.document_id AS document_id, {tag_expr} AS tag_map
                    FROM policy_lines
                    WHERE policy_lines.is_atomic = TRUE
                      AND {tag_expr} IS NOT NULL
                      AND jsonb_typeof({tag_expr}) = 'object'
                ),
                kv AS (
                    SELECT document_id, (e.key)::text AS tag, (e.value)::float AS score
                    FROM base, LATERAL jsonb_each_text(base.tag_map) AS e(key, value)
                )
                SELECT tag, COUNT(*)::int AS hit_lines, COUNT(DISTINCT document_id)::int AS hit_docs, MAX(score)::float AS max_score
                FROM kv
                WHERE score >= :min_score
                GROUP BY tag
                ORDER BY hit_lines DESC
                LIMIT :limit
            """
            stats = (await db.execute(text(stats_sql), {"min_score": float(min_score), "limit": int(limit)})).all()

            # For search, filter by tag code/description/category (best-effort)
            def _tag_spec(code: str) -> dict:
                if k == "p":
                    return (lex.p_tags or {}).get(code) if isinstance((lex.p_tags or {}).get(code), dict) else {}
                if k == "j":
                    return (lex.j_tags or {}).get(code) if isinstance((lex.j_tags or {}).get(code), dict) else {}
                # d: flatten lookup by code; easiest: rely on snapshot flatten
                return d_lookup.get(code) if isinstance(d_lookup.get(code), dict) else {}

            for tag, hit_lines, hit_docs, max_score in stats:
                code = str(tag or "")
                if not code:
                    continue
                spec = _tag_spec(code)
                category = str(spec.get("category") or "") if isinstance(spec, dict) else ""
                desc = str(spec.get("description") or "") if isinstance(spec, dict) else ""
                if q:
                    hay = f"{code} {category} {desc}".lower()
                    if q not in hay:
                        continue
                tag_rows.append(
                    {
                        "id": f"tag:{k}:{code}",
                        "row_type": "tag",
                        "kind": k,
                        "status": "approved",
                        "code": code,
                        "category": category,
                        "description": desc,
                        "hit_lines": int(hit_lines or 0),
                        "hit_docs": int(hit_docs or 0),
                        "max_score": float(max_score or 0.0),
                    }
                )

    # Merge and sort by usage (so high-hit candidates surface in "All").
    merged = tag_rows + cand_rows
    def _usage(row: dict) -> int:
        if row.get("row_type") == "tag":
            return int(row.get("hit_lines") or 0)
        return int(row.get("total_occurrences") or 0)
    merged.sort(
        key=lambda r: (
            -_usage(r),
            0 if r.get("row_type") == "tag" else 1,
        )
    )
    merged = merged[: int(limit)]
    return {
        "lexicon_version": lex.version,
        "lexicon_revision": revision,
        "min_score": float(min_score),
        "rows": merged,
    }


@app.get("/policy/lexicon/tag-details")
async def get_policy_lexicon_tag_details(
    kind: str = Query("d", description="Tag kind: p|d|j"),
    code: str = Query(..., description="Tag code"),
    min_score: float = Query(0.6, ge=0.0, le=1.0),
    top_docs: int = Query(15, ge=0, le=50),
    sample_lines: int = Query(20, ge=0, le=100),
    db: AsyncSession = Depends(get_db),
):
    """Right-panel details: top documents + sample lines for one approved tag."""
    kind_norm = (kind or "").strip().lower()
    if kind_norm not in ("p", "d", "j"):
        raise HTTPException(status_code=400, detail="kind must be p|d|j")
    tag_code = (code or "").strip()
    if not tag_code:
        raise HTTPException(status_code=400, detail="code is required")

    tag_expr = (
        "policy_lines.p_tags"
        if kind_norm == "p"
        else "COALESCE(policy_lines.inferred_d_tags, policy_lines.d_tags)"
        if kind_norm == "d"
        else "policy_lines.j_tags"
    )
    kv_sql = f"""
        WITH base AS (
            SELECT
                policy_lines.document_id AS document_id,
                policy_lines.page_number AS page_number,
                policy_lines.text AS text,
                {tag_expr} AS tag_map
            FROM policy_lines
            WHERE policy_lines.is_atomic = TRUE
              AND {tag_expr} IS NOT NULL
              AND jsonb_typeof({tag_expr}) = 'object'
        ),
        kv AS (
            SELECT document_id, page_number, text, (e.key)::text AS tag, (e.value)::float AS score
            FROM base, LATERAL jsonb_each_text(base.tag_map) AS e(key, value)
        ),
        filtered AS (
            SELECT document_id, page_number, text, score
            FROM kv
            WHERE tag = :tag_code AND score >= :min_score
        ),
        doc_agg AS (
            SELECT document_id, COUNT(*)::int AS hits, MAX(score)::float AS max_score
            FROM filtered
            GROUP BY document_id
            ORDER BY hits DESC
            LIMIT :top_docs
        )
        SELECT
            doc_agg.document_id,
            COALESCE(documents.display_name, documents.filename) AS document_name,
            doc_agg.hits,
            doc_agg.max_score
        FROM doc_agg
        JOIN documents ON documents.id = doc_agg.document_id
        ORDER BY doc_agg.hits DESC
    """
    docs = (await db.execute(text(kv_sql), {"tag_code": tag_code, "min_score": float(min_score), "top_docs": int(top_docs)})).all()
    top_documents = [
        {"document_id": str(did), "document_name": str(name or ""), "hits": int(hits or 0), "max_score": float(ms or 0.0)}
        for did, name, hits, ms in docs
    ]

    lines_sql = f"""
        WITH base AS (
            SELECT
                policy_lines.document_id AS document_id,
                policy_lines.page_number AS page_number,
                policy_lines.text AS text,
                {tag_expr} AS tag_map
            FROM policy_lines
            WHERE policy_lines.is_atomic = TRUE
              AND {tag_expr} IS NOT NULL
              AND jsonb_typeof({tag_expr}) = 'object'
        ),
        kv AS (
            SELECT document_id, page_number, text, (e.key)::text AS tag, (e.value)::float AS score
            FROM base, LATERAL jsonb_each_text(base.tag_map) AS e(key, value)
        )
        SELECT
            kv.document_id,
            COALESCE(documents.display_name, documents.filename) AS document_name,
            kv.page_number,
            kv.score,
            kv.text
        FROM kv
        JOIN documents ON documents.id = kv.document_id
        WHERE kv.tag = :tag_code AND kv.score >= :min_score
        ORDER BY kv.score DESC
        LIMIT :sample_lines
    """
    lines = (await db.execute(text(lines_sql), {"tag_code": tag_code, "min_score": float(min_score), "sample_lines": int(sample_lines)})).all()
    sample = [
        {
            "document_id": str(did),
            "document_name": str(name or ""),
            "page_number": int(pn or 0),
            "score": float(score or 0.0),
            "text": str(txt or ""),
        }
        for did, name, pn, score, txt in lines
    ]

    return {"kind": kind_norm, "code": tag_code, "min_score": float(min_score), "top_documents": top_documents, "sample_lines": sample}


class LexiconTagUpdateBody(BaseModel):
    spec: dict
    active: Optional[bool] = None
    reviewer: Optional[str] = None
    reviewer_notes: Optional[str] = None


@app.patch("/policy/lexicon/tags/{kind}/{code:path}")
async def update_policy_lexicon_tag(
    kind: str,
    code: str,
    body: LexiconTagUpdateBody = Body(...),
    db: AsyncSession = Depends(get_db),
):
    """Update a lexicon tag spec (DB source of truth), bump revision, export YAML, and enqueue B reruns for consistency."""
    kind_norm = (kind or "").strip().lower()
    if kind_norm not in ("p", "d", "j"):
        raise HTTPException(status_code=400, detail="kind must be p|d|j")
    tag_code = (code or "").strip()
    if not tag_code:
        raise HTTPException(status_code=400, detail="code is required")
    if not isinstance(body.spec, dict):
        raise HTTPException(status_code=400, detail="spec must be an object")

    # Apply update
    try:
        from app.services.policy_lexicon_repo import update_tag_in_db, bump_revision, export_yaml_from_db

        await update_tag_in_db(db, kind=kind_norm, code=tag_code, spec=body.spec, active=body.active)
        revision = await bump_revision(db)
        # Persist revision bump + spec change
        await db.commit()

        # Export YAML for compat
        yaml_path = None
        try:
            yaml_path = await export_yaml_from_db(db)
        except Exception:
            yaml_path = None

        # Auto-backfill: on lexicon edit, rerun all past docs (best-effort).
        backfill_enqueued = 0
        try:
            from uuid import UUID
            # Only enqueue for docs that are not already pending/processing B.
            doc_ids = (await db.execute(select(Document.id))).scalars().all()
            if doc_ids:
                existing = (
                    await db.execute(
                        select(ChunkingJob.document_id).where(
                            ChunkingJob.document_id.in_(doc_ids),
                            ChunkingJob.generator_id == "B",
                            ChunkingJob.status.in_(("pending", "processing")),
                        )
                    )
                ).all()
                existing_set = {r[0] for r in existing}
                for did in doc_ids:
                    if did in existing_set:
                        continue
                    db.add(ChunkingJob(document_id=did, generator_id="B", threshold="0.6", status="pending", extraction_enabled="false", critique_enabled="false", max_retries=0))
                    backfill_enqueued += 1
                await db.commit()
        except Exception:
            backfill_enqueued = backfill_enqueued

        return {"status": "ok", "kind": kind_norm, "code": tag_code, "lexicon_revision": revision, "lexicon_yaml_path": yaml_path, "backfill_enqueued": backfill_enqueued}
    except KeyError:
        raise HTTPException(status_code=404, detail="Tag not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update tag: {e}")


@app.get("/policy/lexicon/stats")
async def get_policy_lexicon_stats(
    kind: str = Query("d", description="Tag kind: p|d"),
    min_score: float = Query(0.6, ge=0.0, le=1.0),
    limit: int = Query(200, ge=1, le=2000),
    top_docs: int = Query(5, ge=0, le=20),
    use_inferred_d: bool = Query(True, description="For d-stats, use inferred_d_tags when present (else d_tags)."),
    db: AsyncSession = Depends(get_db),
):
    """Aggregate tag usage (hits + top documents) from policy_lines for UI ranking."""
    kind_norm = (kind or "d").strip().lower()
    if kind_norm not in ("p", "d"):
        raise HTTPException(status_code=400, detail="kind must be p or d")

    tag_expr = "policy_lines.p_tags" if kind_norm == "p" else ("COALESCE(policy_lines.inferred_d_tags, policy_lines.d_tags)" if use_inferred_d else "policy_lines.d_tags")

    base_sql = f"""
        WITH base AS (
            SELECT policy_lines.document_id AS document_id, {tag_expr} AS tag_map
            FROM policy_lines
            WHERE policy_lines.is_atomic = TRUE
              AND {tag_expr} IS NOT NULL
        ),
        kv AS (
            SELECT document_id, (e.key)::text AS tag, (e.value)::float AS score
            FROM base, LATERAL jsonb_each_text(base.tag_map) AS e(key, value)
        )
        SELECT tag, COUNT(*)::int AS hit_lines, COUNT(DISTINCT document_id)::int AS hit_docs, MAX(score)::float AS max_score
        FROM kv
        WHERE score >= :min_score
        GROUP BY tag
        ORDER BY hit_lines DESC
        LIMIT :limit
    """

    rows = (await db.execute(text(base_sql), {"min_score": float(min_score), "limit": int(limit)})).all()
    tags = [r[0] for r in rows]
    stats_by_tag: dict[str, dict] = {
        str(tag): {"tag": str(tag), "hit_lines": int(hit_lines or 0), "hit_docs": int(hit_docs or 0), "max_score": float(max_score or 0.0), "top_documents": []}
        for tag, hit_lines, hit_docs, max_score in rows
    }

    if top_docs > 0 and tags:
        docs_sql = f"""
            WITH base AS (
                SELECT policy_lines.document_id AS document_id, {tag_expr} AS tag_map
                FROM policy_lines
                WHERE policy_lines.is_atomic = TRUE
                  AND {tag_expr} IS NOT NULL
            ),
            kv AS (
                SELECT document_id, (e.key)::text AS tag, (e.value)::float AS score
                FROM base, LATERAL jsonb_each_text(base.tag_map) AS e(key, value)
            ),
            agg AS (
                SELECT tag, document_id, COUNT(*)::int AS hits
                FROM kv
                WHERE score >= :min_score AND tag IN :tags
                GROUP BY tag, document_id
            )
            SELECT agg.tag, agg.document_id, agg.hits, COALESCE(documents.display_name, documents.filename) AS document_name
            FROM agg
            JOIN documents ON documents.id = agg.document_id
            ORDER BY agg.tag, agg.hits DESC
        """
        docs_rows = (
            await db.execute(
                text(docs_sql).bindparams(bindparam("tags", expanding=True)),
                {"min_score": float(min_score), "tags": tags},
            )
        ).all()
        per_tag: dict[str, list[dict]] = {}
        for tag, doc_id, hits, doc_name in docs_rows:
            per_tag.setdefault(str(tag), []).append(
                {"document_id": str(doc_id), "document_name": str(doc_name or ""), "hits": int(hits or 0)}
            )
        for tag, lst in per_tag.items():
            stats_by_tag[tag]["top_documents"] = lst[: int(top_docs)]

    return {"kind": kind_norm, "min_score": float(min_score), "stats": list(stats_by_tag.values())}


@app.get("/policy/lexicon/doc-stats")
async def get_policy_lexicon_document_stats(
    kind: str = Query("d", description="Tag kind: p|d"),
    min_score: float = Query(0.6, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=500),
    use_inferred_d: bool = Query(True, description="For d-stats, use inferred_d_tags when present (else d_tags)."),
    db: AsyncSession = Depends(get_db),
):
    """Rank documents by total tag hits (for lexicon management UI)."""
    kind_norm = (kind or "d").strip().lower()
    if kind_norm not in ("p", "d"):
        raise HTTPException(status_code=400, detail="kind must be p or d")

    tag_expr = "policy_lines.p_tags" if kind_norm == "p" else ("COALESCE(policy_lines.inferred_d_tags, policy_lines.d_tags)" if use_inferred_d else "policy_lines.d_tags")
    sql = f"""
        WITH base AS (
            SELECT policy_lines.document_id AS document_id, {tag_expr} AS tag_map
            FROM policy_lines
            WHERE policy_lines.is_atomic = TRUE
              AND {tag_expr} IS NOT NULL
        ),
        kv AS (
            SELECT document_id, (e.key)::text AS tag, (e.value)::float AS score
            FROM base, LATERAL jsonb_each_text(base.tag_map) AS e(key, value)
        ),
        filtered AS (
            SELECT document_id, tag
            FROM kv
            WHERE score >= :min_score
        )
        SELECT
            filtered.document_id,
            COALESCE(documents.display_name, documents.filename) AS document_name,
            COUNT(*)::int AS hit_lines,
            COUNT(DISTINCT filtered.tag)::int AS distinct_tags
        FROM filtered
        JOIN documents ON documents.id = filtered.document_id
        GROUP BY filtered.document_id, documents.display_name, documents.filename
        ORDER BY hit_lines DESC
        LIMIT :limit
    """
    rows = (await db.execute(text(sql), {"min_score": float(min_score), "limit": int(limit)})).all()
    return {
        "kind": kind_norm,
        "min_score": float(min_score),
        "documents": [
            {"document_id": str(doc_id), "document_name": str(doc_name or ""), "hit_lines": int(hit_lines or 0), "distinct_tags": int(distinct_tags or 0)}
            for doc_id, doc_name, hit_lines, distinct_tags in rows
        ],
    }


@app.get("/policy/lines")
async def list_policy_lines(
    db: AsyncSession = Depends(get_db),
    document_id: Optional[List[str]] = Query(None, description="Filter by document ID(s)"),
    page_number: Optional[int] = Query(None, description="Filter by page number"),
    heading_path: Optional[str] = Query(None, description="Filter by heading_path (contains)"),
    search: Optional[str] = Query(None, description="Search in line text"),
    p_tag_min_scores: Optional[str] = Query(None, description="JSON object of p_tag -> min score, e.g. {\"eligibility\":0.6}"),
    d_tag_min_scores: Optional[str] = Query(None, description="JSON object of d_tag -> min score, e.g. {\"behavioral_health\":0.7}"),
    has_phone: Optional[bool] = Query(None, description="Filter by whether extracted_fields.phones is non-empty"),
    has_email: Optional[bool] = Query(None, description="Filter by whether extracted_fields.emails is non-empty"),
    has_url: Optional[bool] = Query(None, description="Filter by whether extracted_fields.urls is non-empty"),
    has_date: Optional[bool] = Query(None, description="Filter by whether extracted_fields.dates is non-empty"),
    has_code: Optional[bool] = Query(None, description="Filter by whether extracted_fields.codes is non-empty"),
    payer: Optional[List[str]] = Query(None, description="Filter by payer"),
    state: Optional[List[str]] = Query(None, description="Filter by state"),
    program: Optional[List[str]] = Query(None, description="Filter by program"),
    atomic_only: Optional[bool] = Query(True, description="If true, only return is_atomic lines"),
    skip: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=5000),
    sort: str = Query("created_at", description="Sort by: created_at, document, page"),
    sort_dir: str = Query("asc", description="Sort direction: asc, desc"),
):
    """Global list endpoint for Path B lines (for Facts-tab A/B view)."""
    from uuid import UUID

    q = (
        select(
            PolicyLine,
            PolicyParagraph.order_index.label("paragraph_order"),
            Document.filename.label("doc_filename"),
            Document.display_name.label("doc_display_name"),
            Document.payer.label("doc_payer"),
            Document.state.label("doc_state"),
            Document.program.label("doc_program"),
        )
        .join(Document, Document.id == PolicyLine.document_id)
        .join(PolicyParagraph, PolicyParagraph.id == PolicyLine.paragraph_id)
    )

    if atomic_only is True:
        q = q.where(PolicyLine.is_atomic.is_(True))

    if document_id and len(document_id) > 0:
        try:
            uuids = [UUID(d) for d in document_id]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid document_id")
        q = q.where(PolicyLine.document_id.in_(uuids))

    if page_number is not None:
        q = q.where(PolicyLine.page_number == int(page_number))

    if heading_path and heading_path.strip():
        # JSONB heading_path contains string match (best-effort)
        q = q.where(cast(PolicyLine.heading_path, SAText).ilike(f"%{heading_path.strip()}%"))

    if search and search.strip():
        q = q.where(PolicyLine.text.ilike(f"%{search.strip()}%"))

    # P/D tag threshold filters (JSON objects: {tag_code: min_score})
    def _apply_tag_mins(json_str: Optional[str], kind: str) -> None:
        nonlocal q
        if not json_str or not json_str.strip():
            return
        try:
            mins = json.loads(json_str)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid {kind}_tag_min_scores JSON")
        if not isinstance(mins, dict):
            raise HTTPException(status_code=400, detail=f"{kind}_tag_min_scores must be a JSON object")

        for i, (tag, min_val) in enumerate(mins.items()):
            if tag is None:
                continue
            tag_s = str(tag).strip()
            try:
                mv = float(min_val)
            except Exception:
                continue
            if not tag_s or mv <= 0:
                continue

            if kind == "p":
                cond = text(
                    f"COALESCE((policy_lines.p_tags ->> :ptag_{i})::float, 0) >= :pmin_{i}"
                ).bindparams(
                    bindparam(f"ptag_{i}", value=tag_s),
                    bindparam(f"pmin_{i}", value=mv),
                )
            else:
                # Prefer inferred D if present, else asserted D
                cond = text(
                    f"""COALESCE(
                        (policy_lines.inferred_d_tags ->> :dtag_{i})::float,
                        (policy_lines.d_tags ->> :dtag_{i})::float,
                        0
                      ) >= :dmin_{i}"""
                ).bindparams(
                    bindparam(f"dtag_{i}", value=tag_s),
                    bindparam(f"dmin_{i}", value=mv),
                )
            q = q.where(cond)

    _apply_tag_mins(p_tag_min_scores, "p")
    _apply_tag_mins(d_tag_min_scores, "d")

    # Hard-field presence filters from extracted_fields (phones/emails/urls/dates/codes)
    def _apply_has(field: str, val: Optional[bool]) -> None:
        nonlocal q
        if val is None:
            return
        allowed = {
            "phones": "phones",
            "emails": "emails",
            "urls": "urls",
            "dates": "dates",
            "codes": "codes",
        }
        key = allowed.get(field)
        if not key:
            return
        op = ">" if val else "="
        q = q.where(
            text(f"COALESCE(jsonb_array_length(policy_lines.extracted_fields -> '{key}'), 0) {op} 0")
        )

    _apply_has("phones", has_phone)
    _apply_has("emails", has_email)
    _apply_has("urls", has_url)
    _apply_has("dates", has_date)
    _apply_has("codes", has_code)

    def _in(values: Optional[List[str]], col):
        nonlocal q
        if values and len(values) > 0:
            q = q.where(col.in_(values))

    _in(payer, Document.payer)
    _in(state, Document.state)
    _in(program, Document.program)

    total_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(total_q)).scalar_one()

    dir_norm = (sort_dir or "asc").strip().lower()
    if dir_norm not in ("asc", "desc"):
        dir_norm = "asc"
    page_order = PolicyLine.page_number.desc() if dir_norm == "desc" else PolicyLine.page_number.asc()
    created_order = PolicyLine.created_at.desc() if dir_norm == "desc" else PolicyLine.created_at.asc()

    if sort == "document":
        q = q.order_by(Document.display_name, Document.filename, PolicyLine.created_at.desc())
    elif sort == "page":
        # Only reverse page order; keep within-page reading order stable.
        q = q.order_by(page_order, PolicyParagraph.order_index, PolicyLine.order_index)
    else:
        q = q.order_by(created_order, PolicyLine.id.desc())

    q = q.offset(skip).limit(limit)
    rows = (await db.execute(q)).all()

    lines = []
    for ln, paragraph_order, doc_filename, doc_display_name, doc_payer, doc_state, doc_program in rows:
        lines.append(
            {
                "id": str(ln.id),
                "document_id": str(ln.document_id),
                "document_filename": doc_filename,
                "document_display_name": doc_display_name,
                "payer": doc_payer,
                "state": doc_state,
                "program": doc_program,
                "page_number": ln.page_number,
                "paragraph_id": str(ln.paragraph_id),
                "paragraph_order_index": int(paragraph_order) if paragraph_order is not None else None,
                "order_index": ln.order_index,
                "heading_path": ln.heading_path,
                "line_type": ln.line_type,
                "text": ln.text,
                "is_atomic": ln.is_atomic,
                "non_atomic_reason": ln.non_atomic_reason,
                "p_tags": ln.p_tags,
                "d_tags": ln.d_tags,
                "j_tags": ln.j_tags,
                "inferred_d_tags": getattr(ln, "inferred_d_tags", None),
                "conflict_flags": ln.conflict_flags,
                "extracted_fields": ln.extracted_fields,
                "start_offset": getattr(ln, "start_offset", None),
                "end_offset": getattr(ln, "end_offset", None),
                "offset_match_quality": getattr(ln, "offset_match_quality", None),
                "created_at": ln.created_at.isoformat() if getattr(ln, "created_at", None) else None,
            }
        )

    return {"total": int(total), "lines": lines}


@app.get("/policy/candidates/aggregate")
async def list_policy_candidates_aggregate(
    state: str = Query("proposed", description="Filter by candidate state: proposed|approved|rejected|flagged|all"),
    candidate_type: Optional[List[str]] = Query(None, description="Filter by candidate_type (p|d|j|alias)"),
    search: Optional[str] = Query(None, description="Search in normalized phrase"),
    limit: int = Query(200, ge=1, le=2000),
    sort: str = Query("occurrences", description="Sort by: occurrences, docs, confidence"),
    sort_dir: str = Query("desc", description="Sort direction: asc, desc"),
    top_docs: int = Query(10, ge=0, le=50),
    db: AsyncSession = Depends(get_db),
):
    """Aggregate candidates across documents (for lexicon management)."""
    st = (state or "proposed").strip().lower()
    sort_norm = (sort or "occurrences").strip().lower()
    dir_norm = (sort_dir or "desc").strip().lower()
    if dir_norm not in ("asc", "desc"):
        dir_norm = "desc"
    if sort_norm not in ("occurrences", "docs", "confidence"):
        sort_norm = "occurrences"

    where = ["1=1"]
    params: dict[str, Any] = {"limit": int(limit), "top_docs": int(top_docs)}
    if st and st != "all":
        where.append("c.state = :state")
        params["state"] = st
    if candidate_type:
        types = [str(t).strip().lower() for t in candidate_type if str(t).strip()]
        if types:
            where.append("lower(c.candidate_type) = ANY(:types)")
            params["types"] = types
    if search and search.strip():
        where.append("lower(c.normalized) LIKE :q")
        params["q"] = f"%{search.strip().lower()}%"

    order_expr = "total_occurrences" if sort_norm == "occurrences" else "doc_count" if sort_norm == "docs" else "max_confidence"
    order_dir = "ASC" if dir_norm == "asc" else "DESC"

    sql = f"""
        WITH c AS (
            SELECT
                lower(normalized) AS norm_key,
                normalized,
                state,
                candidate_type,
                proposed_tag,
                confidence,
                COALESCE(occurrences, 1) AS occ,
                document_id,
                examples
            FROM policy_lexicon_candidates c
            WHERE {" AND ".join(where)}
        ),
        ex AS (
            SELECT
                c.norm_key,
                jsonb_agg(DISTINCT ex_txt) FILTER (WHERE ex_txt IS NOT NULL) AS examples
            FROM c
            LEFT JOIN LATERAL jsonb_array_elements_text(COALESCE(c.examples, '[]'::jsonb)) AS ex_txt ON TRUE
            GROUP BY c.norm_key
        ),
        agg AS (
            SELECT
                norm_key,
                MAX(normalized) AS normalized,
                COUNT(DISTINCT document_id)::int AS doc_count,
                SUM(occ)::int AS total_occurrences,
                MAX(COALESCE(confidence, 0))::float AS max_confidence,
                ARRAY_AGG(DISTINCT lower(state)) AS states,
                ARRAY_AGG(DISTINCT lower(candidate_type)) AS candidate_types,
                ARRAY_AGG(DISTINCT proposed_tag) FILTER (WHERE proposed_tag IS NOT NULL) AS proposed_tags
            FROM c
            GROUP BY norm_key
        ),
        doc_agg AS (
            SELECT
                c.norm_key,
                c.document_id,
                SUM(c.occ)::int AS occurrences
            FROM c
            GROUP BY c.norm_key, c.document_id
        ),
        ranked_docs AS (
            SELECT
                doc_agg.norm_key,
                doc_agg.document_id,
                doc_agg.occurrences,
                COALESCE(d.display_name, d.filename) AS document_name,
                ROW_NUMBER() OVER (PARTITION BY doc_agg.norm_key ORDER BY doc_agg.occurrences DESC) AS rn
            FROM doc_agg
            JOIN documents d ON d.id = doc_agg.document_id
        )
        SELECT
            agg.norm_key,
            agg.normalized,
            CASE
              WHEN 'rejected' = ANY(agg.states) THEN 'rejected'
              WHEN 'approved' = ANY(agg.states) THEN 'approved'
              WHEN 'flagged' = ANY(agg.states) THEN 'flagged'
              ELSE 'proposed'
            END AS group_state,
            agg.doc_count,
            agg.total_occurrences,
            agg.max_confidence,
            agg.candidate_types,
            agg.proposed_tags,
            COALESCE(ex.examples, '[]'::jsonb) AS examples,
            COALESCE(
                jsonb_agg(
                    jsonb_build_object(
                        'document_id', ranked_docs.document_id::text,
                        'document_name', ranked_docs.document_name,
                        'occurrences', ranked_docs.occurrences
                    )
                    ORDER BY ranked_docs.occurrences DESC
                ) FILTER (WHERE ranked_docs.rn <= :top_docs),
                '[]'::jsonb
            ) AS top_documents
        FROM agg
        LEFT JOIN ex ON ex.norm_key = agg.norm_key
        LEFT JOIN ranked_docs ON ranked_docs.norm_key = agg.norm_key
        GROUP BY agg.norm_key, agg.normalized, agg.doc_count, agg.total_occurrences, agg.max_confidence, agg.candidate_types, agg.proposed_tags, ex.examples, agg.states
        ORDER BY {order_expr} {order_dir}
        LIMIT :limit
    """

    rows = (await db.execute(text(sql), params)).all()
    out = []
    for norm_key, normalized, group_state, doc_count, total_occurrences, max_confidence, candidate_types, proposed_tags, examples, top_documents in rows:
        out.append(
            {
                "key": str(norm_key),
                "normalized": str(normalized or ""),
                "state": str(group_state or "proposed"),
                "doc_count": int(doc_count or 0),
                "total_occurrences": int(total_occurrences or 0),
                "max_confidence": float(max_confidence or 0.0),
                "candidate_types": list(candidate_types or []),
                "proposed_tags": [str(x) for x in (proposed_tags or []) if x],
                "examples": examples,
                "top_documents": top_documents,
            }
        )
    return {"total": len(out), "candidates": out}


class CandidateAggregateBulkReviewBody(BaseModel):
    normalized_list: List[str]
    state: str  # proposed|approved|rejected|flagged
    candidate_type_override: Optional[str] = None  # p|d|j
    reviewer: Optional[str] = None
    reviewer_notes: Optional[str] = None
    # optional: normalized -> tag_code
    tag_code_map: Optional[dict] = None


@app.post("/policy/candidates/aggregate/review-bulk")
async def review_policy_candidates_aggregate_bulk(
    body: CandidateAggregateBulkReviewBody = Body(...),
    db: AsyncSession = Depends(get_db),
):
    """Bulk review candidate groups across documents (keyed by normalized phrase)."""
    raise HTTPException(
        status_code=410,
        detail="Lexicon maintenance has moved to the QA service. Use the QA Lexicon Maintenance API to approve/reject candidates, then publish into RAG.",
    )
    next_state = (body.state or "").strip().lower()
    if next_state not in ("proposed", "approved", "rejected", "flagged"):
        raise HTTPException(status_code=400, detail="Invalid state (expected proposed|approved|rejected|flagged)")
    override_kind = (body.candidate_type_override or "").strip().lower()
    if override_kind and override_kind not in ("p", "d", "j"):
        raise HTTPException(status_code=400, detail="Invalid candidate_type_override (expected p|d|j)")
    if override_kind and next_state != "approved":
        raise HTTPException(status_code=400, detail="candidate_type_override is only allowed when state=approved")

    norms = [str(s).strip() for s in (body.normalized_list or []) if str(s).strip()]
    if not norms:
        raise HTTPException(status_code=400, detail="normalized_list is required")
    norm_lowers = [n.lower() for n in norms]

    # Restore path: state=proposed means "restore" (rejected/flagged → proposed). No lexicon write.
    if next_state == "proposed":
        updated: list[dict] = []
        errors: list[dict] = []
        for norm_key in norm_lowers:
            try:
                await db.execute(
                    text(
                        """
                        UPDATE policy_lexicon_candidates
                        SET state = 'proposed',
                            reviewer = :reviewer,
                            reviewer_notes = :notes
                        WHERE lower(normalized) = :norm_key
                        """
                    ),
                    {
                        "reviewer": (body.reviewer or "").strip() or None,
                        "notes": body.reviewer_notes,
                        "norm_key": norm_key,
                    },
                )
                # Update catalog too so suppression is removed.
                try:
                    await db.execute(
                        text(
                            """
                            UPDATE policy_lexicon_candidate_catalog
                            SET state = 'proposed',
                                reviewer = COALESCE(NULLIF(:reviewer, ''), reviewer),
                                reviewer_notes = COALESCE(:notes, reviewer_notes),
                                decided_at = NOW(),
                                updated_at = NOW()
                            WHERE normalized_key = :norm_key
                            """
                        ),
                        {
                            "reviewer": (body.reviewer or "").strip(),
                            "notes": body.reviewer_notes,
                            "norm_key": norm_key[:200],
                        },
                    )
                except Exception:
                    pass
                updated.append({"normalized": norm_key, "state": "proposed"})
            except Exception as e:
                errors.append({"normalized": norm_key, "error": str(e)})
        await db.commit()
        return {"status": "ok", "updated": updated, "errors": errors, "backfill_enqueued": 0, "lexicon_revision": None, "lexicon_yaml_path": None}

    # Fetch representative rows (highest occurrences / confidence) per normalized
    rep_sql = text(
        """
        SELECT DISTINCT ON (lower(normalized))
            lower(normalized) AS norm_key,
            normalized,
            candidate_type,
            proposed_tag
        FROM policy_lexicon_candidates
        WHERE lower(normalized) = ANY(:norms)
          AND state = 'proposed'
        ORDER BY lower(normalized),
                 COALESCE(occurrences, 1) DESC,
                 COALESCE(confidence, 0) DESC,
                 created_at DESC
        """
    )
    reps = (await db.execute(rep_sql, {"norms": norm_lowers})).all()
    rep_by_key = {str(r[0]): {"normalized": r[1], "candidate_type": r[2], "proposed_tag": r[3]} for r in reps}

    updated: list[dict] = []
    errors: list[dict] = []
    tag_code_map = body.tag_code_map if isinstance(body.tag_code_map, dict) else {}

    approved_norms: set[str] = set()
    for norm_key in norm_lowers:
        rep = rep_by_key.get(norm_key)
        if not rep:
            errors.append({"normalized": norm_key, "error": "not_found"})
            continue
        try:
            kind = override_kind or str(rep.get("candidate_type") or "").strip().lower() or "d"
            if kind not in ("p", "d", "j"):
                kind = "d"

            updated_lexicon = None
            if next_state == "approved" and kind in ("p", "d", "j"):
                from app.services.policy_lexicon_repo import approve_phrase_to_db

                tag_code = None
                try:
                    tag_code = str(tag_code_map.get(norm_key) or "").strip() or None
                except Exception:
                    tag_code = None
                # Default to proposed_tag if present; else let YAML editor generate snake.
                tag_code = tag_code or (str(rep.get("proposed_tag") or "").strip() or None)
                res = await approve_phrase_to_db(
                    db,
                    kind=kind,
                    normalized=str(rep.get("normalized") or ""),
                    target_code=tag_code,
                    tag_spec=None,
                )
                updated_lexicon = {"path": None, "kind": res.get("kind"), "tag_code": res.get("code"), "action": res.get("action")}
                approved_norms.add(norm_key)

                # Update all matching candidate rows across docs to reflect the chosen tag_code.
                await db.execute(
                    text(
                        """
                        UPDATE policy_lexicon_candidates
                        SET state = :state,
                            candidate_type = :kind,
                            proposed_tag = :tag_code,
                            reviewer = :reviewer,
                            reviewer_notes = :notes
                        WHERE lower(normalized) = :norm_key
                        """
                    ),
                    {
                        "state": next_state,
                        "kind": kind,
                        "tag_code": str(res.get("code") or ""),
                        "reviewer": (body.reviewer or "").strip() or None,
                        "notes": body.reviewer_notes,
                        "norm_key": norm_key,
                    },
                )
            else:
                await db.execute(
                    text(
                        """
                        UPDATE policy_lexicon_candidates
                        SET state = :state,
                            reviewer = :reviewer,
                            reviewer_notes = :notes
                        WHERE lower(normalized) = :norm_key
                        """
                    ),
                    {
                        "state": next_state,
                        "reviewer": (body.reviewer or "").strip() or None,
                        "notes": body.reviewer_notes,
                        "norm_key": norm_key,
                    },
                )

            # Maintain global catalog so rejected candidates do not resurface in future runs.
            if next_state in ("rejected", "approved", "proposed"):
                try:
                    prop = None
                    if isinstance(updated_lexicon, dict) and updated_lexicon.get("tag_code"):
                        prop = str(updated_lexicon.get("tag_code") or "").strip() or None
                    if not prop:
                        prop = str(rep.get("proposed_tag") or "").strip() or None
                    prop_key = (prop or "").lower()[:300] if prop else ""
                    await db.execute(
                        text(
                            """
                            INSERT INTO policy_lexicon_candidate_catalog(
                              candidate_type, normalized_key, normalized, proposed_tag_key, proposed_tag,
                              state, reviewer, reviewer_notes, decided_at, updated_at
                            )
                            VALUES (:kind, :norm_key, :normalized, :prop_key, :prop, :state, :reviewer, :notes, NOW(), NOW())
                            ON CONFLICT (candidate_type, normalized_key, proposed_tag_key)
                            DO UPDATE SET
                              state = EXCLUDED.state,
                              normalized = EXCLUDED.normalized,
                              proposed_tag = EXCLUDED.proposed_tag,
                              reviewer = COALESCE(EXCLUDED.reviewer, policy_lexicon_candidate_catalog.reviewer),
                              reviewer_notes = EXCLUDED.reviewer_notes,
                              decided_at = EXCLUDED.decided_at,
                              updated_at = NOW()
                            """
                        ),
                        {
                            "kind": kind,
                            "norm_key": norm_key,
                            "normalized": str(rep.get("normalized") or "")[:200],
                            "prop_key": prop_key,
                            "prop": (prop[:300] if prop else None),
                            "state": next_state,
                            "reviewer": (body.reviewer or "").strip() or None,
                            "notes": body.reviewer_notes,
                        },
                    )
                except Exception:
                    pass

            updated.append({"normalized": norm_key, "state": next_state, "candidate_type": kind, "lexicon_update": updated_lexicon})
        except Exception as e:
            errors.append({"normalized": norm_key, "error": str(e)})

    backfill_enqueued = 0
    revision = None
    if next_state == "approved" and approved_norms:
        try:
            from app.services.policy_lexicon_repo import bump_revision
            from uuid import UUID

            revision = await bump_revision(db)
            doc_rows = (
                await db.execute(
                    text(
                        "SELECT DISTINCT document_id FROM policy_lexicon_candidates WHERE lower(normalized) = ANY(:norms)"
                    ),
                    {"norms": list(approved_norms)},
                )
            ).all()
            doc_ids = [UUID(str(r[0])) for r in doc_rows if r and r[0]]
            if doc_ids:
                existing = (
                    await db.execute(
                        select(ChunkingJob.document_id).where(
                            ChunkingJob.document_id.in_(doc_ids),
                            ChunkingJob.generator_id == "B",
                            ChunkingJob.status.in_(("pending", "processing")),
                        )
                    )
                ).all()
                existing_set = {r[0] for r in existing}
                for did in doc_ids:
                    if did in existing_set:
                        continue
                    db.add(ChunkingJob(document_id=did, generator_id="B", threshold="0.6", status="pending", extraction_enabled="false", critique_enabled="false", max_retries=0))
                    backfill_enqueued += 1
        except Exception:
            pass

    await db.commit()
    yaml_path = None
    if next_state == "approved" and approved_norms:
        try:
            from app.services.policy_lexicon_repo import export_yaml_from_db

            yaml_path = await export_yaml_from_db(db)
        except Exception:
            yaml_path = None
    return {"status": "ok", "updated": updated, "errors": errors, "lexicon_revision": revision, "lexicon_yaml_path": yaml_path, "backfill_enqueued": backfill_enqueued}


@app.get("/policy/candidates/catalog")
async def list_policy_candidate_catalog(
    state: str = Query("rejected", description="Filter: rejected|approved|flagged|all"),
    candidate_type: Optional[List[str]] = Query(None, description="Filter by candidate_type (p|d|j|alias)"),
    search: Optional[str] = Query(None, description="Search in normalized phrase"),
    limit: int = Query(200, ge=1, le=2000),
    db: AsyncSession = Depends(get_db),
):
    """Inspect the global candidate catalog (mostly for rejected suppression)."""
    from app.models import PolicyLexiconCandidateCatalog

    q = select(PolicyLexiconCandidateCatalog)
    st = (state or "rejected").strip().lower()
    if st and st != "all":
        q = q.where(PolicyLexiconCandidateCatalog.state == st)
    if candidate_type:
        types = [str(t).strip().lower() for t in candidate_type if str(t).strip()]
        if types:
            q = q.where(func.lower(PolicyLexiconCandidateCatalog.candidate_type).in_(types))
    if search and search.strip():
        q = q.where(PolicyLexiconCandidateCatalog.normalized.ilike(f"%{search.strip()}%"))
    q = q.order_by(PolicyLexiconCandidateCatalog.decided_at.desc().nullslast(), PolicyLexiconCandidateCatalog.updated_at.desc()).limit(limit)
    rows = (await db.execute(q)).scalars().all()
    return {
        "total": len(rows),
        "items": [
            {
                "id": str(r.id),
                "candidate_type": r.candidate_type,
                "normalized": r.normalized,
                "proposed_tag": r.proposed_tag,
                "state": r.state,
                "reviewer": r.reviewer,
                "reviewer_notes": r.reviewer_notes,
                "decided_at": r.decided_at.isoformat() if r.decided_at else None,
            }
            for r in rows
        ],
    }


class CandidateAggregateBulkClassifyBody(BaseModel):
    normalized_list: List[str]
    candidate_type_override: str  # p|d|j
    reviewer: Optional[str] = None
    reviewer_notes: Optional[str] = None


@app.post("/policy/candidates/aggregate/classify-bulk")
async def classify_policy_candidates_aggregate_bulk(
    body: CandidateAggregateBulkClassifyBody = Body(...),
    db: AsyncSession = Depends(get_db),
):
    """Disabled: classification-only changes cause integrity issues. Approve instead (writes lexicon + triggers backfill)."""
    raise HTTPException(
        status_code=400,
        detail="Classification-only is disabled for integrity. Use /policy/candidates/aggregate/review-bulk with state=approved (and candidate_type_override) to approve into lexicon.",
    )


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


# ─── Policy Line Tags (from Path B pipeline) ─── #

@app.get("/documents/{document_id}/policy-line-tags")
async def get_document_policy_line_tags(
    document_id: str,
    page_number: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Return policy lines with their p_tags, d_tags, and j_tags for rendering highlights."""
    from uuid import UUID
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    query = (
        select(PolicyLine)
        .where(PolicyLine.document_id == doc_uuid)
        .where(
            or_(
                PolicyLine.p_tags.isnot(None),
                PolicyLine.d_tags.isnot(None),
                PolicyLine.j_tags.isnot(None),
            )
        )
    )
    if page_number is not None:
        query = query.where(PolicyLine.page_number == page_number)
    query = query.order_by(PolicyLine.page_number, PolicyLine.order_index)

    result = await db.execute(query)
    lines = result.scalars().all()

    def clean_tags(val):
        """Return a dict of tag->score, or None if empty / JSON null."""
        if val is None:
            return None
        if isinstance(val, dict) and val:
            return val
        return None

    items = []
    for ln in lines:
        p = clean_tags(ln.p_tags)
        d = clean_tags(ln.d_tags)
        j = clean_tags(ln.j_tags)
        if not p and not d and not j:
            continue
        items.append({
            "id": str(ln.id),
            "page_number": ln.page_number,
            "text": ln.text,
            "start_offset": ln.start_offset,
            "end_offset": ln.end_offset,
            "p_tags": p,
            "d_tags": d,
            "j_tags": j,
        })

    return {
        "document_id": document_id,
        "total": len(items),
        "lines": items,
    }


# ─── Document Text Tags (user-applied category tags on text ranges) ─── #

@app.get("/documents/{document_id}/text-tags")
async def list_document_text_tags(
    document_id: str,
    page_number: Optional[int] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """List text tags for a document, optionally filtered by page number."""
    from uuid import UUID
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")
    query = select(DocumentTextTag).where(DocumentTextTag.document_id == doc_uuid)
    if page_number is not None:
        query = query.where(DocumentTextTag.page_number == page_number)
    query = query.order_by(DocumentTextTag.page_number, DocumentTextTag.start_offset)
    result = await db.execute(query)
    tags = result.scalars().all()
    return {
        "document_id": document_id,
        "total": len(tags),
        "tags": [
            {
                "id": str(t.id),
                "document_id": str(t.document_id),
                "page_number": t.page_number,
                "start_offset": t.start_offset,
                "end_offset": t.end_offset,
                "tagged_text": t.tagged_text,
                "tag": t.tag,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in tags
        ],
    }


@app.post("/documents/{document_id}/text-tags")
async def create_document_text_tag(
    document_id: str,
    body: dict = Body(...),
    db: AsyncSession = Depends(get_db),
):
    """Create a text tag (user selects text and applies a category label)."""
    from uuid import UUID
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    # Validate inputs
    tagged_text = (body.get("tagged_text") or "").strip()
    if not tagged_text:
        raise HTTPException(status_code=400, detail="tagged_text is required")
    tag = (body.get("tag") or "").strip()
    if not tag:
        raise HTTPException(status_code=400, detail="tag is required")
    page_number = body.get("page_number")
    if page_number is None:
        raise HTTPException(status_code=400, detail="page_number is required")
    try:
        page_number = int(page_number)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="page_number must be an integer")
    start_offset = body.get("start_offset")
    end_offset = body.get("end_offset")
    if start_offset is None or end_offset is None:
        raise HTTPException(status_code=400, detail="start_offset and end_offset are required")
    try:
        start_offset = int(start_offset)
        end_offset = int(end_offset)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="start_offset and end_offset must be integers")
    if start_offset >= end_offset:
        raise HTTPException(status_code=400, detail="start_offset must be less than end_offset")

    # Verify document exists
    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    text_tag = DocumentTextTag(
        document_id=doc_uuid,
        page_number=page_number,
        start_offset=start_offset,
        end_offset=end_offset,
        tagged_text=tagged_text,
        tag=tag,
    )
    db.add(text_tag)
    await db.commit()
    await db.refresh(text_tag)

    return {
        "id": str(text_tag.id),
        "document_id": str(text_tag.document_id),
        "page_number": text_tag.page_number,
        "start_offset": text_tag.start_offset,
        "end_offset": text_tag.end_offset,
        "tagged_text": text_tag.tagged_text,
        "tag": text_tag.tag,
        "created_at": text_tag.created_at.isoformat() if text_tag.created_at else None,
    }


@app.delete("/documents/{document_id}/text-tags/{tag_id}")
async def delete_document_text_tag(
    document_id: str,
    tag_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a text tag."""
    from uuid import UUID
    from sqlalchemy import delete as sql_delete
    try:
        doc_uuid = UUID(document_id)
        tag_uuid = UUID(tag_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document or tag ID")
    result = await db.execute(
        select(DocumentTextTag).where(
            DocumentTextTag.id == tag_uuid,
            DocumentTextTag.document_id == doc_uuid,
        )
    )
    tag = result.scalar_one_or_none()
    if not tag:
        raise HTTPException(status_code=404, detail="Text tag not found")
    await db.execute(sql_delete(DocumentTextTag).where(DocumentTextTag.id == tag_uuid))
    await db.commit()
    return {"status": "deleted", "id": tag_id}


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


@app.post("/documents/{document_id}/chunking/kill-and-reset")
async def kill_and_reset_chunking(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Kill stuck chunking/embedding jobs for this document and set status to idle.
    Fails any pending/processing chunking_jobs and embedding_jobs, sets ChunkingResult metadata to idle.
    Use when a document is stuck in extraction or embedding."""
    from uuid import UUID

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Document not found")

    _chunking_cancel.add(document_id)

    msg = "Killed and reset via API"
    await db.execute(
        update(ChunkingJob)
        .where(ChunkingJob.document_id == doc_uuid, ChunkingJob.status.in_(["pending", "processing"]))
        .values(status="failed", worker_id=None, completed_at=None, error_message=msg[:2000])
    )
    await db.execute(
        update(EmbeddingJob)
        .where(EmbeddingJob.document_id == doc_uuid, EmbeddingJob.status.in_(["pending", "processing"]))
        .values(status="failed", worker_id=None, started_at=None, completed_at=None, error_message=msg[:2000])
    )
    cr_result = await db.execute(select(ChunkingResult).where(ChunkingResult.document_id == doc_uuid))
    cr = cr_result.scalar_one_or_none()
    if cr:
        meta = dict(cr.metadata_ or {})
        meta["status"] = "idle"
        cr.metadata_ = meta
    await db.commit()
    return {"status": "ok", "document_id": document_id, "message": "Chunking and embedding jobs killed; status set to idle."}


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
    generator_id: Optional[str] = None  # "A" (default) | "B"
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
    generator_id: str | None = None,
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
        if generator_id is None and body.generator_id is not None:
            generator_id = body.generator_id
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
    gen = (generator_id or "A").strip().upper() or "A"
    if gen not in ("A", "B"):
        gen = "A"

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    # Check if document exists
    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # When starting Path B: cancel any pending Path A jobs for this document so the worker doesn't run the wrong path
    if gen == "B":
        from sqlalchemy import or_, update
        where_a = or_(ChunkingJob.generator_id.is_(None), ChunkingJob.generator_id == "A")
        cancel_result = await db.execute(
            update(ChunkingJob)
            .where(
                ChunkingJob.document_id == doc_uuid,
                where_a,
                ChunkingJob.status == "pending",
            )
            .values(status="cancelled")
        )
        if cancel_result.rowcount and cancel_result.rowcount > 0:
            await db.commit()
            logger.info(f"Cancelled {cancel_result.rowcount} pending Path A job(s) for document {document_id} so Path B can run")

    # Check if there's already a pending or processing job for this document (same generator)
    if gen == "A":
        from sqlalchemy import or_
        where_chunk_gen = or_(ChunkingJob.generator_id.is_(None), ChunkingJob.generator_id == "A")
    else:
        where_chunk_gen = (ChunkingJob.generator_id == gen)
    existing_job = await db.execute(
        select(ChunkingJob).where(
            ChunkingJob.document_id == doc_uuid,
            where_chunk_gen,
            ChunkingJob.status.in_(["pending", "processing"])
        )
    )
    if existing_job.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Chunking job already queued or in progress for this document")

    # Create new job (run-configured: prompt_versions, llm_config_version optional)
    job = ChunkingJob(
        document_id=doc_uuid,
        generator_id=gen,
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
        "generator_id": gen,
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
    generator_id: Optional[str] = None  # "A" (default) | "B"


class ChunkingStatusUpdateBody(BaseModel):
    """Body for PATCH chunking status (e.g. mark stuck in_progress as completed)."""
    status: str  # "completed", "idle", "in_progress"


@app.post("/documents/{document_id}/chunking/restart")
async def restart_chunking(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    threshold: float | None = None,
    generator_id: str | None = None,
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
    gen = (generator_id or (body.generator_id if body else None) or "A").strip().upper() or "A"
    if gen not in ("A", "B"):
        gen = "A"

    # Path B restart = re-run deterministic chunking from scratch (no "resume" semantics yet)
    if gen == "B":
        return await start_chunking(
            document_id=document_id,
            db=db,
            threshold=th,
            critique_enabled=False,
            max_retries=0,
            extraction_enabled=False,
            generator_id="B",
            llm_config_version=None,
            body=None,
        )
    
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
    generator_id: str | None = None,
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

    gen = (generator_id or "A").strip().upper() or "A"
    if gen not in ("A", "B"):
        gen = "A"
    if gen == "A":
        from sqlalchemy import or_
        where_gen = or_(EmbeddingJob.generator_id.is_(None), EmbeddingJob.generator_id == "A")
    else:
        where_gen = (EmbeddingJob.generator_id == gen)
    existing = await db.execute(
        select(EmbeddingJob).where(
            EmbeddingJob.document_id == doc_uuid,
            where_gen,
            EmbeddingJob.status == "pending",
        ).limit(1)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Embedding job already queued for this document")

    job = EmbeddingJob(document_id=doc_uuid, generator_id=gen, status="pending")
    db.add(job)
    await db.commit()

    return {"status": "queued", "document_id": document_id, "job_id": str(job.id), "generator_id": gen}


@app.post("/documents/{document_id}/embedding/reset")
async def reset_embedding(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    generator_id: str | None = None,
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

    gen = (generator_id or "A").strip().upper() or "A"
    if gen not in ("A", "B"):
        gen = "A"
    if gen == "A":
        from sqlalchemy import or_
        where_job_gen = or_(EmbeddingJob.generator_id.is_(None), EmbeddingJob.generator_id == "A")
    else:
        where_job_gen = (EmbeddingJob.generator_id == gen)
    jobs_result = await db.execute(
        select(EmbeddingJob).where(
            EmbeddingJob.document_id == doc_uuid,
            where_job_gen,
            EmbeddingJob.status.in_(["pending", "processing"]),
        ).order_by(EmbeddingJob.created_at.desc())
    )
    jobs = jobs_result.scalars().all()
    if not jobs:
        raise HTTPException(status_code=404, detail="No pending or processing embedding job found for this document")

    # Clear partial embeddings for this generator only (back-compat: NULL treated as "A")
    if gen == "A":
        from sqlalchemy import or_
        where_gen = or_(ChunkEmbedding.generator_id.is_(None), ChunkEmbedding.generator_id == "A")
    else:
        where_gen = (ChunkEmbedding.generator_id == gen)
    await db.execute(delete(ChunkEmbedding).where(ChunkEmbedding.document_id == doc_uuid, where_gen))
    vector_store = get_vector_store()
    vector_store.delete_by_document(document_id)

    for job in jobs:
        job.status = "pending"
        job.worker_id = None
        job.started_at = None
        job.completed_at = None
        job.error_message = None

    await db.commit()
    return {"status": "reset", "document_id": document_id, "jobs_reset": len(jobs), "generator_id": gen}


class PublishBody(BaseModel):
    """Optional body for POST /documents/{id}/publish (audit)."""
    published_by: Optional[str] = None
    generator_id: Optional[str] = None  # "A" (default) | "B"


@app.post("/documents/{document_id}/publish")
async def publish_document_endpoint(
    document_id: str,
    body: Optional[PublishBody] = Body(None),
    db: AsyncSession = Depends(get_db),
):
    """Publish a document to rag_published_embeddings (dbt contract) for one generator (A/B)."""
    from uuid import UUID

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    gen = (body.generator_id if body else None) or None
    try:
        result = await publish_document(doc_uuid, db, generator_id=gen)
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
            "source_metadata": getattr(doc, "source_metadata", None),
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
    """Embed query, search vector store (top k), resolve source_id to text.

    Search order:
    - Uses configured vector store (Chroma or Vertex AI Vector Search).
    - Note: pgvector fallback removed - embeddings are in Vertex AI, not PostgreSQL.
    """
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

    # 2. Search vector store (Chroma or Vertex AI Vector Search)
    vector_store = get_vector_store()
    results = vector_store.search(query_embedding, k=k)
    # Note: pgvector fallback removed - embeddings are stored in Vertex AI Vector Search

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


# ---------------------------------------------------------------------------
# SSE stream – real-time push via PostgreSQL LISTEN / NOTIFY
# ---------------------------------------------------------------------------

@app.get("/documents/{document_id}/chunking/events/stream")
async def stream_chunking_events_sse(document_id: str):
    """Server-Sent Events stream of chunking/embedding events for a document.

    1. Establishes ``LISTEN chunking_events`` so no notifications are lost.
    2. Replays all existing events (chronological) from the DB.
    3. Then pushes new events in real-time, deduplicating against the replay set.
    4. Sends a keepalive comment every 15 s to prevent proxy timeouts.

    The frontend connects with ``new EventSource(url)`` which auto-reconnects.
    """
    import asyncpg as _asyncpg
    from uuid import UUID as _UUID
    from app.config import DATABASE_URL
    from app.database import AsyncSessionLocal

    try:
        doc_uuid = _UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    async def _sse_generator():
        """Async generator yielding SSE frames."""
        queue: asyncio.Queue[str] = asyncio.Queue()
        seen_ids: set[str] = set()  # track replayed event IDs to avoid duplicates
        conn: _asyncpg.Connection | None = None

        dsn = DATABASE_URL.replace("+asyncpg", "")  # strip SQLAlchemy dialect prefix

        try:
            # --- Phase 1: Set up LISTEN first so notifications queue up ---
            conn = await _asyncpg.connect(dsn)

            def _on_notify(
                pg_conn: _asyncpg.Connection,
                pid: int,
                channel: str,
                payload: str,
            ) -> None:
                queue.put_nowait(payload)

            await conn.add_listener("chunking_events", _on_notify)

            # --- Phase 2: Replay existing events from DB ---
            async with AsyncSessionLocal() as db:
                q = (
                    select(ChunkingEvent)
                    .where(ChunkingEvent.document_id == doc_uuid)
                    .order_by(ChunkingEvent.created_at, ChunkingEvent.id)
                )
                result = await db.execute(q)
                for ev in result.scalars():
                    ev_id = str(ev.id)
                    seen_ids.add(ev_id)
                    yield (
                        f"id: {ev_id}\n"
                        f"data: {json.dumps({'event': ev.event_type, 'data': ev.event_data})}\n\n"
                    )

            # --- Phase 3: Stream live events, skipping any already replayed ---
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15)
                    try:
                        msg = json.loads(payload)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    # Filter: only forward events for *this* document
                    if msg.get("document_id") != document_id:
                        continue
                    ev_id = msg.get("id", "")
                    if ev_id in seen_ids:
                        continue  # already sent during replay
                    yield (
                        f"id: {ev_id}\n"
                        f"data: {json.dumps({'event': msg.get('event_type', 'message'), 'data': msg.get('event_data', {})})}\n\n"
                    )
                except asyncio.TimeoutError:
                    # Keepalive: SSE comment line keeps the connection alive
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass  # client disconnected
        except Exception as exc:
            logger.warning("[sse] stream error for doc %s: %s", document_id, exc)
        finally:
            if conn is not None:
                try:
                    await conn.remove_listener("chunking_events", _on_notify)
                except Exception:
                    pass
                try:
                    await conn.close()
                except Exception:
                    pass

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


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
            text("DELETE FROM embedding_jobs WHERE document_id = :doc_id"),
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
            text("DELETE FROM chunk_embeddings WHERE document_id = :doc_id"),
            {"doc_id": doc_uuid}
        )
        await db.execute(
            text("DELETE FROM rag_published_embeddings WHERE document_id = :doc_id"),
            {"doc_id": doc_uuid}
        )
        await db.execute(
            text("DELETE FROM publish_events WHERE document_id = :doc_id"),
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
        try:
            from app.services.vector_store import get_vector_store
            get_vector_store().delete_by_document(document_id)
        except Exception as vec_err:
            logger.warning("Vector store delete_by_document failed (non-fatal): %s", vec_err)
        
        return {
            "status": "success",
            "message": f"Document '{document.filename}' and all related records deleted successfully",
            "document_id": document_id
        }
    except Exception as e:
        await db.rollback()
        logger.error(f"Error deleting document cascade: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/{document_id}/retag")
async def retag_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Re-apply the latest lexicon tags to a single document by queueing a Path B job.
    The document must have completed extraction already.
    Intended to be called by the Lexicon module or from the RAG UI.
    """
    from uuid import UUID

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if document.status != "completed":
        raise HTTPException(status_code=400, detail="Document extraction must be completed before retagging")

    job_info = await _enqueue_retag_job(db, doc_uuid, document_id)
    if job_info is None:
        return {"status": "already_queued", "document_id": document_id, "message": "A Path B job is already pending or processing for this document"}

    await db.commit()
    logger.info("Retag queued for document %s  job=%s", document_id, job_info["job_id"])
    return {"status": "queued", **job_info}


# Serve frontend static files (for Cloud Run / single-container deployment)
# Must be last so API routes take precedence
_frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="static")
