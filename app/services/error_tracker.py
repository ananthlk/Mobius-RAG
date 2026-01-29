"""Error tracking service for logging and classifying processing errors."""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models import ProcessingError, Document
from datetime import datetime, timezone
import uuid
import logging

logger = logging.getLogger(__name__)


async def log_error(
    db: AsyncSession,
    document_id: str,
    error_type: str,
    error_message: str,
    severity: str = "warning",
    stage: str = "other",
    paragraph_id: str | None = None,
    error_details: dict | None = None
) -> ProcessingError:
    """
    Log an error to the processing_errors table.
    
    Args:
        db: Database session
        document_id: UUID of the document
        error_type: Type of error (llm_failure, json_parse_error, database_error, stream_error, persistence_error, other)
        error_message: Human-readable error message
        severity: Error severity (critical, warning, info)
        stage: Processing stage (extraction, critique, persistence, other)
        paragraph_id: Optional paragraph identifier (e.g., "1_0")
        error_details: Optional JSONB dict with additional context (stack trace, raw data, etc.)
    
    Returns:
        The created ProcessingError record
    """
    from uuid import UUID
    
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        logger.error(f"Invalid document_id format: {document_id}")
        raise ValueError(f"Invalid document_id format: {document_id}")
    
    error = ProcessingError(
        id=uuid.uuid4(),
        document_id=doc_uuid,
        paragraph_id=paragraph_id,
        error_type=error_type,
        severity=severity,
        error_message=error_message,
        error_details=error_details or {},
        stage=stage,
        resolved="false",
        created_at=datetime.now(timezone.utc).replace(tzinfo=None)
    )
    
    db.add(error)
    
    # Update document error counts
    try:
        doc_result = await db.execute(select(Document).where(Document.id == doc_uuid))
        doc = doc_result.scalar_one_or_none()
        if doc:
            doc.error_count = (doc.error_count or 0) + 1
            if severity == "critical":
                doc.critical_error_count = (doc.critical_error_count or 0) + 1
                doc.has_errors = "true"
            elif doc.has_errors != "true":
                # Set has_errors to true if we have any errors
                doc.has_errors = "true"
    except Exception as e:
        logger.error(f"Failed to update document error counts: {e}", exc_info=True)
        # Continue - error logging shouldn't fail the process
    
    try:
        await db.commit()
        await db.refresh(error)
        logger.info(f"Logged {severity} error for document {document_id}: {error_type} - {error_message[:100]}")
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to commit error log: {e}", exc_info=True)
        raise
    
    return error


def classify_error(error_type: str, error: Exception, recovered: bool = False) -> tuple[str, str]:
    """
    Classify an error by type and determine severity.
    
    Args:
        error_type: Type of error
        error: The exception object
        recovered: Whether the error was recovered from
    
    Returns:
        Tuple of (severity, stage)
    """
    severity_map = {
        "llm_failure": "critical",
        "json_parse_error": "warning" if recovered else "critical",
        "database_error": "critical",
        "stream_error": "critical",
        "persistence_error": "critical",
        "other": "warning"
    }
    
    stage_map = {
        "llm_failure": "extraction",
        "json_parse_error": "extraction",
        "database_error": "persistence",
        "stream_error": "extraction",
        "persistence_error": "persistence",
        "other": "other"
    }
    
    severity = severity_map.get(error_type, "warning")
    stage = stage_map.get(error_type, "other")
    
    return severity, stage
