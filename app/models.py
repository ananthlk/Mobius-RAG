from sqlalchemy import Column, String, DateTime, Integer, Text, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime
import uuid
from app.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), unique=True, nullable=False)
    file_path = Column(String(500), nullable=False)  # GCS path
    payer = Column(String(100))
    state = Column(String(2))
    program = Column(String(100))
    status = Column(String(20), default="uploaded", nullable=False)  # uploaded, extracting, completed, failed, completed_with_errors
    has_errors = Column(String(10), default="false", nullable=False)  # 'true', 'false'
    error_count = Column(Integer, default=0, nullable=False)
    critical_error_count = Column(Integer, default=0, nullable=False)
    review_status = Column(String(20), default="pending", nullable=False)  # pending, approved, rejected, reprocessing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DocumentPage(Base):
    __tablename__ = "document_pages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    text = Column(Text, nullable=True)  # Raw extracted text
    text_markdown = Column(Text, nullable=True)  # Structured markdown for reader
    extraction_status = Column(String(20), default="success", nullable=False)  # success, failed, empty
    extraction_error = Column(Text, nullable=True)  # Error message if extraction failed
    text_length = Column(Integer, default=0, nullable=False)  # Length of extracted text
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class HierarchicalChunk(Base):
    """Hierarchical chunks for atomic fact extraction - paragraph-level with structure."""
    __tablename__ = "hierarchical_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    paragraph_index = Column(Integer, nullable=False)
    
    # Document hierarchy
    section_path = Column(String(500), nullable=True)  # e.g., "Section 3.2"
    chapter_path = Column(String(500), nullable=True)  # e.g., "Chapter 5"
    
    text = Column(Text, nullable=False)
    text_length = Column(Integer, nullable=False)
    
    # Character offset in raw page text where this paragraph body starts (for LLM source highlighting)
    start_offset_in_page = Column(Integer, nullable=True)
    
    # LLM-extracted summary
    summary = Column(Text, nullable=True)  # Summary of the paragraph
    
    # LLM classification (prescriptive questions - more strict)
    is_eligibility_related = Column(String(10), nullable=True)  # 'true', 'false', NULL if pending
    classification_confidence = Column(String(10), nullable=True)  # 0.0-1.0
    classification_reasoning = Column(Text, nullable=True)
    extraction_status = Column(String(20), default="pending", nullable=False)  # pending, extracted, failed
    
    # QA/Critique agent results
    critique_status = Column(String(20), default="pending", nullable=False)  # pending, passed, failed, retrying
    critique_feedback = Column(Text, nullable=True)  # Feedback from critique agent
    retry_count = Column(Integer, default=0, nullable=False)  # Number of retries (max 2)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ChunkingResult(Base):
    """Persisted chunking/extraction output per document. Stored in PostgreSQL."""
    __tablename__ = "chunking_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, unique=True)
    metadata_ = Column("metadata", JSONB, nullable=False, default=dict)
    results = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class ChunkingJob(Base):
    """Job queue for chunking tasks - processed by separate worker process."""
    __tablename__ = "chunking_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    status = Column(String(20), default="pending", nullable=False)  # pending, processing, completed, failed, cancelled
    threshold = Column(String(10), nullable=False)  # Store as string to avoid float precision issues
    worker_id = Column(String(100), nullable=True)  # ID of worker processing this job
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Run-configured: immutable refs to prompt set and LLM config (append-only)
    prompt_versions = Column(JSONB, nullable=True)  # e.g. {"extraction": "v1", "extraction_retry": "v1", "critique": "v1"}
    llm_config_version = Column(String(100), nullable=True)  # e.g. "default" or "v1" or content SHA

    # Run mode: critique on/off, retries on/off
    critique_enabled = Column(String(10), nullable=True)  # 'true', 'false' (store as string for consistency)
    max_retries = Column(Integer, nullable=True)  # 0 = no retry; 1, 2, etc.


class ChunkingEvent(Base):
    """Events generated during chunking - stored in database for SSE streaming."""
    __tablename__ = "chunking_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    event_type = Column(String(50), nullable=False)  # paragraph_start, llm_stream, paragraph_complete, etc.
    event_data = Column(JSONB, nullable=False)  # Event payload
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# Category names for relevance scores (must match extraction prompt)
CATEGORY_NAMES = (
    "contacting_marketing_members",
    "member_eligibility_molina",
    "benefit_access_limitations",
    "prior_authorization_required",
    "claims_authorization_submissions",
    "compliant_claim_requirements",
    "claim_disputes",
    "credentialing",
    "claim_submission_important",
    "coordination_of_benefits",
    "other_important",
)


def category_scores_dict_to_columns(category_scores):
    """Convert category_scores dict from extraction JSON into kwargs for ExtractedFact columns."""
    if not category_scores or not isinstance(category_scores, dict):
        return {}
    kwargs = {}
    for cat in CATEGORY_NAMES:
        val = category_scores.get(cat)
        if isinstance(val, dict):
            s = val.get("score")
            d = val.get("direction")
            kwargs[f"{cat}_score"] = _safe_float(s)
            kwargs[f"{cat}_direction"] = _safe_float(d)
        else:
            kwargs[f"{cat}_score"] = None
            kwargs[f"{cat}_direction"] = None
    return kwargs


def _safe_float(x):
    """Return float or None; handle nan/inf."""
    if x is None:
        return None
    try:
        f = float(x)
        import math
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def fact_to_category_scores_dict(fact):
    """Build category_scores dict from an ExtractedFact instance (for API response)."""
    out = {}
    for cat in CATEGORY_NAMES:
        score = getattr(fact, f"{cat}_score", None)
        direction = getattr(fact, f"{cat}_direction", None)
        out[cat] = {"score": score, "direction": direction}
    return out


class ExtractedFact(Base):
    """Facts extracted from hierarchical chunks with prescriptive question answers."""
    __tablename__ = "extracted_facts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    hierarchical_chunk_id = Column(UUID(as_uuid=True), ForeignKey("hierarchical_chunks.id"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)  # Denormalized for queries
    
    # The extracted fact statement
    fact_text = Column(Text, nullable=False)
    fact_type = Column(String(255), nullable=True)  # who_eligible|verification_method|... (pipe-separated allowed)
    
    # Prescriptive question answers
    who_eligible = Column(Text, nullable=True)  # Answer to WHO question
    how_verified = Column(Text, nullable=True)  # Answer to HOW question
    conflict_resolution = Column(Text, nullable=True)  # Answer to WHAT/conflict question
    when_applies = Column(Text, nullable=True)  # Answer to WHEN question
    limitations = Column(Text, nullable=True)  # Answer to LIMITATIONS question
    
    # Verification status (auto-qualified)
    is_verified = Column(String(10), nullable=True)  # 'true' if explicitly stated, 'false' if inferred
    
    # Classification
    is_eligibility_related = Column(String(10), nullable=True)  # true if ANY question answered
    is_pertinent_to_claims_or_members = Column(String(10), nullable=True)  # 'true' if pertinent to submitting claims or working with members, 'false' otherwise
    confidence = Column(String(10), nullable=True)  # 0.0-1.0
    
    # Category relevance: one score and one direction per category (0.0-1.0 or null)
    contacting_marketing_members_score = Column(Float, nullable=True)
    contacting_marketing_members_direction = Column(Float, nullable=True)
    member_eligibility_molina_score = Column(Float, nullable=True)
    member_eligibility_molina_direction = Column(Float, nullable=True)
    benefit_access_limitations_score = Column(Float, nullable=True)
    benefit_access_limitations_direction = Column(Float, nullable=True)
    prior_authorization_required_score = Column(Float, nullable=True)
    prior_authorization_required_direction = Column(Float, nullable=True)
    claims_authorization_submissions_score = Column(Float, nullable=True)
    claims_authorization_submissions_direction = Column(Float, nullable=True)
    compliant_claim_requirements_score = Column(Float, nullable=True)
    compliant_claim_requirements_direction = Column(Float, nullable=True)
    claim_disputes_score = Column(Float, nullable=True)
    claim_disputes_direction = Column(Float, nullable=True)
    credentialing_score = Column(Float, nullable=True)
    credentialing_direction = Column(Float, nullable=True)
    claim_submission_important_score = Column(Float, nullable=True)
    claim_submission_important_direction = Column(Float, nullable=True)
    coordination_of_benefits_score = Column(Float, nullable=True)
    coordination_of_benefits_direction = Column(Float, nullable=True)
    other_important_score = Column(Float, nullable=True)
    other_important_direction = Column(Float, nullable=True)
    
    # Reader-added facts: source page and highlight range (character offsets in raw page text)
    page_number = Column(Integer, nullable=True)  # Source page in reader; Review can show "From page N"
    start_offset = Column(Integer, nullable=True)  # Character start in raw page text for highlight
    end_offset = Column(Integer, nullable=True)   # Character end in raw page text for highlight

    # Verification (AI or human): who verified, when, status for facts sheet
    verified_by = Column(String(20), nullable=True)  # 'ai' | 'human' | null
    verified_at = Column(DateTime, nullable=True)   # When verified (AI or human)
    verification_status = Column(String(20), nullable=True)  # pending | approved | rejected | deleted

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ProcessingError(Base):
    """Errors encountered during document processing - tracked for review and approval."""
    __tablename__ = "processing_errors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    paragraph_id = Column(String(100), nullable=True)  # e.g., "1_0" for page 1, paragraph 0
    
    error_type = Column(String(50), nullable=False)  # llm_failure, json_parse_error, database_error, stream_error, persistence_error, other
    severity = Column(String(20), nullable=False)  # critical, warning, info
    error_message = Column(Text, nullable=False)
    error_details = Column(JSONB, nullable=True)  # stack trace, context, raw data
    stage = Column(String(50), nullable=False)  # extraction, critique, persistence, other
    
    resolved = Column(String(10), default="false", nullable=False)  # 'true', 'false'
    resolution = Column(String(20), nullable=True)  # approved, rejected, reprocess
    resolved_by = Column(String(255), nullable=True)  # reviewer identifier
    resolved_at = Column(DateTime, nullable=True)
    resolution_notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
