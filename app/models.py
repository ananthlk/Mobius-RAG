from sqlalchemy import Boolean, Column, String, DateTime, Integer, Text, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB
# NOTE: Embeddings are stored as JSONB arrays in this DB (for Vertex sync).
# pgvector may be installed, but the current schema uses JSONB for embedding columns.
from datetime import datetime
import uuid
from app.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=True)  # User-friendly name; when set, UI shows this instead of filename
    file_hash = Column(String(64), unique=True, nullable=False)
    file_path = Column(String(500), nullable=False)  # GCS path
    payer = Column(String(100))
    state = Column(String(2))
    program = Column(String(100))
    authority_level = Column(String(100), nullable=True)
    effective_date = Column(String(20), nullable=True)   # ISO date or free text, e.g. 2024-01-15
    termination_date = Column(String(20), nullable=True)  # ISO date or free text
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

    # Run mode: extraction on/off (Path A extraction may be disabled for deterministic runs)
    extraction_enabled = Column(String(10), nullable=True)  # 'true', 'false'

    # A/B generator (NULL treated as "A" for back-compat)
    generator_id = Column(String(10), nullable=True)  # "A" | "B"

    # Resolved run config snapshot (set once at job start; never mutated).
    chunking_config_snapshot = Column(JSONB, nullable=True)


class ChunkingEvent(Base):
    """Events generated during chunking - stored in database for SSE streaming."""
    __tablename__ = "chunking_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    event_type = Column(String(50), nullable=False)  # paragraph_start, llm_stream, paragraph_complete, etc.
    event_data = Column(JSONB, nullable=False)  # Event payload
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class LlmConfig(Base):
    """LLM provider configs stored in DB; name matches version/ref used by jobs (e.g. default, production)."""
    __tablename__ = "llm_configs"

    name = Column(String(100), primary_key=True)  # e.g. default, production, openai
    provider = Column(String(50), nullable=False)  # ollama, vertex, openai
    model = Column(String(200), nullable=True)
    version_label = Column(String(100), nullable=True)  # optional display version
    options = Column(JSONB, nullable=True, default=dict)  # temperature, num_predict, etc.
    ollama = Column(JSONB, nullable=True, default=dict)  # base_url, etc.
    vertex = Column(JSONB, nullable=True, default=dict)  # project_id, location, etc.
    openai = Column(JSONB, nullable=True, default=dict)  # api_key, base_url, etc.
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class EmbeddingJob(Base):
    """Job queue for embedding tasks - processed by embedding worker after chunking completes."""
    __tablename__ = "embedding_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    status = Column(String(20), default="pending", nullable=False)  # pending, processing, completed, failed
    worker_id = Column(String(100), nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    embedding_config_version = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # A/B generator (NULL treated as "A" for back-compat)
    generator_id = Column(String(10), nullable=True)  # "A" | "B"


class ChunkEmbedding(Base):
    """Embeddings for hierarchical chunks and facts. No text column; link back via source_type + source_id."""
    __tablename__ = "chunk_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    source_type = Column(String(20), nullable=False)  # 'hierarchical' | 'fact'
    source_id = Column(UUID(as_uuid=True), nullable=False)  # hierarchical_chunks.id or extracted_facts.id
    embedding = Column(JSONB, nullable=False)  # list[float] length=1536
    model = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # A/B generator (NULL treated as "A" for back-compat)
    generator_id = Column(String(10), nullable=True)  # "A" | "B"


class PublishEvent(Base):
    """Audit log: one row per Publish action (who published what when)."""
    __tablename__ = "publish_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    published_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    published_by = Column(String(255), nullable=True)
    rows_written = Column(Integer, default=0, nullable=False)
    notes = Column(Text, nullable=True)
    verification_passed = Column(Boolean, nullable=True)  # True/False after integrity check; None for legacy rows
    verification_message = Column(Text, nullable=True)  # Error message if verification failed


class RagPublishedEmbedding(Base):
    """dbt contract table: one row per published embedding (written on user Publish)."""
    __tablename__ = "rag_published_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False)
    source_type = Column(String(20), nullable=False)
    source_id = Column(UUID(as_uuid=True), nullable=False)
    embedding = Column(JSONB, nullable=False)  # list[float] length=1536
    model = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    text = Column(Text, default="", nullable=False)
    page_number = Column(Integer, default=0, nullable=False)
    paragraph_index = Column(Integer, default=0, nullable=False)
    section_path = Column(String(500), default="", nullable=False)
    chapter_path = Column(String(500), default="", nullable=False)
    summary = Column(Text, default="", nullable=False)
    document_filename = Column(String(255), default="", nullable=False)
    document_display_name = Column(String(255), default="", nullable=False)
    document_authority_level = Column(String(100), default="", nullable=False)
    document_effective_date = Column(String(20), default="", nullable=False)
    document_termination_date = Column(String(20), default="", nullable=False)
    document_payer = Column(String(100), default="", nullable=False)
    document_state = Column(String(2), default="", nullable=False)
    document_program = Column(String(100), default="", nullable=False)
    document_status = Column(String(20), default="", nullable=False)
    document_created_at = Column(DateTime, nullable=True)
    document_review_status = Column(String(20), default="", nullable=False)
    document_reviewed_at = Column(DateTime, nullable=True)
    document_reviewed_by = Column(String(255), nullable=True)
    content_sha = Column(String(64), default="", nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    source_verification_status = Column(String(20), default="", nullable=False)


class EmbeddableUnit(Base):
    """Consolidated table of texts to embed.

    Both Path A and Path B write rows here after persisting their own artefacts.
    The embedding worker reads solely from this table, giving a single contract
    regardless of which chunking path produced the data.
    """
    __tablename__ = "embeddable_units"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    generator_id = Column(String(10), nullable=True)  # "A" | "B"
    source_type = Column(String(30), nullable=False)   # "chunk", "fact", "policy_line"
    source_id = Column(UUID(as_uuid=True), nullable=False)  # PK of the source row

    text = Column(Text, nullable=False)                # The text to embed
    page_number = Column(Integer, nullable=True)
    paragraph_index = Column(Integer, nullable=True)
    section_path = Column(String(500), nullable=True)

    metadata_ = Column("metadata", JSONB, nullable=True, default=dict)  # Arbitrary metadata (tags, scores, etc.)

    # Lifecycle
    status = Column(String(20), default="pending", nullable=False)  # pending, embedded, failed
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


# --------------------------------------------------------------------------------------
# Path B (deterministic policy chunking + lexicon candidates)
#
# Note: Lexicon maintenance has moved to the QA service, but the RAG API still exposes
# read-only endpoints for Path B artifacts. These models are required for API import to load.
# If the underlying tables are not present in the current DB, those endpoints will error
# at query time, but the API server can still start (and Path A flows continue to work).
# --------------------------------------------------------------------------------------


class PolicyParagraph(Base):
    """Path B paragraph-level units (structured, taggable)."""
    __tablename__ = "policy_paragraphs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    order_index = Column(Integer, nullable=False)

    paragraph_type = Column(String(50), nullable=True)  # e.g. heading|body|table|list
    heading_path = Column(JSONB, nullable=True)  # e.g. ["Section 1", "Subsection"] or null
    text = Column(Text, nullable=False, default="")

    # Tags (Path B)
    p_tags = Column(JSONB, nullable=True)  # prescriptive tags
    d_tags = Column(JSONB, nullable=True)  # descriptive tags
    j_tags = Column(JSONB, nullable=True)  # jurisdiction tags
    inferred_d_tags = Column(JSONB, nullable=True)
    inferred_j_tags = Column(JSONB, nullable=True)
    conflict_flags = Column(JSONB, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class PolicyLine(Base):
    """Path B line-level units (may be atomic)."""
    __tablename__ = "policy_lines"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    paragraph_id = Column(UUID(as_uuid=True), ForeignKey("policy_paragraphs.id"), nullable=False)
    order_index = Column(Integer, nullable=False)

    parent_line_id = Column(UUID(as_uuid=True), nullable=True)
    heading_path = Column(JSONB, nullable=True)  # e.g. ["Section 1"] or null
    line_type = Column(String(50), nullable=True)  # e.g. bullet|sentence|table_row
    text = Column(Text, nullable=False, default="")

    is_atomic = Column(Boolean, default=False, nullable=False)
    non_atomic_reason = Column(Text, nullable=True)

    # Tags / features
    p_tags = Column(JSONB, nullable=True)
    d_tags = Column(JSONB, nullable=True)
    j_tags = Column(JSONB, nullable=True)
    inferred_d_tags = Column(JSONB, nullable=True)
    inferred_j_tags = Column(JSONB, nullable=True)
    conflict_flags = Column(JSONB, nullable=True)
    extracted_fields = Column(JSONB, nullable=True)

    # Optional offsets in raw page text
    start_offset = Column(Integer, nullable=True)
    end_offset = Column(Integer, nullable=True)
    offset_match_quality = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    # Note: policy_lines table has no updated_at column; do not add it to the model unless the DB is migrated.


class DocumentTags(Base):
    """Document-level tag aggregates (forward propagation from paragraph tags).

    One row per document.  Updated at the end of a Path B run by
    ``aggregate_paragraph_tags_to_document()``.
    """
    __tablename__ = "document_tags"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, unique=True)
    p_tags = Column(JSONB, nullable=True)  # { tag_code: { count, lines_total, avg_weight, ... } }
    d_tags = Column(JSONB, nullable=True)
    j_tags = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class DocumentTextTag(Base):
    """User-applied text-range tags (e.g. category labels on selected text in the reader).

    Unlike DocumentTags (document-level aggregates), each row here represents
    a single highlighted text range on a specific page.
    """
    __tablename__ = "document_text_tags"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True)
    page_number = Column(Integer, nullable=False)
    start_offset = Column(Integer, nullable=False)
    end_offset = Column(Integer, nullable=False)
    tagged_text = Column(Text, nullable=False)       # The selected text
    tag = Column(String(100), nullable=False)         # Category key, e.g. "prior_authorization_required"
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class PolicyLexiconCandidate(Base):
    """Path B lexicon candidates generated from policy artifacts."""
    __tablename__ = "policy_lexicon_candidates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)

    run_id = Column(UUID(as_uuid=True), nullable=True)
    candidate_type = Column(String(20), nullable=False)  # p|d|j|alias
    normalized = Column(String(500), nullable=False, default="")
    proposed_tag = Column(String(500), nullable=True)
    confidence = Column(Float, nullable=True)
    examples = Column(JSONB, nullable=True)
    source = Column(String(50), nullable=True)
    occurrences = Column(Integer, nullable=True)

    state = Column(String(20), nullable=False, default="proposed")  # proposed|approved|rejected|flagged
    reviewer = Column(String(255), nullable=True)
    reviewer_notes = Column(Text, nullable=True)

    # LLM triage columns (populated by llm_triage_candidates after extraction)
    llm_verdict = Column(String(20), nullable=True)         # new_tag | alias | reject
    llm_confidence = Column(Float, nullable=True)            # 0.0 - 1.0
    llm_reason = Column(Text, nullable=True)                 # one-line explanation
    llm_suggested_parent = Column(String(500), nullable=True)
    llm_suggested_code = Column(String(500), nullable=True)
    llm_suggested_kind = Column(String(10), nullable=True)   # p | d | j

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    # Note: policy_lexicon_candidates table has no updated_at column; do not add unless the DB is migrated.


class PolicyLexiconCandidateCatalog(Base):
    """Global catalog of candidate decisions (used to suppress already-rejected candidates, analytics)."""
    __tablename__ = "policy_lexicon_candidate_catalog"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    candidate_type = Column(String(20), nullable=False)  # p|d|j|alias
    normalized_key = Column(String(300), nullable=True)  # normalized key for upsert
    normalized = Column(String(500), nullable=False, default="")
    proposed_tag_key = Column(String(300), nullable=True)  # proposed_tag key for upsert
    proposed_tag = Column(String(500), nullable=True)

    state = Column(String(20), nullable=False, default="rejected")  # rejected|approved|flagged
    reviewer = Column(String(255), nullable=True)
    reviewer_notes = Column(Text, nullable=True)
    decided_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
