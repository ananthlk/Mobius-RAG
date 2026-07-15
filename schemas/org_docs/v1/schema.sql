-- org_docs schema v1
-- Applied per-org into a dedicated schema within the mobius_org_docs database.
-- The db-agent creates the schema (CREATE SCHEMA IF NOT EXISTS <slug>) then
-- runs this file with SET search_path = <slug> so every object lands in that namespace.
-- RAG queries with schema-qualified references (e.g. "aetna_fl".org_chunks) —
-- a global-corpus query structurally cannot return an org-private row.
--
-- Requires: pgvector extension enabled on the host DB instance.
-- Embedding model: text-embedding-004 (1536-dim). Dim pinned here; version tracked via schema_version.

CREATE EXTENSION IF NOT EXISTS vector;

-- ── Schema-version sentinel ───────────────────────────────────────────────────
-- db-agent reads this to confirm which DDL version is live before applying newer ones.
CREATE TABLE IF NOT EXISTS org_schema_version (
    version       INTEGER     NOT NULL,
    applied_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    notes         TEXT
);
INSERT INTO org_schema_version (version, notes)
SELECT 1, 'initial: org_documents + org_chunks + HNSW cosine index'
WHERE NOT EXISTS (SELECT 1 FROM org_schema_version WHERE version = 1);

-- ── Documents ─────────────────────────────────────────────────────────────────
-- One row per uploaded file. RAG writes on ingest; retrieval joins for metadata.
CREATE TABLE IF NOT EXISTS org_documents (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        TEXT        NOT NULL,
    display_name    TEXT        NOT NULL DEFAULT '',
    uploaded_by     TEXT        NOT NULL,          -- user identifier (email or user_id)
    visibility      TEXT        NOT NULL DEFAULT 'org'
                                CHECK (visibility IN ('org', 'member')),
    status          TEXT        NOT NULL DEFAULT 'processing'
                                CHECK (status IN ('processing', 'ready', 'error', 'needs_ocr')),
    content_hash    TEXT        NOT NULL DEFAULT '',  -- SHA-256 of raw file bytes (dedup gate)
    error_detail    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Chunks ────────────────────────────────────────────────────────────────────
-- One row per chunk. Embedding stored as native pgvector column (not JSONB).
-- Inherits visibility from parent document; retrieval always filters on org_documents.visibility.
CREATE TABLE IF NOT EXISTS org_chunks (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID        NOT NULL REFERENCES org_documents(id) ON DELETE CASCADE,
    text            TEXT        NOT NULL,
    embedding       vector(1536),                  -- NULL until embedding worker processes
    page_number     INTEGER     NOT NULL DEFAULT 0,
    paragraph_index INTEGER     NOT NULL DEFAULT 0,
    section_path    TEXT        NOT NULL DEFAULT '',
    content_sha     TEXT        NOT NULL DEFAULT '', -- SHA-256 of text (dedup + neighbor lookup)
    d_tags          JSONB,                          -- topic tags (reuses RAG tagging pipeline)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Indexes ───────────────────────────────────────────────────────────────────
-- HNSW cosine index — matches the shared corpus (rag_published_embeddings) tuning.
-- m=16 ef_construction=64 is the fleet default; adjust after load profiling.
CREATE INDEX IF NOT EXISTS org_chunks_embedding_hnsw
    ON org_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- BM25 / full-text fallback
CREATE INDEX IF NOT EXISTS org_chunks_text_fts
    ON org_chunks
    USING gin (to_tsvector('english', text));

-- Document lookup by status (embedding worker polls)
CREATE INDEX IF NOT EXISTS org_documents_status_idx
    ON org_documents (status)
    WHERE status IN ('processing', 'needs_ocr');

-- content_sha uniqueness within an org (prevents re-chunk of identical text)
CREATE UNIQUE INDEX IF NOT EXISTS org_chunks_content_sha_udx
    ON org_chunks (content_sha)
    WHERE content_sha <> '';
