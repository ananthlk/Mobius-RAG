# Contract: Mobius-RAG and dbt / Data Transformation

## 1. Parties and purpose

- **Publisher:** Mobius-RAG. RAG curates documents, reviews authenticity, and exposes a published output when the user explicitly publishes.
- **Consumer:** dbt-enabled data transformation layer and ingestion. Ingestion reads RAG’s published table and replicates it to BigQuery; dbt sources from BigQuery and produces marts.

RAG maintains one or more **published output table(s)** in PostgreSQL. The only surface the consumer reads is **`rag_published_embeddings`**. The schema of this table is the contract below; ingestion replicates it to BigQuery (same column names and types).

---

## 2. Publish flow

- **Explicit user Publish.** The user reviews facts in the UI (Review Facts tab) and, when satisfied, goes to Document Status and clicks **Publish** for a document.
- **Backend:** `POST /documents/{document_id}/publish` (optional body: `{ "published_by": "user-id" }`). RAG then:
  1. Loads the document and all `chunk_embeddings` for that document.
  2. For each embedding, joins to `hierarchical_chunks` or `extracted_facts` (by `source_type` + `source_id`) and to `documents` to build one row per contract schema (including `content_sha`, `updated_at`).
  3. Deletes all existing rows in `rag_published_embeddings` for that `document_id`.
  4. Inserts the new set of rows.
  5. Inserts one row into `publish_events` (audit: document_id, published_at, published_by, rows_written).
- **Entire document only.** No partial publish; when the user publishes, the whole document’s embeddings (chunks + facts) are written. We stand behind everything in the document at that point.

---

## 3. Table: rag_published_embeddings (dbt contract)

This is the **only** table the consumer reads. One row per published embedding (chunk or fact). All columns below are part of the contract.

### Identity and embedding

| Column        | Type         | Description                                                                 |
|---------------|--------------|-----------------------------------------------------------------------------|
| id            | UUID         | Primary key (e.g. chunk_embedding.id).                                      |
| document_id   | UUID         | Document this row belongs to.                                               |
| source_type   | VARCHAR(20)  | `'hierarchical'` or `'fact'`.                                               |
| source_id     | UUID         | hierarchical_chunks.id or extracted_facts.id.                               |
| embedding     | vector(1536) | The embedding vector (pgvector).                                            |
| model         | VARCHAR(100) | Embedding model (e.g. text-embedding-3-small). Empty string if unknown.     |
| created_at    | TIMESTAMP    | When the embedding was created (from chunk_embeddings.created_at).          |

### Content and structure

| Column           | Type         | Description                                                                 |
|------------------|--------------|-----------------------------------------------------------------------------|
| text             | TEXT         | The text that was embedded (chunk or fact text). Never null; empty string if none. |
| page_number      | INTEGER      | Source page (0 when not applicable).                                        |
| paragraph_index  | INTEGER      | Paragraph index within page (0 when not applicable).                        |
| section_path     | VARCHAR(500) | e.g. "Section 3.2". Empty string when not applicable.                        |
| chapter_path     | VARCHAR(500) | e.g. "Chapter 5". Empty string when not applicable.                         |
| summary          | TEXT         | Chunk summary (hierarchical); for facts use empty string or fact_text.      |

### Document metadata

| Column                     | Type      | Description                                    |
|----------------------------|-----------|------------------------------------------------|
| document_filename          | VARCHAR(255)  | documents.filename.                            |
| document_display_name      | VARCHAR(255)  | documents.display_name. Empty string if null.  |
| document_authority_level   | VARCHAR(100) | documents.authority_level. Empty string if null.|
| document_effective_date    | VARCHAR(20)   | documents.effective_date. Empty string if null.|
| document_termination_date  | VARCHAR(20)   | documents.termination_date. Empty string if null. |
| document_payer             | VARCHAR(100) | documents.payer. Empty string if null.         |
| document_state             | VARCHAR(2)   | documents.state. Empty string if null.         |
| document_program           | VARCHAR(100) | documents.program. Empty string if null.       |
| document_status            | VARCHAR(20)  | documents.status.                              |
| document_created_at        | TIMESTAMP    | documents.created_at.                          |

### Audit

| Column                    | Type        | Description                                          |
|---------------------------|-------------|------------------------------------------------------|
| document_review_status    | VARCHAR(20) | documents.review_status.                             |
| document_reviewed_at      | TIMESTAMP   | When document was last reviewed (null if not tracked).|
| document_reviewed_by      | VARCHAR(255)| Reviewer identifier (empty string or null if not tracked). |

### Change detection

| Column               | Type        | Description                                                                 |
|----------------------|-------------|-----------------------------------------------------------------------------|
| content_sha          | VARCHAR(64) | SHA-256 hex of canonical content (e.g. document_id + source_id + text).    |
| updated_at           | TIMESTAMP   | Last update time (publish time or created_at).                              |

### Per-row review (optional)

| Column                    | Type        | Description                                                                 |
|---------------------------|-------------|-----------------------------------------------------------------------------|
| source_verification_status| VARCHAR(20) | For facts: extracted_facts.verification_status; for chunks: `'n/a'` or empty.|

**Sentinels:** All string columns that are “empty if null” are stored as `''` when the source is null. Use `0` for `page_number` and `paragraph_index` when not applicable.

---

## 4. Change handling

- RAG **writes** to `rag_published_embeddings` only when the user clicks **Publish** for a document.
- On **re-publish** (user publishes the same document again, e.g. after re-embedding or edits), RAG **replaces** all rows for that `document_id`: DELETE existing rows, then INSERT the new set.
- Each row MUST include `content_sha` (64-char hex) and `updated_at` for change detection and idempotent sync by the consumer.

---

## 5. Publish events (audit, not part of dbt contract)

RAG optionally writes to **`publish_events`** for audit:

| Column        | Type      | Description                |
|---------------|-----------|----------------------------|
| id            | UUID      | Primary key.               |
| document_id   | UUID      | Document published.        |
| published_at  | TIMESTAMP | When the publish happened. |
| published_by  | VARCHAR(255) | User/system id (optional). |
| rows_written  | INTEGER   | Number of rows written.    |
| notes         | TEXT      | Optional.                  |

This table is **not** part of the dbt contract; it is for RAG operations and traceability only.

---

## 6. Consumer flow

1. **RAG PostgreSQL** → table `rag_published_embeddings` (with vector column).
2. **Ingestion** → reads `rag_published_embeddings` and loads into **BigQuery landing** (same schema; vector as ARRAY<FLOAT64> or equivalent).
3. **dbt** → sources from BigQuery landing and produces marts.
4. **Downstream** → marts (and optional sync) feed the chat server (e.g. vectors → vector DB, metadata → PostgreSQL).

---

## 7. API (for reference)

- **POST /documents/{document_id}/publish**  
  Body (optional): `{ "published_by": "user-id" }`.  
  Returns: `{ "status": "ok", "document_id": "...", "rows_written": N }`.  
  Errors: 404 document not found; 400 no chunk embeddings (run embedding first).

- **GET /documents/{document_id}/publish-status**  
  Returns: `{ "published": true|false, "document_id": "...", "published_at": "...", "published_by": "...", "rows_written": N }` (when published).
