# Contract: Mobius-RAG and dbt / Data Transformation

**Version:** 2026-02  
**For:** dbt agent and downstream consumers of published RAG data

---

## 1. Parties and purpose

- **Publisher:** Mobius-RAG. RAG ingests documents, extracts/chunks/embeds content, and publishes a single output table when the user explicitly clicks **Publish** (or **Republish**) for a document.
- **Consumer:** dbt-enabled data transformation layer. Ingestion reads RAG’s published table (in PostgreSQL); it may be replicated to BigQuery or another warehouse. dbt sources from that landing layer and produces marts.

RAG exposes one **contract table** in PostgreSQL: **`rag_published_embeddings`**. Its schema (Section 3) is the contract. The consumer **only** reads this table (or its replicated copy). No other RAG tables are part of the contract.

---

## 2. Publish flow

- **Explicit user action.** The user reviews documents in the UI and, when ready, uses **Document Status → Publish** (or **Republish** for a document that was already published).
- **Backend:** `POST /documents/{document_id}/publish`. RAG then:
  1. Loads the document and all `chunk_embeddings` for that document.
  2. For each embedding, joins to `hierarchical_chunks` or `extracted_facts` (by `source_type` + `source_id`) and to `documents` to build one row per contract schema (including `content_sha`, `updated_at`).
  3. **Deletes** all existing rows in `rag_published_embeddings` for that `document_id`.
  4. **Inserts** the new set of rows.
  5. Runs an **integrity check** (row count + spot-check of sample rows). Result is stored in `publish_events` (`verification_passed`, `verification_message`).
  6. Inserts one row into **`publish_events`** (audit: document_id, published_at, published_by, rows_written, verification_passed, verification_message).
- **Whole document only.** There is no partial publish. Each publish/re-publish replaces **all** rows for that `document_id` in `rag_published_embeddings`.
- **Republish.** Publishing again for the same document is allowed and has the same semantics: full replace by `document_id`.

---

## 3. Table: `rag_published_embeddings` (dbt contract)

This is the **only** table the consumer reads. One row per published embedding (chunk or fact). All columns below are part of the contract.

### Identity and embedding

| Column       | Type         | Description                                                                 |
|--------------|--------------|-----------------------------------------------------------------------------|
| id           | UUID         | Primary key (from chunk_embedding.id).                                      |
| document_id  | UUID         | Document this row belongs to.                                               |
| source_type  | VARCHAR(20)  | `'hierarchical'` (chunk) or `'fact'`.                                       |
| source_id    | UUID         | hierarchical_chunks.id or extracted_facts.id.                               |
| embedding    | vector(1536) | The embedding vector (pgvector). In BigQuery replication: ARRAY<FLOAT64>(1536). |
| model        | VARCHAR(100) | Embedding model (e.g. text-embedding-3-small). Empty string if unknown.      |
| created_at   | TIMESTAMP    | When the embedding was created (from chunk_embeddings.created_at).          |

### Content and structure

| Column          | Type         | Description                                                                 |
|-----------------|--------------|-----------------------------------------------------------------------------|
| text            | TEXT         | The text that was embedded (chunk or fact text). Never null; empty string if none. |
| page_number     | INTEGER      | Source page (0 when not applicable).                                       |
| paragraph_index | INTEGER      | Paragraph index within page (0 when not applicable).                        |
| section_path    | VARCHAR(500) | e.g. "Section 3.2". Empty string when not applicable.                       |
| chapter_path    | VARCHAR(500) | e.g. "Chapter 5". Empty string when not applicable.                         |
| summary         | TEXT         | Chunk summary (hierarchical); for facts use empty string or fact_text.     |

### Document metadata (denormalized at publish time)

| Column                     | Type          | Description                                    |
|----------------------------|---------------|------------------------------------------------|
| document_filename          | VARCHAR(255)  | documents.filename.                            |
| document_display_name      | VARCHAR(255)  | documents.display_name. Empty string if null.  |
| document_authority_level   | VARCHAR(100)  | documents.authority_level. Empty string if null. |
| document_effective_date    | VARCHAR(20)   | documents.effective_date. Empty string if null. |
| document_termination_date  | VARCHAR(20)   | documents.termination_date. Empty string if null. |
| document_payer             | VARCHAR(100)  | documents.payer. Empty string if null.        |
| document_state             | VARCHAR(2)    | documents.state. Empty string if null.        |
| document_program           | VARCHAR(100)  | documents.program. Empty string if null.      |
| document_status            | VARCHAR(20)   | documents.status.                              |
| document_created_at        | TIMESTAMP    | documents.created_at.                          |
| document_review_status     | VARCHAR(20)   | documents.review_status.                       |
| document_reviewed_at       | TIMESTAMP    | When document was last reviewed (null if not tracked). |
| document_reviewed_by       | VARCHAR(255)  | Reviewer identifier (empty string or null if not tracked). |

### Change detection

| Column      | Type        | Description                                                                 |
|-------------|-------------|-----------------------------------------------------------------------------|
| content_sha | VARCHAR(64) | SHA-256 hex of canonical content (document_id + source_id + text). Use for change detection and idempotent sync. |
| updated_at  | TIMESTAMP   | Last update time (publish time).                                            |

### Per-row review (optional)

| Column                     | Type        | Description                                                                 |
|----------------------------|-------------|-----------------------------------------------------------------------------|
| source_verification_status | VARCHAR(20) | For facts: extracted_facts.verification_status; for chunks: `'n/a'` or empty. |

**Sentinels:** All string columns that are “empty if null” are stored as `''` when the source is null. Use `0` for `page_number` and `paragraph_index` when not applicable.

---

## 4. Change handling

- RAG **writes** to `rag_published_embeddings` only when the user clicks **Publish** or **Republish** for a document.
- On each publish (including republish), RAG **replaces** all rows for that `document_id`: DELETE existing rows, then INSERT the new set. There are no incremental updates per document.
- Each row includes **`content_sha`** (64-char hex) and **`updated_at`** so the consumer can do change detection and idempotent sync (e.g. by document_id + content_sha or updated_at).
- **Primary key:** `id` (unique per row; stable across republishes for the same chunk/fact until the source embedding changes).

---

## 5. Publish events (audit; not part of dbt contract)

RAG writes to **`publish_events`** for audit and operations. The consumer does **not** need to read this table for the dbt contract. It is listed here for completeness.

| Column                 | Type        | Description                                           |
|------------------------|-------------|-------------------------------------------------------|
| id                     | UUID        | Primary key.                                          |
| document_id            | UUID        | Document published.                                   |
| published_at           | TIMESTAMP   | When the publish happened.                            |
| published_by           | VARCHAR(255)| User/system id (optional).                           |
| rows_written           | INTEGER     | Number of rows written to rag_published_embeddings.   |
| notes                  | TEXT        | Optional.                                             |
| verification_passed    | BOOLEAN     | True if post-write integrity check passed; False if failed; NULL for legacy. |
| verification_message   | TEXT        | Error message if verification failed; NULL if passed. |

---

## 6. Consumer flow

1. **RAG PostgreSQL** → table **`rag_published_embeddings`** (with `embedding` as pgvector).
2. **Ingestion** → reads `rag_published_embeddings` and loads into the warehouse (e.g. BigQuery landing). Map `embedding` to ARRAY<FLOAT64>(1536) or equivalent.
3. **dbt** → sources from the landing table and produces marts (e.g. filtered, aggregated, or synced to a vector store).
4. **Downstream** → marts (and optional sync) feed the chat server or other consumers.

---

## 7. API (for reference)

These endpoints are for RAG UI and automation. The dbt agent only needs to know that the **source of truth** is the table **`rag_published_embeddings`**.

- **POST /documents/{document_id}/publish**  
  Body (optional): `{ "published_by": "user-id" }`.  
  Returns:  
  `{ "status": "ok", "document_id": "...", "rows_written": N, "verification_passed": true|false, "verification_message": null|"..." }`.  
  Errors: 404 document not found; 400 no chunk embeddings (run embedding first).

- **GET /documents/{document_id}/publish-status**  
  Returns:  
  `{ "published": true|false, "document_id": "...", "published_at": "...", "published_by": "...", "rows_written": N, "verification_passed": true|false|null, "verification_message": null|"..." }` (when published).

---

## 8. Summary for dbt agent

- **Single contract table:** `rag_published_embeddings`.
- **Semantics:** Full replace per document on each Publish/Republish; use `content_sha` and `updated_at` for change detection.
- **Schema:** Section 3 is authoritative; replicate as-is (with `embedding` mapped to your warehouse’s vector/array type).
- **Audit:** `publish_events` is for RAG operations only; not required for dbt models.
