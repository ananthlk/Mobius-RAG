# Chunking worker refactor: implementation and testing plan

This document outlines phased implementation steps and how to test each phase. Scope is limited to the chunking worker and its direct dependencies; impacted modules (embedding worker, frontend, main API) are not modified here and are reviewed separately.

---

## Scope and constraints

**Only this submodule may be modified.** No code outside **mobius-rag** (this repo) may be touched in this sprint. Do not change mobius-qa, mobius-chat, mobius-dbt, or any parent or sibling repos.

**Within mobius-rag, do not modify (noted for later review):**
- [app/embedding_worker.py](app/embedding_worker.py) — will need to read from embeddable_units later.
- [app/services/publish.py](app/services/publish.py) — may need to join to embeddable_units later.
- [app/main.py](app/main.py) — chunking events API and in-process stream; consume `message`/`user_message` later.
- [frontend/](frontend/) — live updates and event consumption; update later.

**Other modules outside mobius-rag that may be impacted (do not modify; awareness only):**
- **mobius-qa** — lexicon maintenance, retrieval eval, or any RAG API consumers; may depend on chunking/embedding contracts.
- **mobius-chat** — if it displays document processing status or calls chunking/events APIs.
- Any other service that calls mobius-rag APIs for documents, chunking status, or embeddings.

---

## Implementation phases

### Phase 1: DB handler and config (foundation)

**Goal:** All persistence and run configuration go through a single layer; no direct `db.add()` in paths.

| Step | Task | Notes |
|------|------|-------|
| 1.1 | Add `app/worker/` package with `__init__.py` | Empty or re-export only initially. |
| 1.2 | Add `app/worker/config.py` | Load poll_interval, error_sleep, Path B caps, default threshold/retries from env (and optional DB). Expose a single `get_worker_config()` (or dataclass). |
| 1.3 | Add `app/worker/db.py` (database handler) | Methods: `upsert_chunking_result()`, `write_event()`, `clear_policy_for_document()`, `persist_chunk()`, `persist_fact()`. Session lifecycle and commit/rollback only here. Accept `AsyncSession` from caller (process_job still creates session). |
| 1.4 | Migrate current worker to call db handler | In `worker.py` (or a temporary shim), replace inline `db.add`/commit with calls to `worker.db.*`. Keep behavior identical. |
| 1.5 | Add `ChunkingJob.chunking_config_snapshot` (JSONB) | Migration + set snapshot when job starts (resolved config dict). Optionally copy to ChunkingResult on completion. |

**Testing:** Run existing chunking job (Path A and Path B) end-to-end; confirm ChunkingResult, ChunkingEvent, and policy tables match previous behavior. Unit tests for config loading and for db handler methods (with mocked session).

---

### Phase 2: Context and error helper

**Goal:** One context object for emit/upsert/send_status; one place to record paragraph errors.

| Step | Task | Notes |
|------|------|-------|
| 2.1 | Add `app/worker/context.py` | `ChunkingRunContext` or `ChunkingLoopContext`: holds document_id, job_id, event_callback, total_paragraphs, results_paragraphs; methods `emit(event_type, payload)`, `upsert_progress()`, `send_status(message)`. Payload must include `message` and `user_message` (see plan §9). |
| 2.2 | Add `app/worker/errors.py` | `record_paragraph_error(job_id, document_id, paragraph_id, error, ctx)` → classify_error, log_error, set results_paragraphs[para_id] = failed. |
| 2.3 | Wire context into current loop | Replace nested closures (_upsert, _emit, _send_status) with context; emit dual messages. |

**Testing:** Run chunking; verify ChunkingEvent rows have both `message` and `user_message` in event_data. Force one paragraph to fail (e.g. invalid content); verify record_paragraph_error behavior and that job continues or fails as configured.

---

### Phase 3: Path A and Path B split + coordinator

**Goal:** Single branch in the loop: path_a.run() or path_b.run(); no mixed logic.

| Step | Task | Notes |
|------|------|-------|
| 3.1 | Add `app/worker/path_a.py` | `run(ctx, pages, job_options)` → materialize paragraphs from pages, for each: emit paragraph_start, extraction → critique → retry loop, persist via db handler, emit paragraph_complete/failed. Use ctx.emit, ctx.upsert, errors.record_paragraph_error. |
| 3.2 | Add `app/worker/path_b.py` | `run(ctx, pages, lexicon_snapshot, job_options)` → db_handler.clear_policy_for_document; for each paragraph: build_paragraph_and_lines, apply_lexicon_to_lines (services), db handler persist policy; emit; at end extract_candidates_for_document. |
| 3.3 | Add `app/worker/coordinator.py` | `run_chunking_loop(db, job, document, pages, options)` → build context, materialize (page_number, markdown), branch on generator/extraction_enabled: path_a.run(...) or path_b.run(...). |
| 3.4 | Thin `worker.py` | process_job: setup (load doc, pages, config, prompts, LLM, lexicon), call coordinator.run_chunking_loop(), teardown (job status, enqueue EmbeddingJob). worker_loop and main unchanged. |

**Testing:** Path A: run job with extraction_enabled true; verify HierarchicalChunk + ExtractedFact + events. Path B: run with extraction_enabled false; verify policy_paragraphs, policy_lines, tags, and candidate extraction. Use the small test document (see below).

---

### Phase 4: Embeddable units table and writes

**Goal:** Chunking worker writes to a single table for “what to embed”; no changes to embedding worker in this refactor.

| Step | Task | Notes |
|------|------|-------|
| 4.1 | Add `embeddable_units` table + model | Columns: document_id, generator_id, source_type, source_id (UUID), text, metadata_ (JSONB). Migration. |
| 4.2 | DB handler: `write_embeddable_unit()` / batch | Path A: after persist_chunk/persist_fact, insert one row per chunk/fact (text = same as current embedding_worker builds). Path B: after building paragraphs/lines, insert one row per paragraph (or line) with text to embed. |
| 4.3 | Path A and Path B call write_embeddable_unit | From path_a and path_b via db handler only. |

**Testing:** Run Path A and Path B jobs; query embeddable_units for document_id + generator_id; confirm row count and text content. Embedding worker is not changed; it will be updated in a separate pass to read from embeddable_units.

---

### Phase 5: Tag propagation (Path B)

**Goal:** Line → paragraph → document tags stored; optional effective line-level score.

| Step | Task | Notes |
|------|------|-------|
| 5.1 | Document-level tag storage | Add columns to Document (e.g. policy_tags JSONB) or new table document_policy_tags; migration. |
| 5.2 | `policy_path_b`: aggregate_line_tags_to_paragraph() | For each paragraph, aggregate its lines’ p_tags/d_tags/j_tags (e.g. max per code); write to policy_paragraphs. |
| 5.3 | `policy_path_b`: aggregate_paragraph_tags_to_document() | For document, aggregate paragraph tags; write to document storage. |
| 5.4 | Call aggregation after Path B paragraph loop | In path_b.run(), after all paragraphs/lines and lexicon applied, run line→paragraph then paragraph→document. |
| 5.5 | (Optional) compute_effective_line_tags() | In services/policy_path_b: f(line_tags, paragraph_tags, doc_tags; weights). Used at read time or for materialized column later; no change to embedding/frontend in this refactor. |

**Testing:** Path B run on test doc; verify policy_paragraphs rows have p_tags/d_tags/j_tags populated from lines; document-level tags populated. Unit tests for aggregation and effective_line_tags with mock inputs.

---

### Phase 6: Production hardening

**Goal:** Safe failures, checkpoint commits, clear status, optional heartbeat.

| Step | Task | Notes |
|------|------|-------|
| 6.1 | process_job try/except | Catch all exceptions; set job.status=failed, error_message, completed_at; commit; log; continue worker_loop. |
| 6.2 | Checkpoint commits | DB handler commits after each paragraph (or configurable batch); no single transaction for whole document. |
| 6.3 | Event coverage | Ensure job_start, paragraph_start/complete/failed, progress_update, chunking_complete/chunking_failed all emitted with message + user_message. |
| 6.4 | Optional timeouts | Configurable timeout for LLM calls and/or whole job; on timeout set failed and error_message. |
| 6.5 | Optional heartbeat | Update ChunkingJob.updated_at (or heartbeat_at) periodically during processing for dead-job detection. |

**Testing:** Kill worker mid-job; verify job can be retried or marked failed by monitor. Force timeout; verify job fails with clear message. Run full job with small test document and confirm events and status in DB.

---

## Testing plan

### Unit tests (per phase)

| Phase | Focus | Location |
|-------|-------|----------|
| 1 | config loading; db handler with mocked session | e.g. `tests/test_worker_config.py`, `tests/test_worker_db.py` |
| 2 | context emit/upsert; record_paragraph_error | e.g. `tests/test_worker_context.py`, `tests/test_worker_errors.py` |
| 3 | path_a path_b coordinator with mocked ctx/db | e.g. `tests/test_worker_path_a.py`, `tests/test_worker_path_b.py`, `tests/test_worker_coordinator.py` |
| 4 | write_embeddable_unit and row shape | e.g. in `tests/test_worker_db.py` or integration |
| 5 | aggregate_line_tags_to_paragraph, aggregate_paragraph_tags_to_document, compute_effective_line_tags | extend `tests/test_policy_path_b.py` |
| 6 | job failure and checkpoint behavior | integration or test_worker_coordinator |

### Integration test (end-to-end)

- **Setup:** Start DB (and optional LLM/embedding stubs if needed). Create one document with the small test document content (see below).
- **Path A:** Enqueue ChunkingJob (extraction_enabled=true, generator_id=A). Run worker until job completed. Assert: ChunkingResult status completed; ChunkingEvent has paragraph_start, extraction_complete, critique_complete, paragraph_complete, chunking_complete; HierarchicalChunk and ExtractedFact rows; embeddable_units rows.
- **Path B:** Same document or second document; ChunkingJob (extraction_enabled=false, generator_id=B). Assert: policy_paragraphs, policy_lines with tags; document-level tags if Phase 5 done; embeddable_units rows; ChunkingEvent has paragraph_complete, chunking_complete.
- **Failure:** Run job that fails (e.g. invalid doc or mock LLM failure). Assert: job.status=failed, error_message set, ChunkingEvent chunking_failed or equivalent.

### Manual testing checklist

1. **Small test document:** Upload and process the test document (see below). Path A: confirm extraction and critique in UI or via API; Path B: confirm policy lines and tags.
2. **Live updates:** Poll GET /documents/{id}/chunking/events; confirm events have `message` and `user_message` in data. (Frontend changes are out of scope; manual check with curl or browser.)
3. **Config snapshot:** After run, read ChunkingJob.chunking_config_snapshot (or ChunkingResult.metadata_.chunking_config); confirm it matches run parameters.
4. **Idempotency:** Re-run same document (Path B); confirm no duplicate policy_paragraphs/lines or clear-then-rebuild behavior.

---

## Small test document for chunking

Use the following content as the **small document** for fast, predictable chunking runs. It has three short paragraphs so Path A runs 3 paragraphs (extraction + critique) and Path B produces 3 policy paragraphs with a few lines each.

**Option A:** Save as a `.md` or `.txt` file and convert to PDF (e.g. via pandoc or print-to-PDF), then upload via your document upload API.

**Option B:** If your pipeline can accept raw text or markdown, use the content below directly.

**File to use:** [mobius-rag/tests/fixtures/small_chunking_test_document.md](../tests/fixtures/small_chunking_test_document.md).

### Expected outcomes (for reference)

- **Path A:** 3 paragraphs processed; 3 HierarchicalChunk rows; some ExtractedFact rows; ChunkingEvent: paragraph_start (×3), extraction_complete (×3), critique_* (×3), paragraph_complete (×3), progress_update, chunking_complete.
- **Path B:** 3 policy_paragraphs, multiple policy_lines; tags on lines if lexicon has matches; ChunkingEvent: paragraph_complete (×3), chunking_complete; after Phase 4, embeddable_units rows for generator_id B.
- **Run time:** Small doc keeps each run under a few minutes so tests and manual checks are quick.

---

## Summary

| Phase | Delivers | Test focus |
|-------|----------|------------|
| 1 | DB handler, config, config snapshot | Unit + E2E same as before |
| 2 | Context, errors, dual messages | Events have message + user_message |
| 3 | path_a, path_b, coordinator, thin worker | Path A vs Path B E2E with test doc |
| 4 | embeddable_units table and writes | Rows for A and B |
| 5 | Tag propagation line→para→doc | Path B tags and optional effective score |
| 6 | Production hardening | Failures, checkpoints, status, timeouts |

Impacted modules (embedding worker, frontend, main API) are not modified in this refactor; they are listed in the main plan for separate review and updates.
