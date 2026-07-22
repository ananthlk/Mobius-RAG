# Refactor bug: "completed" embedding-jobs that produced no embeddings

**Status:** documented for P2 refactor · **Found:** 2026-07-22 (corpus-heal 424 forensics)
**Owners:** RAG (embedding worker + nightly), EVAL (calibration impact / recall accounting)
**Severity:** correctness (silent data-integrity gap; not user-facing outage)

## Symptom
424 documents have a `completed` `embedding_jobs` row but **no** rows in
`rag_published_embeddings` (and none in `chunk_embeddings`). Surfaced when the
corpus-heal track tried to "purge 424 phantom rows" — the rows don't exist to
purge; the docs are *absent* from the serving corpus. Decomposition:
- **276** are `status='needs_ocr'` — no extractable text (OCR backlog, expected).
- **148** are `status='completed'`, `has_errors=false`, `review_status='pending'`,
  **0 chunk_embeddings** — web-scraped news/directory pages (CFBHN partner
  listings, LMHC interview pages) that extracted to no usable content.

Neither class is a recall wedge — but the 148 are a genuine integrity bug:
a "completed" job that embedded nothing, invisible to every health signal.

## Root cause 1 — `status="completed"` is overloaded
`app/embedding_worker.py:156-157`:
```python
logger.info("[JOB %s] No chunks or facts to embed for document %s", ...)
job.status = "completed"      # <-- same status as a successful embed
```
When a doc yields zero chunks/facts, the job is marked **`completed`** — the
identical terminal state used for a job that embedded N chunks. Consequence:
**`completed` no longer implies "embeddings exist."** The invariant every
downstream consumer (and the corpus-heal analysis) assumed is silently false.

**Fix (P2):** give the no-content path a distinct terminal state, e.g.
`status='empty'` (or `no_content` / `skipped_empty`). Then `completed` means
exactly "wrote ≥1 embedding," and `empty` routes to the content-less workstream.

## Root cause 2 — no embedding↔rpe reconciliation in the nightly
This is the answer to "why didn't the nightly job clear it — that's its job":
- The nightly `reconcile` step (`nightly_orchestrator.py:376`) is **inheritance
  reconcile only** (re-stamps MCO payor j-tags). There is **no** step that
  reconciles "`completed` embedding_job ⟺ rows in `rag_published_embeddings`."
- `scripts/publish_unpublished_documents.py` is the right tool for the
  *have-chunks-but-unpublished* case (keys on `chunk_embeddings NOT IN rpe`),
  but (a) it is a **standalone script, not wired into the nightly loop**, and
  (b) it cannot help these 148 — they have **0 chunk_embeddings**, so there is
  nothing to publish. The defect is upstream (RC1), not a publish miss.
- The content-less gate targets `needs_ocr`/error-flagged docs. The 148 present
  as healthy (`has_errors=false`, `status='completed'`) → they fall in a blind
  spot no gate inspects.

**Fix (P2):** add a nightly reconciliation + alarm:
1. Assert the invariant: `count(completed jobs) == count(distinct docs in rpe)`;
   any drift emits a metric + the offending doc_ids (this exact drift = 424
   would have alarmed on day one).
2. Route `empty`/0-chunk-completed docs to the content-less workstream instead
   of leaving them `completed`-and-invisible.
3. Wire `publish_unpublished_documents.py` into the nightly (or fold its query
   into the reconcile step) so the have-chunks-unpublished case self-heals.

## What is NOT the fix
- Do **not** purge `rag_published_embeddings` — these docs aren't in it.
- Do **not** publish the 148 — they have no content; publishing adds noise.
- The corpus-heal recall premise (`oracle_recall 0.47→0.60`) does not rest on
  these docs; re-derive the accuracy gap from post-`s`-fix calibration.

## P2 refactor checklist
- [ ] `embedding_worker`: distinct terminal state for no-content jobs (RC1).
- [ ] nightly: embedding_job↔rpe invariant check + doc_id alarm (RC2).
- [ ] nightly: route `empty` docs to content-less remediation.
- [ ] nightly: wire/absorb `publish_unpublished_documents.py`.
- [ ] backfill: re-classify the existing 148 `completed`→`empty` once RC1 lands.
