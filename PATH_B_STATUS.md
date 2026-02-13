# Path B status: what’s broken and what’s intended

Path B is intended to be the **deterministic policy pipeline**: extract structure, apply tags from the lexicon, write `policy_paragraphs` / `policy_lines`, and identify new **lexicon candidates** for review. Right now that pipeline is incomplete or missing in this repo.

---

## Intended Path B behavior

1. **Chunk deterministically**  
   Split document pages into paragraphs and lines (no LLM). Same as current worker “Path B” branch: use existing chunking and write `HierarchicalChunk` with `extraction_status="skipped"`.

2. **Build policy structure**  
   From those chunks, create:
   - **policy_paragraphs**: paragraph-level units with `heading_path`, `text`, `paragraph_type`, etc.
   - **policy_lines**: line-level units (optionally atomic) with `text`, `line_type`, `is_atomic`, linked to a paragraph.

3. **Apply tags**  
   Using the current **policy lexicon** (p/d/j), tag lines and paragraphs:
   - **p_tags** (prescriptive), **d_tags** (descriptive), **j_tags** (jurisdiction), plus optional **inferred_d_tags** / **inferred_j_tags**.

4. **Identify new candidates**  
   Detect phrases that are not in the lexicon and insert **policy_lexicon_candidates** (proposed_tag, confidence, examples, etc.) for human review.

5. **Embed**  
   After policy artifacts are written, run the existing embedding path for generator_id B (unchanged).

---

## What’s actually in the repo

| Component | Status |
|----------|--------|
| **Worker Path B** | Only writes `HierarchicalChunk` with `extraction_status="skipped"`. Does **not** create `policy_paragraphs`, `policy_lines`, or `policy_lexicon_candidates`. |
| **Policy write path** | **Missing.** No code in this repo inserts into `policy_paragraphs` or `policy_lines`. |
| **Tag application** | **Missing.** No code applies the lexicon to lines/paragraphs to fill `p_tags` / `d_tags` / `j_tags`. |
| **Candidate extraction** | **Missing.** No code that creates `PolicyLexiconCandidate` rows from policy artifacts. |
| **policy_lexicon_repo** | **Missing.** `app.services.policy_lexicon_repo` is imported in `main.py` (lexicon snapshot, approve_phrase_to_db, update_tag_in_db, export_yaml_from_db, etc.) but the module **does not exist**. Any endpoint that uses it will raise `ModuleNotFoundError` at runtime. |
| **Policy migrations** | **Missing.** `main.py` runs these at startup: `add_policy_line_offsets`, `add_policy_lexicon_candidate_occurrences`, `add_policy_lexicon_candidate_catalog`. The migration files are not in `app/migrations/`; the imports are caught and logged as “skipped.” |

So: **Path B in this repo is only “chunk + skip LLM + embed”.** Tag extraction, writing policy tables, and candidate identification are not implemented here. The models comment says “Lexicon maintenance has moved to the QA service”; the RAG API still expects to use `policy_lexicon_repo` for review/approve and lexicon read/update, so the current state is inconsistent and broken for those endpoints.

---

## How to fix it

1. **Restore or add Path B pipeline in the worker (or a dedicated service)**  
   After the existing “Path B: persist hierarchical chunk only” step:
   - From `HierarchicalChunk` (or from raw pages/markdown), build **policy_paragraphs** and **policy_lines** (structure only at first).
   - Load the current lexicon (from DB or YAML; see `policy_lexicon_repo` below).
   - For each line (and optionally paragraph), apply matching rules and set **p_tags**, **d_tags**, **j_tags** (and inferred_*).
   - Run candidate extraction: find phrases not in the lexicon, create **policy_lexicon_candidates** with proposed_tag, confidence, examples.

2. **Restore or implement `app.services.policy_lexicon_repo`**  
   Implement (or copy from another branch/repo) a module that provides at least:
   - `load_lexicon_snapshot_db()` – load current p/d/j lexicon from DB (or fallback).
   - `approve_phrase_to_db(...)` – write approved phrase into lexicon and bump revision.
   - `update_tag_in_db(...)` – update a tag’s spec/active state.
   - `bump_revision()` – revision tracking for lexicon.
   - `export_yaml_from_db(db)` – optional YAML export for compatibility.

   Until this exists, all endpoints that import it will fail when first called.

3. **Restore or add policy migrations**  
   Add migration modules under `app/migrations/` for:
   - `add_policy_line_offsets`
   - `add_policy_lexicon_candidate_occurrences`
   - `add_policy_lexicon_candidate_catalog`  
   so that the DB schema and startup behavior match what `main.py` expects.

4. **Optional: stub `policy_lexicon_repo`**  
   If you want the server to start and policy **read** endpoints to work without review/approve, add a minimal stub that raises a clear `NotImplementedError` (or returns empty lexicon) for the missing functions so callers get a clear error instead of `ModuleNotFoundError`.

---

## Quick reference: where things are

- **Path B worker branch:** `mobius-rag/app/worker.py` – `if not extraction_enabled:` block (~lines 304–343). Only creates `HierarchicalChunk` (skipped).
- **Policy API (read):** `main.py` – e.g. `GET /documents/{id}/policy/summary`, `.../policy/paragraphs`, `.../policy/lines`, `.../policy/candidates`. These **read** from `policy_paragraphs`, `policy_lines`, `policy_lexicon_candidates`; they work only if something else has written to those tables.
- **Policy API (review/lexicon):** `main.py` – e.g. `POST /policy/candidates/{id}/review`, `GET /policy/lexicon`, `PATCH /policy/lexicon/tags/...`. These **import** `app.services.policy_lexicon_repo` and will crash until that module exists.

---

## Streaming chunking logs

To start Path B chunking for a document (e.g. **MMA-LTC-Member-Handbook.pdf**) and stream the worker’s chunking events in your terminal:

```bash
# API and worker must be running (same DB). Default document: MMA-LTC-Member-Handbook.pdf
python3 scripts/run_chunking_and_stream_log.py
```

Options: `--document-id <uuid>`, `--filename "substring"`, `--base-url URL`, `--no-start` (only stream existing events). See script docstring for details.

---

## No database updates after Path B chunking

If the stream script runs but you see no new rows in `policy_paragraphs` / `policy_lines`:

1. **Worker must be running** – `mstart` starts `mobius-rag-chunking-worker`. Check `.mobius_logs` (e.g. `mobius-rag-chunking-worker`) for errors.
2. **Same database** – API (8001) and worker must use the same `DATABASE_URL` in `mobius-rag/.env`.
3. **Policy tables exist** – Migrations must have run: `policy_paragraphs`, `policy_lines`, `policy_lexicon_*`. If they’re missing, Path B logs: `Path B policy build/tag (non-fatal): ...` and skips writing.
4. **Worker logs** – Look for:
   - `Path B: built 1 paragraph, N lines for <para_id>` → build succeeded; data should be committed.
   - `Path B policy build/tag (non-fatal): ...` → build failed (e.g. missing table or constraint); no policy rows for that paragraph.
   - `Path B: could not load lexicon ...` → tags may be empty but paragraphs/lines should still be written.

---

## Why are there no p/d/j tags on policy lines?

Tags are applied from the **policy lexicon** stored in `policy_lexicon_meta` and `policy_lexicon_entries`. If those tables are empty (or entries have no `spec.phrases`), the phrase map is empty and **no tags are applied**.

1. **Check worker logs** – On Path B job start you should see either:
   - `Path B: loaded lexicon with N phrases for tagging` → lexicon has data; tags will be applied when line text matches a phrase.
   - `Path B: lexicon has 0 phrases (...); no p/d/j tags will be applied` → no lexicon data; populate `policy_lexicon_entries` with rows that have `spec` containing a `"phrases"` array (e.g. via the approve-candidate API or a seed script).
2. **Populate the lexicon** – Insert into `policy_lexicon_entries`: `kind` (p/d/j), `code`, `spec` JSONB with `{"phrases": ["phrase one", "phrase two"], "description": "..."}`. Path B matches normalized line text against these phrases and sets `p_tags` / `d_tags` / `j_tags` on matching lines.
3. **Per-paragraph log** – After the fix, logs show `Path B: built 1 paragraph, N lines (M with tags)` when any line in that paragraph got a tag.

---

## Candidates vs existing p/d/j tags

**Candidate extraction** proposes new phrases for the lexicon (e.g. n-grams, abbreviations). It **excludes** anything already in the lexicon so we don’t re-propose existing tags.

- **Exclusion source:** The same phrase map used for tagging: `get_phrase_to_tag_map(lexicon_snapshot)`. So `known = phrase_map.keys()` plus rejected catalog and common phrases.
- **If the lexicon has 0 phrases:** Nothing is excluded. You’ll see “Extracted 250 lexicon candidates” and many of those may be phrases you consider “already” p/d/j (e.g. from another system or a lexicon that isn’t loaded). Fix: populate `policy_lexicon_entries` with `spec.phrases` (or `spec.description`) so the phrase map is non-empty; then tags apply and those phrases are excluded from candidates.
- **Diagnostics:** On job start the worker logs `Path B: lexicon has 0 phrases (entries: p=X d=Y j=Z)` or `Path B: loaded lexicon with N phrases (entries: p=X d=Y j=Z)`. If you have entries (X,Y,Z > 0) but 0 phrases, the issue is **spec shape**: ensure each row’s `spec` has a `"phrases"` array (or at least `"description"`) so they’re included in the map.
- **Tag codes and description:** The phrase map also includes tag codes (e.g. `member_services`) and, when `spec.phrases` is missing, `spec.description`, so those are not proposed as new candidates.
