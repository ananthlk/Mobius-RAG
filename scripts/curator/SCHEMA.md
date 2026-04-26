# `discovered_sources` — curator persistence layer

**Purpose**: durable record of every URL the curator has ever observed.
Survives scrape job retention. Drives chat ReAct's
`lookup_authoritative_sources` tool. Replaces the throw-everything-away
behavior that lost ~164 HTML pages on the 945-doc scrape.

## Table

```sql
CREATE TABLE discovered_sources (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Identity (UNIQUE on url)
    url             TEXT NOT NULL UNIQUE,
    host            TEXT NOT NULL,            -- 'sunshinehealth.com'
    path            TEXT NOT NULL,            -- '/providers/Billing-manual.html'

    -- Classification (set on insert, mutable via curator UI)
    payer                       TEXT,         -- canonical 'Sunshine Health'
    state                       TEXT,         -- 'FL'
    program                     TEXT,         -- 'Medicaid' | 'CMS' | nullable
    inferred_authority_level    TEXT,         -- 'payer_manual' | 'payer_policy' | ...
    curated_authority_level     TEXT,         -- human override; NULL = use inferred
    topic_tags                  TEXT[],       -- LLM-derived: ['ECT','PA','criteria']
    content_kind                TEXT NOT NULL,-- 'doc' | 'page'
    extension                   TEXT,         -- 'pdf' | 'html' | 'xlsx'

    -- Liveness
    first_seen_at               TIMESTAMPTZ DEFAULT now(),
    last_seen_at                TIMESTAMPTZ DEFAULT now(),
    last_fetch_status           INT,          -- 200 | 403 | 404 | 451 | -1 (network err)
    last_fetch_at               TIMESTAMPTZ,
    fetch_attempt_count         INT DEFAULT 0,

    -- Content fingerprinting (only when fetched 200)
    content_type                TEXT,
    content_length              BIGINT,
    content_hash                TEXT,         -- sha256 of body; NULL until fetched
    content_changed_at          TIMESTAMPTZ,  -- updated when hash changes vs last fetch

    -- Ingestion linkage
    ingested                    BOOLEAN DEFAULT false,
    ingested_doc_id             UUID REFERENCES documents(id) ON DELETE SET NULL,
    ingested_at                 TIMESTAMPTZ,

    -- Discovery provenance
    discovered_via              TEXT,         -- 'sitemap' | 'scrape' | 'manual_submit'
    seed_url                    TEXT,         -- the BFS seed (or NULL for sitemap-only)
    depth_from_seed             INT,
    scrape_job_id               TEXT,         -- nullable, references scrape job

    -- Curation
    curation_status             TEXT DEFAULT 'auto', -- 'auto'|'canonical'|'noise'|'stale'|'needs_auth'
    curated_by                  TEXT,         -- user_id who set status
    curation_notes              TEXT,
    curated_at                  TIMESTAMPTZ
);

CREATE INDEX idx_ds_host          ON discovered_sources (host);
CREATE INDEX idx_ds_payer_state   ON discovered_sources (payer, state);
CREATE INDEX idx_ds_curation      ON discovered_sources (curation_status);
CREATE INDEX idx_ds_ingested      ON discovered_sources (ingested);
CREATE INDEX idx_ds_authority     ON discovered_sources (
    COALESCE(curated_authority_level, inferred_authority_level)
);
-- For ReAct lookup_authoritative_sources(payer, state, topic):
CREATE INDEX idx_ds_topic_tags_gin ON discovered_sources USING gin (topic_tags);
```

## Endpoints

### `POST /sources/upsert`
Body:
```json
{
  "url": "https://www.sunshinehealth.com/providers/Billing-manual.html",
  "discovered_via": "scrape",
  "seed_url": "https://www.sunshinehealth.com/providers/Billing-manual.html",
  "depth_from_seed": 0,
  "scrape_job_id": "b1946a8b-...",
  "fetch_status": 200,
  "content_type": "text/html",
  "content_length": 161321,
  "content_hash": "sha256:...",
  "payer_hint": "Sunshine Health",
  "state_hint": "FL"
}
```
Behavior: insert or update by `url`. Bumps `last_seen_at`,
`last_fetch_at`, `fetch_attempt_count`. Recomputes hash-changed flag.
Idempotent — same URL fetched twice produces no extra rows.

### `GET /sources/search`
Query params: `payer`, `state`, `program`, `authority_level`, `topic`,
`status` (curation_status), `ingested`. All optional.
Returns: list of `DiscoveredSource` records, ranked by:
1. curated `canonical` first
2. `last_fetch_status=200` and `ingested=true`
3. inferred relevance to topic (if provided)
4. last_seen_at descending

### `POST /sources/{id}/curate`
Body: `{curation_status, curation_notes, curated_authority_level?, topic_tags?}`
Marks human-curated. Sets `curated_by` from auth.

### `POST /sources/{id}/ingest`
Triggers fetch + (if doc) GCS upload + import-from-gcs OR
(if page) import-from-html. Updates `ingested`, `ingested_doc_id`.
Used by ReAct `ingest_url` tool.

## ReAct tool surface

```yaml
- name: lookup_authoritative_sources
  description: |
    Search Mobius's curated registry of authoritative URLs across
    payer/state/program/topic. Returns URLs with their ingestion
    status — useful when you need to tell the user "I know about
    this source but haven't indexed it yet, want me to fetch it?"
  inputs: { payer?, state?, program?, authority_level?, topic? }
  returns: list[{url, host, payer, ingested, last_seen_at, summary?}]

- name: ingest_url
  description: |
    Fetch a single URL and add it to the indexed corpus. Use only
    after lookup_authoritative_sources surfaces a URL the user
    asks about, not for arbitrary URLs.
  inputs: { url }
  returns: { document_id, chunk_count, status }
```

## Freshness model — the moat

Re-fetch loop that detects upstream policy changes within hours, not weeks.

```
Daily cron (or hourly for curation_status='canonical' sources):
  SELECT id, url, content_hash
  FROM discovered_sources
  WHERE last_fetch_at < now() - interval '24 hours'
    AND curation_status = 'canonical'
  ORDER BY last_fetch_at ASC
  LIMIT 200;

  for each:
    1. HEAD request →
       if ETag and Last-Modified both unchanged from last fetch: skip
       (bump last_fetch_at only)

    2. If HEAD changed OR HEAD unsupported (AHCA returns 400 to HEAD):
       GET → compute sha256(body) → compare to stored content_hash

       a. If hash unchanged: bump last_fetch_at only (false alarm — header
          churn without content change, common with CDN cache-busting)

       b. If hash CHANGED:
          i.   UPDATE discovered_sources SET
                  content_hash=$new, content_changed_at=now(),
                  last_fetch_at=now(), last_fetch_status=200;
          ii.  If ingested=true: re-import to rag (replaces the
                 existing documents row, new chunks/embeddings/published).
                 The existing Phase 12.1 publish_sync runs as part of this
                 — Chroma + chat_pg get the new vectors.
          iii. Fire notification email:
                 "Provider Manual.pdf changed at https://sunshinehealth.com/...
                  diff: [old hash] -> [new hash]
                  re-indexed at HH:MM, X chunks updated"
          iv.  Trigger lexicon re-scan (Phase 13.9) for this doc_id.

    3. If status_code IN (404, 410):
       UPDATE discovered_sources SET
         curation_status='stale', last_fetch_status=$sc, last_fetch_at=now();
       Notification: "URL went 404: ..."

    4. If status_code IN (401, 403, 451):
       UPDATE discovered_sources SET
         curation_status='needs_auth' (only if was 'canonical'),
         last_fetch_status=$sc, last_fetch_at=now();
       Notification: "URL is now blocked. Was canonical: ..."
```

Implementation: a single Python module `app/curator/freshness_worker.py`
running as a Cloud Run job (or scheduled task) once daily. Reuses the
existing scrape-side fetch helpers via `httpx`. No new infra.

Why this matters: a payer can update a clinical policy on Tuesday and
chat will cite the stale Monday version forever. With freshness, we
catch it the same day, re-index, and the next chat citation is fresh.

## Migration story

The data model is designed so today's existing `documents` table is the
backing store for *ingested* sources. `discovered_sources` is a strict
*superset* — every doc has a row, but most rows are non-ingested
URLs (HTML pages, blocked PDFs, etc.). FK from `discovered_sources.ingested_doc_id`
to `documents.id` keeps the relationship explicit.

Backfill: for the 951 currently-published docs, we synthesize a
`discovered_sources` row each from the doc's `source_url` (already
captured during scraper-driven imports). One-time SQL.
