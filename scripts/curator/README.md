# Curator scaffolding (Phase 13)

This directory holds **design + scripts for the curator**, the URL
registry that turns Mobius from "scrape what you point us at" into
"intelligently maintained source-of-truth registry."

The runtime code lives in [`app/curator/`](../../app/curator/):
* `__init__.py` — module overview
* `classifier.py` — URL → payer/authority inference (regex, 27 tests)
* `service.py` — SQL layer (upsert, search, curate, mark_ingested, stats)
* `routes.py` — FastAPI router mounted at `/sources/*`

The table lives in [`app/migrations/add_discovered_sources.py`](../../app/migrations/add_discovered_sources.py).

## Files in this directory

| File | What |
|---|---|
| `SCHEMA.md` | Full design — data model, API, freshness model, ReAct surface |
| `scan_v0.py` | The standalone scanner that produced sitemap_data_v0.json |
| `sitemap_data_v0.json` | First scan results: 1066 URLs across Sunshine + AHCA |
| `first_scan_report.md` | Human-readable inventory by host/status/authority |
| `backfill_discovered_sources.py` | Seed the table from scan + existing docs |

## How to run the backfill (after deploy)

```bash
RAG_URL=https://mobius-rag-ortabkknqa-uc.a.run.app \
ADMIN_API_KEY=$(gcloud secrets versions access latest \
                  --secret=rag-admin-api-key --project=mobius-os-dev) \
python3 scripts/curator/backfill_discovered_sources.py
```

Idempotent — re-running bumps `last_seen_at` but doesn't double-row.
Dry run with `--dry-run` first to see what would be sent.

## Status

- ✅ `discovered_sources` model + migration (`d041e47`)
- ✅ `app/curator/` module: routes, service, classifier (`39b20ae`)
- ✅ Backfill script — 1,066 URLs seeded into prod (`2ad5382`)
- ✅ Classifier extension guard (`803a2ce`)
- ✅ HTML pipeline: `/documents/import-from-html` (`dff5f4b`)
- ✅ Table-aware HTML extraction (`88e72d1`)
- ✅ ReAct tools `lookup_authoritative_sources` + `ingest_url` (chat repo `d3a1f23`)
- ✅ Hardening pass: service tests + extractor edges + E2E smoke (`858fcbf`)
- ✅ End-to-end proven in production: chat retrieved freshly-ingested
   `Dental Plan Transition Dates` row with Pasco County / Region 5 precision
- ⬜ Scraper-side push: `web-scraper` POSTs `/sources/upsert` per URL (Phase 13.3c)
- ⬜ Freshness worker — daily HEAD/hash-diff (Phase 13.10)
- ⬜ Auto-publish-on-embed (Phase 13.8)
- ⬜ Lexicon pull cron (Phase 13.9)
- ⬜ Email notifier (Phase 13.7)

## Test inventory

| Suite | Tests | Runtime | What it locks down |
|---|---|---|---|
| `tests/test_curator_classifier.py` | 29 | ~30ms | URL → payer/authority/kind inference, host mapping, extension-detection edge cases |
| `tests/test_curator_service.py` | 7 | ~600ms | Search ranking (canonical→ingested→recency), filter narrowing, only_reachable, curate validation, classifier wire-up |
| `tests/test_html_extractor.py` | 30 | ~110ms | Title derivation, heading splits, boilerplate stripping, table-row context, lists, blockquotes, iframes, edge sizes |
| `tests/test_curator_tools.py` (chat repo) | 11 | ~190ms | ReAct tool handlers — HTTP mocking, 5xx/409 paths, query param forwarding |
| `scripts/curator/smoke_e2e.py` | 7 steps | ~4s | Live chain check against deployed services. Re-runnable any time. |
| **Total** | **84 cases** | **~5s** | Full curator + HTML pipeline coverage |

## Running everything

```bash
# Unit tests (DB-mocked, no deploy needed)
cd /Users/ananth/Mobius/mobius-rag
python3 -m pytest tests/test_curator_classifier.py tests/test_curator_service.py \
                  tests/test_html_extractor.py -v

# Chat-side tests
cd /Users/ananth/Mobius/mobius-chat
python3 -m pytest tests/test_curator_tools.py -v

# Live E2E smoke (requires deploy)
cd /Users/ananth/Mobius/mobius-rag
RAG_URL=https://mobius-rag-ortabkknqa-uc.a.run.app \
ADMIN_API_KEY=$(gcloud secrets versions access latest \
                  --secret=rag-admin-api-key --project=mobius-os-dev) \
python3 scripts/curator/smoke_e2e.py
```

## Adding a new test case

* Pure URL classification → `test_curator_classifier.py`
* DB query / upsert behavior → `test_curator_service.py`
* HTML parsing edge case → `test_html_extractor.py`
* ReAct tool handler logic → `mobius-chat/tests/test_curator_tools.py`
* End-to-end live behavior → add a step function to `smoke_e2e.py`
