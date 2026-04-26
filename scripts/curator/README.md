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

## Status (end of Phase 13.3a)

- ✅ `discovered_sources` model + migration (commit `d041e47`)
- ✅ `app/curator/` module: routes, service, classifier (commit `39b20ae`)
- ✅ 27 classifier unit tests passing
- ✅ Backfill script ready (this commit)
- ⬜ Deploy (next)
- ⬜ Run backfill against prod
- ⬜ Wire scraper-side push: `web-scraper` POSTs `/sources/upsert` per URL
- ⬜ HTML pages → rag (Phase 13.4)
- ⬜ ReAct tools: `lookup_authoritative_sources`, `ingest_url` (Phase 13.5)
- ⬜ Freshness worker (Phase 13.10)
