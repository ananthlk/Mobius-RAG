"""Curator module — durable URL registry + freshness loop.

Owns:
* discovered_sources table (the URL registry)
* /sources/* HTTP endpoints (upsert, search, curate, ingest)
* URL classification (URL → payer/authority inference)
* Freshness worker (daily HEAD/hash-diff re-fetch, runs as cron)

Lives inside mobius-rag rather than a separate skill because:
* shares documents.id FK directly (no cross-service coordination)
* single API surface for chat ReAct
* freshness re-import path reuses rag's existing pipeline

The web-scraper stays dumb (fetch + return). All registry,
classification, and freshness intelligence lives here.

Full design: scripts/curator/SCHEMA.md
"""
