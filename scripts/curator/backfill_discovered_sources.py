"""Phase 13.3b — backfill the discovered_sources table.

Two sources of seed data:

1. sitemap_data_v0.json (tonight's scan) — 1066 URLs across Sunshine
   Health and AHCA, with status_code / content_type / inferred
   authority. These become discovered_via='sitemap' rows.

2. Existing documents table (951 ingested rows) — each has a
   ``source_url`` in source_metadata or the equivalent. These become
   discovered_via='bfs_link' rows with ingested=true and
   ingested_doc_id set.

Run mode:
* By default, talks to the deployed rag's /sources/bulk_upsert endpoint
  (read RAG_URL + ADMIN_API_KEY from env). Idempotent — re-running
  bumps last_seen_at but doesn't double-insert.
* Pass --dry-run to print what would be sent without hitting the API.

Usage:
  RAG_URL=https://mobius-rag-...run.app \\
  ADMIN_API_KEY=$(gcloud secrets versions access latest --secret=rag-admin-api-key) \\
  python3 scripts/curator/backfill_discovered_sources.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
SITEMAP_JSON = ROOT / "sitemap_data_v0.json"


# ── Source 1: sitemap scan v0 ────────────────────────────────────────


def from_sitemap_scan() -> list[dict]:
    """Convert tonight's scan results into upsert-ready dicts."""
    if not SITEMAP_JSON.exists():
        print(f"WARN: {SITEMAP_JSON} not found — sitemap source skipped", file=sys.stderr)
        return []
    raw = json.loads(SITEMAP_JSON.read_text())
    out: list[dict] = []
    for r in raw:
        # The scan recorded status, content_type, inferred authority,
        # payer/state from per-domain config. Translate into the
        # /sources/upsert schema.
        out.append({
            "url":              r["url"],
            "discovered_via":   "sitemap",
            "fetch_status":     r.get("status_code"),
            "content_type":     r.get("content_type") or None,
            "content_length":   r.get("content_length") or None,
            "payer_hint":       r.get("payer"),
            "state_hint":       r.get("state"),
            "authority_hint":   r.get("inferred_authority_level"),
        })
    return out


# ── Source 2: existing documents (the ingested 951) ──────────────────


def from_existing_documents(rag_url: str, admin_key: str) -> list[dict]:
    """Pull the existing /documents list and synthesize one row per
    doc with a usable URL. Marks them as already-ingested.

    Pre-curator imports stored URL in source_metadata or the file_path
    (GCS) — we prefer source_metadata['source_url'] when present.
    Docs without any URL are skipped (manual uploads have no URL).
    """
    print(f"  fetching existing documents from {rag_url}/documents ...", file=sys.stderr)
    req = urllib.request.Request(
        f"{rag_url}/documents?limit=2000",
        headers={"X-Admin-Api-Key": admin_key},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode())
    docs = body.get("documents", [])
    print(f"  got {len(docs)} documents", file=sys.stderr)

    out: list[dict] = []
    skipped_no_url = 0
    for d in docs:
        sm = d.get("source_metadata") or {}
        url = sm.get("source_url") or sm.get("url")
        if not url or not url.startswith(("http://", "https://")):
            skipped_no_url += 1
            continue
        out.append({
            "url":              url,
            "discovered_via":   "bfs_link",  # came in via scrape, not sitemap
            "fetch_status":     200,         # implicit — we ingested it
            "payer_hint":       d.get("payer"),
            "state_hint":       d.get("state"),
            "authority_hint":   d.get("authority_level"),
        })
    if skipped_no_url:
        print(f"  skipped {skipped_no_url} docs without source_url (manual uploads)", file=sys.stderr)
    return out


# ── Bulk upsert via API ──────────────────────────────────────────────


def bulk_upsert(rag_url: str, admin_key: str, batch: list[dict]) -> tuple[int, int]:
    """POST /sources/bulk_upsert. Returns (inserted_or_updated, failed)."""
    body = json.dumps({"sources": batch}).encode()
    req = urllib.request.Request(
        f"{rag_url}/sources/bulk_upsert",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Admin-Api-Key": admin_key,
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read().decode())
    return out.get("inserted_or_updated", 0), out.get("failed", 0)


def chunked(items: list, size: int) -> Iterable[list]:
    """Yield ``items`` in chunks of ``size``."""
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print summary, don't hit the API")
    ap.add_argument("--batch-size", type=int, default=200,
                    help="Records per /sources/bulk_upsert call")
    ap.add_argument("--skip-sitemap", action="store_true",
                    help="Don't seed from sitemap_data_v0.json")
    ap.add_argument("--skip-documents", action="store_true",
                    help="Don't seed from existing /documents rows")
    args = ap.parse_args()

    rag_url = (os.environ.get("RAG_URL") or "").rstrip("/")
    admin_key = os.environ.get("ADMIN_API_KEY") or ""
    if not args.dry_run and (not rag_url or not admin_key):
        print("ERROR: set RAG_URL + ADMIN_API_KEY env (or use --dry-run)", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    if not args.skip_sitemap:
        scan = from_sitemap_scan()
        print(f"  + {len(scan)} rows from sitemap scan", file=sys.stderr)
        rows.extend(scan)
    if not args.skip_documents and not args.dry_run:
        existing = from_existing_documents(rag_url, admin_key)
        print(f"  + {len(existing)} rows from existing documents", file=sys.stderr)
        rows.extend(existing)

    # Dedupe by URL — sitemap and documents overlap (e.g. an AHCA PDF
    # in both lists). Last-write-wins, with documents winning since
    # they carry ingested-implied state.
    by_url: dict[str, dict] = {}
    for r in rows:
        by_url[r["url"]] = r
    deduped = list(by_url.values())
    print(f"\nTotal unique URLs to upsert: {len(deduped)}", file=sys.stderr)

    if args.dry_run:
        from collections import Counter
        by_via = Counter(r.get("discovered_via") for r in deduped)
        by_status = Counter(r.get("fetch_status") for r in deduped)
        by_payer = Counter(r.get("payer_hint") for r in deduped)
        print(f"  by discovered_via: {dict(by_via)}", file=sys.stderr)
        print(f"  by fetch_status:   {dict(by_status)}", file=sys.stderr)
        print(f"  by payer_hint:     {dict(by_payer)}", file=sys.stderr)
        print("\n(dry run; no HTTP calls made)", file=sys.stderr)
        return

    total_ok = 0
    total_fail = 0
    t0 = time.time()
    for i, batch in enumerate(chunked(deduped, args.batch_size), 1):
        ok, fail = bulk_upsert(rag_url, admin_key, batch)
        total_ok += ok
        total_fail += fail
        print(f"  batch {i:3d}: {len(batch):4d} sent, {ok} ok, {fail} fail "
              f"({time.time()-t0:5.1f}s elapsed)", file=sys.stderr)
    print(f"\nDONE: upserted={total_ok} failed={total_fail} in {time.time()-t0:.1f}s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
