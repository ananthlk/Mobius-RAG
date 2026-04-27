"""Backfill the curator + corpus from sources_seed.yaml.

The YAML is the operator's source-of-truth list. This script reads it
and dispatches each source per its declared ingest_strategy:

  scrape         → POST /scrape on web-scraper (tree mode), then
                   bulk-import the resulting GCS PDFs to rag.
  sitemap_only   → fetch sitemap.xml, bulk-upsert URLs to /sources/*
                   without a wide BFS (used when site is JS-heavy or
                   bot-walled at most paths but publishes a sitemap).
  state_mirror   → no scrape; record a placeholder discovered_sources
                   row so chat ReAct can surface "this exists but you
                   need AHCA / partner feed for it."
  partner_feed   → same — placeholder row, manual ingestion path.
  manual_upload  → same — placeholder row.

By default does ALL sources from the YAML. Use --tier P0 to filter,
--id <source_id> to run one source, --dry-run to print plan.

Usage:
  RAG_URL=https://...        \\
  SCRAPER_URL=https://...    \\
  ADMIN_API_KEY=$(...)       \\
  python3 scripts/curator/backfill_from_sources_yaml.py \\
      --tier P0              \\
      [--id samhsa]          \\
      [--dry-run]            \\
      [--max-pages 500]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import yaml  # PyYAML — used by mobius-rag elsewhere; standard dep.
except ImportError:
    print("ERROR: PyYAML not installed. pip install pyyaml", file=sys.stderr)
    sys.exit(2)


YAML_PATH = Path(__file__).resolve().parent / "sources_seed.yaml"


# ── Per-strategy handlers ────────────────────────────────────────────


def _scrape_source(
    *,
    src: dict,
    scraper_url: str,
    rag_url: str,
    admin_key: str,
    max_pages: int,
    max_depth: int,
    dry_run: bool,
) -> dict:
    """Fire a tree scan. Returns {job_id, status} dict."""
    seed_url = src.get("home_url")
    if not seed_url:
        return {"status": "skipped", "reason": "no home_url"}

    # If known_subtrees is set, prefer those as the seeds (more focused
    # than scanning the whole site). One job per subtree.
    seeds = src.get("known_subtrees") or [seed_url]

    job_ids: list[str] = []
    for seed in seeds:
        body = {
            "url": seed,
            "mode": "tree",
            "max_depth": max_depth,
            "max_pages": max_pages,
            "scope_mode": "same_domain",
            "include_content": True,
            "include_summary": False,
        }
        if dry_run:
            print(f"  DRY: POST {scraper_url}/scrape  body={body!r}")
            continue
        req = urllib.request.Request(
            f"{scraper_url}/scrape",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            r = json.loads(resp.read().decode())
        if "job_id" in r:
            job_ids.append(r["job_id"])
    return {"status": "queued" if not dry_run else "dry_run", "job_ids": job_ids}


def _sitemap_only(*, src, rag_url, admin_key, dry_run: bool) -> dict:
    """Fetch sitemap, upsert URLs to discovered_sources without scraping."""
    sitemap_url = src.get("sitemap_url") or (
        src["home_url"].rstrip("/") + "/sitemap.xml"
    )
    if dry_run:
        print(f"  DRY: fetch {sitemap_url} + bulk_upsert")
        return {"status": "dry_run"}
    # Reuse the existing scan_v0 / fetch_sitemap_urls path inline:
    import urllib.request as _u
    req = _u.Request(sitemap_url, headers={
        "User-Agent": "Mobius-WebScraper/1.0 (+https://github.com/mobius)"
    })
    try:
        body = _u.urlopen(req, timeout=20).read().decode()
    except Exception as e:
        return {"status": "error", "error": f"sitemap fetch failed: {e}"}
    # Crude parse — good enough for v1; replace with the production
    # fetch_sitemap_urls when this script becomes a service.
    import re
    locs = re.findall(r"<loc>([^<]+)</loc>", body)
    locs = [u for u in locs if u.startswith(("http://", "https://"))]
    if not locs:
        return {"status": "ok", "upserted": 0, "note": "sitemap empty"}
    # Bulk-upsert
    sources_payload = [{
        "url": u,
        "discovered_via": "sitemap",
        "fetch_status": None,  # not probed yet — operator can re-probe later
        "payer_hint": src.get("payer"),
        "state_hint": src.get("state"),
        "authority_hint": None,
    } for u in locs[:1000]]  # cap for safety
    upsert_req = _u.Request(
        f"{rag_url}/sources/bulk_upsert",
        data=json.dumps({"sources": sources_payload}).encode(),
        headers={"Content-Type": "application/json", "X-Admin-Api-Key": admin_key},
        method="POST",
    )
    with _u.urlopen(upsert_req, timeout=120) as resp:
        out = json.loads(resp.read().decode())
    return {"status": "ok", **out}


def _placeholder_row(*, src, rag_url, admin_key, dry_run: bool) -> dict:
    """For state_mirror / partner_feed / manual_upload sources: create
    one discovered_sources row at the home_url so chat ReAct can
    surface 'we know this source exists but ingestion path is X'.
    """
    if not src.get("home_url"):
        return {"status": "skipped", "reason": "no home_url"}
    if dry_run:
        print(f"  DRY: upsert placeholder for {src['home_url']}")
        return {"status": "dry_run"}
    body = {
        "url": src["home_url"],
        "discovered_via": "manual",
        "payer_hint": src.get("payer"),
        "state_hint": src.get("state"),
    }
    req = urllib.request.Request(
        f"{rag_url}/sources/upsert",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Admin-Api-Key": admin_key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        out = json.loads(resp.read().decode())
    return {"status": "ok", "id": out.get("id")}


# ── Main ─────────────────────────────────────────────────────────────


_HANDLERS = {
    "scrape":         _scrape_source,
    "sitemap_only":   _sitemap_only,
    "state_mirror":   _placeholder_row,
    "partner_feed":   _placeholder_row,
    "manual_upload":  _placeholder_row,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", help="Filter to tier (P0|P1|P2)")
    ap.add_argument("--id", help="Run one specific source id")
    ap.add_argument("--scope", help="Filter by scope (payer|state_agency|federal_agency|professional_org|accreditor|coding_standard)")
    ap.add_argument("--max-pages", type=int, default=300, help="Per-scrape page cap (default 300)")
    ap.add_argument("--max-depth", type=int, default=4, help="BFS depth (default 4)")
    ap.add_argument("--dry-run", action="store_true", help="Print plan, don't fire")
    args = ap.parse_args()

    rag_url = (os.environ.get("RAG_URL") or "").rstrip("/")
    scraper_url = (os.environ.get("SCRAPER_URL") or "").rstrip("/")
    admin_key = os.environ.get("ADMIN_API_KEY") or ""
    if not args.dry_run and (not rag_url or not scraper_url or not admin_key):
        print("ERROR: set RAG_URL + SCRAPER_URL + ADMIN_API_KEY (or --dry-run)", file=sys.stderr)
        sys.exit(1)

    data = yaml.safe_load(YAML_PATH.read_text())
    sources = data.get("sources", [])

    # Apply filters.
    if args.id:
        sources = [s for s in sources if s.get("id") == args.id]
    if args.tier:
        sources = [s for s in sources if s.get("tier") == args.tier]
    if args.scope:
        sources = [s for s in sources if s.get("scope") == args.scope]

    if not sources:
        print("No matching sources after filters.", file=sys.stderr)
        sys.exit(0)

    print(f"Processing {len(sources)} source(s):", file=sys.stderr)
    for s in sources:
        print(f"  - {s['id']:30s} ({s.get('tier')}, {s.get('ingest_strategy')})", file=sys.stderr)
    print("", file=sys.stderr)

    t0 = time.time()
    summary = []
    for src in sources:
        strategy = src.get("ingest_strategy", "scrape")
        handler = _HANDLERS.get(strategy)
        if not handler:
            summary.append((src["id"], strategy, {"status": "unknown_strategy"}))
            continue
        print(f"[{time.time()-t0:5.1f}s] {src['id']} ({strategy})...", file=sys.stderr)
        try:
            if strategy == "scrape":
                result = handler(
                    src=src, scraper_url=scraper_url, rag_url=rag_url,
                    admin_key=admin_key, max_pages=args.max_pages,
                    max_depth=args.max_depth, dry_run=args.dry_run,
                )
            else:
                result = handler(
                    src=src, rag_url=rag_url, admin_key=admin_key,
                    dry_run=args.dry_run,
                )
        except Exception as e:
            result = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        summary.append((src["id"], strategy, result))
        print(f"  → {result}", file=sys.stderr)

    # Final summary
    print("\n=== SUMMARY ===")
    for sid, strat, res in summary:
        status = res.get("status", "?")
        extra = ""
        if "job_ids" in res:
            extra = f" job_ids={res['job_ids']}"
        elif "upserted" in res:
            extra = f" upserted={res.get('inserted_or_updated', 0)}"
        elif "id" in res:
            extra = f" placeholder_id={res['id']}"
        print(f"  {sid:30s} {strat:15s} {status:12s}{extra}")
    print(f"\ntotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
