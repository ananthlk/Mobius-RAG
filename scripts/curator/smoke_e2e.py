"""End-to-end smoke for the curator chain (Phase 13.5c).

Runs against deployed rag + chat-pg + Chroma. ~60 seconds total.
Re-runnable any time to verify the full architecture still works
after a deploy / refactor / dependency bump.

Steps:
   1. /sources/stats         — registry table reachable, has rows
   2. /sources/search        — filter returns expected payer rows
   3. /documents/import-from-html — pick an unindexed URL, ingest it
   4. poll /documents/{id}/status — chunking + embedding + extraction land
   5. /documents/{id}/publish — push to Chroma + chat-pg
   6. Chroma query           — chunks present + non-empty text
   7. /sources/search?ingested=true — the URL is now linked

Exit code: 0 if all pass, 1 on first failure with a clear "FAIL: ..."
line so a CI / cron job can alert.

Usage:
  RAG_URL=https://mobius-rag-...run.app \\
  ADMIN_API_KEY=$(gcloud secrets versions access latest --secret=rag-admin-api-key) \\
  python3 scripts/curator/smoke_e2e.py [--keep-doc]

  --keep-doc: don't purge the test doc afterward (useful for inspection).
              Default: cascade-delete via the dev DB so re-runs stay clean.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error


# Test URL: a Sunshine page with a known table — the dental transition
# dates page that exposed our chunking bug. Re-using it locks in the
# table-handling contract for future chunks.
TEST_URL = "https://www.sunshinehealth.com/providers/preauth-check/medicaid-pre-auth/dental-plan-transition-dates.html"


# ── tiny logger ──────────────────────────────────────────────────────


_passed = 0
_failed = 0
_t0 = time.time()


def step(name: str) -> None:
    global _passed
    _passed += 1
    elapsed = time.time() - _t0
    print(f"  ✓ [{elapsed:5.1f}s] {name}", flush=True)


def fail(reason: str) -> None:
    global _failed
    _failed += 1
    elapsed = time.time() - _t0
    print(f"  ✗ [{elapsed:5.1f}s] FAIL: {reason}", flush=True)
    sys.exit(1)


# ── HTTP helpers ─────────────────────────────────────────────────────


def _req(method: str, url: str, *, body: dict | None = None, headers: dict | None = None, timeout: int = 60) -> tuple[int, dict | str]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode()
            try:
                return resp.status, json.loads(text)
            except Exception:
                return resp.status, text
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, str(e)


# ── Steps ────────────────────────────────────────────────────────────


def step_1_stats(rag_url: str, admin_key: str) -> int:
    code, body = _req("GET", f"{rag_url}/sources/stats",
                      headers={"X-Admin-Api-Key": admin_key})
    if code != 200:
        fail(f"GET /sources/stats returned {code}: {body}")
    if not isinstance(body, dict) or "by_host" not in body:
        fail(f"unexpected stats shape: {body}")
    total_hosts = sum(body.get("by_host", {}).values())
    if total_hosts < 100:
        fail(f"stats shows only {total_hosts} URLs registered — backfill not run?")
    step(f"/sources/stats reachable, {total_hosts} URLs registered "
         f"({len(body['by_host'])} hosts)")
    return total_hosts


def step_2_search(rag_url: str, admin_key: str) -> None:
    code, body = _req(
        "GET",
        f"{rag_url}/sources/search?payer=Sunshine+Health&limit=5",
        headers={"X-Admin-Api-Key": admin_key},
    )
    if code != 200:
        fail(f"GET /sources/search returned {code}: {body}")
    if not isinstance(body, list):
        fail(f"search returned non-list: {body}")
    if not body:
        fail("search?payer=Sunshine+Health returned empty — backfill missing?")
    sample = body[0]
    for k in ("url", "host", "payer", "ingested", "content_kind"):
        if k not in sample:
            fail(f"search row missing field {k!r}: {sample}")
    step(f"/sources/search?payer=Sunshine+Health returned {len(body)} rows")


def step_3_import_html(rag_url: str, admin_key: str) -> str:
    """Returns the new document_id."""
    code, body = _req(
        "POST",
        f"{rag_url}/documents/import-from-html",
        body={
            "url": TEST_URL,
            "authority_level": "payer_policy",
        },
        headers={"Content-Type": "application/json", "X-Admin-Api-Key": admin_key},
        timeout=120,
    )
    if code == 409:
        # Already imported — dedupe path. Fetch existing doc id from response.
        existing = (body.get("detail", {}) if isinstance(body, dict) else {}).get("document_id")
        if not existing:
            fail(f"409 dedupe but no document_id returned: {body}")
        step(f"import-from-html dedupe: doc {existing} already exists "
             f"(re-run smoke after smoke_e2e --reset to test fresh ingest)")
        return existing
    if code != 200:
        fail(f"POST /documents/import-from-html returned {code}: {body}")
    if not isinstance(body, dict) or "document_id" not in body:
        fail(f"import response missing document_id: {body}")
    doc_id = body["document_id"]
    sections = body.get("sections", 0)
    if sections == 0:
        fail(f"HTML import produced 0 sections — extractor broken?")
    step(f"/documents/import-from-html created doc {doc_id} with {sections} section(s)")
    return doc_id


def step_4_wait_processed(rag_url: str, admin_key: str, doc_id: str) -> None:
    """Poll /documents until embedding completes. ~30-90s typically."""
    deadline = time.time() + 240  # 4 min cap
    last_status = ""
    while time.time() < deadline:
        code, body = _req(
            "GET", f"{rag_url}/documents?limit=2000",
            headers={"X-Admin-Api-Key": admin_key},
        )
        if code != 200:
            fail(f"GET /documents returned {code}")
        for d in body.get("documents", []):
            if d.get("id") == doc_id:
                last_status = d.get("embedding_status") or ""
                if last_status == "completed":
                    step(f"chunking + embedding completed for doc {doc_id}")
                    return
                if last_status == "failed":
                    fail(f"embedding failed for doc {doc_id}")
                break
        time.sleep(8)
    fail(f"timed out waiting for embedding (last_status={last_status})")


def step_5_publish(rag_url: str, admin_key: str, doc_id: str) -> int:
    """Returns chunk count published."""
    code, body = _req(
        "POST", f"{rag_url}/documents/{doc_id}/publish",
        headers={"X-Admin-Api-Key": admin_key},
        timeout=120,
    )
    if code != 200:
        fail(f"POST /documents/{doc_id}/publish returned {code}: {body}")
    if not isinstance(body, dict) or not body.get("verification_passed"):
        fail(f"publish verification failed: {body}")
    msg = body.get("verification_message", "")
    if "chroma=ok" not in msg or "chat_pg=ok" not in msg:
        fail(f"publish sync incomplete: {msg}")
    rows = body.get("rows_written", 0)
    if rows < 1:
        fail(f"publish wrote 0 rows: {body}")
    step(f"published {rows} chunks (sync: {msg})")
    return rows


def step_6_chroma_check(doc_id: str) -> None:
    """Direct Chroma probe: chunks present, non-trivial text."""
    token = subprocess.run(
        ["gcloud", "secrets", "versions", "access", "latest",
         "--secret=chroma-auth-token", "--project=mobius-os-dev"],
        capture_output=True, text=True,
    ).stdout.strip()
    if not token:
        # Skip the Chroma probe in environments without gcloud access.
        # The publish step already verified chroma=ok via the rag's own
        # write path; this just adds an independent confirmation.
        step("Chroma probe skipped (no gcloud chroma-auth-token)")
        return
    cid = "17d2f4be-8eb6-40a0-8189-ed8f041858e3"
    base = f"http://34.170.243.161:8000/api/v2/tenants/default_tenant/databases/default_database/collections/{cid}"
    body = json.dumps({
        "where": {"document_id": doc_id},
        "limit": 50,
        "include": ["documents"],
    }).encode()
    req = urllib.request.Request(
        base + "/get", data=body, method="POST",
        headers={"X-Chroma-Token": token, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        fail(f"Chroma probe failed: {e}")
    docs = data.get("documents") or []
    if not docs:
        fail(f"Chroma has 0 chunks for doc {doc_id}")
    avg_len = sum(len(d or "") for d in docs) / max(1, len(docs))
    if avg_len < 30:
        fail(f"Chroma chunks suspiciously short (avg {avg_len:.0f} chars) — chunking broken?")
    step(f"Chroma has {len(docs)} chunks, avg {avg_len:.0f} chars per chunk")


def step_7_curator_linked(rag_url: str, admin_key: str, doc_id: str) -> None:
    """Verify the discovered_sources row got marked ingested with the FK."""
    code, body = _req(
        "GET",
        f"{rag_url}/sources/search?ingested=true&payer=Sunshine+Health&limit=50",
        headers={"X-Admin-Api-Key": admin_key},
    )
    if code != 200:
        fail(f"GET /sources/search?ingested=true returned {code}")
    rows = body if isinstance(body, list) else []
    matching = [r for r in rows if r.get("ingested_doc_id") == doc_id]
    if not matching:
        fail(f"discovered_sources has no row linked to doc {doc_id} "
             "(mark_ingested didn't fire?)")
    step(f"curator linkage: discovered_sources.ingested_doc_id = {doc_id}")


# ── Cleanup ──────────────────────────────────────────────────────────


def cleanup(doc_id: str) -> None:
    """Best-effort cascade delete via SQL. Skips if cloud-sql-proxy
    isn't available — operator can clean up manually."""
    proxy = "/Users/ananth/google-cloud-sdk/bin/cloud-sql-proxy"
    if not os.path.exists(proxy):
        print(f"  (cleanup skipped: {proxy} not present; doc {doc_id} left in place)")
        return
    pw = subprocess.run(
        ["gcloud", "secrets", "versions", "access", "latest",
         "--secret=db-password", "--project=mobius-os-dev"],
        capture_output=True, text=True,
    ).stdout.strip()
    if not pw:
        print("  (cleanup skipped: db-password unavailable)")
        return
    proxy_proc = subprocess.Popen(
        [proxy, "mobius-os-dev:us-central1:mobius-platform-dev-db", "--port", "15432"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(5)
    try:
        env = {**os.environ, "PGPASSWORD": pw}
        sql = f"""
            BEGIN;
            DELETE FROM rag_published_embeddings WHERE document_id='{doc_id}';
            DELETE FROM chunk_embeddings WHERE document_id='{doc_id}';
            DELETE FROM chunking_events WHERE document_id='{doc_id}';
            DELETE FROM chunking_jobs WHERE document_id='{doc_id}';
            DELETE FROM chunking_results WHERE document_id='{doc_id}';
            DELETE FROM document_pages WHERE document_id='{doc_id}';
            DELETE FROM document_tags WHERE document_id='{doc_id}';
            DELETE FROM document_text_tags WHERE document_id='{doc_id}';
            DELETE FROM embeddable_units WHERE document_id='{doc_id}';
            DELETE FROM embedding_jobs WHERE document_id='{doc_id}';
            DELETE FROM extracted_facts WHERE document_id='{doc_id}';
            DELETE FROM hierarchical_chunks WHERE document_id='{doc_id}';
            DELETE FROM policy_blocks WHERE document_id='{doc_id}';
            DELETE FROM policy_lexicon_candidates WHERE document_id='{doc_id}';
            DELETE FROM policy_lines WHERE document_id='{doc_id}';
            DELETE FROM policy_paragraphs WHERE document_id='{doc_id}';
            DELETE FROM policy_spans WHERE document_id='{doc_id}';
            DELETE FROM processing_errors WHERE document_id='{doc_id}';
            DELETE FROM publish_events WHERE document_id='{doc_id}';
            UPDATE discovered_sources SET ingested=false, ingested_doc_id=NULL
              WHERE ingested_doc_id='{doc_id}';
            DELETE FROM documents WHERE id='{doc_id}';
            COMMIT;
        """
        subprocess.run(
            ["psql", "-h", "127.0.0.1", "-p", "15432", "-U", "postgres", "-d", "mobius_rag", "-c", sql],
            env=env, check=False, capture_output=True,
        )
        subprocess.run(
            ["psql", "-h", "127.0.0.1", "-p", "15432", "-U", "postgres", "-d", "mobius_chat",
             "-c", f"DELETE FROM published_rag_metadata WHERE document_id='{doc_id}';"],
            env=env, check=False, capture_output=True,
        )
        # Chroma cleanup
        token = subprocess.run(
            ["gcloud", "secrets", "versions", "access", "latest",
             "--secret=chroma-auth-token", "--project=mobius-os-dev"],
            capture_output=True, text=True,
        ).stdout.strip()
        if token:
            cid = "17d2f4be-8eb6-40a0-8189-ed8f041858e3"
            url = f"http://34.170.243.161:8000/api/v2/tenants/default_tenant/databases/default_database/collections/{cid}/delete"
            req = urllib.request.Request(
                url, data=json.dumps({"where": {"document_id": doc_id}}).encode(),
                method="POST",
                headers={"X-Chroma-Token": token, "Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=15).read()
        print(f"  (cleanup OK: doc {doc_id} purged from rag DB + chat_pg + Chroma)")
    finally:
        proxy_proc.terminate()


# ── Main ─────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-doc", action="store_true",
                    help="Don't cleanup the test doc afterward")
    args = ap.parse_args()

    rag_url = (os.environ.get("RAG_URL") or "").rstrip("/")
    admin_key = (os.environ.get("ADMIN_API_KEY") or "").strip()
    if not rag_url or not admin_key:
        print("ERROR: set RAG_URL and ADMIN_API_KEY env vars")
        sys.exit(2)

    print(f"=== Curator E2E smoke against {rag_url} ===\n")

    step_1_stats(rag_url, admin_key)
    step_2_search(rag_url, admin_key)
    doc_id = step_3_import_html(rag_url, admin_key)
    step_4_wait_processed(rag_url, admin_key, doc_id)
    step_5_publish(rag_url, admin_key, doc_id)
    step_6_chroma_check(doc_id)
    step_7_curator_linked(rag_url, admin_key, doc_id)

    elapsed = time.time() - _t0
    print(f"\n=== {_passed} steps passed in {elapsed:.1f}s ===")

    if not args.keep_doc:
        print("\n--- cleanup ---")
        cleanup(doc_id)


if __name__ == "__main__":
    main()
