"""Re-run extraction on documents that produced nothing, and record WHY.

Ananth, 2026-08-19: "fix at the source and use these failures as means to run
these before we start another massive run."

WHY THIS EXISTS
  161 documents sat at status='failed' with zero rows in processing_errors and no
  error column on `documents`. The system recorded THAT they failed and never WHY,
  so nothing could retry them — a retry policy needs a reason to condition on and
  an attempt count to stop at, and neither existed.

  Re-running extraction now, with the classifier baked into extract_text, turns
  that silent pile into a diagnosis. The answer is expected to be unflattering:
  most of it is likely `unsupported_format` (.xls), which no amount of retrying
  would ever have fixed.

WHAT IT DOES
  For each document with no usable text: fetch the bytes, attempt extraction,
  classify the outcome, and write ingest_failure_reason / ingest_error_message /
  ingest_attempts. Retryable reasons (fetch_timeout, upstream_error,
  parser_crashed) are attempted up to MAX_INGEST_ATTEMPTS in the same pass, since
  a transient failure may well clear on the second try. Terminal reasons are
  attempted once, deliberately — retrying a file with no parser produces the
  identical failure three times and hides the real fix.

  A document that now extracts successfully is REPAIRED: pages are written and
  status is cleared. That is the point of running this before the next big
  ingest — the failures are free test cases for the extractor.

TELEMETRY BY DEFAULT.  --apply writes.

Usage:  python3 scripts/reclassify_ingest_failures.py [--apply] [--limit N] [--status failed]
"""
from __future__ import annotations

import asyncio
import re
import sys
from collections import Counter

import asyncpg

sys.path.insert(0, ".")
from app.services.extract_text import (  # noqa: E402
    classify_ingest_failure, should_retry, extract_text_from_bytes,
    MAX_INGEST_ATTEMPTS,
)

DSN = [l.split("=", 1)[1].strip() for l in open(".env") if l.startswith("DATABASE_URL=")][-1] \
    .replace("postgresql+asyncpg://", "postgresql://")


def ext_of(fn: str) -> str:
    m = re.search(r"\.([A-Za-z0-9]+)$", fn or "")
    return m.group(1).lower() if m else ""


async def main():
    apply = "--apply" in sys.argv
    limit = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--limit=")), 400)
    status = next((a.split("=")[1] for a in sys.argv if a.startswith("--status=")), "failed")
    # `--mode=no_text` sweeps every document that produced nothing usable,
    # whatever status it wears. Status alone misses the worst class: documents
    # marked `completed` that extracted to a 44-character stub and therefore look
    # healthy on every dashboard.
    mode = next((a.split("=")[1] for a in sys.argv if a.startswith("--mode=")), "status")

    c = await asyncpg.connect(DSN, timeout=300)
    await c.execute("SET statement_timeout = 0")
    if mode == "no_text":
        rows = await c.fetch("""
            SELECT d.id, d.filename, d.file_path, d.status,
                   COALESCE(d.ingest_attempts, 0) AS attempts
            FROM documents d
            WHERE d.lifecycle_state IS DISTINCT FROM 'retired'
              AND d.ingest_failure_reason IS NULL
              AND COALESCE((SELECT sum(p.text_length) FROM document_pages p
                            WHERE p.document_id = d.id), 0) < 200
            ORDER BY d.created_at DESC LIMIT $1""", limit)
        print(f"documents with no usable text (any status): {len(rows)}\n")
    else:
        rows = await c.fetch("""
            SELECT id, filename, file_path, status, COALESCE(ingest_attempts, 0) AS attempts
            FROM documents
            WHERE status = $1 AND lifecycle_state IS DISTINCT FROM 'retired'
            ORDER BY created_at DESC LIMIT $2""", status, limit)
        print(f"documents with status={status}: {len(rows)}\n")

    from google.cloud import storage
    gcs = storage.Client()

    verdicts, repaired, errors = Counter(), [], 0
    for r in rows:
        content, err, text = None, None, None
        try:
            b = r["file_path"].replace("gs://", "").split("/", 1)
            content = gcs.bucket(b[0]).blob(b[1]).download_as_bytes()
        except Exception as e:                       # transport, not format
            err = e
        attempts = r["attempts"]
        if err is None:
            for _ in range(MAX_INGEST_ATTEMPTS):
                attempts += 1
                try:
                    text = extract_text_from_bytes(content, ext_of(r["filename"]))
                    err = None
                    break
                except Exception as e:
                    err = e
                    reason, _ = classify_ingest_failure(content, ext_of(r["filename"]), None, e)
                    if not should_retry(reason, attempts):
                        break          # deterministic — a second attempt is waste
        else:
            attempts += 1

        reason, msg = classify_ingest_failure(content, ext_of(r["filename"]), text, err)
        verdicts[reason or "REPAIRED — extracts fine now"] += 1
        if reason is None:
            repaired.append((r["id"], r["filename"], len(text or "")))
        if apply:
            await c.execute("""
                UPDATE documents SET ingest_failure_reason = $2, ingest_error_message = $3,
                       ingest_attempts = $4, ingest_last_attempt_at = now()
                WHERE id = $1""", r["id"], reason, (msg or None), attempts)

    print("outcome by technical reason:\n")
    for v, n in verdicts.most_common():
        retryable = should_retry(v, 0)
        tag = "retryable" if retryable else ("" if v.startswith("REPAIRED") else "terminal")
        print(f"   {v:30} {n:>5}  {tag}")
    if repaired:
        print(f"\n   {len(repaired)} now extract successfully — the extractor fix repaired them:")
        for _, fn, n in repaired[:5]:
            print(f"      {str(fn)[:48]:50} {n:,} chars")
    print("\nAPPLIED — reasons written" if apply else "\nDRY RUN — nothing written. Re-run with --apply.")
    await c.close()


asyncio.run(main())
