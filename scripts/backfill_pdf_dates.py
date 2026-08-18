"""One-time backfill: extract publication dates from every stored PDF.

Spec: docs/versioning-dedup-gate-spec.md §18 — publication time is the ordering
key for version chains, and we already discard it for 85% of PDF-backed docs.

WHAT IT WRITES
  documents.source_metadata->'pdf_meta'  ->  {creation_date, mod_date, producer, extracted_by}

  Written with jsonb_set, so every other key in source_metadata is preserved.
  Nothing else is touched — no lifecycle, no dates on the row, no chunks, no index.
  This is the existing pdf_meta convention (already present on 949 docs), NOT the
  new §9 columns — those await the DB seat's contract, and this promotes into them
  later without re-reading a single file.

HOW IT READS
  PDF /Info dictionaries are usually literal uncompressed bytes, so a byte-RANGE
  read of the head and tail finds /CreationDate without downloading whole files.
  Falls back to a full read only when the ranges miss. Avoids ~10GB of transfer.

Usage:  python3 scripts/backfill_pdf_dates.py [limit] [--dry-run]
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import asyncpg
from google.cloud import storage

DSN = [l.split("=", 1)[1].strip() for l in open(".env") if l.startswith("DATABASE_URL=")][-1] \
    .replace("postgresql+asyncpg://", "postgresql://")

HEAD = 256 * 1024
TAIL = 256 * 1024
WORKERS = 24

CREATED = re.compile(rb"/CreationDate\s*\(\s*(D:[^)]{8,})\)")
MODDATE = re.compile(rb"/ModDate\s*\(\s*(D:[^)]{8,})\)")
PRODUCER = re.compile(rb"/Producer\s*\(([^)]{0,80})\)")

_client = None


def client():
    global _client
    if _client is None:
        _client = storage.Client()
    return _client


def clean(v):
    """PDF metadata strings can carry NUL and control bytes. Postgres text cannot
    hold \\u0000, and asyncpg raises UntranslatableCharacterError on the whole
    batch — so one bad file would otherwise kill the run."""
    if v is None:
        return None
    v = v.replace("\x00", "")
    v = "".join(ch for ch in v if ch >= " " or ch == "\t")
    return v.strip()[:120] or None


def scan(buf: bytes):
    c = CREATED.search(buf)
    m = MODDATE.search(buf)
    p = PRODUCER.search(buf)
    return (c.group(1).decode("latin-1", "replace") if c else None,
            m.group(1).decode("latin-1", "replace") if m else None,
            p.group(1).decode("latin-1", "replace").strip() if p else None)


def fetch_meta(path: str):
    """Raw-byte regex first (cheap); pypdf fallback when /Info sits in a
    compressed object stream, which raw bytes cannot see. Verified: where the
    regex fires it agrees with pypdf exactly; where it misses, pypdf still finds
    the date on most files."""
    try:
        _, _, bucket, *rest = path.split("/")
        data = client().bucket(bucket).blob("/".join(rest)).download_as_bytes()
    except Exception as e:
        return {"_error": f"fetch:{type(e).__name__}"}

    if not data.lstrip()[:5].startswith(b"%PDF"):
        return {"_error": "not-a-pdf"}          # some .pdf files are HTML

    cr, md, pr = scan(data)
    if cr or md:
        return {"creation_date": clean(cr), "mod_date": clean(md), "producer": clean(pr),
                "extracted_by": "backfill_pdf_dates:regex"}
    try:
        import io as _io
        from pypdf import PdfReader
        meta = PdfReader(_io.BytesIO(data)).metadata or {}
        cr = str(meta.get("/CreationDate")) if meta.get("/CreationDate") else None
        md = str(meta.get("/ModDate")) if meta.get("/ModDate") else None
        pr = str(meta.get("/Producer"))[:80] if meta.get("/Producer") else None
        if cr or md:
            return {"creation_date": clean(cr), "mod_date": clean(md), "producer": clean(pr),
                    "extracted_by": "backfill_pdf_dates:pypdf"}
        return None
    except Exception as e:
        return {"_error": f"parse:{type(e).__name__}"}


async def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = int(args[0]) if args else 100000
    dry = "--dry-run" in sys.argv

    c = await asyncpg.connect(DSN, timeout=60)
    rows = await c.fetch("""
        SELECT id, file_path, filename FROM documents
        WHERE file_path ILIKE '%.pdf'
          AND NOT (coalesce(source_metadata->'pdf_meta','{}'::jsonb) ? 'creation_date')
        ORDER BY payer NULLS LAST, filename
        LIMIT $1""", limit)
    print(f"{len(rows)} PDFs to read{'  (DRY RUN — no writes)' if dry else ''}")

    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=WORKERS)
    ok = miss = err = 0
    batch = []

    SQL = """UPDATE documents SET source_metadata =
             jsonb_set(coalesce(source_metadata,'{}'::jsonb),'{pdf_meta}',$2::jsonb,true)
             WHERE id=$1"""

    async def flush():
        nonlocal err
        if batch and not dry:
            try:
                await c.executemany(SQL, batch)
            except Exception:
                # executemany is all-or-nothing; fall back per row so one bad
                # value costs one document, not the whole batch
                for did, js in batch:
                    try:
                        await c.execute(SQL, did, js)
                    except Exception:
                        err += 1
        batch.clear()

    for i in range(0, len(rows), WORKERS * 4):
        window = rows[i:i + WORKERS * 4]
        metas = await asyncio.gather(*[
            loop.run_in_executor(pool, fetch_meta, r["file_path"]) for r in window])
        for r, m in zip(window, metas):
            if m is None:
                miss += 1
            elif "_error" in m:
                err += 1
            else:
                ok += 1
                batch.append((r["id"], json.dumps(m)))
        await flush()
        done = i + len(window)
        print(f"  {done:>5}/{len(rows)}   extracted={ok}  no-date={miss}  error={err}", flush=True)

    print(f"\nDONE  extracted={ok}  no-date-in-file={miss}  error={err}")
    if not dry:
        n = await c.fetchval(
            "SELECT count(*) FROM documents WHERE source_metadata->'pdf_meta' ? 'creation_date'")
        print(f"corpus now carrying a publication date: {n}")
    await c.close()


asyncio.run(main())
