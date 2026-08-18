"""Duplicate determination — the dedup half of the gate.

TELEMETRY BY DEFAULT. `--apply` writes lifecycle_state / supersedes_id, and only
for the one kind where the determination is not a judgement call (exact_text).

WHY THIS IS A SEPARATE PASS FROM gate_run_corpus.py
---------------------------------------------------
The versioning gate only ever compares documents that share a doc_key, and
doc_key is NULL for most of the corpus (episodic documents by design, plus
anything matching neither the rule-number nor the revisable-title pattern). Two
copies of the same drug-policy PDF therefore never met inside that gate: both
scored `first_version` against an empty sibling list. That is the whole reason
dedup had no persisted status while versioning did — not a missing writer, a
missing comparison.

Duplication is not lineage. It needs its own candidate generator.

THE INSTRUMENT
--------------
Shingle Jaccard (8-word) over normalized `document_pages` text, NOT chunk
identity. Chunk-identity overlap reads ~0 whenever chunk boundaries shift, and
near-identical editions almost always shift boundaries: an 8-character edit
re-flows every downstream chunk. Measured, on known revisions:

    Panretin ~ CMS-Panretin    chunk 0.000 "unrelated"   text 0.804  version
    Fuzeon   ~ CMS-Fuzeon      chunk 0.000 "unrelated"   text 0.924  version

So chunk identity confirms exact duplicates and cannot separate the two verdicts
that matter. `content_digest` equality stays as the fast exact pre-pass.

CANDIDATE GENERATION, AND THE BUG IT FIXES
------------------------------------------
The first generator normalized filenames with

    re.sub(r'^(cms|ahca|fl|sh|bh)[-_]+', '', stem)

which strips product and plan markers so stems match — and thereby deletes the
only signal that separates a *version* from a *product variant*. It manufactured
exactly those pairs and destroyed their discriminator in the same line.

Here the prefix is stripped for RECALL (so the pair is generated) and retained
for DETERMINATION (so it can be classified). Same normalization, two outputs.

PRODUCT IS READ FROM THE PAGE FIRST, THE NAME SECOND
----------------------------------------------------
`CMS-` carries both meanings in this corpus: of 45 CMS-prefixed documents, 8 say
"Children's Medical Services" in their text and 4 say "Centers for Medicare &
Medicaid Services". A rule reading the prefix alone misfiles one group or the
other. Text declaration wins; the prefix is a prior; where both are silent the
pair is `product_unknown` and nobody guesses.

Usage:  python3 scripts/gate_duplicates.py [--apply] [--limit N]
"""
from __future__ import annotations

import asyncio
import itertools
import json
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import date

import asyncpg

DSN = [l.split("=", 1)[1].strip() for l in open(".env") if l.startswith("DATABASE_URL=")][-1] \
    .replace("postgresql+asyncpg://", "postgresql://")

# "Duplicate" is reserved for pairs where EVERY signal agrees — Ananth, 2026-08-18:
# match only if everything matches. Text overlap alone was never sufficient: a
# blank annual form is identical every year, and one product's copy of a policy is
# byte-identical to another's. Both are legitimate separate documents. So a pair is
# a duplicate only when text, length, page count, reporting period and product ALL
# agree; any single disagreement routes it to a holding bucket instead.
TAU_EXACT = 1.0           # identical text, not merely near-identical
TAU_NEAR = 0.35           # below this the pair is unrelated, not a duplicate
SHINGLE = 8
NORM_VERSION = "v2-text-shingle"

# Product / plan markers that appear as filename prefixes. Stripped to generate
# candidates, kept to classify them.
PREFIX = re.compile(r"^(cms|ahca|fl|sh|bh)[-_]+", re.I)
PERIOD = re.compile(r"20[0-9]{2}([-_][0-9]{1,2})*")
# Reporting-period tokens. Two documents whose names differ ONLY by one of these
# are a PERIOD SERIES — the same blank form issued for a different year/quarter —
# not two copies of one document. Their text is identical *by design*, so text
# overlap scores 1.000 and says "duplicate" with total confidence. Retiring the
# later one would delete this year's attestation in favour of last year's, which
# is what the first --apply dry check was about to do to 21 documents.
PERIOD_TOK = re.compile(r"(sfy[-_ ]?\d{2,4}([-_]\d{2,4})?|fy[-_ ]?\d{2,4}([-_]\d{2,4})?|"
                        r"20\d{2}([-_]\d{1,2})*|\bq[1-4]\b|\d{4}[-_]\d{2})", re.I)
FNDATE = re.compile(r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})|(\d{1,2})[-_](\d{1,2})[-_](\d{2})(?!\d)")
PDFD = re.compile(r"D:(\d{4})(\d{2})(\d{2})")

# Product declarations, read out of the document's own text.
PRODUCT_TEXT = [
    ("CMS", re.compile(r"children'?s medical services", re.I)),
    ("FEDERAL", re.compile(r"centers for medicare\s*&?\s*medicaid|centers for medicaid", re.I)),
    ("LTC", re.compile(r"\blong[- ]term care\b", re.I)),
    ("MMA", re.compile(r"\bmanaged medical assistance\b|\bMMA\b")),
]


def stem_for_recall(fn: str) -> str:
    s = fn.lower().rsplit(".", 1)[0]
    s = PREFIX.sub("", s)
    s = PERIOD.sub("", s)
    return re.sub(r"[^a-z0-9]", "", s)


def period_tokens(fn: str) -> tuple:
    """Period markers present in a filename, normalized for comparison."""
    return tuple(sorted({m.group(0).lower().replace("_", "-").replace(" ", "")
                         for m in PERIOD_TOK.finditer(fn.rsplit(".", 1)[0])}))


def prefix_of(fn: str) -> str:
    m = PREFIX.match(fn or "")
    return m.group(1).lower() if m else ""


def edition_date(fn, pdf_meta, created_at):
    """§19.3 ladder — stated edition outranks publication metadata."""
    m = FNDATE.search(str(fn or ""))
    if m:
        try:
            return ((date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m.group(1)
                     else date(2000 + int(m.group(6)), int(m.group(4)), int(m.group(5)))), "filename")
        except ValueError:
            pass
    for key, rung in (("mod_date", "source_modified"), ("creation_date", "source_created")):
        raw = (pdf_meta or {}).get(key) if isinstance(pdf_meta, dict) else None
        mm = PDFD.match(str(raw or ""))
        if mm:
            try:
                return date(int(mm.group(1)), int(mm.group(2)), int(mm.group(3))), rung
            except ValueError:
                pass
    # first_seen is deliberately NOT a rung here. It records when we fetched the
    # document, which says nothing about which edition is newer — using it would
    # manufacture ordering confidence the corpus does not have.
    return None, "none"


def shingles(text: str) -> set:
    w = re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", (text or "").lower())).split()
    return {tuple(w[i:i + SHINGLE]) for i in range(max(0, len(w) - SHINGLE + 1))}


def product_of(text: str, fn: str):
    """(product, source). Text declaration beats the filename prefix."""
    for name, rx in PRODUCT_TEXT:
        if rx.search(text or ""):
            return name, "text"
    p = prefix_of(fn)
    if p in ("cms", "sh", "bh"):
        return p.upper(), "prefix(weak)"
    return None, "none"


def classify(tj, pa, sa, pb, sb, da, db, prefix_differs, per_a, per_b,
             same_len, same_pages, chars_a, pages_a):
    """Returns (duplicate_kind, reason). Ordering of the branches is the policy."""
    # FIRST, before any overlap reasoning. A period series is the one case where
    # identical text is positive evidence that the documents are DIFFERENT: a
    # blank annual form is identical every year, and each year's copy is the
    # record for its own year. Nothing here is ever retired.
    if per_a and per_b and per_a != per_b:
        return "period_series", (f"overlap {tj:.3f} but reporting periods differ "
                                 f"({'/'.join(per_a)} vs {'/'.join(per_b)}) — "
                                 f"same form, different period; both stay")

    # ASYMMETRIC period: one name carries a period, the other does not. Almost
    # always an undated master form plus a filed copy for one year
    # (GME_Startup_Bonus_Attestation.pdf ~ ..._2016-17.pdf). Which is canonical
    # is a real question — the master may be current, or may be a stale artifact —
    # and it is not answerable from the filename, so nothing is retired.
    if bool(per_a) != bool(per_b):
        return "period_series", (f"overlap {tj:.3f}; one side carries a period "
                                 f"({'/'.join(per_a or per_b)}) and the other does not — "
                                 f"master-vs-filed-copy, needs a human; both stay")

    # Different product, EVEN AT IDENTICAL TEXT. The same form issued for two
    # products is byte-identical and still two documents: CMS-Compound-over-300 ~
    # Compound-over-300 scores 1.000 and retiring either one removes a product's
    # copy of its own policy. This must precede the exact_text branch, which would
    # otherwise claim the pair with total confidence.
    if (pa or pb) and pa != pb:
        conf = "text" if (sa == "text" and sb == "text") else "name prefix"
        return "product_variant", (f"overlap {tj:.3f}; products differ "
                                   f"({pa or 'undeclared'} vs {pb or 'undeclared'}, by {conf}) "
                                   f"— both stay")
    if prefix_differs:
        return "product_unknown", (f"overlap {tj:.3f}; name prefixes differ but neither "
                                   f"document declares a product — cannot tell variant "
                                   f"from copy; both stay")
    if tj >= TAU_EXACT and same_len and same_pages:
        return "duplicate", (f"every signal agrees — identical text, {chars_a} chars, "
                             f"{pages_a} pages, same period, same product")
    if tj >= 0.98:
        return "near_identical_review", (f"overlap {tj:.3f} but "
                                         + ("lengths differ" if not same_len else "page counts differ")
                                         + " — not every signal agrees; both stay")

    if da and db and da != db:
        return "near_duplicate", f"overlap {tj:.3f}; dated {da} -> {db}"

    return "ordering_unknown", f"overlap {tj:.3f}; no usable edition date"


async def main():
    apply = "--apply" in sys.argv
    limit = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--limit=")), None)
    t0 = time.time()
    c = await asyncpg.connect(DSN, timeout=300)
    await c.execute("SET statement_timeout = 0")
    # asyncpg returns jsonb as str unless told otherwise. Without this codec the
    # `isinstance(pdf_meta, dict)` guard in the date ladder is ALWAYS False, so the
    # publication-date rung silently never fires and every document falls through
    # to filename-or-nothing. Same silent-inert class as the \b(oct)\b fix.
    await c.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads,
                           schema="pg_catalog")

    docs = await c.fetch("""
        SELECT d.id, d.filename, d.payer, d.state, d.created_at, d.authority_level,
               d.source_metadata->'pdf_meta' AS pdf_meta,
               d.source_metadata->'payor_classification'->>'importance' AS importance
        FROM documents d WHERE d.filename IS NOT NULL""")
    print(f"documents: {len(docs)}", flush=True)

    meta = {}
    groups = defaultdict(list)
    for d in docs:
        ed, rung = edition_date(d["filename"], d["pdf_meta"], d["created_at"])
        meta[d["id"]] = dict(d=d, fn=d["filename"], edate=ed, rung=rung)
        groups[stem_for_recall(d["filename"])].append(d["id"])

    pairs = [(a, b) for k, v in groups.items() if 2 <= len(v) <= 6
             for a, b in itertools.combinations(v, 2)]
    if limit:
        pairs = pairs[:limit]
    print(f"candidate pairs (stem, prefix-stripped for recall): {len(pairs)}", flush=True)

    ids = {i for p in pairs for i in p}
    text = {}
    idl = list(ids)
    for i in range(0, len(idl), 300):
        for r in await c.fetch("""
            SELECT document_id AS id, string_agg(text, ' ' ORDER BY page_number) AS t
            FROM document_pages WHERE document_id = ANY($1::uuid[]) GROUP BY document_id""",
                               idl[i:i + 300]):
            text[r["id"]] = r["t"] or ""
    print(f"  text loaded for {len(text)} documents  ({time.time()-t0:.0f}s)", flush=True)

    npages = {r["id"]: r["n"] for r in await c.fetch("""
        SELECT document_id AS id, count(*) AS n FROM document_pages
        WHERE document_id = ANY($1::uuid[]) GROUP BY document_id""", idl)}
    sh = {i: shingles(t) for i, t in text.items()}
    prod = {i: product_of(text.get(i, ""), meta[i]["fn"]) for i in ids}

    run_id = uuid.uuid4()
    rows, kinds, skipped = [], Counter(), Counter()

    for a, b in pairs:
        if a not in sh or b not in sh or not sh[a] or not sh[b]:
            skipped["no_text"] += 1
            continue
        inter = len(sh[a] & sh[b])
        tj = inter / max(len(sh[a] | sh[b]), 1)
        if tj < TAU_NEAR:
            skipped["unrelated"] += 1
            continue

        pa, sa = prod[a]
        pb, sb = prod[b]
        da, db = meta[a]["edate"], meta[b]["edate"]
        prefix_differs = prefix_of(meta[a]["fn"]) != prefix_of(meta[b]["fn"])
        ca, cb = len(text[a]), len(text[b])
        pga, pgb = npages.get(a, 0), npages.get(b, 0)
        kind, reason = classify(tj, pa, sa, pb, sb, da, db, prefix_differs,
                                period_tokens(meta[a]["fn"]), period_tokens(meta[b]["fn"]),
                                ca == cb, pga == pgb, ca, pga)
        kinds[kind] += 1

        # Canonical: earliest edition wins for exact_text (agreed rule — identical
        # text means origin time is the only signal left). For every other kind the
        # canonical pick is Fact Store's call, so this side records the pair and
        # names no winner.
        if kind == "duplicate" and da and db:
            keep, drop = (a, b) if da <= db else (b, a)
        elif kind == "duplicate":
            keep, drop = (a, b) if str(a) < str(b) else (b, a)   # stable, arbitrary, disclosed
            reason += "; no dates — canonical pick arbitrary, needs review"
        else:
            keep, drop = None, None

        for did, other in ((a, b), (b, a)):
            is_dropped = (drop == did)
            rows.append((
                run_id, "duplicates", did, None, None, other, None,
                "duplicate", reason, round(tj, 4), None, inter, None, None, None,
                "dedup", meta[did]["d"]["importance"], meta[did]["d"]["authority_level"],
                "n/a",
                ("superseded" if (is_dropped and kind == "duplicate") else "active"),
                ("retire duplicate" if (is_dropped and kind == "duplicate") else "keep"),
                (None if kind == "duplicate" else "fact_store"),
                ("retire; canonical is the earlier edition" if is_dropped
                 else "canonical" if keep == did else f"-> Fact Store: {kind}"),
                None, None, False,
                (meta[did]["rung"] if meta[did]["edate"] else "none"),
                NORM_VERSION, None, 0, kind))

    print(f"  scored {len(pairs)} pairs  ({time.time()-t0:.0f}s)\n", flush=True)

    SQL = """INSERT INTO gate_decisions
      (run_id,mode,document_id,doc_key,content_digest,prior_document_id,prior_digest,decision,reason,
       overlap_ratio,chunks_total,chunks_carried,chunks_changed,chunks_dropped,n_pages,lane,importance,
       authority_level,promotion_gate,lifecycle_state,index_action,adjudication_target,remediation,
       effective_date,termination_date,termination_trusted,ordering_confidence,normalization_version,
       generator_id,latency_ms,duplicate_kind)
      VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,
              $24,$25,$26,$27,$28,$29,$30,$31)"""
    for i in range(0, len(rows), 1000):
        await c.executemany(SQL, rows[i:i + 1000])
    print(f"PERSISTED {len(rows)} decision rows   run_id={run_id}")

    print("\n" + "=" * 74)
    print("DUPLICATE DETERMINATION")
    print("=" * 74)
    for k, n in kinds.most_common():
        print(f"   {k:<20}{n:>6} pairs")
    for k, n in skipped.most_common():
        print(f"   ({k:<18}{n:>6} pairs, no row written)")

    if apply:
        # Only exact_text, and only where the canonical pick had dates to stand on.
        act = [(r[2], r[5]) for r in rows
               if r[30] == "duplicate" and r[19] == "superseded"
               and "canonical pick arbitrary" not in r[8]]
        print(f"\n--apply: retiring {len(act)} documents as exact-text duplicates")
        async with c.transaction():
            for did, canonical in act:
                await c.execute("""
                    UPDATE documents SET lifecycle_state='superseded', supersedes_id=$2
                    WHERE id=$1 AND lifecycle_state IS DISTINCT FROM 'superseded'""", did, canonical)
        # Read the write back in the same run — a write path ships with its reader.
        n = await c.fetchval("SELECT count(*) FROM documents WHERE lifecycle_state='superseded'")
        m = await c.fetchval("SELECT count(*) FROM documents WHERE supersedes_id IS NOT NULL")
        print(f"  read-back: lifecycle_state='superseded' {n} · supersedes_id set {m}")
    else:
        print("\nTELEMETRY ONLY — no document was retired. Re-run with --apply to act "
              "on exact_text pairs only.")

    await c.close()


asyncio.run(main())
