"""Corpus-wide gate run — TELEMETRY ONLY.

Scores every document in the corpus and writes one gate_decisions row each.

WRITES:  gate_decisions (append-only observation)
DOES NOT WRITE: documents, chunks, embeddings, the index. Nothing is retired,
deleted, promoted or re-triggered. The gate records what it WOULD do.

That is not only caution — acting also requires doc_key / lifecycle_state /
supersedes_id, which do not exist yet (DB seat contract unsigned, §11.4).

SCALE
  ~9,900 documents over ~2M chunks. Digests are computed SERVER-SIDE in one
  aggregate rather than pulling chunk text into Python. Hash SETS (needed for
  pairwise overlap) are fetched only for documents that share a doc_key with
  another document — a small fraction — so the expensive part stays bounded.

Usage:  python3 scripts/gate_run_corpus.py [--dry-run]
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
import uuid
from collections import defaultdict
from datetime import date

import asyncpg

DSN = [l.split("=", 1)[1].strip() for l in open(".env") if l.startswith("DATABASE_URL=")][-1] \
    .replace("postgresql+asyncpg://", "postgresql://")

TAU_HIGH, TAU_LOW = 0.70, 0.35
NORM_VERSION = "v1-prose-sql"

# Normalisation performed in SQL so 2M chunks never cross the wire. Prose-level
# only (lower + whitespace collapse); the structure-preserving rules in §3 are
# not yet implemented, so duplicate counts here are a LOWER bound.
CHUNK_HASH = "md5(regexp_replace(lower(c.text), '\\s+', ' ', 'g'))"

RULE = re.compile(r"59[A-Z]?-?\s?(\d{1,3})\.(\d{1,4})", re.I)
EPISODIC = re.compile(r"minutes|agenda|notice|meeting|workshop|hearing|presentation|"
                      r"newsletter|bulletin|\bq[1-4]\b|qtr|quarter|sfy\s?\d{2}", re.I)
REVISABLE = re.compile(r"coverage polic|handbook|manual|fee schedule|companion guide|"
                       r"provider guide|billing guide|reimbursement polic|contract", re.I)
# Period tokens to strip from a tier-2 key. MONTH NAMES belong here: the
# corpus-wide run split one contract chain into four keys because Oct_2025 /
# October_2025 / April_1 survived a numeric-only strip and became part of the
# identity, while 2021-10-01 did not.
_MONTHS = (r"jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|"
           r"aug(ust)?|sep(t|tember)?|oct(ober)?|nov(ember)?|dec(ember)?")
PERIOD = re.compile(r"(19|20)\d{2}([-_/]\d{2,4})?|sfy\s?\d{2}[-_]?\d{0,2}|"
                    r"\b(" + _MONTHS + r")\b", re.I)
CATEGORY = re.compile(r"^[A-Za-z ]+ — ")
FNDATE = re.compile(r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})|(\d{1,2})[-_](\d{1,2})[-_](\d{2})(?!\d)")
PDFD = re.compile(r"D:(\d{4})(\d{2})(\d{2})")


def title_of(d):
    dn = d["display_name"]
    return str(dn) if dn and not CATEGORY.match(str(dn)) else str(d["filename"] or "")


def doc_key(d):
    name = f"{title_of(d)} {d['filename'] or ''}"
    if EPISODIC.search(name):
        return None                       # §2.2 — announcements are never versions
    m = RULE.search(name)
    if m:
        return f"{d['payer']}|{d['state']}|59G-{m.group(1)}.{m.group(2)}"
    if REVISABLE.search(name):
        # Separators MUST become spaces before PERIOD runs. `_` is a word
        # character, so \b(oct)\b never matches `_Oct_` — the first attempt at
        # this fix was silently inert and left the chain fragmented.
        flat = re.sub(r"[^a-z0-9]+", " ", name.lower())
        stem = re.sub(r"[^a-z]+", " ", PERIOD.sub(" ", flat)).strip()
        stem = re.sub(r"\s+", " ", stem)
        if len(stem) > 8:
            return f"{d['payer']}|{d['state']}|{stem[:60]}"
    return None


def edition_date(d):
    """§19.3 ladder — filename (stated edition) outranks publication (noisy proxy)."""
    m = FNDATE.search(str(d["filename"] or ""))
    if m:
        try:
            return ((date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m.group(1)
                     else date(2000 + int(m.group(6)), int(m.group(4)), int(m.group(5)))),
                    "filename")
        except ValueError:
            pass
    for key, rung in (("mod_date", "source_modified"), ("creation_date", "source_created")):
        raw = (d["pdf_meta"] or {}).get(key) if isinstance(d["pdf_meta"], dict) else None
        mm = PDFD.match(str(raw or ""))
        if mm:
            try:
                return date(int(mm.group(1)), int(mm.group(2)), int(mm.group(3))), rung
            except ValueError:
                pass
    if d["created_at"]:
        return d["created_at"].date(), "first_seen(weak)"
    return None, "none"


async def main():
    dry = "--dry-run" in sys.argv
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
        SELECT d.id, d.filename, d.display_name, d.payer, d.state, d.status,
               d.effective_date, d.termination_date, d.created_at, d.authority_level,
               d.source_metadata->'pdf_meta' AS pdf_meta,
               d.source_metadata->'payor_classification'->>'importance' AS importance
        FROM documents d""")
    print(f"documents: {len(docs)}")

    print("computing digests server-side …", flush=True)
    rows = await c.fetch(f"""
        SELECT c.document_id AS id,
               md5(string_agg({CHUNK_HASH}, '' ORDER BY c.page_number, c.paragraph_index)) AS digest,
               count(*) AS n
        FROM hierarchical_chunks c GROUP BY c.document_id""")
    digest = {r["id"]: r["digest"] for r in rows}
    nchunks = {r["id"]: r["n"] for r in rows}
    print(f"  digests for {len(digest)} documents  ({time.time()-t0:.0f}s)", flush=True)

    pages = {r["id"]: r["n"] for r in await c.fetch("""
        SELECT d.id, count(p.id) AS n FROM documents d
        LEFT JOIN document_pages p ON p.document_id = d.id GROUP BY d.id""")}

    # cluster, then pull hash SETS only where a comparison is actually possible
    st = {}
    clusters = defaultdict(list)
    for d in docs:
        k = doc_key(d)
        ed, rung = edition_date(d)
        st[d["id"]] = dict(d=d, key=k, edate=ed, rung=rung,
                           digest=digest.get(d["id"]), n=nchunks.get(d["id"], 0),
                           pages=pages.get(d["id"], 0), hset=None)
        if k and digest.get(d["id"]):
            clusters[k].append(d["id"])

    need = [i for k, v in clusters.items() if len(v) > 1 for i in v]
    print(f"  {len(clusters)} clusters · {sum(1 for v in clusters.values() if len(v)>1)} multi-doc "
          f"· fetching hash sets for {len(need)} documents", flush=True)
    for i in range(0, len(need), 400):
        for r in await c.fetch(f"""
            SELECT c.document_id AS id, array_agg(DISTINCT {CHUNK_HASH}) AS hs
            FROM hierarchical_chunks c WHERE c.document_id = ANY($1::uuid[])
            GROUP BY c.document_id""", need[i:i + 400]):
            st[r["id"]]["hset"] = set(r["hs"])

    run_id = uuid.uuid4()
    out = []
    for did, s in st.items():
        t = time.perf_counter()
        d = s["d"]
        prior = prior_dig = ov = carried = changed = dropped = None
        order_conf = s["rung"] if s["key"] else None
        gate_state = "n/a"

        if s["pages"] == 0:
            dec, life, idx, rem, adj = ("unpublishable", "shelved", "none",
                                        "delete derived, re-trigger EXTRACTION", None)
            reason = "0 pages"
        elif not s["digest"]:
            dec, life, idx, rem, adj = ("unpublishable", "shelved", "none",
                                        "delete partial chunks, re-enqueue CHUNKING", None)
            reason = "pages but 0 chunks"
        elif d["status"] in ("failed", "completed_with_errors"):
            dec, life, idx, rem, adj = ("unpublishable", "shelved", "none",
                                        "delete ALL derived, re-trigger from raw", None)
            reason = f"upstream status={d['status']}"
        else:
            sibs = [x for x in clusters.get(s["key"], []) if x != did and st[x]["digest"]]
            mine = s["edate"] or date(1900, 1, 1)
            earlier = [x for x in sibs if (st[x]["edate"] or date(1900, 1, 1)) < mine]
            if not earlier:
                dec, life, idx, rem, adj = ("first_version", "active", "admitted", "publish v1", None)
                reason = "no earlier edition at doc_key"
            else:
                prior = max(earlier, key=lambda x: st[x]["edate"] or date(1900, 1, 1))
                ps = st[prior]
                prior_dig = ps["digest"]
                # Degeneracy is a property of THIS comparison, not of the cluster.
                # Judging it cluster-wide let one undated sibling freeze every
                # well-dated pair around it — the same failure shape as bug 4.
                if s["edate"] is None or ps["edate"] is None or s["edate"] == ps["edate"]:
                    order_conf = "degenerate"
                elif s["rung"] == "first_seen(weak)" or ps["rung"] == "first_seen(weak)":
                    order_conf = "weak"
                if ps["digest"] == s["digest"]:
                    dec, life, idx, rem, adj = ("unchanged", "active", "none",
                                                "bump last_validated_at", None)
                    reason, ov = "identical content_digest", 1.0
                    carried, changed, dropped = s["n"], 0, 0
                elif s["hset"] and ps["hset"]:
                    carried = len(s["hset"] & ps["hset"])
                    changed = len(s["hset"] - ps["hset"])
                    dropped = len(ps["hset"] - s["hset"])
                    ov = carried / max(len(s["hset"] | ps["hset"]), 1)
                    if order_conf == "degenerate":
                        dec, life, idx = "ambiguous_order", "active", "admitted; prior NOT retired"
                        rem, adj = "-> Fact Store: chain order unreliable", "fact_store"
                        reason = f"overlap={ov:.3f} but ordering degenerate"
                    elif ov >= TAU_HIGH:
                        dec, life, idx = "successor", "active", "admitted + retire prior"
                        rem, adj = "promote; retire prior (retired_at only)", None
                        reason, gate_state = f"overlap={ov:.3f} >= tau_high", "would run"
                    elif ov >= TAU_LOW:
                        dec, life, idx = "ambiguous_revision", "active", "admitted; prior NOT retired"
                        rem, adj = "-> Fact Store: heavy revision?", "fact_store"
                        reason = f"overlap={ov:.3f} in [{TAU_LOW},{TAU_HIGH})"
                    else:
                        dec, life, idx = "ambiguous_tail", "active", "admitted; prior NOT retired"
                        rem, adj = "-> Fact Store: likely unrelated", "fact_store"
                        reason = f"overlap={ov:.3f} < tau_low"
                else:
                    dec, life, idx = "ambiguous_tail", "active", "admitted; prior NOT retired"
                    rem, adj, reason = "-> Fact Store: no comparable chunks", "fact_store", "hash set unavailable"

        out.append((run_id, "corpus_wide", did, s["key"], s["digest"], prior, prior_dig, dec, reason,
                    round(ov, 4) if ov is not None else None, s["n"], carried, changed, dropped,
                    s["pages"], "tracked" if s["key"] else "untracked", d["importance"],
                    d["authority_level"], gate_state, life, idx, adj, rem,
                    d["effective_date"], d["termination_date"], False, order_conf,
                    NORM_VERSION, None, int((time.perf_counter() - t) * 1000)))

    print(f"  scored {len(out)} documents  ({time.time()-t0:.0f}s)", flush=True)
    if not dry:
        SQL = """INSERT INTO gate_decisions
          (run_id,mode,document_id,doc_key,content_digest,prior_document_id,prior_digest,decision,reason,
           overlap_ratio,chunks_total,chunks_carried,chunks_changed,chunks_dropped,n_pages,lane,importance,
           authority_level,promotion_gate,lifecycle_state,index_action,adjudication_target,remediation,
           effective_date,termination_date,termination_trusted,ordering_confidence,normalization_version,
           generator_id,latency_ms)
          VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,
                  $24,$25,$26,$27,$28,$29,$30)"""
        for i in range(0, len(out), 1000):
            await c.executemany(SQL, out[i:i + 1000])
        print(f"PERSISTED {len(out)} rows   run_id={run_id}")

    print("\n" + "=" * 74)
    print("CORPUS-WIDE RESULT")
    print("=" * 74)
    agg = defaultdict(int)
    for r in out:
        agg[r[7]] += 1
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"   {k:<22}{v:>7}")
    adj_n = sum(1 for r in out if r[21])
    succ = [r for r in out if r[7] == "successor"]
    print(f"\n   would go to Fact Store   {adj_n}")
    print(f"   would auto-promote       {len(succ)}")
    print(f"   chunks carried forward   {sum(r[11] or 0 for r in out):,}")
    print(f"   chunks needing re-embed  {sum(r[12] or 0 for r in out):,}")
    print("\nNothing was retired, deleted, promoted or re-triggered.")
    await c.close()


asyncio.run(main())
