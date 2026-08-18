"""Wire §1.1 telemetry to PERSIST, then run N documents through the gate as if incoming.

Spec: docs/versioning-dedup-gate-spec.md §1.1

WHAT THIS WRITES
  · CREATE TABLE IF NOT EXISTS gate_decisions   (new, additive)
  · INSERT one row per document processed        (append-only observation)

WHAT THIS DOES NOT WRITE
  · nothing on documents, document_pages, hierarchical_chunks or any index.
    No retire, no delete, no lifecycle change, no re-trigger. The gate's
    decisions are RECORDED, not applied.

That separation is the point of §1.1: the decision row is written BEFORE any
mutation, so a decision with no corresponding corpus change is a detectable
defect rather than a silent one. Here there is deliberately no mutation at all,
so the telemetry stands alone and can be inspected before anything acts on it.

DDL ownership: the DB seat owns the column contract (§11.4, unsigned). This
table is created in DEV so the telemetry can be exercised; it is expected to be
revised to their shape, and is flagged to them on creation.

Usage:  python3 scripts/gate_persist_run.py [N]     (default 500)
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import sys
import time
import unicodedata
import uuid
from collections import defaultdict
from datetime import date

import asyncpg

DSN = [l.split("=", 1)[1].strip() for l in open(".env") if l.startswith("DATABASE_URL=")][-1] \
    .replace("postgresql+asyncpg://", "postgresql://")

PAYER = "AHCA"
TAU_HIGH, TAU_LOW = 0.70, 0.35
NORM_VERSION = "v1-prose"

RULE = re.compile(r"59[A-Z]?-?\s?(\d{1,3})\.(\d{1,4})", re.I)
EPISODIC = re.compile(r"minutes|agenda|notice|meeting|workshop|hearing|presentation|"
                      r"newsletter|bulletin|\bq[1-4]\b|qtr|quarter|sfy\s?\d{2}", re.I)
REVISABLE = re.compile(r"coverage polic|handbook|manual|fee schedule|companion guide|"
                       r"provider guide|billing guide|reimbursement polic|contract", re.I)
PERIOD = re.compile(r"(19|20)\d{2}([-_/]\d{2,4})?|sfy\s?\d{2}[-_]?\d{0,2}", re.I)
CATEGORY_LABEL = re.compile(r"^[A-Za-z ]+ — ")
# §14.3 bug 3 fix — the real dates live in filenames
FNDATE = re.compile(r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})|(\d{1,2})[-_](\d{1,2})[-_](\d{2})(?!\d)")

DDL = """
CREATE TABLE IF NOT EXISTS gate_decisions (
    decision_id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    decided_at            timestamptz NOT NULL DEFAULT now(),
    run_id                uuid        NOT NULL,
    mode                  text        NOT NULL,
    document_id           uuid        NOT NULL,
    doc_key               text,
    content_digest        text,
    prior_document_id     uuid,
    prior_digest          text,
    decision              text        NOT NULL,
    reason                text,
    overlap_ratio         numeric(6,4),
    chunks_total          int,
    chunks_carried        int,
    chunks_changed        int,
    chunks_dropped        int,
    n_pages               int,
    lane                  text,
    importance            text,
    authority_level       text,
    promotion_gate        text,
    lifecycle_state       text,
    index_action          text,
    adjudication_target   text,
    remediation           text,
    effective_date        date,
    termination_date      date,
    termination_trusted   boolean,
    ordering_confidence   text,
    normalization_version text,
    generator_id          text,
    latency_ms            int
);
COMMENT ON TABLE gate_decisions IS
  'Versioning/dedup gate decisions, spec docs/versioning-dedup-gate-spec.md 1.1. '
  'APPEND-ONLY: a re-decision writes a NEW row so history reads as history. '
  'Written BEFORE any corpus mutation. DEV — column contract pending DB seat.';
CREATE INDEX IF NOT EXISTS ix_gate_decisions_run ON gate_decisions(run_id);
CREATE INDEX IF NOT EXISTS ix_gate_decisions_doc ON gate_decisions(document_id, decided_at DESC);
CREATE INDEX IF NOT EXISTS ix_gate_decisions_decision ON gate_decisions(decision, decided_at DESC);
"""


def normalize(t: str) -> str:
    if not t:
        return ""
    s = unicodedata.normalize("NFKC", t).replace(" ", " ").replace("­", "")
    s = re.sub(r"[‘’]", "'", s)
    s = re.sub(r"[“”]", '"', s)
    s = re.sub(r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$", " ", s, flags=re.I | re.M)
    s = re.sub(r"(printed on|last updated:?).*$", " ", s, flags=re.I | re.M)
    return re.sub(r"\s+", " ", s).strip().lower()


def sha(s): return hashlib.sha256(s.encode()).hexdigest()[:32]


def title_of(d):
    dn = d["display_name"]
    return str(dn) if dn and not CATEGORY_LABEL.match(str(dn)) else str(d["filename"] or "")


def doc_key(d):
    name = f"{title_of(d)} {d['filename'] or ''}"
    if EPISODIC.search(name):
        return None
    m = RULE.search(name)
    if m:
        return f"{PAYER}|{d['state']}|59G-{m.group(1)}.{m.group(2)}"
    if REVISABLE.search(name):
        stem = re.sub(r"[^a-z]+", " ", PERIOD.sub(" ", name.lower())).strip()
        if len(stem) > 8:
            return f"{PAYER}|{d['state']}|{stem[:60]}"
    return None


def filename_date(d):
    """§14.3 — the real edition date, when the filename carries one."""
    m = FNDATE.search(str(d["filename"] or ""))
    if not m:
        return None
    try:
        if m.group(1):
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return date(2000 + int(m.group(6)), int(m.group(4)), int(m.group(5)))
    except ValueError:
        return None


async def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    c = await asyncpg.connect(DSN, timeout=120)
    await c.execute(DDL)
    print("gate_decisions: ready (created if absent)")

    # all keyed/clusterable docs + a random sample, so version logic is exercised
    docs = await c.fetch(f"""
        SELECT d.id,d.filename,d.display_name,d.state,d.status,d.effective_date,
               d.termination_date,d.created_at,d.authority_level,
               d.source_metadata->'payor_classification'->>'importance' AS importance
        FROM documents d WHERE d.payer=$1
        ORDER BY (d.filename ILIKE '%contract%' OR d.filename ILIKE '%polic%'
                  OR d.filename ILIKE '%59G%' OR d.filename ILIKE '%schedule%') DESC,
                 md5(d.id::text)
        LIMIT $2""", PAYER, n_target)
    ids = [d["id"] for d in docs]
    print(f"processing {len(docs)} documents as INCOMING\n")

    pg = {r["id"]: r["n"] for r in await c.fetch(
        """SELECT d.id,count(p.id) n FROM documents d LEFT JOIN document_pages p ON p.document_id=d.id
           WHERE d.id=ANY($1::uuid[]) GROUP BY d.id""", ids)}
    hs = {}
    for r in await c.fetch(
        """SELECT document_id, array_agg(text) txt FROM hierarchical_chunks
           WHERE document_id=ANY($1::uuid[]) GROUP BY document_id""", ids):
        hs[r["document_id"]] = {sha(normalize(t)) for t in r["txt"]}
    gen = {r["document_id"]: r["generator_id"] for r in await c.fetch(
        """SELECT DISTINCT ON (document_id) document_id, generator_id FROM chunking_jobs
           WHERE document_id=ANY($1::uuid[]) ORDER BY document_id, created_at DESC""", ids)}

    st = {}
    for d in docs:
        hset = hs.get(d["id"], set())
        st[d["id"]] = dict(d=d, pages=pg.get(d["id"], 0), hset=hset,
                           digest=sha("".join(sorted(hset))) if hset else None,
                           key=doc_key(d), fdate=filename_date(d))

    # cluster, then walk each cluster in edition order (filename date beats fabricated effective_date)
    clusters = defaultdict(list)
    for did, s in st.items():
        if s["key"]:
            clusters[s["key"]].append(did)

    run_id = uuid.uuid4()
    rows = []
    for did, s in st.items():
        t0 = time.perf_counter()
        d = s["d"]
        trusted_term = False          # §6.3 — every current value is created_at+182d
        key = s["key"]
        prior = prior_dig = None
        ov = carried = changed = dropped = None
        order_conf = None

        if s["pages"] == 0:
            dec, life, idx, rem, adj = ("unpublishable", "shelved", "none",
                                        "delete derived, re-trigger EXTRACTION", None)
            reason = "0 pages"
        elif not s["hset"]:
            dec, life, idx, rem, adj = ("unpublishable", "shelved", "none",
                                        "delete partial chunks, re-enqueue CHUNKING", None)
            reason = "pages but 0 chunks"
        else:
            sibs = [x for x in clusters.get(key, []) if x != did and st[x]["hset"]]
            # §14.3: order by filename date when present; fall back to effective_date
            def edition(x):
                return (st[x]["fdate"] or st[x]["d"]["effective_date"] or date(1900, 1, 1))
            mine = s["fdate"] or d["effective_date"] or date(1900, 1, 1)
            earlier = [x for x in sibs if edition(x) < mine]
            if key and sibs:
                distinct = len({edition(x) for x in sibs + [did]})
                order_conf = ("filename-date" if s["fdate"]
                              else ("degenerate" if distinct < len(sibs) + 1 else "effective-date"))
            if not earlier:
                dec, life, idx, rem, adj = ("first_version", "active", "admitted", "publish v1", None)
                reason = "no earlier edition at doc_key"
            else:
                prior = max(earlier, key=edition)
                ps = st[prior]
                prior_dig = ps["digest"]
                if ps["digest"] == s["digest"]:
                    dec, life, idx, rem, adj = ("unchanged", "active", "none",
                                                "bump last_validated_at", None)
                    reason, ov = "identical content_digest", 1.0
                    carried, changed, dropped = len(s["hset"]), 0, 0
                else:
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
                        reason = f"overlap={ov:.3f} >= tau_high"
                    elif ov >= TAU_LOW:
                        dec, life, idx = "ambiguous_revision", "active", "admitted; prior NOT retired"
                        rem, adj = "-> Fact Store: heavy revision?", "fact_store"
                        reason = f"overlap={ov:.3f} in [{TAU_LOW},{TAU_HIGH})"
                    else:
                        dec, life, idx = "ambiguous_tail", "active", "admitted; prior NOT retired"
                        rem, adj = "-> Fact Store: likely unrelated", "fact_store"
                        reason = f"overlap={ov:.3f} < tau_low"

        rows.append((run_id, "forward", did, key, s["digest"], prior, prior_dig, dec, reason,
                     round(ov, 4) if ov is not None else None, len(s["hset"]), carried, changed,
                     dropped, s["pages"], "tracked" if key else "untracked", d["importance"],
                     d["authority_level"], "n/a" if dec != "successor" else "would run",
                     life, idx, adj, rem, d["effective_date"], d["termination_date"], trusted_term,
                     order_conf, NORM_VERSION, gen.get(did),
                     int((time.perf_counter() - t0) * 1000)))

    await c.executemany("""INSERT INTO gate_decisions
        (run_id,mode,document_id,doc_key,content_digest,prior_document_id,prior_digest,decision,reason,
         overlap_ratio,chunks_total,chunks_carried,chunks_changed,chunks_dropped,n_pages,lane,importance,
         authority_level,promotion_gate,lifecycle_state,index_action,adjudication_target,remediation,
         effective_date,termination_date,termination_trusted,ordering_confidence,normalization_version,
         generator_id,latency_ms)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,
                $24,$25,$26,$27,$28,$29,$30)""", rows)
    print(f"PERSISTED {len(rows)} telemetry rows   run_id={run_id}\n")

    print("=" * 84)
    print("HOW THEY RESPONDED — queried back from gate_decisions")
    print("=" * 84)
    for r in await c.fetch("""SELECT decision, lane, count(*) n, round(avg(overlap_ratio),3) avg_ov,
        sum(chunks_carried) carried, sum(chunks_changed) changed
        FROM gate_decisions WHERE run_id=$1 GROUP BY 1,2 ORDER BY n DESC""", run_id):
        ov = f"{r['avg_ov']}" if r["avg_ov"] is not None else "—"
        print(f"  {r['decision']:<20}{r['lane']:<11}{r['n']:>5}   avg_overlap={ov:<8}"
              f"carried={r['carried'] or 0:<7}re-embed={r['changed'] or 0}")
    print("\n  adjudications that would go to Fact Store:",
          await c.fetchval("SELECT count(*) FROM gate_decisions WHERE run_id=$1 AND adjudication_target IS NOT NULL", run_id))
    print("  ordering confidence:")
    for r in await c.fetch("""SELECT coalesce(ordering_confidence,'(n/a — no sibling)') oc, count(*) n
        FROM gate_decisions WHERE run_id=$1 GROUP BY 1 ORDER BY n DESC""", run_id):
        print(f"     {r['oc']:<28}{r['n']}")
    print(f"\n  median decision latency: "
          f"{await c.fetchval('SELECT percentile_disc(0.5) WITHIN GROUP (ORDER BY latency_ms) FROM gate_decisions WHERE run_id=$1', run_id)} ms")
    print("\nNo document, chunk or index was modified. Only gate_decisions was written.")
    await c.close()


asyncio.run(main())
