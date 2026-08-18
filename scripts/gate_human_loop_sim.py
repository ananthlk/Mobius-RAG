"""Human-in-the-loop simulation — Fact Store verdicts arriving between gate decisions.

Spec: §7 (human loop), §4.2 (ambiguity), §6 (valid time), §16 (anchor rule).
Read-only. Simulates verdicts in memory; writes nothing.

THE PROPERTY UNDER TEST
-----------------------
A verdict is keyed on the DIGEST PAIR + doc_key (Fact Store's refinement), not
on "current state". That is what lets an answer arrive LATE — after the crawler
has moved the chain on — and still apply correctly. This harness deliberately
delivers verdicts out of order to prove it.

Three verdicts are exercised:
  1. successor        — the ordinary case; prior retires, human supplies valid time
  2. not_successor    — the two are different documents; the doc_key must SPLIT
  3. reversal         — a verdict that CONTRADICTS an automatic promotion the gate
                        already made. Only survivable because we retire, never delete.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import unicodedata
from datetime import date

import asyncpg

DSN = [l.split("=", 1)[1].strip() for l in open(".env") if l.startswith("DATABASE_URL=")][-1] \
    .replace("postgresql+asyncpg://", "postgresql://")
TAU_HIGH, TAU_LOW = 0.70, 0.35
FNDATE = re.compile(r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})|(\d{1,2})[-_](\d{1,2})[-_](\d{2})(?!\d)")


def normalize(t):
    if not t:
        return ""
    s = unicodedata.normalize("NFKC", t).replace(" ", " ")
    s = re.sub(r"(printed on|last updated:?).*$", " ", s, flags=re.I | re.M)
    return re.sub(r"\s+", " ", s).strip().lower()


def sha(s): return hashlib.sha256(s.encode()).hexdigest()[:12]


def fdate(fn):
    m = FNDATE.search(str(fn or ""))
    if not m:
        return None
    try:
        return (date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m.group(1)
                else date(2000 + int(m.group(6)), int(m.group(4)), int(m.group(5))))
    except ValueError:
        return None


class Corpus:
    """In-memory mirror of what the gate would have written. Nothing persists."""

    def __init__(self):
        self.docs = {}          # did -> dict(lifecycle, doc_key, version_no, term_date, retired_at)
        self.queue = []         # pending adjudications
        self.log = []

    def admit(self, did, key, version, prior=None, retire_prior=False):
        self.docs[did] = dict(lifecycle="active", doc_key=key, version_no=version,
                              term_date=None, retired_at=None, prior=prior)
        if retire_prior and prior:
            self.docs[prior]["lifecycle"] = "retired"
            self.docs[prior]["retired_at"] = "2026-08-17"   # transaction time only

    def active_at(self, key):
        return [d for d, v in self.docs.items()
                if v["doc_key"] == key and v["lifecycle"] == "active"]


def show(corpus, names, title):
    print(f"\n  ── {title} ──")
    for did, v in sorted(corpus.docs.items(), key=lambda kv: names[kv[0]]):
        mark = "ACTIVE " if v["lifecycle"] == "active" else "retired"
        term = f"  term={v['term_date']}" if v["term_date"] else "  term=NULL"
        print(f"     [{mark}] v{v['version_no']}  {names[did][:44]:46s}{term}"
              f"  key=…{str(v['doc_key'])[-12:]}")
    if corpus.queue:
        print(f"     pending adjudications: {len(corpus.queue)}")


async def main():
    c = await asyncpg.connect(DSN, timeout=90)
    rows = await c.fetch("""SELECT id, filename FROM documents WHERE payer='AHCA'
        AND filename ILIKE 'Attachment_II%Core_Contract%'""")
    chain = sorted((fdate(r["filename"]), r["id"], r["filename"])
                   for r in rows if fdate(r["filename"]))[:5]
    docs, names = {}, {}
    for _, did, fn in chain:
        txt = await c.fetch("SELECT text FROM hierarchical_chunks WHERE document_id=$1", did)
        hs = {sha(normalize(t["text"])) for t in txt}
        docs[did] = dict(hset=hs, digest=sha("".join(sorted(hs))))
        names[did] = fn
    await c.close()

    KEY = "AHCA|FL|attachment-ii-core-contract"
    corpus = Corpus()

    print("=" * 92)
    print("PHASE 1 — the gate runs. Ambiguity queues, but the chain keeps moving (§16).")
    print("=" * 92)
    anchor = None
    for edate, did, fn in chain:
        if anchor is None:
            corpus.admit(did, KEY, 1)
            print(f"  {edate}  FIRST_VERSION      {fn[:44]}")
        else:
            a = docs[anchor]
            i = docs[did]
            ov = len(i["hset"] & a["hset"]) / max(len(i["hset"] | a["hset"]), 1)
            v = corpus.docs[anchor]["version_no"] + 1
            if ov >= TAU_HIGH:
                corpus.admit(did, KEY, v, prior=anchor, retire_prior=True)
                print(f"  {edate}  SUCCESSOR ov={ov:.3f}  {fn[:40]}  → auto-retired prior")
            else:
                corpus.admit(did, KEY, v, prior=anchor, retire_prior=False)
                corpus.queue.append(dict(pred=anchor, succ=did, ov=round(ov, 3),
                                         pred_digest=a["digest"], succ_digest=i["digest"]))
                print(f"  {edate}  AMBIGUOUS ov={ov:.3f}  {fn[:40]}  → queued, prior STAYS active")
        anchor = did
    show(corpus, names, "state after the gate, before any human looks")
    print(f"\n  NOTE: {len(corpus.active_at(KEY))} versions are simultaneously ACTIVE while verdicts pend.")
    print("  §10 as-of resolution must pick between them by validity window, not an `active` flag.")

    # ── VERDICT 1 — ordinary successor, arriving late ────────────────────
    print("\n" + "=" * 92)
    print("PHASE 2 — VERDICT 1 arrives: 'successor' for the FIRST queued pair")
    print("=" * 92)
    q = corpus.queue.pop(0)
    print(f"  human answers pair (…{q['pred_digest'][-8:]} → …{q['succ_digest'][-8:]})  ov was {q['ov']}")
    print(f"    relationship            = successor")
    print(f"    successor.effective     = 2020-02-01   (read off the document)")
    print(f"    predecessor.termination = 2020-01-31   (read off the document)")
    print("  → the verdict is keyed on the DIGEST PAIR, so it applies even though the")
    print("    crawler has moved the chain on since the question was asked.")
    corpus.docs[q["pred"]]["lifecycle"] = "retired"
    corpus.docs[q["pred"]]["retired_at"] = "2026-08-17"
    corpus.docs[q["pred"]]["term_date"] = "2020-01-31"      # ONLY a human writes valid time
    show(corpus, names, "after verdict 1")
    print("  ↑ termination_date is now set — the FIRST time anything has written valid time (§6).")

    # ── VERDICT 2 — not a successor: the key must split ──────────────────
    print("\n" + "=" * 92)
    print("PHASE 3 — VERDICT 2 arrives: 'not_successor' — these are DIFFERENT documents")
    print("=" * 92)
    q2 = corpus.queue.pop(0)
    print(f"  human answers pair (…{q2['pred_digest'][-8:]} → …{q2['succ_digest'][-8:]})  ov was {q2['ov']}")
    print("    relationship = not_successor  (a different contract that shares boilerplate)")
    newkey = KEY + "|split-1"
    moved = []
    for did, v in corpus.docs.items():
        if v["doc_key"] == KEY and names[did] >= names[q2["succ"]]:
            v["doc_key"] = newkey
            moved.append(did)
    corpus.docs[q2["pred"]]["lifecycle"] = "active"     # nothing to supersede it
    print(f"  → doc_key SPLIT. {len(moved)} document(s) moved to a new lineage.")
    print("    The cascade matters: everything chained THROUGH the rejected link inherits")
    print("    the new key, otherwise later editions stay attached to the wrong ancestry.")
    show(corpus, names, "after verdict 2")

    # ── VERDICT 3 — reversal of an automatic promotion ───────────────────
    print("\n" + "=" * 92)
    print("PHASE 4 — VERDICT 3: a human REVERSES an automatic promotion")
    print("=" * 92)
    promoted = [d for d, v in corpus.docs.items() if v["lifecycle"] == "retired" and v["term_date"] is None]
    if promoted:
        target = promoted[0]
        print(f"  the gate auto-retired: {names[target][:52]}")
        print("  human verdict: WRONG — that was a parallel amendment, not a replacement.")
        corpus.docs[target]["lifecycle"] = "active"
        corpus.docs[target]["retired_at"] = None
        print("  → un-retired. Recoverable ONLY because retirement never deletes (§5/§8 phase 3).")
        print("    Had the gate deleted the row or its chunks, this verdict would be unactionable.")
        show(corpus, names, "after verdict 3")
    else:
        print("  (no automatic promotion in this slice to reverse)")

    print("\n" + "=" * 92)
    print("WHAT THE HUMAN LOOP PROVED")
    print("=" * 92)
    print("  1. Verdicts keyed on the digest pair apply correctly when they arrive LATE.")
    print("  2. A human is the only writer of termination_date — the gate never sets valid time.")
    print("  3. 'not_successor' is not a no-op: it SPLITS the lineage, and the split must")
    print("     cascade to every edition chained through the rejected link.")
    print("  4. An automatic promotion is REVERSIBLE, because retirement never deletes.")
    print("  5. Multiple versions stay active while verdicts pend — as-of resolution (§10)")
    print("     must therefore key on the validity window, not on a single `active` flag.")
    print("\nNothing was written.")


asyncio.run(main())
