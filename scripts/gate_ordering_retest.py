"""RETEST §18 — does the publication clock actually order version chains?

Run after scripts/backfill_pdf_dates.py. Read-only.

Three questions:
  1. GROUND TRUTH — on chains whose true order is known from filename dates,
     does the publication date reproduce that order?
  2. COVERAGE — what fraction of the corpus now has an ordering date, and
     from which rung of the §18.3 ladder?
  3. EFFECT — how many back-prop chains now resolve automatically that
     previously went to a human because ordering was degenerate (§14.3)?
"""
from __future__ import annotations

import asyncio
import json
import re
from collections import Counter, defaultdict
from datetime import date

import asyncpg

DSN = [l.split("=", 1)[1].strip() for l in open(".env") if l.startswith("DATABASE_URL=")][-1] \
    .replace("postgresql+asyncpg://", "postgresql://")

PDFD = re.compile(r"D:(\d{4})(\d{2})(\d{2})")
FN = re.compile(r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})|(\d{1,2})[-_](\d{1,2})[-_](\d{2})(?!\d)")
RULE = re.compile(r"59[A-Z]?-?\s?(\d{1,3})\.(\d{1,4})", re.I)
EPISODIC = re.compile(r"minutes|agenda|notice|meeting|workshop|hearing|presentation|"
                      r"newsletter|bulletin|\bq[1-4]\b|qtr|quarter|sfy\s?\d{2}", re.I)
REVISABLE = re.compile(r"coverage polic|handbook|manual|fee schedule|companion guide|"
                       r"provider guide|billing guide|reimbursement polic|contract", re.I)
PERIOD = re.compile(r"(19|20)\d{2}([-_/]\d{2,4})?|sfy\s?\d{2}[-_]?\d{0,2}", re.I)
CATEGORY = re.compile(r"^[A-Za-z ]+ — ")


def pdfdate(pm, key):
    if isinstance(pm, str):
        try:
            pm = json.loads(pm)
        except Exception:
            return None
    if not isinstance(pm, dict):
        return None
    m = PDFD.match(str(pm.get(key) or ""))
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def fndate(fn):
    m = FN.search(str(fn or ""))
    if not m:
        return None
    try:
        return (date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m.group(1)
                else date(2000 + int(m.group(6)), int(m.group(4)), int(m.group(5))))
    except ValueError:
        return None


def title_of(d):
    dn = d["display_name"]
    return str(dn) if dn and not CATEGORY.match(str(dn)) else str(d["filename"] or "")


def doc_key(d):
    name = f"{title_of(d)} {d['filename'] or ''}"
    if EPISODIC.search(name):
        return None
    m = RULE.search(name)
    if m:
        return f"AHCA|{d['state']}|59G-{m.group(1)}.{m.group(2)}"
    if REVISABLE.search(name):
        stem = re.sub(r"[^a-z]+", " ", PERIOD.sub(" ", name.lower())).strip()
        if len(stem) > 8:
            return f"AHCA|{d['state']}|{stem[:60]}"
    return None


def edition_date(d):
    """§18.3 ordering ladder. Returns (date, which rung)."""
    pm = d["pm"]
    md = pdfdate(pm, "mod_date")
    if md:
        return md, "source_modified"
    cr = pdfdate(pm, "creation_date")
    if cr:
        return cr, "source_created"
    fd = fndate(d["filename"])
    if fd:
        return fd, "filename"
    if d["created_at"]:
        return d["created_at"].date(), "first_seen(weak)"
    return None, "none"


def h(t):
    print("\n" + "=" * 88 + f"\n{t}\n" + "=" * 88)


async def main():
    c = await asyncpg.connect(DSN, timeout=90)
    docs = await c.fetch("""SELECT id, filename, display_name, state, effective_date, created_at,
        source_metadata->'pdf_meta' pm FROM documents WHERE payer='AHCA'""")
    print(f"AHCA documents: {len(docs)}")

    # ── 1. ground truth ──────────────────────────────────────────────────
    h("1. GROUND TRUTH — chains whose true order is known from filename dates")
    clusters = defaultdict(list)
    for d in docs:
        k = doc_key(d)
        if k:
            clusters[k].append(d)

    tested = agree = disagree = 0
    for k, v in clusters.items():
        known = [(fndate(x["filename"]), x) for x in v if fndate(x["filename"])]
        if len(known) < 3:
            continue
        pub = [(edition_date(x)[0], fd, x) for fd, x in known if edition_date(x)[1].startswith("source")]
        if len(pub) < 3:
            continue
        tested += 1
        by_true = [id(x) for _, fd, x in sorted(pub, key=lambda t: t[1])]
        by_pub = [id(x) for _, fd, x in sorted(pub, key=lambda t: t[0])]
        match = by_true == by_pub
        agree += match
        disagree += (not match)
        print(f"\n  {k[-46:]}   {len(pub)} dated editions   "
              f"{'ORDER REPRODUCED' if match else 'ORDER DIFFERS'}")
        for pubd, fd, x in sorted(pub, key=lambda t: t[1])[:6]:
            lag = (pubd - fd).days
            print(f"     true {fd}   published {pubd}  ({lag:+5d}d)  {str(x['filename'])[:38]}")
    if tested:
        print(f"\n  chains tested: {tested}   order reproduced: {agree}   differs: {disagree}")
    else:
        print("  (no chain has 3+ editions with BOTH a filename date and a publication date)")

    # ── 2. coverage of the ladder ────────────────────────────────────────
    h("2. COVERAGE — which rung of the §18.3 ladder supplies the ordering date")
    rungs = Counter(edition_date(d)[1] for d in docs)
    for rung in ("source_modified", "source_created", "filename", "first_seen(weak)", "none"):
        n = rungs.get(rung, 0)
        bar = "#" * int(n / max(len(docs), 1) * 46)
        print(f"   {rung:20s}{n:5d}  ({n/len(docs)*100:4.1f}%) {bar}")
    trust = rungs.get("source_modified", 0) + rungs.get("source_created", 0) + rungs.get("filename", 0)
    print(f"\n   TRUSTWORTHY ordering (not first_seen): {trust}  ({trust/len(docs)*100:.1f}%)")
    print(f"   was, before the backfill: filename-only = "
          f"{sum(1 for d in docs if fndate(d['filename']))} ({sum(1 for d in docs if fndate(d['filename']))/len(docs)*100:.1f}%)")

    # ── 3. effect on back-prop ───────────────────────────────────────────
    h("3. EFFECT — chains that can now be ordered without a human (§14.3)")
    multi = {k: v for k, v in clusters.items() if len(v) > 1}
    before = after = 0
    for k, v in multi.items():
        eff = {x["effective_date"] for x in v}
        if len(eff) >= len(v):
            before += 1
        dates = [edition_date(x) for x in v]
        good = [d for d, r in dates if d and r != "first_seen(weak)"]
        if len(set(good)) == len(v):
            after += 1
    print(f"   multi-document chains          {len(multi)}")
    print(f"   orderable BEFORE (effective_date distinct)   {before}")
    print(f"   orderable NOW  (publication/filename)        {after}")
    if multi:
        print(f"\n   → {after - before} chains moved from 'needs a human' to 'resolves automatically'")
    await c.close()


asyncio.run(main())
