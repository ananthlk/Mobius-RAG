"""FORWARD-gate simulation by hold-out. The honest version of the earlier run.

WHY THIS EXISTS
---------------
The previous 500-doc run was labelled mode='forward'. It was not. Verified:

  · 0 URLs in the corpus have ever been fetched more than once — a
    re-observation has never occurred, so `unchanged` was unreachable.
  · All 12 Attachment II editions were ingested on ONE day (2026-04-29)
    from an archive page. That is a historical pile, not a revision stream.

So that run was a REPLAY over static data. Its two "promotions" were
back-propagation chain reconstruction, which is a different question from
"a new document just arrived — does it supersede what is live?"

This harness answers the forward question properly, by HOLD-OUT: we hide
edition N, treat edition N-1 as the live corpus, then present N as arriving.
Nothing about the corpus changes; the hold-out is constructed in memory.

  CASE A  re-present a document that is already active
          → MUST be `unchanged`, and MUST make zero index writes (§12.1)
  CASE B  present edition N against N-1 as live
          → the real forward promotion question

Read-only. Writes nothing, not even telemetry.
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
    s = re.sub(r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$", " ", s, flags=re.I | re.M)
    s = re.sub(r"(printed on|last updated:?).*$", " ", s, flags=re.I | re.M)
    return re.sub(r"\s+", " ", s).strip().lower()


def sha(s): return hashlib.sha256(s.encode()).hexdigest()[:32]


def fdate(fn):
    m = FNDATE.search(str(fn or ""))
    if not m:
        return None
    try:
        if m.group(1):
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return date(2000 + int(m.group(6)), int(m.group(4)), int(m.group(5)))
    except ValueError:
        return None


def decide(incoming, live):
    """The forward gate. `live` is the currently-active doc at this doc_key, or None."""
    if live is None:
        return "first_version", "no active version at doc_key", None, "admitted", 0
    if incoming["digest"] == live["digest"]:
        return "unchanged", "identical content_digest", 1.0, "NONE — must be free", 0
    carried = len(incoming["hset"] & live["hset"])
    ov = carried / max(len(incoming["hset"] | live["hset"]), 1)
    changed = len(incoming["hset"] - live["hset"])
    if ov >= TAU_HIGH:
        return "successor", f"overlap={ov:.3f} >= tau_high", ov, "admitted + retire prior", changed
    if ov >= TAU_LOW:
        return "ambiguous_revision", f"overlap={ov:.3f}", ov, "admitted; prior NOT retired", changed
    return "ambiguous_tail", f"overlap={ov:.3f} < tau_low", ov, "admitted; prior NOT retired", changed


async def main():
    c = await asyncpg.connect(DSN, timeout=90)
    rows = await c.fetch("""SELECT id, filename FROM documents WHERE payer='AHCA'
        AND filename ILIKE 'Attachment_II%Core_Contract%'""")
    chain = []
    for r in rows:
        d = fdate(r["filename"])
        if d:
            chain.append((d, r["id"], r["filename"]))
    chain.sort()

    docs = {}
    for _, did, fn in chain:
        txt = await c.fetch("SELECT text FROM hierarchical_chunks WHERE document_id=$1", did)
        hset = {sha(normalize(t["text"])) for t in txt}
        docs[did] = dict(hset=hset, digest=sha("".join(sorted(hset))), fn=fn)

    print("=" * 90)
    print("FORWARD-GATE SIMULATION — hold-out over the Attachment II chain")
    print("=" * 90)
    print(f"{len(chain)} dated editions, presented in publication order.\n")

    live = None
    live_fn = None
    for edate, did, fn in chain:
        inc = docs[did]
        dec, why, ov, idx, changed = decide(inc, docs[live] if live else None)
        ovs = f"{ov:.3f}" if ov is not None else "  —  "
        print(f"  {str(edate)}  {fn[:46]:48s}")
        print(f"     vs live: {live_fn[:44] if live_fn else '(nothing yet)'}")
        print(f"     → {dec.upper():<20} ov={ovs}  index: {idx}"
              + (f"  re-embed={changed}" if changed else ""))
        # BUG 4 FIX — the comparison anchor is the most recently ADMITTED version,
        # not the most recently PROMOTED one. Admission and retirement are separate
        # decisions: every branch above admits the incoming document to the index;
        # they differ only in whether the PRIOR gets retired. Anchoring on the last
        # promotion freezes the chain at its oldest edition the moment one link is
        # ambiguous, and overlap against that frozen ancestor then decays with every
        # further edition — so the chain gets progressively LESS able to resolve.
        if dec != "unchanged":
            live, live_fn = did, fn
        print()

    # ── CASE A — the nightly path. Re-present what is already live. ──────
    print("=" * 90)
    print("CASE A — re-present the ACTIVE document unchanged (the nightly path)")
    print("=" * 90)
    dec, why, ov, idx, changed = decide(docs[live], docs[live])
    print(f"  incoming : {live_fn[:60]}")
    print(f"  live     : {live_fn[:60]}")
    print(f"  → {dec.upper()}   {why}")
    print(f"  index_action = {idx}")
    print(f"  chunks re-embedded = {changed}")
    ok = (dec == "unchanged" and changed == 0)
    print(f"\n  ACCEPTANCE (§12.1): {'PASS — the nightly path is free' if ok else 'FAIL'}")

    # ── CASE A' — a cosmetic change that normalization should absorb ─────
    print("\n" + "=" * 90)
    print("CASE A' — same document, cosmetic drift only (whitespace + a print date)")
    print("=" * 90)
    raw = await c.fetch("SELECT text FROM hierarchical_chunks WHERE document_id=$1", live)
    drifted = {sha(normalize("  " + t["text"] + "\nPrinted on 2026-08-18\n")) for t in raw}
    inc2 = dict(hset=drifted, digest=sha("".join(sorted(drifted))))
    dec2, why2, ov2, idx2, ch2 = decide(inc2, docs[live])
    print(f"  → {dec2.upper()}   {why2}")
    print(f"  index_action = {idx2}   re-embed={ch2}")
    print(f"\n  {'PASS — normalization absorbed the drift' if dec2 == 'unchanged' else 'FAIL — cosmetic drift produced a false version'}")
    await c.close()


asyncio.run(main())
