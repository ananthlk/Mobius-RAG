"""Versioning & dedup gate — DRY RUN over a narrow pilot set.

Spec: docs/versioning-dedup-gate-spec.md
This is §12 step 4 ("gate: decision logic + lane rules, dry-run only").

STRICTLY READ-ONLY. Every statement is a SELECT. The gate's decisions are
computed and printed — including the §1.1 telemetry row it WOULD write —
but nothing is written, retired, deleted or re-triggered.

The pilot set is chosen to exercise every branch:
  · 59G-4.130      — real version pair + episodic notices sharing a rule number
  · a duplicate group — exact content duplicates at different URLs
  · zero-page docs   — §5.0 remediation: re-trigger extraction
  · zero-chunk docs  — §5.0 remediation: re-enqueue chunking
  · an episodic cluster — must NOT be version-linked

Usage:  python3 scripts/gate_dryrun.py
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import unicodedata
from collections import defaultdict
from datetime import date

import asyncpg

DSN = [l.split("=", 1)[1].strip() for l in open(".env") if l.startswith("DATABASE_URL=")][-1] \
    .replace("postgresql+asyncpg://", "postgresql://")

TAU_HIGH = 0.70          # §4 — confident successor at or above this
TAU_LOW = 0.35           # below this, "successor" is not a safe default
PAYER = "AHCA"

# ── §2.2 key evidence ────────────────────────────────────────────────────
RULE = re.compile(r"59[A-Z]?-?\s?(\d{1,3})\.(\d{1,4})", re.I)
PERIOD = re.compile(r"(19|20)\d{2}([-_/]\d{2,4})?|sfy\s?\d{2}[-_]?\d{0,2}|\bq[1-4]\b|\d(st|nd|rd|th)[ _]qtr", re.I)
# interim stand-in for Fact Store's asset_type (§11.2 ask) — NOT the proposed impl
REVISABLE = re.compile(r"coverage polic|handbook|manual|fee schedule|companion guide|"
                       r"provider guide|billing guide|reimbursement polic|contract", re.I)
EPISODIC = re.compile(r"minutes|agenda|notice|meeting|workshop|hearing|presentation|"
                      r"newsletter|bulletin|\bq[1-4]\b|qtr|quarter|sfy\s?\d{2}", re.I)
CATEGORY_LABEL = re.compile(r"^[A-Za-z ]+ — ")     # the contaminated display_name (§2.2a)


# ── §3 normalization ─────────────────────────────────────────────────────
def normalize(text: str) -> str:
    """Prose normalization for hashing. Structure-preserving is NOT yet
    implemented — flagged in §3 as the fee-schedule caution."""
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text)
    s = s.replace(" ", " ").replace("­", "")
    s = re.sub(r"[‘’]", "'", s)
    s = re.sub(r"[“”]", '"', s)
    s = re.sub(r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$", " ", s, flags=re.I | re.M)
    s = re.sub(r"printed on .*$", " ", s, flags=re.I | re.M)
    s = re.sub(r"last updated:?.*$", " ", s, flags=re.I | re.M)
    return re.sub(r"\s+", " ", s).strip().lower()


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def title_of(d) -> str:
    """display_name is a category label for 28.5% of the corpus (§2.2a) —
    fall back to filename when it is."""
    dn = d["display_name"]
    if dn and not CATEGORY_LABEL.match(str(dn)):
        return str(dn)
    return str(d["filename"] or "")


# ── §2.2 doc_key, evidence-gated, precedence order ───────────────────────
def compute_doc_key(d) -> tuple[str | None, str]:
    name = f"{title_of(d)} {d['filename'] or ''}"
    # EPISODIC GATE RUNS FIRST. A "Notice of Proposed Rule: 59G-4.130" carries the
    # rule number but IS NOT the rule — it is a dated announcement about it. Keying
    # it as the rule makes a notice supersede the policy. Same citation-vs-identity
    # distinction as §2.2b, but occurring in the filename rather than the body.
    if EPISODIC.search(name):
        return None, "episodic (announcement/record) — no key even with a rule#"
    m = RULE.search(name)
    if m:                                                    # tier 1
        return f"{PAYER}|{d['state']}|59G-{m.group(1)}.{m.group(2)}", "tier1 rule# in filename"
    if REVISABLE.search(name) and not EPISODIC.search(name):  # tier 2
        stem = PERIOD.sub(" ", name.lower())
        stem = re.sub(r"\.(pdf|html?|docx?|xlsx?)", " ", stem)
        stem = re.sub(r"[^a-z]+", " ", stem).strip()
        if len(stem) > 8:
            return f"{PAYER}|{d['state']}|{stem[:60]}", "tier2 revisable + filename"
    return None, "episodic — no key, singleton forever"       # §2.2 default


# ── §5.0 publishability + remediation ────────────────────────────────────
def publishability(n_pages: int, n_chunks: int, status: str | None):
    if n_pages == 0:
        return "shelved", "0 pages", "delete derived, re-trigger EXTRACTION"
    if n_chunks == 0:
        return "shelved", "pages but 0 chunks", "delete partial chunks, re-enqueue CHUNKING"
    if status in ("failed", "completed_with_errors"):
        return "shelved", f"upstream status={status}", "delete ALL derived, re-trigger from raw"
    return None, None, None


def h(t):
    print("\n" + "=" * 92)
    print(t)
    print("=" * 92)


async def main():
    c = await asyncpg.connect(DSN, timeout=60)

    # ── assemble the pilot set ───────────────────────────────────────────
    pilot = {}

    async def add(rows, tag):
        for r in rows:
            pilot.setdefault(r["id"], (dict(r), tag))

    cols = """d.id, d.filename, d.display_name, d.state, d.status,
              d.effective_date, d.termination_date, d.created_at, d.authority_level,
              d.source_metadata->>'source_url' AS src"""

    await add(await c.fetch(f"""SELECT {cols} FROM documents d WHERE d.payer=$1
        AND (d.filename ILIKE '%59G-4.130%' OR d.display_name ILIKE '%59G-4.130%')""", PAYER), "59G-4.130")
    await add(await c.fetch(f"""SELECT {cols} FROM documents d WHERE d.payer=$1
        AND d.filename ILIKE '%CS24%' LIMIT 6""", PAYER), "dup-group")
    await add(await c.fetch(f"""SELECT {cols} FROM documents d WHERE d.payer=$1
        AND NOT EXISTS (SELECT 1 FROM document_pages p WHERE p.document_id=d.id) LIMIT 2""", PAYER), "zero-page")
    await add(await c.fetch(f"""SELECT {cols} FROM documents d WHERE d.payer=$1
        AND EXISTS (SELECT 1 FROM document_pages p WHERE p.document_id=d.id)
        AND NOT EXISTS (SELECT 1 FROM hierarchical_chunks hc WHERE hc.document_id=d.id) LIMIT 2""", PAYER), "zero-chunk")
    await add(await c.fetch(f"""SELECT {cols} FROM documents d WHERE d.payer=$1
        AND d.filename ILIKE '%Notice of Meeting%' LIMIT 3""", PAYER), "episodic")

    ids = list(pilot)
    print(f"PILOT SET: {len(ids)} documents  (read-only; nothing will be written)")

    # ── facts ────────────────────────────────────────────────────────────
    pg = {r["id"]: r["n"] for r in await c.fetch(
        """SELECT d.id, count(p.id) n FROM documents d LEFT JOIN document_pages p ON p.document_id=d.id
           WHERE d.id=ANY($1::uuid[]) GROUP BY d.id""", ids)}
    chunks = defaultdict(list)
    for r in await c.fetch(
        """SELECT document_id, text FROM hierarchical_chunks
           WHERE document_id=ANY($1::uuid[]) ORDER BY page_number, paragraph_index""", ids):
        chunks[r["document_id"]].append(r["text"])

    state = {}
    for did, (d, tag) in pilot.items():
        hashes = [sha(normalize(t)) for t in chunks.get(did, [])]
        key, why = compute_doc_key(d)
        state[did] = dict(
            d=d, tag=tag, n_pages=pg.get(did, 0), n_chunks=len(hashes),
            hashes=set(hashes),
            digest=sha("".join(hashes)) if hashes else None,
            doc_key=key, key_why=why,
        )

    # ── run the gate ─────────────────────────────────────────────────────
    h("GATE DECISIONS  (dry run)")
    by_key = defaultdict(list)
    for did, s in state.items():
        if s["doc_key"]:
            by_key[s["doc_key"]].append(did)

    decisions = {}
    for did, s in sorted(state.items(), key=lambda kv: (kv[1]["tag"], str(kv[1]["d"]["filename"]))):
        d = s["d"]
        life, reason, action = publishability(s["n_pages"], s["n_chunks"], d["status"])
        if life:
            decisions[did] = dict(decision="unpublishable", lifecycle=life, reason=reason,
                                  action=action, index="none", prior=None, overlap=None)
            continue

        # prior = earlier doc sharing the key
        prior = None
        if s["doc_key"]:
            sibs = [x for x in by_key[s["doc_key"]] if x != did and state[x]["digest"]]
            earlier = [x for x in sibs
                       if (state[x]["d"]["effective_date"] or date(1900, 1, 1))
                       < (d["effective_date"] or date(1900, 1, 1))]
            if earlier:
                prior = max(earlier, key=lambda x: state[x]["d"]["effective_date"] or date(1900, 1, 1))

        if prior is None:
            decisions[did] = dict(decision="first_version", lifecycle="active", reason="no prior at doc_key",
                                  action="publish v1", index="admitted", prior=None, overlap=None)
            continue

        ps = state[prior]
        if ps["digest"] == s["digest"]:
            decisions[did] = dict(decision="unchanged", lifecycle="active(prior)",
                                  reason="identical content_digest", action="bump last_validated_at",
                                  index="none  <-- MUST be free", prior=prior, overlap=1.0)
            continue

        ov = len(s["hashes"] & ps["hashes"]) / max(len(s["hashes"] | ps["hashes"]), 1)
        # Default is ASK, never PROMOTE. Promotion retires a live document, so it
        # requires positive evidence; ambiguity must never resolve to the
        # destructive branch.
        if ov >= TAU_HIGH:
            dec = "successor"
            act = "promote v2; retire prior (retired_at only, termination_date stays NULL)"
            idx = "admitted + retire prior"
        elif ov >= TAU_LOW:
            dec = "ambiguous_revision"
            act = "-> Fact Store: heavy revision or different doc? prior STAYS ACTIVE"
            idx = "admitted; prior NOT retired"
        else:
            dec = "ambiguous_tail"
            act = "-> Fact Store: likely unrelated. prior STAYS ACTIVE"
            idx = "admitted; prior NOT retired"
        decisions[did] = dict(decision=dec, lifecycle="active",
                              reason=f"overlap={ov:.3f}  (tau_low={TAU_LOW} tau_high={TAU_HIGH})",
                              action=act, index=idx, prior=prior, overlap=ov)

    for did, s in sorted(state.items(), key=lambda kv: (kv[1]["tag"], str(kv[1]["d"]["filename"]))):
        dec = decisions[did]
        d = s["d"]
        print(f"\n[{s['tag']}] {str(d['filename'])[:66]}")
        print(f"   pages={s['n_pages']:<4} chunks={s['n_chunks']:<4} eff={d['effective_date']} "
              f"term={d['termination_date']}")
        print(f"   doc_key   : {s['doc_key'] or '(none)'}   [{s['key_why']}]")
        print(f"   DECISION  : {dec['decision'].upper():<20} {dec['reason']}")
        print(f"   ACTION    : {dec['action']}")
        print(f"   index     : {dec['index']}")

    # ── §1.1 telemetry rows the gate WOULD write ─────────────────────────
    h("§1.1 TELEMETRY — rows that WOULD be written before publish (nothing written)")
    print(f"{'decision':<20}{'lane':<11}{'overlap':>8}  {'index_action':<22}doc")
    print("-" * 92)
    for did, s in sorted(state.items(), key=lambda kv: kv[1]["tag"]):
        dec = decisions[did]
        lane = "tracked" if s["doc_key"] else "untracked"
        ov = f"{dec['overlap']:.3f}" if dec["overlap"] is not None else "  —"
        print(f"{dec['decision']:<20}{lane:<11}{ov:>8}  {dec['index'][:20]:<22}"
              f"{str(s['d']['filename'])[:30]}")

    h("SUMMARY BY DECISION")
    agg = defaultdict(int)
    for dec in decisions.values():
        agg[dec["decision"]] += 1
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"   {k:<24} {v}")
    print("\nNOTHING WAS WRITTEN. No retire, no delete, no re-trigger, no index change.")
    await c.close()


asyncio.run(main())
