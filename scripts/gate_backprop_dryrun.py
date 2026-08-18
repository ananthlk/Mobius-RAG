"""Back-propagation dry run + the Fact Store seam payload.

Spec: docs/versioning-dedup-gate-spec.md §8 (back-propagation), §7 (human loop).
STRICTLY READ-ONLY — prints what it WOULD do and what it WOULD send. No writes.

WHY BACK-PROP IS A DIFFERENT ALGORITHM FROM THE FORWARD GATE
------------------------------------------------------------
The forward gate sees one document arriving against a known-current prior:
"is this the successor of that?" — a pairwise question.

Back-propagation has no arrival order. It has a pile of documents and must
RECONSTRUCT the chain. Two consequences:

  1. Ordering is inferred, not observed. effective_date where trustworthy,
     else created_at as an explicitly-flagged weak proxy.
  2. THE UNIT OF ADJUDICATION IS THE CLUSTER, NOT THE PAIR. One ambiguous
     link in the middle poisons every ordering decision downstream of it,
     so a human resolves the whole chain at once. Sending three separate
     pair questions invites three answers that don't compose.

Usage:  python3 scripts/gate_backprop_dryrun.py
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from datetime import date

import asyncpg

DSN = [l.split("=", 1)[1].strip() for l in open(".env") if l.startswith("DATABASE_URL=")][-1] \
    .replace("postgresql+asyncpg://", "postgresql://")

PAYER = "AHCA"
TAU_HIGH, TAU_LOW = 0.70, 0.35

RULE = re.compile(r"59[A-Z]?-?\s?(\d{1,3})\.(\d{1,4})", re.I)
EPISODIC = re.compile(r"minutes|agenda|notice|meeting|workshop|hearing|presentation|"
                      r"newsletter|bulletin|\bq[1-4]\b|qtr|quarter|sfy\s?\d{2}", re.I)
REVISABLE = re.compile(r"coverage polic|handbook|manual|fee schedule|companion guide|"
                       r"provider guide|billing guide|reimbursement polic|contract", re.I)
PERIOD = re.compile(r"(19|20)\d{2}([-_/]\d{2,4})?|sfy\s?\d{2}[-_]?\d{0,2}", re.I)
CATEGORY_LABEL = re.compile(r"^[A-Za-z ]+ — ")
CH = "md5(regexp_replace(lower(c.text), '\\s+', ' ', 'g'))"


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


def h(t):
    print("\n" + "=" * 92 + f"\n{t}\n" + "=" * 92)


async def main():
    c = await asyncpg.connect(DSN, timeout=90)
    docs = await c.fetch(f"""SELECT d.id,d.filename,d.display_name,d.state,d.status,
        d.effective_date,d.termination_date,d.created_at,d.authority_level,
        d.source_metadata->>'source_url' src FROM documents d WHERE d.payer=$1""", PAYER)
    ids = [d["id"] for d in docs]
    pg = {r["id"]: r["n"] for r in await c.fetch(
        """SELECT d.id,count(p.id) n FROM documents d LEFT JOIN document_pages p ON p.document_id=d.id
           WHERE d.id=ANY($1::uuid[]) GROUP BY d.id""", ids)}
    dig = {r["id"]: (r["dg"], r["n"]) for r in await c.fetch(
        f"""SELECT c.document_id id, md5(string_agg({CH},'' ORDER BY c.page_number,c.paragraph_index)) dg,
            count(*) n FROM hierarchical_chunks c WHERE c.document_id=ANY($1::uuid[])
            GROUP BY c.document_id""", ids)}

    # ── PASS 1 — publishability (§5.0) ───────────────────────────────────
    h("BACK-PROP PASS 1 — publishability, then dedup, then chains")
    unpub = defaultdict(list)
    live = []
    for d in docs:
        n_pages, (dg, n_ch) = pg.get(d["id"], 0), dig.get(d["id"], (None, 0))
        if n_pages == 0:
            unpub["0 pages → re-trigger EXTRACTION"].append(d)
        elif n_ch == 0:
            unpub["pages, 0 chunks → re-enqueue CHUNKING"].append(d)
        elif d["status"] in ("failed", "completed_with_errors"):
            unpub[f"status={d['status']} → delete derived, re-trigger from raw"].append(d)
        else:
            live.append((d, dg, n_ch))
    print(f"  {'documents':<52}{len(docs):>6}")
    for k, v in sorted(unpub.items(), key=lambda kv: -len(kv[1])):
        print(f"  − {k:<50}{len(v):>6}")
    print(f"  {'= publishable':<52}{len(live):>6}")

    # ── PASS 2 — exact duplicates (§8 phase 3) ───────────────────────────
    bydg = defaultdict(list)
    for d, dg, n in live:
        bydg[dg].append((d, n))
    AUTH = {"contract_source_of_truth": 3, "payer_website": 2, "fyi_not_citable": 0}
    dupes = []
    canon = []
    for dg, group in bydg.items():
        if len(group) == 1:
            canon.append(group[0][0])
            continue
        # canonical = highest authority, then earliest created (authority is a property of origin)
        ranked = sorted(group, key=lambda t: (-AUTH.get(t[0]["authority_level"], 1), t[0]["created_at"]))
        canon.append(ranked[0][0])
        dupes += [t for t in ranked[1:]]
    print(f"  − {'exact duplicate (retire, delete their chunks)':<50}{len(dupes):>6}"
          f"   [{sum(n for _, n in dupes):,} indexed chunks]")
    print(f"  {'= distinct documents':<52}{len(canon):>6}")

    # ── PASS 3 — reconstruct chains (§8 phases 4-6) ──────────────────────
    clusters = defaultdict(list)
    for d in canon:
        k = doc_key(d)
        if k:
            clusters[k].append(d)
    multi = {k: v for k, v in clusters.items() if len(v) > 1}
    print(f"\n  keyed documents          {sum(len(v) for v in clusters.values()):>6}"
          f"   ({len(clusters)} clusters)")
    print(f"  episodic / unkeyed       {len(canon)-sum(len(v) for v in clusters.values()):>6}"
          f"   → singleton forever, version 1, never superseded")
    print(f"  multi-doc chains         {len(multi):>6}   ← the only back-prop version work")

    auto, to_human = [], []
    for k, v in multi.items():
        weak = any(x["effective_date"] is None for x in v)
        chain = sorted(v, key=lambda x: (x["effective_date"] or date(1900, 1, 1), x["created_at"]))
        links = []
        for a, b in zip(chain, chain[1:]):
            ha = {r["x"] for r in await c.fetch(
                f"SELECT DISTINCT {CH} x FROM hierarchical_chunks c WHERE c.document_id=$1", a["id"])}
            hb = {r["x"] for r in await c.fetch(
                f"SELECT DISTINCT {CH} x FROM hierarchical_chunks c WHERE c.document_id=$1", b["id"])}
            ov = len(ha & hb) / max(len(ha | hb), 1) if ha and hb else 0.0
            links.append((a, b, ov, len(ha & hb), len(hb - ha)))
        # ONE ambiguous link sends the WHOLE cluster to a human
        if weak or any(ov < TAU_HIGH for _, _, ov, _, _ in links):
            to_human.append((k, chain, links, weak))
        else:
            auto.append((k, chain, links))

    print(f"\n  → chains resolvable automatically   {len(auto):>4}")
    print(f"  → chains needing Fact Store         {len(to_human):>4}"
          f"   (one weak link sends the whole cluster)")

    # ── the Fact Store seam ──────────────────────────────────────────────
    h("THE FACT STORE SEAM — the adjudication request that would be emitted")
    if not to_human:
        print("  (none)")
    for k, chain, links, weak in to_human[:2]:
        a, b, ov, carried, changed = links[0]
        payload = {
            "adjudication_id": "(assigned on emit)",
            "source": "mobius-rag:versioning-gate",
            "mode": "back_propagation",
            "doc_key": k,
            # Fact Store refinement #2: verdict keys on the DIGEST PAIR + doc_key,
            # so tonight's re-crawl cannot recompute the human's answer away.
            "verdict_key": {
                "doc_key": k,
                "predecessor_digest": dig[a["id"]][0],
                "successor_digest": dig[b["id"]][0],
            },
            "cluster_size": len(chain),
            "ordering_confidence": "weak — effective_date missing" if weak else "date-ordered",
            "candidates": [
                {"document_id": str(x["id"]), "title": title_of(x),
                 "effective_date": str(x["effective_date"]), "authority": x["authority_level"],
                 "chunks": dig[x["id"]][1]} for x in chain],
            "diff": {
                "overlap_ratio": round(ov, 4),
                "chunks_carried": carried,
                "chunks_changed": changed,
                "rendered_by": "mobius-rag",   # Fact Store Q2: RAG pre-renders, they display
            },
            "question": "Is the successor a new version of the predecessor?",
            "answers_required": [
                "relationship: successor | not_successor | unrelated",
                # Fact Store refinement #1: TWO separate valid-time dates.
                # Never derive one from the other; NULL if the boundary is unstated.
                "successor.effective_date: date | null",
                "predecessor.termination_date: date | null",
                "promote: true | false",
            ],
            "rag_will_not": [
                "write termination_date (§6 — gate never sets valid time)",
                "retire the predecessor before a verdict (§4.2 — prior stays ACTIVE)",
            ],
        }
        print(json.dumps(payload, indent=2)[:2100])

    h("WHAT BACK-PROP WOULD ACTUALLY CHANGE")
    print(f"  re-trigger extraction        {len(unpub['0 pages → re-trigger EXTRACTION']):>6}")
    print(f"  re-enqueue chunking          {len(unpub['pages, 0 chunks → re-enqueue CHUNKING']):>6}")
    print(f"  retire duplicates            {len(dupes):>6}   [{sum(n for _, n in dupes):,} chunks deleted from index]")
    print(f"  auto-resolve version chains  {len(auto):>6}")
    print(f"  emit to Fact Store           {len(to_human):>6}")
    print("\nNOTHING WAS WRITTEN.")
    await c.close()


asyncio.run(main())
