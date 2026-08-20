"""Reingest a batch of table-heavy documents through the full pipeline.

WHY THESE DOCUMENTS. The purge manifest records how many no-substance chunks each
document contributed. That number IS a table-bleed score: a document that shed
3,568 orphaned cells is a document whose tables were being flattened into prose.
Ranking by it puts the reingest where table capture has the most to recover —
and, as it happens, on the documents that actually rank for LIP queries
(Model_16 and Model_10 were the top hits of the trace query that had no tables
to attach).

GROUP RULE (binding, TABLE_CAPTURE_PROGRAM.md). Re-extraction rewrites
`document_pages` and therefore the normalized md5 that duplicate determination
and the cleanup ledger both rest on. Reingesting one member of a duplicate group
while its twin keeps the old generation would let a verdict flip on a TRANSFORM
rather than on content. So the batch is expanded to whole connected components
before anything runs.

PIPELINE PER DOCUMENT
  extract/restart  -- capture excises tables -> document_tables + breadcrumb,
                      AND re-classifies (wired into restart_extraction, because
                      a verdict derived from the old page text is stale the
                      moment excision rewrites it)
    -> chunking kill-and-reset + start (generator B)
    -> auto-publish on embed

NO EXPLICIT PUBLISH. `AUTO_PUBLISH_ON_EMBED=1` already copies vectors into
rag_published_embeddings when the embedding job completes. Calling publish as
well raced it and lost: `duplicate key value violates unique constraint
"rag_published_embeddings_pkey"`, a 500 on a document that had in fact
succeeded. The stage was redundant, not broken.

THE GATE RUNS AFTER THE WHOLE BATCH, not per document. Duplicate determination
is a SET operation -- it compares a document against every candidate in the
corpus -- so it cannot be scoped to one document mid-run, and running it 30
times would be 30 corpus scans. Run once at the end, telemetry-only: verdicts
are recomputed against the NEW md5s so we can see what the reingest changed
before anything acts on it.

WHY THE GATE MATTERS HERE. 161 documents were retired on a proof that their
normalized page text was md5-identical to a canonical. Reingesting either side
of such a pair invalidates that proof silently -- the ledger still claims
identity that no longer holds. Recomputing is how that surfaces.

Usage:  python3 scripts/batch_reingest.py --top 30 [--apply] [--no-gate]
"""
from __future__ import annotations

import json, sys, time
from collections import defaultdict

import psycopg2, requests

API = "https://mobius-rag-ortabkknqa-uc.a.run.app"
DSN = [l.split("=", 1)[1].strip() for l in open(".env") if l.startswith("DATABASE_URL=")][-1] \
      .replace("postgresql+asyncpg://", "postgresql://")
MANIFEST = "/private/tmp/claude-502/-Users-ananth-Mobius--claude-worktrees-gallant-jepsen/1e498910-b9e9-45c0-b743-9135a1b6d878/scratchpad/purge_manifest.json"
DONE = {"Model_10A.pdf", "Model_10A_2011-01-27.pdf", "Model_19B.pdf",
        "LIP_Model_5_2012-13_unlinked_nbm.pdf", "Sunshine_State_Health_Plan__Inc.__CW_.pdf"}


def connect():
    c = psycopg2.connect(DSN, connect_timeout=30)
    cur = c.cursor(); cur.execute("SET statement_timeout='300s'"); cur.close()
    return c


def groups(c):
    """Connected components over every duplicate-ish edge in the corpus."""
    cur = c.cursor()
    cur.execute("""SELECT DISTINCT document_id::text, prior_document_id::text
                   FROM gate_decisions WHERE prior_document_id IS NOT NULL""")
    par = {}
    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for a, b in cur.fetchall():
        ra, rb = find(a), find(b)
        if ra != rb: par[ra] = rb
    comp = defaultdict(list)
    for x in list(par): comp[find(x)].append(x)
    cur.close()
    return par, comp, find


def poll(c, sql, args, want, label, timeout=1800):
    """Poll one condition to completion. Returns True on success."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        cur = c.cursor(); cur.execute(sql, args); v = cur.fetchone()[0]; cur.close()
        if want(v):
            return True
        time.sleep(10)
    print(f"      TIMEOUT waiting for {label}")
    return False


def main():
    apply = "--apply" in sys.argv
    top = next((int(a.split("=")[1]) for a in sys.argv if a.startswith("--top=")), 30)

    man = json.load(open(MANIFEST))
    c = connect(); cur = c.cursor()
    cur.execute("""SELECT id::text, filename FROM documents
                   WHERE id::text = ANY(%s) AND lifecycle_state IS DISTINCT FROM 'retired'""",
                (list(man),))
    ranked = sorted(((man[i], i, f) for i, f in cur.fetchall() if f not in DONE), reverse=True)
    seed = ranked[:top]

    par, comp, find = groups(c)
    batch, seen = [], set()
    for n, i, f in seed:
        members = comp.get(find(i), [i]) if i in par else [i]
        for m in members:
            if m in seen: continue
            seen.add(m); batch.append(m)

    cur.execute("""SELECT id::text, filename FROM documents
                   WHERE id::text = ANY(%s) AND lifecycle_state IS DISTINCT FROM 'retired'""",
                (batch,))
    names = dict(cur.fetchall())
    batch = [b for b in batch if b in names]

    print(f"seed documents (top {top} table bleeders) : {len(seed)}")
    print(f"after duplicate-group expansion           : {len(batch)}")
    print(f"  (+{len(batch) - len(seed)} pulled in by the group rule)")
    if not apply:
        for b in batch[:40]: print(f"   {names[b][:62]}")
        print("\nDRY RUN — nothing reingested. Re-run with --apply.")
        return

    ok, failed = [], []
    for n, did in enumerate(batch, 1):
        fn = names[did][:44]
        print(f"\n[{n}/{len(batch)}] {fn}", flush=True)
        try:
            r = requests.post(f"{API}/documents/{did}/extract/restart", timeout=300)
            if r.status_code != 200:
                print(f"      extract -> {r.status_code}"); failed.append((fn, "extract")); continue
            if not poll(c, "SELECT status FROM documents WHERE id=%s::uuid", (did,),
                        lambda v: v == "completed", "extraction"):
                failed.append((fn, "extract-timeout")); continue

            requests.post(f"{API}/documents/{did}/chunking/kill-and-reset", timeout=180)
            r = requests.post(f"{API}/documents/{did}/chunking/start?generator_id=B", timeout=300)
            if r.status_code != 200:
                print(f"      chunk -> {r.status_code}"); failed.append((fn, "chunk")); continue
            if not poll(c, """SELECT status FROM chunking_jobs WHERE document_id=%s::uuid
                              ORDER BY created_at DESC LIMIT 1""", (did,),
                        lambda v: v not in ("processing", "queued", "pending"), "chunking"):
                failed.append((fn, "chunk-timeout")); continue

            # Publishing is AUTO_PUBLISH_ON_EMBED's job; calling it here raced it.
            cur2 = c.cursor()
            cur2.execute("""SELECT
                 (SELECT count(*) FROM document_tables t WHERE t.document_id=%s::uuid),
                 (SELECT count(*) FROM rag_published_embeddings e WHERE e.document_id=%s::uuid),
                 (SELECT count(*) FROM document_pages p
                    WHERE p.document_id=%s::uuid AND p.text LIKE '%%[Table:%%'),
                 (SELECT source_metadata->'payor_classification'->>'classified_at'
                    FROM documents WHERE id=%s::uuid)""",
                         (did, did, did, did))
            tabs, pub, crumbs, clf_at = cur2.fetchone(); cur2.close()
            # Read the classification back rather than assuming the wired call ran:
            # it fails open by design, so a silent miss is exactly what would hide.
            stale = " CLASSIFY-STALE" if not clf_at else ""
            print(f"      OK — {tabs} tables, {crumbs} breadcrumb pages, "
                  f"{pub:,} published{stale}", flush=True)
            ok.append((fn, tabs, pub))
        except Exception as e:
            print(f"      ERROR {type(e).__name__}: {str(e)[:90]}")
            failed.append((fn, type(e).__name__))

    print(f"\n=== BATCH DONE ===\nsucceeded: {len(ok)}   failed: {len(failed)}")
    print(f"tables captured this batch: {sum(t for _, t, _ in ok):,}")
    for f, why in failed[:15]: print(f"   FAILED {f} @ {why}")

    if "--no-gate" in sys.argv:
        print("\n--no-gate: duplicate determination NOT re-run. Verdicts for every "
              "document above now rest on an md5 that no longer matches their text.")
        c.close(); return

    # --- Duplicate / versioning gate, telemetry only -------------------------
    # Re-extraction changed the normalized page text of every document above, so
    # every duplicate verdict touching them was computed against text that no
    # longer exists. This recomputes; it does NOT act. --apply on the gate is a
    # separate, deliberate decision after reading what changed.
    print("\n=== RE-RUNNING DUPLICATE DETERMINATION (telemetry only) ===", flush=True)
    import subprocess
    r = subprocess.run([sys.executable, "scripts/gate_duplicates.py"],
                       capture_output=True, text=True, timeout=5400)
    print(r.stdout[-4000:] if r.stdout else "(no output)")
    if r.returncode != 0:
        print(f"GATE FAILED rc={r.returncode}\n{r.stderr[-1500:]}")
    c.close()


main()
