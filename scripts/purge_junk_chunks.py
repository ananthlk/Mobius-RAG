"""Purge no-substance chunks from the live vector index.

Sourcing ruled GO NOW (S-2, 2026-08-19) and widened the scope beyond ASCII '-'
to U+2010 '‐'. Ananth approved: "purge the junk chunks and fix the chunker".

WHAT IS REMOVED.  Chunks carrying fewer than 3 alphanumeric characters. This is
table-extraction bleed: a markdown table flattened one-cell-per-line turns every
empty cell into its own chunk. 185,261 were literally "-", 14,234 were "‐".
They embed to near-identical vectors, so they occupy top-k at a uniform
similarity and crowd out real content.

PARITY (A-43).  The identical rule lives in app/services/chunking.py as
MIN_SUBSTANCE_ALNUM / has_min_substance(), which stops the chunker re-creating
what this removes. Changing one without the other silently re-admits the noise.

REVERSIBLE.  Touches rag_published_embeddings ONLY. hierarchical_chunks is left
intact, so any document can be restored by re-publishing — no re-fetch, no
re-extraction. scratchpad/purge_manifest.json records document_id -> count.

RESILIENCE.  The first run died at 60,000 when the cloud-sql-proxy dropped the
connection (a known degradation on long uptime). Batches commit independently,
so that work survived; this version reconnects and resumes rather than losing a
long transaction. Ids are harvested ONCE into a local list — the original
re-ran the regexp predicate over 1.9M rows per batch, which is what made it slow
enough to hit the proxy's failure window in the first place.
"""
import json, os, sys, time
import psycopg2

BATCH = 5000
DSN = [l.split("=",1)[1].strip() for l in open(".env") if l.startswith("DATABASE_URL=")][-1] \
      .replace("postgresql+asyncpg://","postgresql://")
PRED = "length(regexp_replace(trim(text),'[^[:alnum:]]','','g')) < 3"
SCRATCH = os.environ.get("SCRATCH", "scratchpad")


def connect():
    c = psycopg2.connect(DSN, connect_timeout=30)
    c.autocommit = False
    cur = c.cursor(); cur.execute("SET statement_timeout = 0"); cur.close()
    return c


def main():
    apply = "--apply" in sys.argv
    c = connect(); cur = c.cursor()

    # Manifest first — after the delete the evidence is gone by construction.
    cur.execute(f"""SELECT document_id::text, count(*) FROM rag_published_embeddings
                    WHERE {PRED} GROUP BY 1""")
    manifest = {d: n for d, n in cur.fetchall()}
    total = sum(manifest.values())
    print(f"manifest: {len(manifest):,} documents / {total:,} chunks")
    os.makedirs(SCRATCH, exist_ok=True)
    with open(f"{SCRATCH}/purge_manifest.json", "w") as f:
        json.dump(manifest, f)

    # Documents that would be emptied entirely — a purge must never silently
    # remove a document's last chunk and leave it "published" with nothing.
    cur.execute(f"""SELECT count(*) FROM (
        SELECT document_id FROM rag_published_embeddings GROUP BY document_id
        HAVING count(*) FILTER (WHERE {PRED}) = count(*)) x""")
    emptied = cur.fetchone()[0]
    print(f"documents that would be left with ZERO chunks: {emptied}")
    if emptied:
        print("ABORT — refusing to unpublish a document via a noise filter.")
        return

    if not apply:
        print("\nDRY RUN — nothing removed. Re-run with --apply.")
        return

    # Harvest ids ONCE, then delete by primary key.
    print("harvesting ids ...", flush=True)
    t0 = time.time()
    cur.execute(f"SELECT id FROM rag_published_embeddings WHERE {PRED}")
    ids = [r[0] for r in cur.fetchall()]
    print(f"   {len(ids):,} ids in {time.time()-t0:.0f}s", flush=True)
    cur.close()

    done = 0
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i+BATCH]
        for attempt in range(5):
            try:
                cur = c.cursor()
                cur.execute("DELETE FROM rag_published_embeddings WHERE id = ANY(%s::uuid[])",
                            ([str(x) for x in chunk],))
                n = cur.rowcount
                c.commit(); cur.close()
                done += n
                break
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                # TRANSPORT ONLY. The proxy dropped us; the batch either committed
                # or it did not, and DELETE by id is idempotent either way, so a
                # retry is safe. Programming errors are deliberately NOT caught —
                # retrying one can never succeed, and logging it as 'reconnecting'
                # hides the real fault behind a plausible-looking recovery.
                print(f"   reconnecting after {type(e).__name__} (attempt {attempt+1})", flush=True)
                try: c.close()
                except Exception: pass
                time.sleep(3 * (attempt + 1))
                c = connect()
        else:
            print(f"GIVING UP at {done:,} purged — batch failed 5 times."); break
        if (i // BATCH) % 5 == 0:
            print(f"   {done:,} purged  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nPURGED {done:,}", flush=True)

    # Read the writes back — a purge with no reader is the defect class that
    # keeps recurring here. Independent query, same run.
    cur = c.cursor()
    cur.execute(f"SELECT count(*) FROM rag_published_embeddings WHERE {PRED}")
    left = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM rag_published_embeddings")
    tot = cur.fetchone()[0]
    print(f"read-back: junk remaining={left:,} (expect 0) · index total={tot:,}")
    c.close()


main()
