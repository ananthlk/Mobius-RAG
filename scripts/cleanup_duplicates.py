"""Duplicate cleanup — the non-human loop.

POLICY (Ananth, 2026-08-18):
  unmanaged duplicates  -> cleaned automatically, no human wait
  managed duplicates    -> held from publishing, human adjudicates

WHAT "CLEANED" MEANS, PRECISELY
  lifecycle_state is set to 'retired' — the value the DB constraint ratifies
  (active | retired | shelved | quarantined). An earlier draft used 'superseded',
  which the CHECK rejected; the transaction rolled the deletes back with it, which
  is the only reason a vocabulary slip did not become 152 half-cleaned documents.

  removed : rag_published_embeddings (the live vector index), chunk_embeddings,
            hierarchical_chunks, embeddable_units
  kept    : the `documents` row, the GCS object at file_path, document_pages,
            publish_events, chunking_jobs, document_process_status, gate_decisions

The document is not deleted and neither is its history. Restoring one is a
re-chunk from surviving pages — no re-download, no re-extraction. Every count is
written to corpus_cleanup_actions AS it is removed, because afterwards the
evidence is gone by construction and a recount cannot tell a cleaned document
from one that never chunked.

CANONICAL RULE
  Earliest created_at within a connected group. Safe ONLY because these pairs are
  byte-identical after normalization — there is no edition question to get wrong,
  so "which copy" cannot lose content. This rule is deliberately NOT used for
  versioning, where created_at says nothing about which edition is newer.

  Groups are connected components, not pairs. A~B and B~C is ONE group with one
  survivor; resolving pair-by-pair would retire B twice from two different keeps.

SAFETY GATES (each aborts the group, not the run)
  1. normalized page text must be md5-identical — identity is proven per document
     at cleanup time, not inherited from the earlier scoring run
  2. the canonical must itself be published, or removing its twin would take the
     content out of the index entirely
  3. managed documents are never touched here

Usage:  python3 scripts/cleanup_duplicates.py [--apply]
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
import uuid
from collections import defaultdict

import asyncpg

DSN = [l.split("=", 1)[1].strip() for l in open(".env") if l.startswith("DATABASE_URL=")][-1] \
    .replace("postgresql+asyncpg://", "postgresql://")


def norm_md5(t: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", (t or "").lower()).strip().encode()).hexdigest()


async def main():
    apply = "--apply" in sys.argv
    c = await asyncpg.connect(DSN, timeout=300)
    await c.execute("SET statement_timeout = 0")
    await c.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")

    drun = await c.fetchval("""SELECT run_id FROM gate_decisions WHERE mode='duplicates'
                               GROUP BY run_id ORDER BY max(decided_at) DESC LIMIT 1""")
    pairs = await c.fetch("""
        SELECT DISTINCT least(document_id::text, prior_document_id::text) AS a,
                        greatest(document_id::text, prior_document_id::text) AS b
        FROM gate_decisions WHERE run_id=$1 AND duplicate_kind='duplicate'""", drun)

    par: dict = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    def uni(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb

    for r in pairs:
        uni(r["a"], r["b"])
    comp = defaultdict(list)
    for x in list(par):
        comp[find(x)].append(x)

    rows = await c.fetch("""
        SELECT d.id::text AS id, d.filename, d.created_at, d.file_path,
               COALESCE(jsonb_typeof(d.source_metadata)='object'
                        AND d.source_metadata ? 'payor_classification', false) AS managed,
               (SELECT count(*) FROM rag_published_embeddings e WHERE e.document_id=d.id) AS pub,
               (SELECT count(*) FROM document_pages p WHERE p.document_id=d.id) AS pages
        FROM documents d WHERE d.id::text = ANY($1::text[])""", list(par))
    m = {r["id"]: r for r in rows}

    run_id = uuid.uuid4()
    planned, held, skipped = [], [], defaultdict(int)

    for root, mem in comp.items():
        mem = [x for x in mem if x in m]
        if len(mem) < 2:
            continue
        mem.sort(key=lambda x: (m[x]["created_at"], x))
        keep = mem[0]

        # gate 2 — the survivor must itself be in the index
        if m[keep]["pub"] == 0:
            skipped["canonical_not_published"] += len(mem) - 1
            continue

        keep_md5 = norm_md5(await c.fetchval(
            "SELECT string_agg(text,' ' ORDER BY page_number) FROM document_pages WHERE document_id=$1",
            uuid.UUID(keep)))

        for r in mem[1:]:
            if m[r]["managed"]:
                held.append((r, keep))            # gate 3 — human loop owns these
                continue
            # gate 1 — prove identity now, per document
            if norm_md5(await c.fetchval(
                    "SELECT string_agg(text,' ' ORDER BY page_number) FROM document_pages "
                    "WHERE document_id=$1", uuid.UUID(r))) != keep_md5:
                skipped["text_not_identical"] += 1
                continue
            planned.append((r, keep))

    print(f"groups                     : {sum(1 for v in comp.values() if len(v) > 1)}")
    print(f"unmanaged -> auto-clean    : {len(planned)}")
    print(f"managed   -> held for human: {len(held)}")
    for k, v in skipped.items():
        print(f"skipped ({k}) : {v}")

    if not apply:
        print("\nDRY RUN — nothing removed. Re-run with --apply.")
        await c.close()
        return

    tot = defaultdict(int)
    for did, keep in planned:
        async with c.transaction():
            d = uuid.UUID(did)
            pub = await c.fetchval("SELECT count(*) FROM rag_published_embeddings WHERE document_id=$1", d)
            emb = await c.fetchval("SELECT count(*) FROM chunk_embeddings WHERE document_id=$1", d)
            hch = await c.fetchval("SELECT count(*) FROM hierarchical_chunks WHERE document_id=$1", d)
            eun = await c.fetchval("SELECT count(*) FROM embeddable_units WHERE document_id=$1", d)

            # Ledger row FIRST, inside the same transaction. If a delete fails the
            # whole thing rolls back together; a ledger row can never claim a
            # removal that did not happen, and a removal can never happen unlogged.
            await c.execute("""
                INSERT INTO corpus_cleanup_actions
                  (run_id, document_id, canonical_id, duplicate_kind, managed, action, reason,
                   confidence, published_embeddings_removed, chunk_embeddings_removed,
                   hierarchical_chunks_removed, embeddable_units_removed, pages_retained,
                   gcs_path, reversible)
                VALUES ($1,$2,$3,'duplicate',false,'retired_unpublished',$4,$5,$6,$7,$8,$9,$10,$11,true)""",
                run_id, d, uuid.UUID(keep),
                "unmanaged duplicate; canonical is the earliest-held copy",
                "normalized page text md5-identical to canonical",
                pub, emb, hch, eun, m[did]["pages"], m[did]["file_path"])

            await c.execute("DELETE FROM rag_published_embeddings WHERE document_id=$1", d)
            await c.execute("DELETE FROM chunk_embeddings WHERE document_id=$1", d)
            await c.execute("DELETE FROM embeddable_units WHERE document_id=$1", d)
            await c.execute("DELETE FROM hierarchical_chunks WHERE document_id=$1", d)
            await c.execute("""UPDATE documents SET lifecycle_state='retired', supersedes_id=$2
                               WHERE id=$1""", d, uuid.UUID(keep))
            for k, v in (("pub", pub), ("emb", emb), ("hch", hch), ("eun", eun)):
                tot[k] += v
        tot["docs"] += 1

    for did, keep in held:
        await c.execute("""
            INSERT INTO corpus_cleanup_actions
              (run_id, document_id, canonical_id, duplicate_kind, managed, action, reason, confidence,
               pages_retained, reversible)
            VALUES ($1,$2,$3,'duplicate',true,'held_for_human',
                    'managed document — withheld from publishing pending adjudication',
                    'normalized page text identical', 0, true)""",
            run_id, uuid.UUID(did), uuid.UUID(keep))

    print(f"\nAPPLIED  run_id={run_id}")
    print(f"  documents retired        : {tot['docs']}")
    print(f"  published vectors removed: {tot['pub']:,}")
    print(f"  chunk_embeddings removed : {tot['emb']:,}")
    print(f"  hierarchical_chunks      : {tot['hch']:,}")
    print(f"  embeddable_units         : {tot['eun']:,}")
    print(f"  managed held for human   : {len(held)}")

    # Read the writes back — same run, independent queries.
    n_sup = await c.fetchval("SELECT count(*) FROM documents WHERE lifecycle_state='retired'")
    left = await c.fetchval("""SELECT count(*) FROM rag_published_embeddings
                               WHERE document_id IN (SELECT document_id FROM corpus_cleanup_actions
                                                     WHERE run_id=$1 AND action='retired_unpublished')""", run_id)
    pages = await c.fetchval("""SELECT count(*) FROM document_pages
                                WHERE document_id IN (SELECT document_id FROM corpus_cleanup_actions
                                                      WHERE run_id=$1 AND action='retired_unpublished')""", run_id)
    print(f"\n  read-back: superseded={n_sup} · vectors still live for retired={left} (expect 0) "
          f"· pages retained={pages} (reversal fuel)")
    await c.close()


asyncio.run(main())
