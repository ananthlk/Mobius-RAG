"""Spine verification for eval/priors_store.py + priors_lab.cell_sha256.

Pure checks (no DB): the canonical-sha contract Eval-RAG ruled (2026-08-12).
DB checks: the three structural guards, run end-to-end inside ONE transaction
that is rolled back — so it exercises the REAL store code against the live
tables yet leaves no residue (critical: published_priors is append-only, cannot
be cleaned up after).

    .venv/bin/python -m eval.test_priors_store
"""
from __future__ import annotations

import asyncio

from app.services.retriever.priors_lab import PriorsLabRow, cell_sha256
from eval.db import close_pool, get_pool
from eval import priors_store as ps

RULER = "factcheck/gemini-2.5-pro"
FCV = "fact_check_v1.2026-07-31"
_fail = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        _fail.append(name)


def pure_sha_checks():
    print("[pure] cell_sha256 canonical contract")
    base = {"recall_lift": 0.6440, "accuracy_estimate": 0.4200, "authority": 0.9800, "n": 65, "k0": 9}
    s = cell_sha256(3, "a", base)

    check("deterministic across calls", s == cell_sha256(3, "a", dict(base)))
    check("differs on a changed appliable field",
          s != cell_sha256(3, "a", {**base, "recall_lift": 0.6441}))
    check("differs on depth_bucket", s != cell_sha256(4, "a", base))
    check("differs on strategy", s != cell_sha256(3, "b", base))

    # latency is OUT of the applied identity — extra key must not move the sha
    check("latency_p50_ms excluded from identity",
          s == cell_sha256(3, "a", {**base, "latency_p50_ms": 999}))

    # -0.0 normalizes to 0.0
    check("-0.0 authority == 0.0 authority",
          cell_sha256(3, "a", {**base, "authority": -0.0}) ==
          cell_sha256(3, "a", {**base, "authority": 0.0}))

    # None -> null token, no crash, distinct from 0
    none_k0 = cell_sha256(3, "a", {**base, "k0": None})
    check("k0=None hashes without crashing", isinstance(none_k0, str) and len(none_k0) == 64)
    check("k0=None != k0=0", none_k0 != cell_sha256(3, "a", {**base, "k0": 0}))

    # fixed-4dp: trailing-zero-losing repr must NOT change the hash
    # 0.42 and 0.4200 are the same value; both format to "0.4200"
    check("fixed-4dp: 0.42 == 0.4200",
          cell_sha256(3, "a", {**base, "accuracy_estimate": 0.42}) ==
          cell_sha256(3, "a", {**base, "accuracy_estimate": 0.4200}))


async def db_guard_checks():
    print("[db] structural guards (rolled-back txn — no residue)")
    pool = await get_pool()
    async with pool.acquire() as con:
        tr = con.transaction()
        await tr.start()
        try:
            # FK fail-closed: off-ruler is refused
            try:
                await ps.resolve_ruler_id("factcheck/gpt-4o", "whatever", conn=con)
                check("off-ruler refused", False)
            except ps.PriorsStoreError:
                check("off-ruler refused", True)

            # happy: valid ruler -> bank run
            run_id = await ps.start_bank_run(RULER, FCV, "cmhc_test", conn=con)
            await ps.persist_rows(run_id, [
                {"query_id": "q1", "caller_mode": "chat.default", "strategy": "a",
                 "recall": 0.8, "recall_answer": 0.4, "pool_size": 900, "capacity": 10, "fillers_ms": 46},
            ], conn=con)

            # fold a RECONCILING cell (recall_answer <= recall so clamp doesn't bite)
            rows = [
                PriorsLabRow("q1", "chat.default", "a", 900, 0.8, 0.4, 0.98, 10, 46),
                PriorsLabRow("q2", "chat.default", "a", 900, 0.6, 0.3, 0.98, 10, 50),
            ]
            cell_id, sha = await ps.fold_and_persist_cell(run_id, 3, "a", rows, conn=con)
            check("cell persisted with sha", isinstance(cell_id, int) and len(sha) == 64)

            # approved-sha guard: wrong approved sha refused
            try:
                await ps.publish_cell(cell_id, "deadbeef" * 8, "test", conn=con)
                check("approved-sha mismatch refused", False)
            except ps.PriorsStoreError:
                check("approved-sha mismatch refused", True)

            # happy publish with the correct sha
            pub_id = await ps.publish_cell(cell_id, sha, "test", conn=con)
            check("happy publish landed", isinstance(pub_id, int))
            prov = await con.fetchrow(
                "SELECT ruler, cell_sha256 FROM eval.eval_published_priors WHERE publish_id=$1", pub_id)
            check("published row carries locked-ruler provenance",
                  prov and prov["ruler"] == RULER and prov["cell_sha256"] == sha)

            # read-side fetch: the display contract, with currently_published joined
            cells = await ps.fetch_computed_cells(run_id, conn=con)
            mine = [c for c in cells if c["cell_id"] == cell_id]
            check("fetch returns the computed cell", len(mine) == 1)
            fc = mine[0]
            check("fetch sha matches", fc["sha256"] == sha)
            check("fetch appliable shape", set(fc["appliable"]) == {"recall_lift", "accuracy_estimate", "authority", "k0"})
            check("fetch bucket label", fc["bucket"] == "depth_3")
            check("fetch currently_published joined after publish",
                  fc["currently_published"] is not None and fc["currently_published"]["sha256"] == sha)

            # reconciliation HARD BLOCK: a non-reconciling computed cell cannot publish
            bad_id = await con.fetchval(
                "INSERT INTO eval.eval_computed_cells "
                "(bank_run_id,depth_bucket,strategy,recall_lift,accuracy_estimate,authority,n,k0,"
                " reconciliation_ok,population_rules_version,cell_sha256) "
                "VALUES ($1,3,'z',0.5,0.5,0.9,10,9,false,1,'x') RETURNING cell_id", run_id)
            try:
                await ps.publish_cell(bad_id, "x", "test", conn=con)
                check("reconciliation_ok=false blocked", False)
            except ps.PriorsStoreError as e:
                check("reconciliation_ok=false blocked", "HARD BLOCK" in str(e))
        finally:
            await tr.rollback()
            print("  (rolled back — no test rows persist)")


async def main():
    pure_sha_checks()
    await db_guard_checks()
    await close_pool()
    print(f"\n{'ALL PASS' if not _fail else 'FAILURES: ' + ', '.join(_fail)}")
    if _fail:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
