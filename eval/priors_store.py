"""Persist + publish Router priors through the ratified eval-workflow spine.

The compute LOGIC lives in app/services/retriever/priors_lab.py (compute_cell,
cell_sha256) and is NOT reimplemented here — this module only PERSISTS its
outputs onto the `eval` schema (Database-ratified 2026-08-12) and enforces the
three structural guards that make the provenance spine sound:

  1. FK FAIL-CLOSED on the ruler — a bank run graded by any ruler not on the
     eval_valid_rulers allowlist cannot be recorded (resolve_ruler_id raises;
     the bank_runs.ruler_id FK would reject it anyway). No off-ruler cell exists.
  2. RECONCILIATION HARD-BLOCK on publish — a cell whose recall_lift *
     accuracy_estimate does not reconcile to mean(answer_recall) (reconciliation_ok
     False) can never be published (§7).
  3. APPROVED-SHA TOCTOU GUARD on publish — publish takes the sha the reviewer
     signed off on, RECOMPUTES it from the stored appliable fields at write time,
     and refuses if recomputed != approved OR recomputed != stored. "Sign off on a
     sha, only that sha can land" — closes review->write drift the same way
     apply_cell_to_yaml_text's round-trip closes compute->file drift.

Phase 0/1: priors_bootstrap.yaml stays authoritative; this is compute + audit.
eval_published_priors is the append-only record of what landed in the file.
"""
from __future__ import annotations

import json
import uuid
from typing import Optional

from app.services.retriever.priors_lab import (
    PriorsApplyError,
    PriorsLabRow,
    _APPLIABLE_FIELDS,
    cell_sha256,
    compute_cell,
)
from eval.db import execute, fetchrow, get_pool

# The seeded ruleset the compute_cell logic implements (eval_population_rules row 1).
POPULATION_RULES_VERSION = 1


class _PoolDB:
    """Default executor: the retrying eval.db pool. Every store fn also accepts
    an explicit asyncpg Connection (same execute/fetchrow/executemany surface) so
    a caller — or a test — can run a whole bank-run + fold + publish inside ONE
    transaction (atomic, or rolled back to leave no residue against the
    append-only published_priors)."""

    execute = staticmethod(execute)
    fetchrow = staticmethod(fetchrow)

    @staticmethod
    async def fetch(sql, *args):
        pool = await get_pool()
        return await pool.fetch(sql, *args)

    @staticmethod
    async def executemany(sql, records):
        pool = await get_pool()
        async with pool.acquire() as con:
            await con.executemany(sql, records)


_POOL_DB = _PoolDB()


def _db(conn):
    return conn if conn is not None else _POOL_DB

# eval_bank_run_rows columns the persist path writes (superset of PriorsLabRow).
_ROW_COLS = (
    "query_id", "caller_mode", "strategy",
    "recall", "recall_answer", "authority",
    "n_contradicted", "n_hallucinated_claims", "n_facts_total",
    "top_score_percentile", "distinct_content_topk", "pool_size", "capacity", "fillers_ms",
)


class PriorsStoreError(Exception):
    pass


async def resolve_ruler_id(ruler: str, fact_checker_version: str, conn=None) -> int:
    """FK fail-closed: the locked-ruler allowlist gate. Raises if the pair is
    not on eval_valid_rulers — no bank run can be recorded off-ruler."""
    row = await _db(conn).fetchrow(
        "SELECT ruler_id FROM eval.eval_valid_rulers WHERE ruler=$1 AND fact_checker_version=$2",
        ruler, fact_checker_version,
    )
    if row is None:
        raise PriorsStoreError(
            f"ruler ({ruler!r}, {fact_checker_version!r}) not on eval_valid_rulers allowlist "
            f"— refusing to record an off-ruler bank run (locked-ruler parity)"
        )
    return row["ruler_id"]


async def start_bank_run(
    ruler: str, fact_checker_version: str, query_set: str,
    corpus_version: Optional[str] = None, conn=None,
) -> uuid.UUID:
    ruler_id = await resolve_ruler_id(ruler, fact_checker_version, conn=conn)
    run_id = uuid.uuid4()
    await _db(conn).execute(
        "INSERT INTO eval.eval_bank_runs (bank_run_id, ruler_id, corpus_version, query_set, status) "
        "VALUES ($1, $2, $3, $4, 'running')",
        run_id, ruler_id, corpus_version, query_set,
    )
    return run_id


async def finish_bank_run(bank_run_id: uuid.UUID, status: str = "done", conn=None) -> None:
    if status not in ("done", "failed"):
        raise PriorsStoreError(f"invalid terminal status {status!r}")
    await _db(conn).execute(
        "UPDATE eval.eval_bank_runs SET status=$2, finished_at=now() WHERE bank_run_id=$1",
        bank_run_id, status,
    )


async def persist_rows(bank_run_id: uuid.UUID, rows: list[dict], conn=None) -> int:
    """Bulk-insert per-(query,mode,strategy) observations. Idempotent on the
    natural key (bank_run_id, query_id, caller_mode, strategy)."""
    if not rows:
        return 0
    cols = ("bank_run_id",) + _ROW_COLS
    placeholders = ", ".join(f"${i+1}" for i in range(len(cols)))
    sql = (
        f"INSERT INTO eval.eval_bank_run_rows ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT (bank_run_id, query_id, caller_mode, strategy) DO NOTHING"
    )
    records = [tuple([bank_run_id] + [r.get(c) for c in _ROW_COLS]) for r in rows]
    await _db(conn).executemany(sql, records)
    return len(records)


def _cell_appliable(cell) -> dict:
    """The five appliable fields as a dict, from a PriorsCell (or a stored row)."""
    if hasattr(cell, "to_priors_dict"):
        d = cell.to_priors_dict()
    else:
        d = dict(cell)
    return {k: d.get(k) for k in _APPLIABLE_FIELDS}


async def fold_and_persist_cell(
    bank_run_id: uuid.UUID, depth_bucket: int, strategy: str,
    rows: list[PriorsLabRow], caller_mode: Optional[str] = None, conn=None,
) -> tuple[int, str]:
    """Fold one (depth_bucket, strategy[, caller_mode]) population into a cell
    via the ratified compute_cell, hash its applied identity, and persist to
    eval_computed_cells. Returns (cell_id, cell_sha256).

    depth_bucket is the INTEGER Router bucket (0..4) the cell applies to — the
    persisted identity, distinct from priors_lab's exploratory quantile labels."""
    cell = compute_cell(rows, bucket=str(depth_bucket), strategy=strategy, caller_mode=caller_mode)
    appliable = _cell_appliable(cell)
    sha = cell_sha256(depth_bucket, strategy, appliable)
    row = await _db(conn).fetchrow(
        "INSERT INTO eval.eval_computed_cells "
        "(bank_run_id, depth_bucket, strategy, caller_mode, recall_lift, accuracy_estimate, "
        " authority, authority_measured_at_bucket, n, authority_n, k0, latency_p50_ms, "
        " reconciliation_ok, population_rules_version, cell_sha256, warnings) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16::jsonb) "
        "RETURNING cell_id",
        bank_run_id, depth_bucket, strategy, caller_mode,
        cell.recall_lift, cell.accuracy_estimate, cell.authority, depth_bucket,
        cell.n, cell.authority_n, cell.k0, cell.latency_p50_ms,
        cell.reconciliation_ok, POPULATION_RULES_VERSION, sha, json.dumps(cell.warnings),
    )
    return row["cell_id"], sha


async def publish_cell(cell_id: int, approved_sha: str, published_by: str, conn=None) -> int:
    """Publish a computed cell to the append-only audit — the review->write gate.

    Enforces, in order: cell exists -> reconciliation_ok (HARD BLOCK) ->
    recompute sha from stored appliable fields -> refuse if recomputed != approved
    (reviewer signed a different value) -> refuse if recomputed != stored (tamper).
    Only then insert eval_published_priors with denormalized locked-ruler provenance.
    Returns publish_id."""
    db = _db(conn)
    c = await db.fetchrow(
        "SELECT c.cell_id, c.depth_bucket, c.strategy, c.recall_lift, c.accuracy_estimate, "
        "       c.authority, c.n, c.k0, c.cell_sha256, c.reconciliation_ok, "
        "       c.population_rules_version, v.ruler, v.fact_checker_version "
        "FROM eval.eval_computed_cells c "
        "JOIN eval.eval_bank_runs r ON r.bank_run_id = c.bank_run_id "
        "JOIN eval.eval_valid_rulers v ON v.ruler_id = r.ruler_id "
        "WHERE c.cell_id = $1",
        cell_id,
    )
    if c is None:
        raise PriorsStoreError(f"computed cell {cell_id} not found")
    if not c["reconciliation_ok"]:
        raise PriorsStoreError(
            f"cell {cell_id} has reconciliation_ok=false — HARD BLOCK, cannot publish "
            f"(recall_lift*accuracy_estimate does not reconcile to mean(answer_recall))"
        )
    appliable = {k: c[k] for k in _APPLIABLE_FIELDS}
    recomputed = cell_sha256(c["depth_bucket"], c["strategy"], appliable)
    if recomputed != approved_sha:
        raise PriorsStoreError(
            f"approved-sha guard: reviewer approved {approved_sha[:12]}… but the cell now "
            f"recomputes to {recomputed[:12]}… — refusing (review->write drift)"
        )
    if recomputed != c["cell_sha256"]:
        raise PriorsStoreError(
            f"tamper guard: stored cell_sha256 {c['cell_sha256'][:12]}… != recomputed "
            f"{recomputed[:12]}… — refusing"
        )
    row = await db.fetchrow(
        "INSERT INTO eval.eval_published_priors "
        "(depth_bucket, strategy, cell_id, cell_sha256, ruler, fact_checker_version, "
        " population_rules_version, n, published_by) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING publish_id",
        c["depth_bucket"], c["strategy"], cell_id, recomputed, c["ruler"],
        c["fact_checker_version"], c["population_rules_version"], c["n"], published_by,
    )
    return row["publish_id"]


async def fetch_computed_cells(bank_run_id: uuid.UUID, conn=None) -> list[dict]:
    """Read-side fetch backing Eval-RAG's §7 review gate (the display contract).

    One row per computed cell in the run, with `currently_published` for the
    same (depth_bucket, strategy) slot JOINED in (its sha + appliable values,
    resolved through the published row's source cell) so the gate can render the
    sha-diff without a second call. This endpoint hands RAW fields only — the
    approvability predicate (block/ack/comparability) is Eval-RAG's lane on top.

    Pending schema v1.1 (routed to Database, NOT applied unilaterally):
      - `reconciliation_delta` (the NUMBER abs(rl*ae − mean_answer_recall)) needs
        a stored column — returned None until it lands.
      - `superseded_by` / active-cell filter — returned as-is until the
        immutable-recompute model is ratified; today one cell per slot per run.
    """
    rows = await _db(conn).fetch(
        "SELECT c.cell_id, c.depth_bucket, c.strategy, c.caller_mode, c.cell_sha256, "
        "       c.reconciliation_ok, c.recall_lift, c.accuracy_estimate, c.authority, "
        "       c.k0, c.n, c.authority_n, c.latency_p50_ms, c.warnings, "
        "       c.population_rules_version, v.fact_checker_version, "
        "       pub.pub_sha, pub.pub_n, pub.pub_at, pub.pub_fcv, "
        "       pub.pub_rl, pub.pub_ae, pub.pub_auth, pub.pub_k0 "
        "FROM eval.eval_computed_cells c "
        "JOIN eval.eval_bank_runs br ON br.bank_run_id = c.bank_run_id "
        "JOIN eval.eval_valid_rulers v ON v.ruler_id = br.ruler_id "
        "LEFT JOIN LATERAL ("
        "    SELECT p.cell_sha256 AS pub_sha, p.n AS pub_n, p.published_at AS pub_at, "
        "           p.fact_checker_version AS pub_fcv, cc.recall_lift AS pub_rl, "
        "           cc.accuracy_estimate AS pub_ae, cc.authority AS pub_auth, cc.k0 AS pub_k0 "
        "    FROM eval.eval_published_priors p "
        "    JOIN eval.eval_computed_cells cc ON cc.cell_id = p.cell_id "
        "    WHERE p.depth_bucket = c.depth_bucket AND p.strategy = c.strategy "
        "    ORDER BY p.published_at DESC LIMIT 1"
        ") pub ON true "
        "WHERE c.bank_run_id = $1 "
        "ORDER BY c.depth_bucket, c.strategy, c.caller_mode NULLS FIRST",
        bank_run_id,
    )
    out = []
    for r in rows:
        warnings = r["warnings"]
        if isinstance(warnings, str):
            warnings = json.loads(warnings)
        currently_published = None
        if r["pub_sha"] is not None:
            currently_published = {
                "sha256": r["pub_sha"],
                "appliable": {"recall_lift": r["pub_rl"], "accuracy_estimate": r["pub_ae"],
                              "authority": r["pub_auth"], "k0": r["pub_k0"]},
                "n": r["pub_n"],
                "published_at": r["pub_at"].isoformat() if r["pub_at"] else None,
                "fact_checker_version": r["pub_fcv"],
            }
        out.append({
            "cell_id": r["cell_id"],
            "bucket": f"depth_{r['depth_bucket']}",
            "strategy": r["strategy"],
            "caller_mode": r["caller_mode"],
            "sha256": r["cell_sha256"],
            "reconciliation_ok": r["reconciliation_ok"],
            "reconciliation_delta": None,  # pending v1.1 column
            "n": r["n"],
            "authority_n": r["authority_n"],
            "appliable": {"recall_lift": r["recall_lift"], "accuracy_estimate": r["accuracy_estimate"],
                          "authority": r["authority"], "k0": r["k0"]},
            "latency_p50_ms": r["latency_p50_ms"],  # display-only, curated/out-of-sha
            "warnings": warnings or [],
            "fact_checker_version": r["fact_checker_version"],
            "population_rules_version": r["population_rules_version"],
            "currently_published": currently_published,
        })
    return out
