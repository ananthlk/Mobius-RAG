"""Eval + routing read endpoints — backs the EvalTab and RoutingTab in
the RAG frontend.

These are read-only views over ``rag_eval_runs`` / ``rag_eval_results``
/ ``rag_routing_decisions``. The agent persists into these tables on
every call; the frontend shows them. No new write paths.

Endpoints
---------

GET /api/eval/runs                 — list eval runs, newest first
GET /api/eval/runs/{run_id}        — run summary + per-query result list
GET /api/eval/results/{result_id}  — single query result drill-down
GET /api/routing/decisions         — live ticker of recent agent calls
GET /api/routing/decisions/{id}    — single routing-decision drill-down
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Body, HTTPException, Query
from sqlalchemy import text as sql_text

from app.database import get_db  # noqa: F401  (kept for future use)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["eval"])


# ---------------------------------------------------------------------------
# Eval runs
# ---------------------------------------------------------------------------

@router.get("/eval/runs")
async def list_eval_runs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List recent eval runs (newest first). Aggregate columns only —
    drill into a single run for per-query details.
    """
    from app.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        rows = await db.execute(sql_text(
            """
            SELECT id::text, ts, bank_path, bank_version, priors_version,
                   notes, n_queries, n_correct, n_partial, n_wrong, n_unable,
                   routing_accuracy, citation_hit_rate,
                   median_latency_ms, p95_latency_ms, completed_at
            FROM rag_eval_runs
            ORDER BY ts DESC
            LIMIT :lim OFFSET :off
            """
        ), {"lim": limit, "off": offset})
        out = [dict(r) for r in rows.mappings()]
    return {"runs": out, "n": len(out)}


@router.get("/eval/runs/{run_id}")
async def get_eval_run(run_id: str):
    """Run summary + per-query results list (no chunks_summary in list —
    fetch results/{id} for full drilldown)."""
    from app.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        run_row = (await db.execute(sql_text(
            """
            SELECT id::text, ts, bank_path, bank_version, priors_version,
                   caller_mode_filter, notes, config_dump,
                   n_queries, n_correct, n_partial, n_wrong, n_unable,
                   routing_accuracy, citation_hit_rate,
                   median_latency_ms, p95_latency_ms, completed_at
            FROM rag_eval_runs WHERE id::text = :id
            """
        ), {"id": run_id})).mappings().first()
        if not run_row:
            raise HTTPException(status_code=404, detail="run not found")

        results = await db.execute(sql_text(
            """
            SELECT id::text, ts, query_id, query, expected,
                   strategy_chosen, strategy_executed, confidence,
                   total_ms, n_chunks, top_rerank,
                   routing_correct, citation_hit, fail_fast_correct,
                   judge_verdict, judge_score, judge_reasoning,
                   judge_model, judge_ms,
                   human_verdict, human_reasoning, human_verdict_at, human_verdict_by,
                   COALESCE(human_verdict, judge_verdict) AS effective_verdict,
                   routing_decision_id::text
            FROM rag_eval_results
            WHERE run_id::text = :id
            ORDER BY query_id, ts
            """
        ), {"id": run_id})
        results_list = [dict(r) for r in results.mappings()]

    return {"run": dict(run_row), "results": results_list}


@router.get("/eval/results/{result_id}")
async def get_eval_result(result_id: str):
    """Single result with the full chunks_summary + llm_answer + linked
    routing_decision row (if any). This is the page that shows you
    exactly what the system retrieved and what the judge saw."""
    from app.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        row = (await db.execute(sql_text(
            """
            SELECT id::text, run_id::text, ts, query_id, query, expected,
                   strategy_chosen, strategy_executed, confidence,
                   total_ms, n_chunks, top_rerank, llm_answer, chunks_summary,
                   routing_correct, citation_hit, fail_fast_correct,
                   judge_verdict, judge_score, judge_reasoning,
                   judge_model, judge_ms,
                   human_verdict, human_reasoning, human_verdict_at, human_verdict_by,
                   COALESCE(human_verdict, judge_verdict) AS effective_verdict,
                   routing_decision_id::text,
                   full_response
            FROM rag_eval_results
            WHERE id::text = :id
            """
        ), {"id": result_id})).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="result not found")
        result = dict(row)

        # Pull the linked routing decision for full pipeline state.
        routing = None
        if result.get("routing_decision_id"):
            r2 = (await db.execute(sql_text(
                """
                SELECT id::text, ts, agent_id, query_class,
                       routing_method, scores, self_assessments, withdrawn,
                       strategy_chosen, strategy_executed, fallback_strategy,
                       priors_version, fail_fast_reason,
                       confidence, n_chunks, top_rerank, total_ms,
                       prefs_received, prefs_resolved,
                       per_strategy_telemetry
                FROM rag_routing_decisions WHERE id::text = :id
                """
            ), {"id": result["routing_decision_id"]})).mappings().first()
            if r2:
                routing = dict(r2)

    return {"result": result, "routing": routing}


# ---------------------------------------------------------------------------
# Routing decisions (live ticker — every agent call)
# ---------------------------------------------------------------------------

@router.get("/routing/decisions")
async def list_routing_decisions(
    limit: int = Query(50, ge=1, le=500),
    strategy: str | None = Query(None, description="Filter by strategy_executed"),
    caller_mode: str | None = Query(None),
    since_minutes: int | None = Query(None, description="Only show last N minutes"),
):
    """Recent agent calls. Each row is one call with its routing
    decision + immediate outcome. Frontend's RoutingTab renders this
    as a live ticker.
    """
    from app.database import AsyncSessionLocal
    where_clauses = []
    params: dict[str, Any] = {"lim": limit}
    if strategy:
        where_clauses.append("strategy_executed = :strategy")
        params["strategy"] = strategy
    if caller_mode:
        where_clauses.append("caller_mode = :caller_mode")
        params["caller_mode"] = caller_mode
    if since_minutes:
        where_clauses.append("ts > now() - (:mins || ' minutes')::interval")
        params["mins"] = str(since_minutes)
    where = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = (
        "SELECT id::text, ts, agent_id, query, query_class, "
        "       routing_method, caller_mode, "
        "       strategy_chosen, strategy_executed, fallback_strategy, "
        "       confidence, n_chunks, top_rerank, total_ms, "
        "       fail_fast_reason, scores "
        "FROM rag_routing_decisions" + where + " "
        "ORDER BY ts DESC LIMIT :lim"
    )
    async with AsyncSessionLocal() as db:
        rows = await db.execute(sql_text(sql), params)
        out = [dict(r) for r in rows.mappings()]
    return {"decisions": out, "n": len(out)}


@router.get("/routing/decisions/{decision_id}")
async def get_routing_decision(decision_id: str):
    """Full single-decision drilldown — every column. Use this to
    inspect ‘why did the router pick X for this query?'."""
    from app.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        row = (await db.execute(sql_text(
            "SELECT id::text, ts, agent_id, query, "
            "  query_type, query_class, coverage, has_d_tag, has_literal, "
            "  is_exploratory, tag_matches, literal_anchors, untagged_meaningful, "
            "  caller_mode, prefs_received, prefs_resolved, "
            "  routing_method, scores, self_assessments, withdrawn, "
            "  strategy_chosen, strategy_executed, fallback_strategy, "
            "  priors_version, fail_fast_reason, "
            "  confidence, n_chunks, top_rerank, total_ms, per_strategy_telemetry, "
            "  eval_run_id::text, routing_correct, citation_hit, keyword_hit, "
            "  llm_judge_verdict, llm_judge_score, llm_judge_reasoning, "
            "  user_feedback, critique_verdict "
            "FROM rag_routing_decisions WHERE id::text = :id"
        ), {"id": decision_id})).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="decision not found")
    return dict(row)


# ---------------------------------------------------------------------------
# Aggregate stats — for dashboard tiles
# ---------------------------------------------------------------------------

@router.get("/routing/stats")
async def routing_stats(
    since_hours: int = Query(24, ge=1, le=24 * 30),
):
    """Aggregate counts by strategy + qclass over the last N hours."""
    from app.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        # Per-strategy.
        s_rows = await db.execute(sql_text(
            "SELECT strategy_executed, COUNT(*) AS n, "
            "       AVG(total_ms) AS avg_ms, "
            "       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_ms) AS p50_ms, "
            "       PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_ms) AS p95_ms, "
            "       SUM(CASE WHEN confidence='high' THEN 1 ELSE 0 END) AS n_high, "
            "       SUM(CASE WHEN confidence='medium' THEN 1 ELSE 0 END) AS n_med, "
            "       SUM(CASE WHEN confidence='low' THEN 1 ELSE 0 END) AS n_low "
            "FROM rag_routing_decisions "
            "WHERE ts > now() - (:hours || ' hours')::interval "
            "GROUP BY strategy_executed "
            "ORDER BY n DESC"
        ), {"hours": str(since_hours)})
        per_strategy = [dict(r) for r in s_rows.mappings()]

        # Per-query-class.
        c_rows = await db.execute(sql_text(
            "SELECT query_class, COUNT(*) AS n, "
            "       SUM(CASE WHEN strategy_chosen='a' THEN 1 ELSE 0 END) AS n_a, "
            "       SUM(CASE WHEN strategy_chosen='b' THEN 1 ELSE 0 END) AS n_b, "
            "       SUM(CASE WHEN strategy_chosen='c' THEN 1 ELSE 0 END) AS n_c, "
            "       SUM(CASE WHEN strategy_chosen='d' THEN 1 ELSE 0 END) AS n_d, "
            "       SUM(CASE WHEN strategy_chosen='e' THEN 1 ELSE 0 END) AS n_e "
            "FROM rag_routing_decisions "
            "WHERE ts > now() - (:hours || ' hours')::interval "
            "GROUP BY query_class "
            "ORDER BY n DESC"
        ), {"hours": str(since_hours)})
        per_class = [dict(r) for r in c_rows.mappings()]

    return {
        "window_hours": since_hours,
        "per_strategy": per_strategy,
        "per_query_class": per_class,
    }


# ---------------------------------------------------------------------------
# Bank editor + trigger
# ---------------------------------------------------------------------------

# Repo-relative path. The agent runs from the repo root via uvicorn so
# the cwd is mobius-rag/. We resolve to absolute when reading/writing
# so the location is unambiguous.
_DEFAULT_BANK_REL = "eval/queries.yaml"


def _bank_path(rel: str | None = None) -> Path:
    rel = rel or _DEFAULT_BANK_REL
    # Lock down: only allow paths inside the repo's eval/ dir.
    p = Path(rel).resolve()
    repo = Path(__file__).resolve().parent.parent.parent
    eval_dir = (repo / "eval").resolve()
    try:
        p.relative_to(eval_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"bank path must live under {eval_dir}")
    return p


@router.get("/eval/bank")
async def get_eval_bank(path: str | None = Query(None)):
    """Read the YAML bank as JSON. If ``path`` omitted, defaults to
    ``eval/queries.yaml``.
    """
    rel = path or _DEFAULT_BANK_REL
    bank_path = _bank_path(rel if Path(rel).is_absolute() else None) if Path(rel).is_absolute() else (
        Path(__file__).resolve().parent.parent.parent / rel
    )
    if not bank_path.exists():
        raise HTTPException(status_code=404, detail=f"bank not found: {bank_path}")
    raw = bank_path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=500, detail=f"yaml parse error: {exc}")
    queries = data.get("queries") or []
    return {
        "path": str(bank_path),
        "rel_path": rel,
        "n_queries": len(queries),
        "queries": queries,
        "raw_yaml": raw,
    }


@router.put("/eval/bank")
async def put_eval_bank(body: dict = Body(...)):
    """Write a new bank file from a JSON ``{queries: [...]}`` payload.

    Validates that every query has at minimum ``id`` and ``query``
    fields; expected dict is optional. Backs up the previous version
    to ``queries.yaml.bak`` so a bad save can be recovered.
    """
    rel = body.get("rel_path") or _DEFAULT_BANK_REL
    queries = body.get("queries")
    if not isinstance(queries, list):
        raise HTTPException(status_code=400, detail="body.queries must be a list")

    # Validate.
    seen_ids: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    for i, q in enumerate(queries):
        if not isinstance(q, dict):
            raise HTTPException(status_code=400, detail=f"query[{i}] not a dict")
        qid = (q.get("id") or "").strip()
        text = (q.get("query") or "").strip()
        if not qid:
            raise HTTPException(status_code=400, detail=f"query[{i}] missing id")
        if not text:
            raise HTTPException(status_code=400, detail=f"query[{i}] ({qid}) missing query text")
        if qid in seen_ids:
            raise HTTPException(status_code=400, detail=f"duplicate id: {qid}")
        seen_ids.add(qid)
        # Drop empty/None fields for cleaner YAML.
        cleaned.append({k: v for k, v in q.items() if v not in (None, "", [], {})})

    bank_path = Path(__file__).resolve().parent.parent.parent / rel
    bank_path.parent.mkdir(parents=True, exist_ok=True)

    # Backup previous.
    if bank_path.exists():
        try:
            bank_path.with_suffix(bank_path.suffix + ".bak").write_text(
                bank_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("bank backup failed: %s", exc)

    # Dump with stable ordering.
    out = yaml.safe_dump(
        {"queries": cleaned},
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=120,
    )
    bank_path.write_text(out, encoding="utf-8")
    return {
        "path": str(bank_path),
        "rel_path": rel,
        "n_queries": len(cleaned),
        "ok": True,
    }


# Tracks the currently-running eval (in-process). One concurrent run
# at a time keeps the LLM judge and DB load predictable.
_RUN_LOCK = asyncio.Lock()
_LATEST_RUN_ID: dict[str, str | None] = {"id": None}


async def _run_eval_task(rel_bank: str, notes: str, endpoint: str):
    """Background task: import + invoke the eval driver."""
    try:
        import sys
        from pathlib import Path as _P
        repo = _P(__file__).resolve().parent.parent.parent
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from eval.run import run_eval

        async with _RUN_LOCK:
            run_id = await run_eval(
                bank_path=repo / rel_bank,
                endpoint=endpoint,
                notes=notes,
            )
            _LATEST_RUN_ID["id"] = run_id
            logger.info("[eval-trigger] run finished: %s", run_id)
    except Exception as exc:
        logger.exception("[eval-trigger] run failed: %s", exc)


async def _run_calibration_task(rel_bank: str, notes: str, endpoint: str):
    """Background task: run the calibration driver against the given bank.

    Calibration runs ~22 × 4 = 88 forced-strategy passes. Each result
    writes to ``rag_eval_results`` immediately (incremental persistence)
    so the EvalTab can stream rows live.
    """
    try:
        import sys
        from pathlib import Path as _P
        repo = _P(__file__).resolve().parent.parent.parent
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from eval.calibrate import run_calibration, load_bank
        import hashlib

        async with _RUN_LOCK:
            bank = load_bank(repo / rel_bank)
            bank_sha = hashlib.sha256(
                (repo / rel_bank).read_text(encoding="utf-8").encode()
            ).hexdigest()[:12]
            run_id, _rows, _cells = await run_calibration(
                bank, endpoint,
                notes=notes,
                bank_path=rel_bank,
                bank_version=bank_sha,
            )
            _LATEST_RUN_ID["id"] = run_id
            logger.info("[calibrate-trigger] finished: %s", run_id)
    except Exception as exc:
        logger.exception("[calibrate-trigger] failed: %s", exc)


@router.post("/eval/calibrate/trigger")
async def trigger_calibration(body: dict = Body(default={})):
    """Kick off the full bank × all-strategies calibration.

    88 forced-strategy passes against the agent endpoint, each judged
    by the LLM, persisted live to ``rag_eval_results`` so the EvalTab
    streams progress. Returns immediately with status; client polls
    ``/api/eval/active`` and ``/api/eval/runs/{id}/progress`` for live
    updates and per-query rows.
    """
    if _RUN_LOCK.locked():
        raise HTTPException(status_code=409, detail="another eval/calibration is already running")
    rel_bank = body.get("bank_path") or _DEFAULT_BANK_REL
    notes = body.get("notes") or "from-ui"
    endpoint = (
        body.get("endpoint")
        or os.environ.get("EVAL_TARGET_ENDPOINT")
        or "http://localhost:8001/api/skills/v1/corpus_search_agent"
    )
    asyncio.create_task(_run_calibration_task(rel_bank, notes, endpoint))
    return {
        "status": "started",
        "kind": "calibration",
        "bank_path": rel_bank,
        "notes": notes,
        "endpoint": endpoint,
    }


@router.post("/eval/trigger")
async def trigger_eval(body: dict = Body(default={})):
    """Kick off ``eval/run.py`` as a background task. Returns immediately
    with a status sentinel; client polls ``/api/eval/runs`` (newest row
    is the active run) and ``/api/eval/runs/{id}/progress`` for the
    live result count.
    """
    if _RUN_LOCK.locked():
        raise HTTPException(status_code=409, detail="another eval is already running")
    rel_bank = body.get("bank_path") or _DEFAULT_BANK_REL
    notes = body.get("notes") or "from-ui"
    # Default endpoint = same host. Allow override for cross-host runs.
    endpoint = (
        body.get("endpoint")
        or os.environ.get("EVAL_TARGET_ENDPOINT")
        or "http://localhost:8001/api/skills/v1/corpus_search_agent"
    )
    asyncio.create_task(_run_eval_task(rel_bank, notes, endpoint))
    return {
        "status": "started",
        "bank_path": rel_bank,
        "notes": notes,
        "endpoint": endpoint,
    }


@router.get("/eval/runs/{run_id}/progress")
async def get_run_progress(run_id: str):
    """Live counter — how many results have been written for this run.

    Lets the frontend show ``X / N`` while a run is in flight without
    re-fetching the full results list every poll.
    """
    from app.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        row = (await db.execute(sql_text(
            """
            SELECT
              n_queries,
              completed_at,
              (SELECT COUNT(*) FROM rag_eval_results
                 WHERE run_id::text = :id) AS n_results
            FROM rag_eval_runs WHERE id::text = :id
            """
        ), {"id": run_id})).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="run not found")
    return {
        "run_id": run_id,
        "n_queries": row["n_queries"],
        "n_completed": row["n_results"],
        "is_running": row["completed_at"] is None,
        "completed_at": row["completed_at"],
    }


# ---------------------------------------------------------------------------
# Human override — single endpoint that updates one result's verdict +
# rolls the parent run's aggregates so the dashboards stay consistent.
# ---------------------------------------------------------------------------

_VALID_VERDICTS = {"correct", "partial", "wrong", "unable_to_verify"}


@router.patch("/eval/results/{result_id}/verdict")
async def patch_eval_result_verdict(
    result_id: str,
    body: dict = Body(...),
):
    """Override the LLM judge's verdict on one result.

    Body: ``{verdict: 'correct'|'partial'|'wrong'|'unable_to_verify',
    reasoning?, by?}``. Pass ``verdict: null`` to clear an override
    and revert to the judge's call. After updating, the parent run's
    aggregates are recomputed using the COALESCE(human, judge) rule
    so the runs-list and per-run summary reflect the human verdict.
    """
    verdict = body.get("verdict")
    if verdict is not None and verdict not in _VALID_VERDICTS:
        raise HTTPException(
            status_code=400,
            detail=f"verdict must be one of {sorted(_VALID_VERDICTS)} or null",
        )
    reasoning = body.get("reasoning")
    by = body.get("by")

    from app.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        # 1. Update the row.
        if verdict is None:
            # Clear the override.
            res = await db.execute(sql_text(
                """
                UPDATE rag_eval_results
                SET human_verdict = NULL,
                    human_reasoning = NULL,
                    human_verdict_at = NULL,
                    human_verdict_by = NULL
                WHERE id::text = :id
                RETURNING run_id::text AS run_id
                """
            ), {"id": result_id})
        else:
            res = await db.execute(sql_text(
                """
                UPDATE rag_eval_results
                SET human_verdict     = :v,
                    human_reasoning   = :r,
                    human_verdict_at  = now(),
                    human_verdict_by  = :b
                WHERE id::text = :id
                RETURNING run_id::text AS run_id
                """
            ), {"id": result_id, "v": verdict, "r": reasoning, "b": by})
        row = res.mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="result not found")
        run_id = row["run_id"]

        # 2. Recompute the parent run's aggregates using effective verdict.
        await _recompute_run_aggregates(db, run_id)
        await db.commit()

    return {"ok": True, "result_id": result_id, "run_id": run_id, "verdict": verdict}


async def _recompute_run_aggregates(db, run_id: str) -> None:
    """Roll up per-result effective verdicts into the run row."""
    await db.execute(sql_text(
        """
        UPDATE rag_eval_runs r
        SET
          n_correct = sub.n_correct,
          n_partial = sub.n_partial,
          n_wrong   = sub.n_wrong,
          n_unable  = sub.n_unable,
          routing_accuracy = sub.routing_acc,
          citation_hit_rate = sub.cite_rate
        FROM (
          SELECT
            run_id,
            COUNT(*) FILTER (WHERE COALESCE(human_verdict, judge_verdict) = 'correct')          AS n_correct,
            COUNT(*) FILTER (WHERE COALESCE(human_verdict, judge_verdict) = 'partial')          AS n_partial,
            COUNT(*) FILTER (WHERE COALESCE(human_verdict, judge_verdict) = 'wrong')            AS n_wrong,
            COUNT(*) FILTER (WHERE COALESCE(human_verdict, judge_verdict) = 'unable_to_verify') AS n_unable,
            (COUNT(*) FILTER (WHERE routing_correct = true)::float
              / NULLIF(COUNT(*) FILTER (WHERE routing_correct IS NOT NULL), 0)) AS routing_acc,
            (COUNT(*) FILTER (WHERE citation_hit = true)::float
              / NULLIF(COUNT(*) FILTER (WHERE citation_hit IS NOT NULL), 0))  AS cite_rate
          FROM rag_eval_results
          WHERE run_id::text = :id
          GROUP BY run_id
        ) sub
        WHERE r.id::text = :id AND sub.run_id = r.id
        """
    ), {"id": run_id})


@router.get("/eval/calibration/status")
async def get_calibration_status(
    log_path: str = Query("/tmp/calibration_full.log"),
    tail_lines: int = Query(20, ge=1, le=200),
):
    """Read the calibration log file and return parsed progress.

    The standalone calibration script (``python -m eval.calibrate``)
    writes JSONL-style INFO lines like:
        [INFO] [q005/c] qclass=literal_anchor answered=False verdict=unable_to_verify(0.50) 17710ms

    Until calibrate.py is refactored to write to ``rag_eval_runs``,
    this endpoint is the live-progress feed for the EvalTab.
    """
    p = Path(log_path)
    if not p.exists():
        return {
            "running": False, "log_present": False,
            "n_completed": 0, "n_total": 88, "tail": [],
        }
    raw = p.read_text(encoding="utf-8", errors="replace")
    lines = raw.splitlines()

    # Pull "[qid/strategy] ... verdict=X" events.
    import re as _re
    pat = _re.compile(
        r"\[(q\d+)/([a-d])\]\s+qclass=(\w+)\s+answered=(\w+)\s+verdict=(\w+)\(([\d.]+)\)\s+(\d+)ms"
    )
    events: list[dict[str, Any]] = []
    for ln in lines:
        m = pat.search(ln)
        if m:
            events.append({
                "qid": m.group(1),
                "strategy": m.group(2),
                "qclass": m.group(3),
                "answered": m.group(4) == "True",
                "verdict": m.group(5),
                "score": float(m.group(6)),
                "total_ms": int(m.group(7)),
            })

    # Tally
    tally = {"correct": 0, "partial": 0, "wrong": 0, "unable_to_verify": 0}
    for e in events:
        tally[e["verdict"]] = tally.get(e["verdict"], 0) + 1

    # Total + N from the first INFO line ("Calibrating N queries × M strategies = K runs")
    total_match = _re.search(r"Calibrating (\d+) queries × (\d+) strategies = (\d+) runs", raw)
    n_total = int(total_match.group(3)) if total_match else 88

    # Determine running: log was written to recently AND no "Wrote raw"
    # finalisation line yet.
    finished = "Wrote raw + aggregates" in raw or "EMPIRICAL vs CURRENT" in raw
    import os as _os
    mtime = p.stat().st_mtime
    seconds_since_write = max(0, int(time.time() - mtime))
    running = (not finished) and seconds_since_write < 120  # 2-min idle = stuck

    return {
        "running": running,
        "finished": finished,
        "log_present": True,
        "log_path": str(p),
        "seconds_since_write": seconds_since_write,
        "n_completed": len(events),
        "n_total": n_total,
        "tally": tally,
        "events": events,            # all 88 events
        "tail": lines[-tail_lines:], # last N raw log lines
    }


@router.get("/eval/active")
async def get_active_run():
    """Returns the in-flight run (if any) so the frontend can resume
    polling after a refresh."""
    from app.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        # Only consider runs from the last 30 minutes — older "in
        # flight" rows are abandoned (server crash / redeploy).
        row = (await db.execute(sql_text(
            "SELECT id::text AS id, ts, notes, n_queries, "
            "       (SELECT COUNT(*) FROM rag_eval_results "
            "          WHERE run_id = rag_eval_runs.id) AS n_completed "
            "FROM rag_eval_runs "
            "WHERE completed_at IS NULL "
            "  AND ts > now() - interval '30 minutes' "
            "ORDER BY ts DESC LIMIT 1"
        ))).mappings().first()
    if not row:
        return {"active": False}
    return {
        "active": True,
        "run_id": row["id"],
        "ts": row["ts"],
        "notes": row["notes"],
        "n_queries": row["n_queries"],
        "n_completed": row["n_completed"],
    }


# ---------------------------------------------------------------------------
# PR-curve endpoint
# ---------------------------------------------------------------------------
#
# Sweeps a confidence threshold τ over [0, 1] and at each step computes
# precision and recall PER STRATEGY for a given run. The "answered" set
# at threshold τ is rows where the chosen confidence axis ≥ τ AND the
# strategy didn't abstain (n_chunks > 0 OR strategy_executed != 'e').
#
# Effective verdict prefers the human verdict if set (manual override),
# else the LLM judge verdict. partial = 0.5 credit on precision.
#
# Confidence axis options:
#   * top_rerank      — the single highest-scoring chunk's rerank score
#   * mean_top3       — mean of the top-3 chunks (smoother, less noisy)
#   * confidence_tier — agent's own self-rated label, mapped low/med/high
#                       to 0.33/0.66/1.00 for sweep-comparability
#
# Returns: { axis: str, points: { strategy: [ { tau, p, r, n_answered, n_correct, n_wrong, n_partial, n_total } ] } }

@router.get("/eval/runs/{run_id}/pr_curve")
async def pr_curve(
    run_id: str,
    axis: str = Query(
        "top_rerank",
        pattern="^(top_rerank|mean_top3|confidence_tier)$",
        description="Which confidence signal to threshold on",
    ),
    n_steps: int = Query(21, ge=5, le=101,
                         description="Number of τ samples in [0,1]"),
):
    """Per-strategy PR curve for a completed run."""
    from app.database import AsyncSessionLocal

    # Map confidence_tier label → numeric for thresholding.
    tier_to_score = {"low": 0.33, "medium": 0.66, "high": 1.0}

    async with AsyncSessionLocal() as db:
        rows = await db.execute(sql_text(
            """
            SELECT
                query_id,
                strategy_executed,
                strategy_chosen,
                top_rerank,
                mean_top3_rerank,
                confidence,
                n_chunks,
                judge_verdict,
                human_verdict,
                expected->>'strategy' AS expected_strategy,
                full_response->'routing'->>'strategy' AS routing_strategy
            FROM rag_eval_results
            WHERE run_id = :run_id
            """
        ), {"run_id": run_id})
        results = list(rows.mappings().all())

    if not results:
        raise HTTPException(404, f"no results for run {run_id}")

    # Group by strategy_chosen (the FORCED strategy in calibration);
    # strategy_executed may differ if the router fell back, but for the
    # PR curve we want "what does THIS strategy produce when forced".
    # Outside calibration, strategy_chosen == strategy_executed always.
    by_strategy: dict[str, list[dict]] = {}
    for r in results:
        s = r["strategy_chosen"] or r["strategy_executed"] or "?"
        by_strategy.setdefault(s, []).append(dict(r))

    # ── Synthetic "system" strategies ────────────────────────────────
    #
    # These represent the END-TO-END pipeline: parser → router → strategy.
    # On forced calibration each query has a row per strategy (a/b/c/d);
    # the system curve picks one row per unique base query (the leading
    # ``q###`` of ``query_id``) according to the routing rule, then sweeps
    # on that subset. This is what we'd actually ship to users — the
    # "best of all worlds" curve when the router does its job right.
    #
    #  * system_oracle  — picks the cell where strategy_chosen ==
    #                     expected.strategy (perfect routing; ceiling)
    #  * system_router  — picks the cell that the live router would have
    #                     fired (current routing logic in production).
    #                     For the calibration override case the router's
    #                     "would have chosen" lives in routing_strategy
    #                     when no override fired. For now we approximate
    #                     by also using expected_strategy when the router
    #                     was bypassed; once we have a non-forced eval
    #                     run this becomes accurate.

    def _base_qid(qid: str) -> str:
        # ``q001/a`` → ``q001``
        return (qid or "").split("/", 1)[0] or qid

    by_base_q: dict[str, list[dict]] = {}
    for r in results:
        by_base_q.setdefault(_base_qid(r["query_id"]), []).append(dict(r))

    oracle_rows: list[dict] = []
    for base, rows_for_q in by_base_q.items():
        expected = (rows_for_q[0].get("expected_strategy") or "").strip()
        if not expected:
            continue
        # Find the row where strategy_chosen matches expected. If the
        # expected strategy is 'e' (fail-fast), any of the four rows
        # works — they all carry the same query — so pick the first.
        if expected == "e":
            oracle_rows.append(rows_for_q[0])
            continue
        match = next((r for r in rows_for_q
                      if r.get("strategy_chosen") == expected), None)
        if match is not None:
            oracle_rows.append(match)
    if oracle_rows:
        by_strategy["system_oracle"] = oracle_rows

    # system_best — per-query, pick whichever strategy actually got the
    # best verdict (correct > partial > unable > wrong). This is the
    # absolute ceiling: what a perfect router with hindsight could
    # achieve. The gap between system_best and system_oracle quantifies
    # how much the bank's a-priori `expected.strategy` labels under-count
    # what the strategies can do; the gap between system_best and any
    # real router is the room a smarter router has to improve.
    verdict_rank = {
        "correct": 3, "partial": 2, "unable_to_verify": 1, "wrong": 0,
    }

    def _row_score(row: dict) -> tuple[int, float]:
        v = (row.get("human_verdict") or row.get("judge_verdict")
             or "unable_to_verify")
        # Tie-break by top_rerank so cells with the same verdict but
        # higher confidence sort first.
        return (verdict_rank.get(v, 0), float(row.get("top_rerank") or 0.0))

    best_rows: list[dict] = []
    for base, rows_for_q in by_base_q.items():
        if not rows_for_q:
            continue
        best = max(rows_for_q, key=_row_score)
        best_rows.append(best)
    if best_rows:
        by_strategy["system_best"] = best_rows

    def _confidence(row: dict) -> float | None:
        if axis == "top_rerank":
            return row.get("top_rerank")
        if axis == "mean_top3":
            return row.get("mean_top3_rerank")
        if axis == "confidence_tier":
            label = (row.get("confidence") or "").lower().strip()
            return tier_to_score.get(label)
        return None

    def _verdict(row: dict) -> str:
        # Manual override wins. Falls back to LLM judge.
        return (row.get("human_verdict") or row.get("judge_verdict")
                or "unable_to_verify")

    def _is_pure_skip(row: dict) -> bool:
        # A "pure skip" is an abstain whose effective verdict is still
        # ``unable_to_verify`` — i.e. the strategy gave up AND we don't
        # know if that was right or wrong. These get excluded from both
        # numerator and answered counts because they're an evaluation
        # gap, not a strategy outcome.
        if (row.get("n_chunks") or 0) > 0:
            return False
        return _verdict(row) == "unable_to_verify"

    taus = [i / (n_steps - 1) for i in range(n_steps)]

    points: dict[str, list[dict]] = {}
    for strategy, rs in by_strategy.items():
        n_total = len(rs)
        strategy_points: list[dict] = []
        for tau in taus:
            n_answered = 0
            n_correct = 0
            n_partial = 0
            n_wrong = 0
            for row in rs:
                if _is_pure_skip(row):
                    # Excluded from both P and R — an unscored cell.
                    continue
                conf = _confidence(row)
                # Abstain rows with a definitive verdict (correct=matched
                # expected fail-fast, wrong=false negative) carry no
                # confidence signal. Treat them as conf=0 so they only
                # count at τ=0 — at higher thresholds they correctly
                # drop out (a strategy that abstains can't be "high
                # confidence about its abstain" — the user is asking
                # "how confident are answered chunks", and there are no
                # chunks).
                if conf is None:
                    conf = 0.0
                if conf < tau:
                    continue
                n_answered += 1
                v = _verdict(row)
                if v == "correct":
                    n_correct += 1
                elif v == "partial":
                    n_partial += 1
                elif v == "wrong":
                    n_wrong += 1
                # unable_to_verify: counted in n_answered but not in any
                # verdict bucket — visible in the panel but doesn't move
                # precision either way.

            # Precision: correct + 0.5*partial out of (correct + partial + wrong).
            # We exclude "unable_to_verify" from the precision denominator
            # because it's an evaluation gap, not a strategy fault.
            decided = n_correct + n_partial + n_wrong
            precision = ((n_correct + 0.5 * n_partial) / decided) if decided else None

            # Recall: correct + 0.5*partial out of total bank size for
            # this strategy. Includes abstains and unable_to_verify in
            # the denominator — abstaining is a recall hit.
            recall = (n_correct + 0.5 * n_partial) / n_total if n_total else None

            strategy_points.append({
                "tau": round(tau, 4),
                "precision": round(precision, 4) if precision is not None else None,
                "recall": round(recall, 4) if recall is not None else None,
                "n_answered": n_answered,
                "n_correct": n_correct,
                "n_partial": n_partial,
                "n_wrong": n_wrong,
                "n_total": n_total,
            })
        points[strategy] = strategy_points

    return {
        "run_id": run_id,
        "axis": axis,
        "n_steps": n_steps,
        "n_strategies": len(by_strategy),
        "n_total_per_strategy": {s: len(rs) for s, rs in by_strategy.items()},
        "points": points,
    }

