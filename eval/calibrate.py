"""Empirical prior calibration — run every query through every strategy,
score the results, and output recommended router priors.

The router's `_BASE_PRIORS` table currently holds hand-set values for
each (strategy, query_class) cell. This script measures the real
behavior of each strategy on the labeled query bank and proposes
empirically-grounded replacements.

Per (query, strategy) we capture:

  * judge verdict (correct / partial / wrong / unable_to_verify)
  * total latency (ms)
  * whether the strategy effectively "answered" — chunks returned or
    LLM answer non-empty (vs. fail-fast / withdrew / empty)

Per (strategy, query_class) we then aggregate:

  * **accuracy** = correct / (correct + wrong)
      — precision among answered queries. Withdrawals don't count
      against accuracy ("can't go negative").
  * **recall_capacity** = answered / total
      — what fraction of queries this strategy could attempt.
  * **mean_ms** → normalised to a speed prior in [0..1]

Usage:
    python -m eval.calibrate \\
        --bank eval/queries.yaml \\
        --endpoint http://localhost:8001/api/skills/v1/corpus_search_agent

Output:
    eval/calibration/<ts>.json — per-cell raw counts + aggregates
    stdout — table of recommended priors with deltas vs current

Note: forcing a strategy via the agent's `mode` override BYPASSES the
router's score function (no withdrawal). That's intentional — we want
to see how the strategy actually performs WHEN forced to run, not how
its self-assessment tells it to back off.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any
import urllib.request

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import yaml  # noqa: E402

from eval.judge import adjudicate  # noqa: E402
from eval.db import close_pool, execute as db_execute, fetchrow as db_fetchrow  # noqa: E402
from eval.run import insert_run, insert_result, finalize_run, _strip_nulls  # noqa: E402
from app.services.fact_checker import check_facts  # noqa: E402  — retrieval critic (chunk-only recall)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("calibrate")


STRATEGIES = ["a", "b", "c", "d", "natural"]   # a-d forced; "natural" = router decides.
# The 4 forced paths give per-strategy priors + the derived oracle (per-query max);
# "natural" is the router's ACTUAL pick — measured alongside so a run shows
# router-vs-oracle-vs-strategy in one place. "e" is a gate, not a competing path.


# ---------------------------------------------------------------------------
# Bank loading
# ---------------------------------------------------------------------------

def load_bank(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    queries = data.get("queries") or []
    return [q for q in queries if q.get("query")]


# ---------------------------------------------------------------------------
# Agent call (forced strategy)
# ---------------------------------------------------------------------------

# Transient, connection-level errors worth a retry — NOT application-level
# failures (a real 4xx/5xx with a JSON error body is left alone, since
# retrying would mask an actual bug). 2026-07-07: calibration runs share
# the single pinned mobius-rag instance (min=max=1, deliberately — see
# feedback_rag_api_max1_inprocess_state) with concurrent EvalTab dashboard
# polling, and that contention can drop or stall an in-flight connection.
# That isn't a signal about the strategy being tested — retrying recovers
# the real data point instead of silently losing it as an error.
_TRANSIENT_ERROR_NAMES = (
    "RemoteDisconnected", "ConnectionResetError", "ConnectionAbortedError",
    "ConnectionRefusedError", "IncompleteRead",
    # httpx's own exception names (see call_agent's docstring for why
    # httpx, not urllib, is what actually raises these now):
    "RemoteProtocolError", "ConnectError", "ReadError", "WriteError",
    "ReadTimeout", "ConnectTimeout", "PoolTimeout",
)
_AGENT_CALL_RETRIES = 2
_AGENT_CALL_RETRY_BACKOFF_S = 2.0
_AGENT_CALL_PER_ATTEMPT_TIMEOUT_S = 40.0


async def call_agent(
    endpoint: str,
    query: str,
    forced_strategy: str,
    *,
    caller_mode: str | None = None,
    timeout: float = _AGENT_CALL_PER_ATTEMPT_TIMEOUT_S,
) -> dict[str, Any]:
    """POST with mode=<strategy> override. Returns the response dict.

    Uses httpx.AsyncClient — natively async, unlike the urllib-in-a-thread
    version this replaced (2026-07-07). That distinction matters a lot
    here: run_calibration wraps this call in
    ``asyncio.wait_for(..., timeout=agent_timeout_s)``, and when THAT
    outer timeout fires on a urllib+asyncio.to_thread call, the blocked OS
    thread underneath keeps running regardless — asyncio has no way to
    kill it. Over a long run those orphaned threads accumulate, and
    eventually the process's own DB connection pool destabilizes (this
    was caught live: a "did not finish joining its threads within 300
    seconds" warning immediately preceded a DB connection failure that
    killed two sigma-baseline runs at 70-95% complete). httpx's async
    requests are genuinely cancellable, so an outer timeout actually
    tears down the in-flight request — no orphans possible.

    Retries a small, fixed number of times on transient connection-level
    errors (see _TRANSIENT_ERROR_NAMES) with a short linear backoff —
    everything else (a real timeout, an application error response) is
    returned as-is on the first attempt. The per-attempt timeout is kept
    well under run_calibration's usual 90s outer budget so at least one
    retry fits inside it.
    """
    import httpx

    body: dict[str, Any] = {
        "query": query,
        "k": 5,
        "skip_synthesis": True,         # retrieval-only — we score the CHUNKS, not the synthesized answer
    }
    if forced_strategy != "natural":
        body["mode"] = forced_strategy  # force the strategy; "natural" → router decides
    if caller_mode:
        body["caller_mode"] = caller_mode

    result: dict[str, Any] = {"error": "unreachable"}
    for attempt in range(_AGENT_CALL_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(endpoint, json=body)
                resp.raise_for_status()
                result = resp.json()
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}

        err = result.get("error", "")
        is_transient = any(name in err for name in _TRANSIENT_ERROR_NAMES)
        if not is_transient or attempt == _AGENT_CALL_RETRIES:
            return result
        logger.warning(
            "transient agent-call error (attempt %d/%d), retrying in %.0fs: %s",
            attempt + 1, _AGENT_CALL_RETRIES, _AGENT_CALL_RETRY_BACKOFF_S, err,
        )
        await asyncio.sleep(_AGENT_CALL_RETRY_BACKOFF_S)
    return result


def _answered(response: dict[str, Any]) -> bool:
    """Did the strategy effectively produce an answer (vs. empty / fail-fast)?

    A response counts as "answered" if it returned chunks OR an LLM
    answer (c/d). Fail-fast (e) and 0-chunks count as not answered.
    """
    if response.get("strategy_used") == "e":
        return False
    if (response.get("chunks") or []):
        return True
    if response.get("llm_answer"):
        return True
    return False


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def aggregate(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Group raw rows by (strategy, query_class) and compute the priors."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r["strategy"], r["query_class"])
        grouped[key].append(r)

    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for (strategy, qclass), items in grouped.items():
        n_total = len(items)
        n_correct = sum(1 for it in items if it["verdict"] == "correct")
        n_partial = sum(1 for it in items if it["verdict"] == "partial")
        n_wrong = sum(1 for it in items if it["verdict"] == "wrong")
        n_unable = sum(1 for it in items if it["verdict"] == "unable_to_verify")
        n_answered = sum(1 for it in items if it["answered"])

        # Accuracy: precision among answered queries.
        attempted_with_judgement = n_correct + n_wrong
        accuracy = (n_correct / attempted_with_judgement) if attempted_with_judgement > 0 else None
        # Soft accuracy: include partials at half weight.
        soft_accuracy = (
            (n_correct + 0.5 * n_partial) / max(1, n_correct + n_partial + n_wrong)
            if (n_correct + n_partial + n_wrong) > 0 else None
        )
        # Recall capacity: what fraction of queries did the strategy actually attempt
        recall_capacity = n_answered / n_total if n_total else 0.0
        # Mean latency
        latencies = [it["total_ms"] for it in items if it.get("total_ms")]
        mean_ms = sum(latencies) / len(latencies) if latencies else None

        cells[(strategy, qclass)] = {
            "n_total": n_total,
            "n_correct": n_correct,
            "n_partial": n_partial,
            "n_wrong": n_wrong,
            "n_unable": n_unable,
            "n_answered": n_answered,
            "accuracy": round(accuracy, 3) if accuracy is not None else None,
            "soft_accuracy": round(soft_accuracy, 3) if soft_accuracy is not None else None,
            "recall_capacity": round(recall_capacity, 3),
            "mean_ms": int(mean_ms) if mean_ms else None,
        }
    return cells


def speed_prior(mean_ms: int | None, all_means: list[int]) -> float:
    """Normalise mean latency into a [0..1] speed prior.

    Reference: 1s → 1.00, 30s → 0.10. Linear interpolation in log-space.
    """
    if not mean_ms:
        return 0.5
    import math
    fastest = max(500, min(all_means)) if all_means else 1000
    slowest = max(all_means) if all_means else 30000
    if fastest >= slowest:
        return 0.5
    log_ms = math.log(max(mean_ms, fastest))
    log_min = math.log(fastest)
    log_max = math.log(slowest)
    # Faster = higher score
    return round(1.0 - (log_ms - log_min) / (log_max - log_min) * 0.9 - 0.05, 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_JUDGE_MODEL_LOCKED = "gemini-2.5-pro"  # locked adjudicator stage rag_eval_adjudicate


async def _capture_fingerprint(endpoint: str, bank_version: str) -> dict[str, Any]:
    """Capture the run's version fingerprint from AUTHORITATIVE sources — the
    provenance foundation for eval observability. Invocation-agnostic: every
    caller of run_calibration inherits this, so a result can't be stored
    without its fingerprint. Agent identity comes from the DEPLOYED agent's
    /version (NOT the local driver — the mixed-build guard); corpus/lexicon
    state from the DB; judge + bank from config.
    """
    fp: dict[str, Any] = {"judge_model": _JUDGE_MODEL_LOCKED, "bank_hash": bank_version}
    base = endpoint.split("/api/")[0]
    try:
        def _get() -> dict:
            req = urllib.request.Request(base + "/version")
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        v = await asyncio.to_thread(_get)
        fp["agent_revision"] = v.get("revision")
        fp["agent_git_sha"] = v.get("git_sha")
        fp["priors_version"] = v.get("priors_version")
        fp["retrieval_config_hash"] = v.get("retrieval_config_hash")
    except Exception as exc:                       # /version unreachable → record the miss
        fp["agent_version_error"] = f"{type(exc).__name__}: {exc}"
    try:
        row = await db_fetchrow(
            "SELECT max(lexicon_revision) lex, max(tagged_at) snap FROM document_tags"
        )
        fp["lexicon_revision"] = row["lex"]
        fp["corpus_snapshot_at"] = row["snap"].isoformat() if row["snap"] else None
    except Exception as exc:
        fp["corpus_state_error"] = f"{type(exc).__name__}: {exc}"
    return fp


async def _finalize_fingerprint_stable(run_id: str) -> None:
    """Set fingerprint.stable — did every cell run on ONE build? Reads each
    cell's response priors_version; >1 distinct ⇒ the run straddled a deploy
    (mixed-build confound) ⇒ flag it so the dashboard excludes it from
    attribution. Turns the trap that cost us hours into a data flag."""
    try:
        row = await db_fetchrow(
            """
            SELECT count(DISTINCT full_response->'routing'->>'priors_version') n,
                   array_agg(DISTINCT full_response->'routing'->>'priors_version')
                     FILTER (WHERE full_response->'routing'->>'priors_version' IS NOT NULL) vs
            FROM rag_eval_results WHERE run_id::text = $1
            """, run_id)
        stable = (row["n"] or 0) <= 1
        await db_execute(
            """
            UPDATE rag_eval_runs SET config_dump = jsonb_set(
                jsonb_set(coalesce(config_dump, '{}'::jsonb),
                          '{fingerprint,stable}', $2::jsonb),
                '{fingerprint,observed_priors_versions}', $3::jsonb)
            WHERE id::text = $1
            """, run_id, json.dumps(stable), json.dumps(row["vs"] or []))
    except Exception as exc:
        logger.warning("fingerprint_stable finalize failed for %s: %s", run_id, exc)


async def run_calibration(
    bank: list[dict[str, Any]],
    endpoint: str,
    *,
    skip_judge: bool = False,
    judge_timeout_s: float = 30.0,
    agent_timeout_s: float = 90.0,
    notes: str = "calibration",
    bank_path: str = "eval/queries.yaml",
    bank_version: str = "?",
    priors_version: str = "v1.2026-05-01",
) -> tuple[str, list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Run all (query × strategy) combos and PERSIST each result to
    ``rag_eval_results`` as it lands.

    Returns ``(run_id, raw_rows, cells)``. Bulletproofed:
      * ``asyncio.wait_for`` per agent + judge call (hung LLM bounded)
      * each row written immediately (frontend can stream live)
      * resume-on-restart via ``rag_eval_results`` rowmap
    """
    # 1. Open one eval_run for the full calibration. Notes carry
    # "calibration-…" so the EvalTab can distinguish; the 88 results
    # will share this run_id.
    n_total_planned = len(bank) * len(STRATEGIES)
    # Provenance fingerprint — captured HERE (in the eval agent), so every
    # invocation path (button / driver / nightly / cron) stamps it identically.
    fingerprint = await _capture_fingerprint(endpoint, bank_version)
    # Prefer the DEPLOYED agent's priors_version over the caller-supplied one.
    priors_version = fingerprint.get("priors_version") or priors_version
    run_id = await insert_run(
        bank_path=bank_path,
        bank_version=bank_version,
        priors_version=priors_version,
        notes=f"calibration · {notes}",
        config_dump={
            "endpoint": endpoint,
            "skip_judge": skip_judge,
            "judge_timeout_s": judge_timeout_s,
            "agent_timeout_s": agent_timeout_s,
            "n_strategies_per_query": len(STRATEGIES),
            "fingerprint": fingerprint,
        },
        n_queries=n_total_planned,
    )
    logger.info("Calibration run_id=%s (writes to rag_eval_results live)", run_id)

    raw_rows: list[dict[str, Any]] = []

    for q in bank:
        qid = q["id"]
        text = q["query"]
        caller_mode = q.get("caller_mode")
        # Merge top-level rubric fields into ``expected`` so the judge
        # sees them via the same kwarg. This keeps the call sites
        # backward-compatible with the old ``answer_keywords``-based
        # bank format. ``golden_answer`` presence is the dispatch flag
        # inside the judge to choose rubric vs keyword scoring.
        expected = dict(q.get("expected") or {})
        for rubric_key in (
            "golden_answer", "must_facts", "bonus_facts",
            "forbidden_facts", "golden_citation", "persona",
        ):
            if rubric_key in q and rubric_key not in expected:
                expected[rubric_key] = q[rubric_key]
        for strategy in STRATEGIES:
            t0 = time.monotonic()
            try:
                response = await asyncio.wait_for(
                    call_agent(endpoint, text, strategy, caller_mode=caller_mode),
                    timeout=agent_timeout_s,
                )
            except asyncio.TimeoutError:
                response = {"error": f"agent_timeout after {agent_timeout_s}s"}
            except Exception as exc:
                response = {"error": f"agent_exception: {exc}"}
            elapsed_ms = int((time.monotonic() - t0) * 1000)

            if "error" in response:
                logger.warning("[%s/%s] agent error: %s", qid, strategy, response["error"])
                row = {
                    "run_id": run_id,
                    "query_id": f"{qid}/{strategy}",      # disambiguate per-strategy
                    "query": text,
                    "expected": expected,
                    "strategy_chosen": strategy,
                    "strategy_executed": strategy,
                    "confidence": "low",
                    "total_ms": elapsed_ms,
                    "n_chunks": 0,
                    "top_rerank": None,
                    "llm_answer": None,
                    "chunks_summary": [],
                    "routing_correct": None,
                    "citation_hit": None,
                    "fail_fast_correct": None,
                    "judge_verdict": "unable_to_verify",
                    "judge_score": 0.0,
                    "judge_reasoning": f"agent_error: {response['error']}",
                    "judge_model": "agent_error",
                    "judge_ms": 0,
                    "full_response": None,
                }
                await insert_result(row)
                raw_rows.append({
                    "qid": qid, "strategy": strategy, "query_class": "?",
                    "answered": False, "verdict": "unable_to_verify",
                    "score": 0.0, "total_ms": elapsed_ms,
                })
                continue

            qclass = (response.get("routing") or {}).get("query_class") or "unknown"
            answered = _answered(response)

            # Deterministic verdict for empty / abstain responses BEFORE
            # falling through to the LLM judge. Two cases:
            #   1. Strategy abstained AND expected.strategy == 'e'   → CORRECT
            #      (the bank explicitly says "you should refuse to answer";
            #       an empty response is exactly the right thing.)
            #   2. Strategy abstained AND expected expected an answer → WRONG
            #      (false negative — the strategy gave up on something it
            #       should've solved.)
            # This was previously stamped 'unable_to_verify' which conflated
            # two very different failure modes and hid both lift signals
            # behind the abstain veil. Fixing here removes the post-hoc
            # backfill the analyst was doing manually.
            expected_strategy = (expected or {}).get("strategy")
            must_facts = (expected or {}).get("must_facts") or []
            forbidden = (expected or {}).get("forbidden_facts") or []
            chunks = response.get("chunks") or []
            recall_val: float | None = None
            precision_val: float | None = None
            n_cited: int | None = None
            n_contra = 0
            if skip_judge:
                verdict, score, reasoning, model, judge_ms = (
                    "unable_to_verify", 0.5, "skipped", "skipped", 0)
            elif not answered:
                if expected_strategy == "e":
                    verdict, score, reasoning, model = (
                        "correct", 1.0,
                        "abstain matched expected fail-fast",
                        "deterministic/abstain")
                else:
                    # Silent abstain on an answerable query. NOTE: whether
                    # this is an honest abstain (nothing retrievable) or a
                    # failure-to-answer is resolved downstream by joining
                    # recall — recall≈0 here (no chunks) means honest.
                    verdict, score, reasoning, model = (
                        "false_negative", 0.10,
                        "false negative — silently abstained on answerable query",
                        "deterministic/abstain")
                judge_ms = 0
                recall_val = 0.0
            elif must_facts:
                # RETRIEVAL scoring: are the golden facts present in the CHUNKS?
                # (chunk-only fact_checker — answer-independent, so a strategy
                # isn't penalised for a synthesizer that failed to use a chunk.)
                try:
                    fc = await asyncio.wait_for(
                        check_facts(query=text, must_facts=must_facts,
                                    chunks=chunks, forbidden_facts=forbidden),
                        timeout=judge_timeout_s)
                    recall_val = fc.score
                    n_contra = sum(1 for v in fc.verdicts if v.contradicted)
                    # Precision = distinct chunks that were RELEVANT (cited as
                    # supporting a golden fact). Indexed to the THEORETICAL MAX
                    # so it reads on a true [0,1] scale: a query with n_facts
                    # golden facts can cite at most min(n_chunks, n_facts) chunks
                    # (raw cited/n_chunks caps at n_facts/n_chunks ≈ 0.4-0.6 even
                    # for perfect retrieval, which silently discounts gains).
                    # theoretical_max = min(n_chunks, n_facts)/n_chunks →
                    # precision_score = cited / min(n_chunks, n_facts), clamped 1.
                    cited = {v.passage for v in fc.verdicts if v.support > 0 and v.passage}
                    n_cited = len(cited)
                    denom = min(len(chunks), len(must_facts)) or 1
                    precision_val = round(min(1.0, n_cited / denom), 3)
                    score = recall_val
                    verdict = ("correct" if recall_val >= 0.67
                               else "partial" if recall_val >= 0.34 else "wrong")
                    reasoning = fc.reasoning
                    model = fc.model
                    judge_ms = fc.elapsed_ms
                except asyncio.TimeoutError:
                    verdict, score, reasoning, model, judge_ms = (
                        "unable_to_verify", 0.5,
                        f"factcheck_timeout after {judge_timeout_s}s",
                        "timeout", int(judge_timeout_s * 1000))
                    logger.warning("[%s/%s] fact_checker timed out", qid, strategy)
                except Exception as exc:
                    verdict, score, reasoning, model, judge_ms = (
                        "unable_to_verify", 0.5, f"factcheck_error: {exc}", "error", 0)
            else:
                # Legacy bank (no must_facts) → answer-quality judge fallback.
                try:
                    verdict, score, reasoning, model, judge_ms = await asyncio.wait_for(
                        adjudicate(text, expected, response),
                        timeout=judge_timeout_s)
                except asyncio.TimeoutError:
                    verdict, score, reasoning, model, judge_ms = (
                        "unable_to_verify", 0.5,
                        f"judge_timeout after {judge_timeout_s}s",
                        "timeout", int(judge_timeout_s * 1000))
                except Exception as exc:
                    verdict, score, reasoning, model, judge_ms = (
                        "unable_to_verify", 0.5, f"judge_error: {exc}", "error", 0)

            total_ms = (response.get("telemetry") or {}).get("total_ms") or elapsed_ms

            # Calibration metrics for the summary endpoint (5-axis aggregation).
            response["_calibration"] = {
                "recall": recall_val,               # facts present in chunks (None if not scored)
                "precision": precision_val,          # indexed: cited / min(n_chunks, n_facts) ∈ [0,1]
                "n_cited": (n_cited if must_facts and recall_val is not None else None),  # raw components
                "n_facts": (len(must_facts) or None),          # for transparency + post-hoc recompute
                "n_contradicted": n_contra,          # chunks asserting a wrong fact (retrieval error)
                "answered": answered,                # attempted vs abstained (answer-rate axis)
                "strategy": strategy,
                "query_class": qclass,
                "latency_ms": total_ms,
                "n_chunks": len(chunks),
            }

            chunks_summary = []
            for c in (response.get("chunks") or [])[:5]:
                chunks_summary.append({
                    "document_name": c.get("document_name"),
                    "page_number": c.get("page_number"),
                    "rerank_score": c.get("rerank_score"),
                    "text": (c.get("text") or "")[:300],
                })

            chunk_scores = sorted(
                ((c.get("rerank_score") or 0.0)
                 for c in (response.get("chunks") or [])),
                reverse=True,
            )
            top_rerank = chunk_scores[0] if chunk_scores else None
            # Mean of the top-3 rerank scores. Smoother confidence axis than
            # top_rerank alone — a single anomalous high-scoring chunk can't
            # carry a strategy across the threshold.
            top3 = chunk_scores[:3]
            mean_top3_rerank = (sum(top3) / len(top3)) if top3 else None

            row = _strip_nulls({
                "run_id": run_id,
                "query_id": f"{qid}/{strategy}",
                "query": text,
                "expected": expected,
                "strategy_chosen": strategy,
                "strategy_executed": response.get("strategy_used") or strategy,
                "confidence": response.get("confidence"),
                "total_ms": total_ms,
                "n_chunks": len(response.get("chunks") or []),
                "top_rerank": top_rerank,
                "mean_top3_rerank": mean_top3_rerank,
                "llm_answer": response.get("llm_answer"),
                "chunks_summary": chunks_summary,
                "routing_correct": None,    # not meaningful for forced runs
                "citation_hit": None,
                "fail_fast_correct": None,
                "judge_verdict": verdict,
                "judge_score": score,
                "judge_reasoning": reasoning,
                "judge_model": model,
                "judge_ms": judge_ms,
                "full_response": response,
            })
            await insert_result(row)

            raw_rows.append({
                "qid": qid, "strategy": strategy,
                "query_class": qclass,
                "answered": answered,
                "verdict": verdict, "score": score,
                "total_ms": total_ms,
            })
            logger.info(
                "[%s/%s] qclass=%s answered=%s verdict=%s(%.2f) %dms",
                qid, strategy, qclass, answered, verdict, score, total_ms,
            )

    # Roll up aggregates (n_correct/wrong/etc.) so the run row reflects reality.
    summary_rows = [
        {"judge_verdict": r["verdict"], "routing_correct": None,
         "citation_hit": None, "total_ms": r.get("total_ms")}
        for r in raw_rows
    ]
    await finalize_run(run_id, summary_rows)
    # Mixed-build guard: flag the run confounded if cells straddled a deploy.
    await _finalize_fingerprint_stable(run_id)

    cells = aggregate(raw_rows)
    return run_id, raw_rows, cells


def render_priors_diff(cells: dict[tuple[str, str], dict[str, Any]]) -> str:
    """Compare empirical aggregates to current router priors and print a
    side-by-side table of recommended updates."""
    from app.services.corpus_search_router import _BASE_PRIORS, PRIORS_VERSION

    lines: list[str] = []
    lines.append(f"\nEMPIRICAL vs CURRENT priors (version: {PRIORS_VERSION})")
    lines.append("=" * 110)
    header = f"{'strat':<5} {'qclass':<18} {'n':<4} {'cor/par/wrong/unable/unanswered':<26} "
    header += f"{'accuracy':<14} {'recall_cap':<14} {'speed':<14}"
    lines.append(header)
    lines.append("-" * 110)

    all_means = [c["mean_ms"] for c in cells.values() if c.get("mean_ms")]

    for sid in STRATEGIES:
        for qclass in ("literal_anchor", "tight_pool", "wide_pool", "exploratory", "vague", "conceptual"):
            cell = cells.get((sid, qclass))
            if cell is None or cell["n_total"] == 0:
                continue
            try:
                cur = _BASE_PRIORS[sid][qclass]
                cur_acc = cur.accuracy
                cur_rec = cur.recall_capacity
                cur_speed = cur.speed
            except KeyError:
                cur_acc = cur_rec = cur_speed = None

            emp_acc = cell["accuracy"]
            emp_rec = cell["recall_capacity"]
            emp_speed = speed_prior(cell.get("mean_ms"), all_means)

            distrib = (
                f"{cell['n_correct']}/{cell['n_partial']}/{cell['n_wrong']}/"
                f"{cell['n_unable']}/{cell['n_total'] - cell['n_answered']}"
            )

            def _fmt(emp, cur):
                if emp is None:
                    return "—"
                if cur is None:
                    return f"{emp:.2f}"
                delta = emp - cur
                arrow = "↑" if delta > 0.02 else ("↓" if delta < -0.02 else "·")
                return f"{emp:.2f} (cur {cur:.2f}) {arrow}"

            lines.append(
                f"{sid:<5} {qclass:<18} {cell['n_total']:<4} {distrib:<26} "
                f"{_fmt(emp_acc, cur_acc):<14} {_fmt(emp_rec, cur_rec):<14} "
                f"{_fmt(emp_speed, cur_speed):<14}"
            )
    return "\n".join(lines)


async def main_async(args):
    bank = load_bank(Path(args.bank))
    if args.limit:
        bank = bank[: args.limit]
    n_runs = len(bank) * len(STRATEGIES)
    logger.info(
        "Calibrating %d queries × %d strategies = %d runs",
        len(bank), len(STRATEGIES), n_runs,
    )

    # Bank-version hash for the run row (so we know which question
    # bank a calibration was run against).
    import hashlib
    bank_raw = Path(args.bank).read_text(encoding="utf-8")
    bank_sha = hashlib.sha256(bank_raw.encode()).hexdigest()[:12]

    try:
        run_id, raw_rows, cells = await run_calibration(
            bank, args.endpoint,
            skip_judge=args.skip_judge,
            judge_timeout_s=args.judge_timeout,
            agent_timeout_s=args.agent_timeout,
            notes=args.notes,
            bank_path=args.bank,
            bank_version=bank_sha,
        )
    finally:
        # CLI-only: close before asyncio.run() tears the loop down. The
        # in-process eval router keeps its (long-lived loop's) pool open.
        await close_pool()

    print(render_priors_diff(cells))
    print(f"\nrun_id: {run_id}")
    print(f"View live in EvalTab — runs labelled 'calibration · {args.notes}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default="eval/queries.yaml")
    parser.add_argument("--endpoint", default="http://localhost:8001/api/skills/v1/corpus_search_agent")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N queries (debug).")
    parser.add_argument("--judge-timeout", type=float, default=30.0,
                        help="Per-judge-call timeout in seconds.")
    parser.add_argument("--agent-timeout", type=float, default=90.0,
                        help="Per-agent-call timeout in seconds.")
    global STRATEGIES
    parser.add_argument("--notes", default="from-cli",
                        help="Notes string for the run row (shown in EvalTab).")
    parser.add_argument("--strategies", default=",".join(STRATEGIES),
                        help="Comma-separated subset of strategies to run "
                             "(default: a,b,c,d). E.g. 'a,c,d' to skip (b).")
    args = parser.parse_args()
    # Override the module-level STRATEGIES with the caller's subset so
    # both run_calibration and any aggregation downstream use the same
    # list. We only allow values that are already in the canonical set —
    # typos shouldn't silently route to garbage.
    chosen = [s.strip().lower() for s in (args.strategies or "").split(",") if s.strip()]
    invalid = [s for s in chosen if s not in STRATEGIES]
    if invalid:
        parser.error(f"unknown strategies: {invalid}; allowed: {STRATEGIES}")
    if chosen:
        STRATEGIES = chosen
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
