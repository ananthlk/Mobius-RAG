"""Eval harness driver — runs a labeled query bank against the
``corpus_search_agent`` endpoint, scores responses with a Thompson-
bandit-routed LLM judge, and persists everything to ``rag_eval_runs``
+ ``rag_eval_results``.

Inspired by ``mobius-qa/retrieval-eval/retrieval_eval.py`` (the older
Vertex-direct driver) but rewired to talk to OUR agent endpoint so we
exercise the full router → strategy → response pipeline. The old
driver and its companion FastAPI studio (``retrieval-eval-studio``)
are now archived; this is the system of record.

Usage:

    python -m eval.run \\
        --bank eval/queries.yaml \\
        --endpoint http://localhost:8001/api/skills/v1/corpus_search_agent \\
        --notes "baseline priors v1.2026-05-01 + tier-cooc"

Env vars:
    DATABASE_URL      Postgres URL for persistence
    CHAT_INTERNAL_LLM_URL  / VERTEX_PROJECT_ID  for the judge
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4

# Add project root to path so we can import app.* modules.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import yaml

from eval.db import close_pool, execute as db_execute, fetchrow as db_fetchrow
from eval.judge import adjudicate


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eval")


# ---------------------------------------------------------------------------
# Bank loading + checks
# ---------------------------------------------------------------------------

def load_bank(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load queries.yaml. Returns (queries, content_sha for versioning)."""
    raw = path.read_text(encoding="utf-8")
    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    data = yaml.safe_load(raw) or {}
    queries = data.get("queries") or []
    if not isinstance(queries, list):
        raise ValueError("queries.yaml must contain top-level 'queries' list")
    # Normalize: the cmhc rubric bank stores golden_answer / must_facts /
    # bonus_facts / forbidden_facts (and keyword-bank fields) as SIBLINGS of
    # ``expected``, but the judge + deterministic_checks read them from INSIDE
    # ``expected``. Without this fold the judge never sees golden_answer and
    # silently runs in keyword mode — the whole rubric (must/bonus/forbidden)
    # scoring path is dead. Fold siblings in without clobbering existing keys.
    _EXPECTED_KEYS = (
        "golden_answer", "must_facts", "bonus_facts", "forbidden_facts",
        "golden_citation", "answer_keywords", "must_cite_doc",
        "must_cite_url_contains", "fail_fast_reason", "notes",
    )
    for q in queries:
        if not isinstance(q, dict):
            continue
        exp = dict(q.get("expected") or {})
        for k in _EXPECTED_KEYS:
            if k in q and k not in exp:
                exp[k] = q[k]
        q["expected"] = exp
    return queries, sha


def deterministic_checks(
    expected: dict[str, Any],
    response: dict[str, Any],
) -> tuple[bool, bool, bool]:
    """Cheap, non-LLM scoring. Returns:
        (routing_correct, citation_hit, fail_fast_correct)
    """
    exp_strategy = (expected.get("strategy") or "").lower()
    exec_strategy = (response.get("strategy_used") or "").lower()
    routing_correct = (exp_strategy == exec_strategy) if exp_strategy else None

    # Fail-fast correctness (only meaningful when expected==e)
    if exp_strategy == "e":
        ff = response.get("fail_fast") or {}
        exp_reason = expected.get("fail_fast_reason")
        if exp_reason:
            fail_fast_correct = (ff.get("reason") == exp_reason)
        else:
            fail_fast_correct = bool(ff)
    else:
        fail_fast_correct = None

    # Citation hit — does any chunk's document_name / URL match expected?
    citation_hit = None
    must_cite_doc = expected.get("must_cite_doc") or []
    must_cite_url = expected.get("must_cite_url_contains") or []
    if must_cite_doc or must_cite_url:
        chunks = response.get("chunks") or []
        validated = response.get("validated_citations") or []
        names = {(c.get("document_name") or "").lower() for c in chunks}
        names |= {
            ((v.get("document_display_name")
              or v.get("document_filename")
              or "") or (v.get("candidate") or {}).get("document_title") or "").lower()
            for v in validated
        }
        urls = []
        for v in validated:
            u = v.get("discovered_source_url") or (v.get("candidate") or {}).get("url")
            if u:
                urls.append(u.lower())

        doc_hit = any(
            (d or "").lower() in n or (n in (d or "").lower())
            for d in must_cite_doc for n in names if n
        ) if must_cite_doc else False
        url_hit = any(
            substr.lower() in u
            for substr in must_cite_url for u in urls
        ) if must_cite_url else False

        citation_hit = bool(doc_hit or url_hit)

    return routing_correct, citation_hit, fail_fast_correct


# ---------------------------------------------------------------------------
# Calling the agent
# ---------------------------------------------------------------------------

async def call_agent(
    endpoint: str,
    query: str,
    caller_mode: str | None,
    *,
    eval_run_id: str | None = None,
    must_facts: list[str] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """POST to the agent endpoint, return response dict."""
    body: dict[str, Any] = {"query": query, "k": 5}
    if caller_mode:
        body["caller_mode"] = caller_mode
    if eval_run_id:
        body["eval_run_id"] = eval_run_id
    if must_facts:
        body["eval_must_facts"] = must_facts

    def _do() -> dict[str, Any]:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    return await asyncio.to_thread(_do)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

# All writes go through the shared pool in eval/db.py (one pool per
# process, acquire per statement). 2026-07-07: the previous open-a-fresh-
# asyncpg-connection-per-call pattern (~111 connects per calibration run)
# both lost runs to single transient blips (two sigma-baseline runs got
# 0/110 and 32/110 rows) and fed the local cloud-sql-proxy's degraded
# hang-on-handshake state. The pool replaces the interim per-call retry
# patch that lived here.


async def insert_run(
    *,
    bank_path: str,
    bank_version: str,
    priors_version: str,
    notes: str,
    config_dump: dict[str, Any],
    n_queries: int,
) -> str:
    # Client-generated id + ON CONFLICT DO NOTHING makes the INSERT safe to
    # retry (a lost ack on a committed row is a no-op, not a duplicate run).
    run_id = str(uuid4())
    await db_execute(
        """
        INSERT INTO rag_eval_runs (
            id, bank_path, bank_version, priors_version, caller_mode_filter,
            notes, config_dump, n_queries
        ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
        ON CONFLICT (id) DO NOTHING
        """,
        run_id, bank_path, bank_version, priors_version, "all",
        notes, json.dumps(config_dump), n_queries,
    )
    return run_id


def _strip_nulls(obj):
    """Recursively strip NUL bytes (\\u0000) from strings — Postgres
    JSONB rejects them. Some PDF-extracted chunk text contains them."""
    if isinstance(obj, str):
        return obj.replace("\x00", "")
    if isinstance(obj, list):
        return [_strip_nulls(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _strip_nulls(v) for k, v in obj.items()}
    return obj


async def insert_result(row: dict[str, Any]) -> None:
    row = _strip_nulls(row)
    full_response = row.get("full_response")
    if full_response is None:
        full_response_json = None
    else:
        full_response_json = json.dumps(full_response)
    await db_execute(
        """
        INSERT INTO rag_eval_results (
            id, run_id, query_id, query, expected,
            strategy_chosen, strategy_executed, confidence,
            total_ms, n_chunks, top_rerank, mean_top3_rerank,
            llm_answer, chunks_summary,
            routing_correct, citation_hit, fail_fast_correct,
            judge_verdict, judge_score, judge_reasoning,
            judge_model, judge_ms, full_response
        ) VALUES (
            $1, $2, $3, $4, $5::jsonb,
            $6, $7, $8,
            $9, $10, $11, $12,
            $13, $14::jsonb,
            $15, $16, $17,
            $18, $19, $20,
            $21, $22, $23::jsonb
        )
        ON CONFLICT (id) DO NOTHING
        """,
        str(uuid4()),
        row["run_id"], row["query_id"], row["query"], json.dumps(row["expected"]),
        row.get("strategy_chosen"), row.get("strategy_executed"),
        row.get("confidence"),
        row.get("total_ms"), row.get("n_chunks"), row.get("top_rerank"),
        row.get("mean_top3_rerank"),
        row.get("llm_answer"), json.dumps(row.get("chunks_summary") or []),
        row.get("routing_correct"), row.get("citation_hit"),
        row.get("fail_fast_correct"),
        row.get("judge_verdict"), row.get("judge_score"), row.get("judge_reasoning"),
        row.get("judge_model"), row.get("judge_ms"),
        full_response_json,
    )


async def finalize_run(run_id: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics and update the run row."""
    if not results:
        return {}
    n = len(results)
    n_correct = sum(1 for r in results if r.get("judge_verdict") == "correct")
    n_partial = sum(1 for r in results if r.get("judge_verdict") == "partial")
    n_wrong = sum(1 for r in results if r.get("judge_verdict") == "wrong")
    n_unable = sum(1 for r in results if r.get("judge_verdict") == "unable_to_verify")

    routing_judgements = [r["routing_correct"] for r in results
                          if r.get("routing_correct") is not None]
    routing_acc = (sum(1 for x in routing_judgements if x) / len(routing_judgements)
                   if routing_judgements else None)
    citation_judgements = [r["citation_hit"] for r in results
                           if r.get("citation_hit") is not None]
    citation_rate = (sum(1 for x in citation_judgements if x) / len(citation_judgements)
                     if citation_judgements else None)

    latencies = [r.get("total_ms") or 0 for r in results]
    median_ms = int(statistics.median(latencies)) if latencies else None
    p95 = int(sorted(latencies)[int(len(latencies) * 0.95)]) if latencies else None

    summary = {
        "n_queries": n,
        "n_correct": n_correct,
        "n_partial": n_partial,
        "n_wrong": n_wrong,
        "n_unable": n_unable,
        "routing_accuracy": routing_acc,
        "citation_hit_rate": citation_rate,
        "median_latency_ms": median_ms,
        "p95_latency_ms": p95,
    }

    await db_execute(
        """
        UPDATE rag_eval_runs SET
          n_correct = $2, n_partial = $3, n_wrong = $4, n_unable = $5,
          routing_accuracy = $6, citation_hit_rate = $7,
          median_latency_ms = $8, p95_latency_ms = $9,
          completed_at = now()
        WHERE id = $1
        """,
        run_id,
        n_correct, n_partial, n_wrong, n_unable,
        routing_acc, citation_rate, median_ms, p95,
    )
    return summary


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_eval(
    bank_path: Path,
    endpoint: str,
    notes: str,
    *,
    priors_version: str = "v1.2026-05-01",
    skip_judge: bool = False,
) -> str:
    queries, bank_sha = load_bank(bank_path)
    logger.info("Loaded %d queries from %s (sha=%s)", len(queries), bank_path, bank_sha)

    run_id = await insert_run(
        bank_path=str(bank_path),
        bank_version=bank_sha,
        priors_version=priors_version,
        notes=notes,
        config_dump={"endpoint": endpoint, "skip_judge": skip_judge},
        n_queries=len(queries),
    )
    logger.info("Eval run started: run_id=%s", run_id)

    results: list[dict[str, Any]] = []

    for q in queries:
        qid = q.get("id") or f"q{len(results)+1:03d}"
        query = q.get("query") or ""
        caller_mode = q.get("caller_mode")
        expected = q.get("expected") or {}

        if not query:
            logger.warning("[%s] missing query; skip", qid)
            continue

        logger.info("[%s] %s", qid, query[:80])
        t_call = time.monotonic()
        must_facts = (expected.get("must_facts") or []) if expected else []
        response = await call_agent(
            endpoint, query, caller_mode,
            eval_run_id=run_id,
            must_facts=must_facts or None,
        )
        call_ms = int((time.monotonic() - t_call) * 1000)
        if "error" in response:
            logger.warning("[%s] agent ERROR: %s", qid, response["error"])
            results.append({
                "run_id": run_id, "query_id": qid, "query": query,
                "expected": expected, "judge_verdict": "unable_to_verify",
                "judge_reasoning": f"agent_error: {response['error']}",
                "total_ms": call_ms,
            })
            await insert_result(results[-1])
            continue

        # Cheap deterministic checks.
        rcorrect, chit, ffcorrect = deterministic_checks(expected, response)

        # LLM judge.
        if skip_judge:
            verdict, score, reasoning, model, judge_ms = ("unable_to_verify", 0.5, "skipped", "skipped", 0)
        else:
            verdict, score, reasoning, model, judge_ms = await adjudicate(
                query, expected, response,
                correlation_id=run_id,
            )

        # Build chunks_summary (denormalized for fast SQL queries).
        chunks_summary = []
        for c in (response.get("chunks") or [])[:5]:
            chunks_summary.append({
                "document_name": c.get("document_name"),
                "page_number": c.get("page_number"),
                "rerank_score": c.get("rerank_score"),
                "text": (c.get("text") or "")[:300],
            })

        try:
            top_rerank = max((c.get("rerank_score") or 0.0)
                             for c in (response.get("chunks") or []))
        except ValueError:
            top_rerank = None

        row = {
            "run_id": run_id,
            "query_id": qid,
            "query": query,
            "expected": expected,
            "strategy_chosen": (response.get("routing") or {}).get("strategy"),
            "strategy_executed": response.get("strategy_used"),
            "confidence": response.get("confidence"),
            "total_ms": (response.get("telemetry") or {}).get("total_ms") or call_ms,
            "n_chunks": len(response.get("chunks") or []),
            "top_rerank": top_rerank,
            "llm_answer": response.get("llm_answer"),
            "chunks_summary": chunks_summary,
            "routing_correct": rcorrect,
            "citation_hit": chit,
            "fail_fast_correct": ffcorrect,
            "judge_verdict": verdict,
            "judge_score": score,
            "judge_reasoning": reasoning,
            "judge_model": model,
            "judge_ms": judge_ms,
            # Stash the entire agent response so the frontend can render
            # the full thinking pipeline (parser → partition → pool →
            # router → strategies → assembler) without us pre-projecting.
            "full_response": response,
        }
        await insert_result(row)
        results.append(row)
        logger.info(
            "[%s] strategy=%s/%s conf=%s judge=%s(%.2f) %s",
            qid, row["strategy_chosen"], row["strategy_executed"],
            row["confidence"], verdict, score,
            (reasoning[:80] + "...") if len(reasoning) > 80 else reasoning,
        )

    summary = await finalize_run(run_id, results)
    logger.info("=" * 60)
    logger.info("Run summary (run_id=%s):", run_id)
    for k, v in summary.items():
        logger.info("  %s: %s", k, v)
    return run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default="eval/queries.yaml")
    parser.add_argument("--endpoint", default="http://localhost:8001/api/skills/v1/corpus_search_agent")
    parser.add_argument("--notes", default="")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM judge (deterministic checks only).")
    args = parser.parse_args()

    bank = Path(args.bank)
    if not bank.exists():
        logger.error("Bank file not found: %s", bank)
        sys.exit(1)
    async def _cli() -> None:
        # Close the pool before asyncio.run() tears down the loop — the
        # in-process eval router never calls this (its loop is long-lived).
        try:
            await run_eval(bank, args.endpoint, args.notes,
                           skip_judge=args.skip_judge)
        finally:
            await close_pool()

    asyncio.run(_cli())


if __name__ == "__main__":
    main()
