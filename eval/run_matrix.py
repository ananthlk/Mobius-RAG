"""Strategy × query verdict matrix collector.

For every query in the bank and every forceable strategy in {a, b, c, d}, this
runner forces that strategy via ``mode=<letter>``, judges the response with the
same rubric used by ``eval/run.py``, and persists a flat JSON row per cell.

Output schema (one row per query × strategy):

    {
        "qid":               str,
        "strategy_forced":   "a" | "b" | "c" | "d",
        "repeat":            int,                  # 0..N-1, when --repeats > 1
        "strategy_executed": str | None,           # what the agent reports
        "judge_verdict":     str,                  # correct | partial | ...
        "judge_score":       float,                # 0..1
        "agent_confidence":  "high" | "medium" | "low",
        "latency_ms":        int,
        "n_chunks":          int,
        "top_doc":           str | None,
        "top_page":          int | None,
        "top_rerank":        float | None,
        "llm_answer":        str | None,
        "features": {
            "literal_anchors":     list[str],
            "tag_matches":         list[str],
            "query_type":          str,
            "cascade_level":       str | None,
            "pool_size":           int | None,
            "payer_specificity":   str,
            "answer_shape":        str,
            "persona":             str,
            "n_must_facts":        int,
            "n_bonus_facts":       int,
        },
        "judge_reasoning":   str,
    }

Why a flat JSON instead of a DB table: this is a tuning artifact, not
production telemetry. It lives next to ``eval/calibration/`` and gets re-run
whenever priors change. A single dictable file is easier to load into a
notebook for fitting weights than a parameterised SQL view.

Usage:

    python -m eval.run_matrix \\
        --bank eval/queries_cmhc.yaml \\
        --strategies a,b,c,d \\
        --out eval/calibration/strategy_matrix_$(date +%Y%m%d-%H%M%S).json

Reuses ``eval.run.call_agent`` (HTTP client) and ``eval.judge.adjudicate``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import yaml

from eval.judge import adjudicate
from eval.run import call_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("matrix")


def _extract_features(query_yaml: dict, response: dict) -> dict[str, Any]:
    """Pull the per-query feature vector out of the agent response + yaml."""
    qp = response.get("query_profile") or {}
    cp = response.get("candidate_pool") or {}
    expected = query_yaml.get("expected") or {}
    return {
        "literal_anchors":   qp.get("literal_anchors") or [],
        "tag_matches":       qp.get("tag_matches") or [],
        "query_type":        qp.get("query_type"),
        "cascade_level":     cp.get("cascade_level"),
        "pool_size":         cp.get("size"),
        "payer_specificity": query_yaml.get("payer_specificity"),
        "answer_shape":      query_yaml.get("answer_shape") or expected.get("answer_shape"),
        "persona":           query_yaml.get("persona"),
        "n_must_facts":      len(query_yaml.get("must_facts") or []),
        "n_bonus_facts":     len(query_yaml.get("bonus_facts") or []),
    }


def _flatten_chunk_top(response: dict) -> dict[str, Any]:
    chunks = response.get("chunks") or []
    if not chunks:
        return {"top_doc": None, "top_page": None, "top_rerank": None, "n_chunks": 0}
    top = chunks[0]
    return {
        "top_doc":     top.get("document_name"),
        "top_page":    top.get("page_number"),
        "top_rerank": (top.get("rerank_score") or 0.0),
        "n_chunks":    len(chunks),
    }


async def _run_cell(
    endpoint: str,
    qid: str,
    query_yaml: dict,
    strategy: str,
    *,
    skip_judge: bool,
    correlation_id: str,
    repeat: int = 0,
) -> dict[str, Any]:
    """Force `strategy` on `query`, judge, return one matrix row."""
    query_text = query_yaml["query"]
    expected = query_yaml.get("expected") or {}

    body: dict[str, Any] = {"query": query_text, "k": 5}
    if strategy != "natural":
        body["mode"] = strategy
    # call_agent's signature wants caller_mode positional, but we want raw body
    # control — replicate its inner POST here.
    import urllib.request
    def _post() -> dict:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    t0 = time.monotonic()
    response = await asyncio.to_thread(_post)
    latency_ms = int((time.monotonic() - t0) * 1000)

    if "error" in response:
        logger.warning("[%s/%s/r%d] agent ERROR: %s", qid, strategy, repeat, response["error"])
        return {
            "qid":               qid,
            "strategy_forced":   strategy,
            "repeat":            repeat,
            "strategy_executed": None,
            "judge_verdict":     "agent_error",
            "judge_score":       0.0,
            "agent_confidence":  None,
            "latency_ms":        latency_ms,
            "n_chunks":          0,
            "top_doc":           None,
            "top_page":          None,
            "top_rerank":        None,
            "llm_answer":        None,
            "features":          _extract_features(query_yaml, {}),
            "judge_reasoning":   f"agent_error: {response['error']}",
        }

    if skip_judge:
        verdict, score, reasoning = ("skipped", 0.5, "judge skipped")
    else:
        verdict, score, reasoning, _model, _judge_ms = await adjudicate(
            query_text, expected, response, correlation_id=correlation_id,
        )

    row = {
        "qid":               qid,
        "strategy_forced":   strategy,
        "repeat":            repeat,
        "strategy_executed": response.get("strategy_used"),
        "judge_verdict":     verdict,
        "judge_score":       score,
        "agent_confidence":  response.get("confidence"),
        "latency_ms":        latency_ms,
        "llm_answer":       (response.get("llm_answer") or "")[:600],
        "features":          _extract_features(query_yaml, response),
        "judge_reasoning":  (reasoning or "")[:400],
    }
    row.update(_flatten_chunk_top(response))
    return row


async def run_matrix(
    bank_path: Path,
    endpoint: str,
    strategies: list[str],
    out_path: Path,
    *,
    skip_judge: bool = False,
    repeats: int = 1,
    concurrency: int = 6,
) -> None:
    data = yaml.safe_load(bank_path.read_text())
    queries = data["queries"] if isinstance(data, dict) and "queries" in data else data
    n_cells = len(queries) * len(strategies) * repeats
    logger.info("Loaded %d queries from %s", len(queries), bank_path)
    logger.info("Strategies: %s  repeats: %d  total cells: %d  | endpoint: %s",
                strategies, repeats, n_cells, endpoint)

    correlation_id = str(uuid4())
    rows: list[dict[str, Any]] = []
    # Semaphore limits concurrent agent+judge calls to avoid overwhelming the server.
    sem = asyncio.Semaphore(concurrency)
    cell_counter = [0]
    rows_lock = asyncio.Lock()

    async def _run_and_log(q: dict, strategy: str, r_idx: int) -> None:
        async with sem:
            qid = q.get("id") or "q???"
            t = time.monotonic()
            row = await _run_cell(
                endpoint, qid, q, strategy,
                skip_judge=skip_judge, correlation_id=correlation_id,
                repeat=r_idx,
            )
            elapsed = int((time.monotonic() - t) * 1000)
            async with rows_lock:
                cell_counter[0] += 1
                idx = cell_counter[0]
                verdict = row["judge_verdict"]
                score = row["judge_score"]
                top_doc = (row.get("top_doc") or "")[:36]
                logger.info(
                    "[%d/%d %s/%s/r%d] verdict=%s(%.2f) conf=%s top=%s p.%s  %dms",
                    idx, n_cells, qid, strategy, r_idx, verdict, score,
                    row.get("agent_confidence"), top_doc, row.get("top_page"),
                    elapsed,
                )
                rows.append(row)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(rows, indent=2, default=str))

    tasks = []
    for q in queries:
        for strategy in strategies:
            for r_idx in range(repeats):
                tasks.append(_run_and_log(q, strategy, r_idx))
    await asyncio.gather(*tasks)

    # Final summary
    print()
    print("=" * 90)
    print(f"{'qid':10} {'a':>20} {'b':>20} {'c':>20} {'d':>20}")
    print("-" * 90)
    by_qid: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_qid.setdefault(r["qid"], {})[r["strategy_forced"] or "natural"] = r
    for qid, cells in by_qid.items():
        line = f"{qid:10}"
        for s in ("a", "b", "c", "d"):
            cell = cells.get(s)
            if cell is None:
                line += f" {'—':>20}"
            else:
                line += f" {cell['judge_verdict'][:8]:>8}({cell['judge_score']:.2f}) {cell['agent_confidence'] or '?':>4}"
        print(line)
    print("=" * 90)
    print(f"\nMatrix saved to {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="eval/queries_cmhc.yaml")
    p.add_argument("--endpoint",
                   default="http://localhost:8001/api/skills/v1/corpus_search_agent")
    p.add_argument("--strategies", default="a,b,c,d",
                   help="comma-separated subset of {a,b,c,d}")
    p.add_argument("--out",
                   default="eval/calibration/strategy_matrix.json")
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--repeats", type=int, default=1,
                   help="Run each (query, strategy) cell N times to estimate variance.")
    p.add_argument("--concurrency", type=int, default=6,
                   help="Max concurrent agent+judge calls.")
    args = p.parse_args()

    asyncio.run(run_matrix(
        Path(args.bank), args.endpoint,
        [s.strip() for s in args.strategies.split(",") if s.strip()],
        Path(args.out),
        skip_judge=args.skip_judge,
        repeats=args.repeats,
        concurrency=args.concurrency,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
