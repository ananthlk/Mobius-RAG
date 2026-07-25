"""Three-way allocator batch over the 22-query cmhc bank — REAL pool depths.

Replaces the throwaway generator behind traces/cmhc22_three_way_batch.json,
fixing its two defects:
  1. It simulated pool depths from query_class (put 20/22 queries at depth 1);
     Eval's backfilled run shows the bank really lives in buckets 2/3
     (pool_size 335-836). This script reads the REAL per-query pool_metadata
     from eval/artifacts/forced_filler_bank_run.json.
  2. It didn't record its inputs, so the run wasn't replayable. This one
     embeds pool_metadata + posture + j_codes in the output.

Runs every query through all three allocators TWICE — once against the frozen
pre-fold seed priors (app/services/router/testdata/priors_frozen_seed.yaml)
and once against the live folded file (eval/priors_bootstrap.yaml) — so the
diff isolates the effect of the 2026-07-23 Beta-update fold on real decisions.

Usage:  python3 scripts/run_router_three_way_batch.py
Output: traces/cmhc22_three_way_batch_v2.json
"""

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app.services.router.allocation import AnswerSlot, allocate_strategies  # noqa: E402
from app.services.router.bayesian_optimizer import optimize_allocation_bayesian  # noqa: E402
from app.services.router.optimizer import optimize_allocation  # noqa: E402
from app.services.router.priors import compute_depth_bucket, load_priors  # noqa: E402

ARTIFACT = REPO / "eval" / "artifacts" / "forced_filler_bank_run.json"
OLD_BATCH = REPO / "traces" / "cmhc22_three_way_batch.json"
OUT = REPO / "traces" / "cmhc22_three_way_batch_v2.json"

FROZEN_PRIORS = REPO / "app" / "services" / "router" / "testdata" / "priors_frozen_seed.yaml"
LIVE_PRIORS = REPO / "eval" / "priors_bootstrap.yaml"

POSTURE_BASE = {
    "speed_budget": "interactive",
    "confidence_bar": 0.85,
    "caller_mode": "chat.default",
    "max_attempts_per_slot": 6,
}

ALLOCATORS = {
    "greedy": allocate_strategies,
    "optimizer": optimize_allocation,
    "bayesian": optimize_allocation_bayesian,
}


def _slot() -> AnswerSlot:
    return AnswerSlot(slot_id="slot_0", slot_semantics="direct_answer", capacity=5,
                      rewritten_query="", required=True, priority=0)


def _plan(ladder) -> dict:
    return {
        "chain": list(ladder.per_slot["slot_0"]),
        "mean": round(ladder.per_slot_confidence["slot_0"], 4),
        "lb": round(ladder.per_slot_lb["slot_0"], 4),
        "status": ladder.per_slot_status["slot_0"],
        "terminal": ladder.per_slot_terminal.get("slot_0"),
        "helpers": list(ladder.helpers or []),
        "cost": ladder.total_estimated_cost,
        "latency_ms": ladder.total_estimated_ms,
        "outcome": ladder.outcome,
    }


def run_pass(rows, j_codes_by_id, priors_path: Path) -> dict:
    os.environ["ROUTER_PRIORS_PATH"] = str(priors_path)
    load_priors(force_reload=True)  # bust cache across passes
    out = {}
    for r in rows:
        pm = dict(r["pool_metadata"])
        posture = dict(POSTURE_BASE)
        posture["gate_j_codes"] = j_codes_by_id.get(r["id"], [])
        plans = {}
        for name, fn in ALLOCATORS.items():
            ladder = fn([_slot()], {"slot_0": pm}, posture)
            plans[name] = _plan(ladder)
        out[r["id"]] = {
            "depth_bucket": compute_depth_bucket(pm),
            "plans": plans,
        }
    return out


def main():
    rows = json.load(open(ARTIFACT))["results"]
    # payor j_codes: carry over the old batch's keyword-derived tags verbatim
    # (query text unchanged; keeps the s-gating inputs identical across runs)
    old = json.load(open(OLD_BATCH))["rows"]
    j_codes_by_id = {r["id"]: r["j_codes"] for r in old}

    pre = run_pass(rows, j_codes_by_id, FROZEN_PRIORS)
    post = run_pass(rows, j_codes_by_id, LIVE_PRIORS)

    merged, diffs = [], []
    for r in rows:
        qid = r["id"]
        entry = {
            "id": qid,
            # replayability: record the actual inputs
            "pool_metadata": r["pool_metadata"],
            "j_codes": j_codes_by_id.get(qid, []),
            "depth_bucket": post[qid]["depth_bucket"],
            "pre_fold": pre[qid]["plans"],
            "post_fold": post[qid]["plans"],
        }
        merged.append(entry)
        for alloc in ALLOCATORS:
            a, b = pre[qid]["plans"][alloc], post[qid]["plans"][alloc]
            if a["chain"] != b["chain"] or a["status"] != b["status"]:
                diffs.append({
                    "id": qid, "allocator": alloc,
                    "chain": [a["chain"], b["chain"]],
                    "status": [a["status"], b["status"]],
                    "lb": [a["lb"], b["lb"]],
                })

    payload = {
        "note": ("v2: REAL pool_metadata from forced_filler_bank_run.json "
                 "(v1 simulated depths — 20/22 at depth 1, unrealistic). "
                 "pre_fold = frozen seed priors; post_fold = live file after "
                 "the 2026-07-23 Beta-update fold (buckets 2/3, a/b/c/d, "
                 "n=19/17). Posture: " + json.dumps(POSTURE_BASE)),
        "priors": {"pre": str(FROZEN_PRIORS.relative_to(REPO)),
                   "post": str(LIVE_PRIORS.relative_to(REPO))},
        "rows": merged,
        "decision_diffs": diffs,
    }
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"wrote {OUT} — {len(merged)} rows, {len(diffs)} allocator-decisions changed")
    for dchg in diffs:
        print(f"  {dchg['id']} [{dchg['allocator']}]: {dchg['chain'][0]} {dchg['status'][0]} "
              f"(lb {dchg['lb'][0]}) -> {dchg['chain'][1]} {dchg['status'][1]} (lb {dchg['lb'][1]})")


if __name__ == "__main__":
    main()
