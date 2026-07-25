"""Run Filler a (BM25) before/after calibration against eval/queries_cmhc.yaml.

Baseline: uniform top-N by bm25_score (no slot semantics).
Proposed: Filler a v0.1 (per-slot BM25 ranking).

Runs Shape (Gate → Reformat → Structure → Slots) → Pool → both fillers per query.
Collects per-query metrics: occupancy, under-fill, chunks.
Verifies acceptance criteria from filler-a-calibration-plan.md.

Verified:
- PublicSourceAdapter(db) constructor ✓
- FAN_OUT handling: run_pool_fanout for multiple rewritten_queries ✓
- Single-query: run_pool_for_query ✓

Usage (from mobius-rag/):
    .venv/bin/python scripts/run_filler_a_calibration_cmhc26.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import AsyncSessionLocal  # noqa: E402
from app.services.retriever.shape.gate import run_gate  # noqa: E402
from app.services.retriever.shape.reformat import run_reformat  # noqa: E402
from app.services.retriever.shape.structure import run_structure  # noqa: E402
from app.services.retriever.shape.slots import run_slots  # noqa: E402
from app.services.retriever.pool.pool import run_pool_for_query, run_pool_fanout  # noqa: E402
from app.services.retriever.pool.public_adapter import PublicSourceAdapter  # noqa: E402
from app.services.retriever.fillers.filler_a import fill_shape_bm25  # noqa: E402
from app.services.retriever.fillers.filler_baseline import fill_shape_uniform_topn  # noqa: E402

BANK_PATH = Path(__file__).resolve().parent.parent / "eval" / "queries_cmhc.yaml"


@dataclass
class QueryResult:
    """Result for one query (baseline vs proposed)."""

    qid: str
    query_text: str
    # Baseline metrics
    baseline_total_capacity: int
    baseline_total_occupancy: int
    baseline_under_filled: int
    # Proposed metrics
    proposed_total_capacity: int
    proposed_total_occupancy: int
    proposed_under_filled: int
    # Computed
    occupancy_diff: float = 0.0
    under_fill_diff: int = 0
    total_ms: int = 0

    def __post_init__(self):
        self.occupancy_diff = self.proposed_total_occupancy - self.baseline_total_occupancy
        self.under_fill_diff = self.proposed_under_filled - self.baseline_under_filled


async def main() -> None:
    """Load cmhc-26, run Shape/Pool/fillers per query, report metrics."""
    # Load query bank (same format as run_reformat_on_postures_bank.py).
    raw = BANK_PATH.read_text()
    yaml_part = raw.split("\n---\n", 1)[0]
    bank = yaml.safe_load(yaml_part)
    queries = bank["queries"]
    print(f"bank: {bank.get('bank_version')} — {len(queries)} queries\n")

    results: list[QueryResult] = []
    errors: list[tuple[str, str]] = []

    async with AsyncSessionLocal() as db:
        for q in queries:
            qid = q["id"]
            text = q["query"]

            try:
                t0 = time.monotonic()

                # Shape pipeline: Gate → Reformat → Structure → Slots
                gr = await run_gate(db, text)
                rr = await run_reformat(db, gr)
                sr = run_structure(rr)  # sync, no db
                shape_result = run_slots(sr)  # sync, no db — produces AnswerShapeResult
                print(f"[{qid}] Shape OK: {sr.resource_posture.breadth}*{len(shape_result.slots)} = {sum(s.capacity for s in shape_result.slots)} capacity", flush=True)

                # Pool: get candidates via PublicSourceAdapter
                adapter = PublicSourceAdapter(db)

                # Handle FAN_OUT (multiple rewritten_queries) vs single-query cases
                if sr.posture.value == "fan_out" and len(sr.rewritten_queries) > 1:
                    # FAN_OUT: run pool for each theme query in parallel
                    pool_results = await run_pool_fanout(db, sr.rewritten_queries, gr, sr.resource_posture, adapter)
                    # For calibration: merge all pools into one for fillers to rank from
                    # (future: could keep separate pools per theme, but v1 merges)
                    all_candidates = []
                    for pr in pool_results:
                        all_candidates.extend(pr.candidates)
                    # Deduplicate by chunk_id
                    seen = set()
                    deduped_candidates = []
                    for c in all_candidates:
                        if c.chunk_id not in seen:
                            deduped_candidates.append(c)
                            seen.add(c.chunk_id)
                    # Create merged pool_result
                    pool_result = pool_results[0]
                    pool_result.candidates = deduped_candidates
                else:
                    # Single-query case
                    query = sr.rewritten_queries[0] if sr.rewritten_queries else text
                    pool_result = await run_pool_for_query(db, query, gr, sr.resource_posture, adapter)

                # Fillers: baseline vs proposed
                baseline_filled = fill_shape_uniform_topn(pool_result, shape_result)
                proposed_filled = fill_shape_bm25(pool_result, shape_result)

                total_ms = int((time.monotonic() - t0) * 1000)

                # Extract metrics
                baseline_capacity = sum(s.capacity for s in baseline_filled.slots)
                baseline_occupancy = sum(s.occupancy for s in baseline_filled.slots)
                baseline_under_filled = len([s for s in baseline_filled.slots if s.under_filled])

                proposed_capacity = sum(s.capacity for s in proposed_filled.slots)
                proposed_occupancy = sum(s.occupancy for s in proposed_filled.slots)
                proposed_under_filled = len([s for s in proposed_filled.slots if s.under_filled])

                result = QueryResult(
                    qid=qid,
                    query_text=text[:80],
                    baseline_total_capacity=baseline_capacity,
                    baseline_total_occupancy=baseline_occupancy,
                    baseline_under_filled=baseline_under_filled,
                    proposed_total_capacity=proposed_capacity,
                    proposed_total_occupancy=proposed_occupancy,
                    proposed_under_filled=proposed_under_filled,
                    total_ms=total_ms,
                )
                results.append(result)

                # Print per-query result
                occ_ratio_base = baseline_occupancy / baseline_capacity if baseline_capacity > 0 else 0.0
                occ_ratio_prop = proposed_occupancy / proposed_capacity if proposed_capacity > 0 else 0.0
                print(
                    f"[{qid:3s}] base: {baseline_occupancy:2d}/{baseline_capacity:2d} ({occ_ratio_base:.2f}) "
                    f"| prop: {proposed_occupancy:2d}/{proposed_capacity:2d} ({occ_ratio_prop:.2f}) "
                    f"| under_fill: {baseline_under_filled:2d} → {proposed_under_filled:2d} | ms={total_ms}"
                )

            except Exception as e:
                errors.append((qid, str(e)))
                print(f"[{qid:3s}] ERROR: {e}")

    # Aggregate results and verify acceptance criteria
    print(f"\n{'='*80}")
    print(f"Calibration Summary ({len(results)}/{len(queries)} queries processed)")
    print(f"{'='*80}\n")

    if results:
        # Compute aggregates
        total_base_occupancy = sum(r.baseline_total_occupancy for r in results)
        total_base_capacity = sum(r.baseline_total_capacity for r in results)
        total_prop_occupancy = sum(r.proposed_total_occupancy for r in results)
        total_prop_capacity = sum(r.proposed_total_capacity for r in results)
        total_base_under = sum(r.baseline_under_filled for r in results)
        total_prop_under = sum(r.proposed_under_filled for r in results)

        base_occ_rate = total_base_occupancy / total_base_capacity if total_base_capacity > 0 else 0.0
        prop_occ_rate = total_prop_occupancy / total_prop_capacity if total_prop_capacity > 0 else 0.0

        print(f"Baseline Occupancy Rate: {total_base_occupancy}/{total_base_capacity} = {base_occ_rate:.4f}")
        print(f"Proposed Occupancy Rate: {total_prop_occupancy}/{total_prop_capacity} = {prop_occ_rate:.4f}")
        print(f"Occupancy Difference:    {prop_occ_rate - base_occ_rate:+.4f}")
        print(f"Baseline Under-filled:   {total_base_under} slots")
        print(f"Proposed Under-filled:   {total_prop_under} slots")
        print(f"Under-fill Difference:   {total_prop_under - total_base_under:+d} slots\n")

        # Acceptance criteria (from calibration-plan.md):
        # 1. proposed_occupancy >= baseline - 0.02 * capacity
        # 2. under_fill_increase <= 5% of total slots
        criterion_1 = prop_occ_rate >= (base_occ_rate - 0.02)
        criterion_2 = (total_prop_under - total_base_under) <= 0.05 * total_prop_capacity

        print(f"Acceptance Criteria:")
        print(f"  1. Occupancy: proposed >= baseline - 0.02: {criterion_1} ({prop_occ_rate:.4f} >= {base_occ_rate - 0.02:.4f})")
        print(f"  2. Under-fill: increase <= 5%: {criterion_2} ({total_prop_under - total_base_under} <= {0.05 * total_prop_capacity:.0f})")

        acceptance = criterion_1 and criterion_2
        print(f"\n{'✅ ACCEPTANCE: PASS' if acceptance else '❌ ACCEPTANCE: FAIL'}\n")

    if errors:
        print(f"\n{len(errors)} errors:")
        for qid, err in errors:
            print(f"  {qid}: {err}")


if __name__ == "__main__":
    asyncio.run(main())
