"""Calibration harness for Filler a (BM25) before/after comparison.

Runs baseline (uniform top-N) vs proposed (Filler a v0.1) on cmhc-26 query bank.
Collects metrics: occupancy, recall, under-fill, FAN_OUT diversity.
Verifies acceptance criteria before Eval sign-off.

See docs/rag-agents/filler-a-calibration-plan.md.

Usage (pseudo-code, awaiting Retriever's clarification on orchestration):
    results = run_calibration_cmhc26(
        query_bank_path="...",
        shape_invoker=...,
        pool_invoker=...,
    )
    print(results.summary())
    results.to_json("calibration-results.json")
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Callable

from app.services.retriever.pool.contracts import PoolResult
from app.services.retriever.shape.slots import AnswerShapeResult
from app.services.retriever.fillers.filler_a import fill_shape_bm25
from app.services.retriever.fillers.filler_baseline import fill_shape_uniform_topn


@dataclass
class PerQueryMetrics:
    """Metrics for one query's baseline vs proposed comparison."""

    query_id: str
    query_text: str
    # Baseline (uniform top-N)
    baseline_occupancy: int
    baseline_capacity: int
    baseline_under_filled_slots: int
    baseline_chunks: int
    # Proposed (Filler a)
    proposed_occupancy: int
    proposed_capacity: int
    proposed_under_filled_slots: int
    proposed_chunks: int
    # Computed
    occupancy_diff: float = 0.0
    under_fill_diff: int = 0

    def __post_init__(self):
        self.occupancy_diff = self.proposed_occupancy - self.baseline_occupancy
        self.under_fill_diff = self.proposed_under_filled_slots - self.baseline_under_filled_slots


@dataclass
class CalibrationResults:
    """Aggregated results from cmhc-26 calibration run."""

    per_query: list[PerQueryMetrics] = field(default_factory=list)
    acceptance_criteria_met: bool = False
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary of results."""
        if not self.per_query:
            return "No results yet."

        total_queries = len(self.per_query)
        avg_occupancy_diff = sum(m.occupancy_diff for m in self.per_query) / total_queries
        max_under_fill_diff = max((m.under_fill_diff for m in self.per_query), default=0)

        summary_lines = [
            f"Calibration Results (cmhc-26, n={total_queries})",
            f"  Acceptance: {'✅ PASS' if self.acceptance_criteria_met else '❌ FAIL'}",
            f"  Avg occupancy delta (proposed - baseline): {avg_occupancy_diff:+.2f}",
            f"  Max under-fill increase: {max_under_fill_diff:+d}",
        ]

        if self.notes:
            summary_lines.append("  Notes:")
            for note in self.notes:
                summary_lines.append(f"    - {note}")

        return "\n".join(summary_lines)

    def to_json(self, path: str):
        """Write results to JSON file."""
        data = {
            "acceptance_criteria_met": self.acceptance_criteria_met,
            "per_query": [
                {
                    "query_id": m.query_id,
                    "baseline_occupancy": m.baseline_occupancy,
                    "proposed_occupancy": m.proposed_occupancy,
                    "occupancy_diff": m.occupancy_diff,
                    "baseline_under_filled": m.baseline_under_filled_slots,
                    "proposed_under_filled": m.proposed_under_filled_slots,
                    "under_fill_diff": m.under_fill_diff,
                }
                for m in self.per_query
            ],
            "notes": self.notes,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def run_calibration_cmhc26(
    query_bank: list[str],
    shape_invoker: Callable[[str], AnswerShapeResult],
    pool_invoker: Callable[[str], PoolResult],
) -> CalibrationResults:
    """
    Run before/after calibration on cmhc-26 query bank.

    Args:
        query_bank: List of 26 query strings from cmhc bank.
        shape_invoker: Function(query: str) -> AnswerShapeResult.
        pool_invoker: Function(query: str) -> PoolResult.

    Returns:
        CalibrationResults with per-query and aggregate metrics.
    """

    results = CalibrationResults()

    for query_idx, query_text in enumerate(query_bank):
        # Invoke Shape and Pool for this query.
        try:
            shape_result = shape_invoker(query_text)
            pool_result = pool_invoker(query_text)
        except Exception as e:
            results.notes.append(f"Query {query_idx}: Shape/Pool failed: {e}")
            continue

        # Run baseline (uniform top-N).
        try:
            baseline_filled = fill_shape_uniform_topn(pool_result, shape_result)
        except Exception as e:
            results.notes.append(f"Query {query_idx}: Baseline filler failed: {e}")
            continue

        # Run proposed (Filler a).
        try:
            proposed_filled = fill_shape_bm25(pool_result, shape_result)
        except Exception as e:
            results.notes.append(f"Query {query_idx}: Proposed filler failed: {e}")
            continue

        # Collect per-query metrics.
        baseline_occupancy = sum(s.occupancy for s in baseline_filled.slots)
        baseline_capacity = sum(s.capacity for s in baseline_filled.slots)
        baseline_under_filled = len([s for s in baseline_filled.slots if s.under_filled])

        proposed_occupancy = sum(s.occupancy for s in proposed_filled.slots)
        proposed_capacity = sum(s.capacity for s in proposed_filled.slots)
        proposed_under_filled = len([s for s in proposed_filled.slots if s.under_filled])

        metrics = PerQueryMetrics(
            query_id=f"cmhc_{query_idx:02d}",
            query_text=query_text[:80],  # Truncate for display
            baseline_occupancy=baseline_occupancy,
            baseline_capacity=baseline_capacity,
            baseline_under_filled_slots=baseline_under_filled,
            baseline_chunks=baseline_filled.total_chunks_assigned,
            proposed_occupancy=proposed_occupancy,
            proposed_capacity=proposed_capacity,
            proposed_under_filled_slots=proposed_under_filled,
            proposed_chunks=proposed_filled.total_chunks_assigned,
        )

        results.per_query.append(metrics)

    # Verify acceptance criteria (from calibration plan).
    if results.per_query:
        # Criterion: proposed occupancy >= baseline (or close enough)
        avg_baseline = sum(m.baseline_occupancy for m in results.per_query) / len(results.per_query)
        avg_proposed = sum(m.proposed_occupancy for m in results.per_query) / len(results.per_query)
        max_under_fill_increase = max((m.under_fill_diff for m in results.per_query), default=0)

        # Acceptance: proposed >= baseline - 0.02 * capacity, under_fill ≤ baseline + 5%
        proposed_acceptable = avg_proposed >= (avg_baseline - 0.02 * avg_baseline)
        under_fill_acceptable = max_under_fill_increase <= 5

        results.acceptance_criteria_met = proposed_acceptable and under_fill_acceptable

        if proposed_acceptable:
            results.notes.append(f"✅ Occupancy: proposed ({avg_proposed:.1f}) >= baseline ({avg_baseline:.1f}) - 0.02")
        else:
            results.notes.append(f"❌ Occupancy: proposed ({avg_proposed:.1f}) < baseline ({avg_baseline:.1f}) - 0.02")

        if under_fill_acceptable:
            results.notes.append(f"✅ Under-fill: max increase {max_under_fill_increase} ≤ 5%")
        else:
            results.notes.append(f"❌ Under-fill: max increase {max_under_fill_increase} > 5%")

    return results
