"""Calibration utilities for Router + Eval collaboration.

STANDALONE OPS TOOL ONLY — invoked by hand/CLI for calibration analysis.
Never import or call this from the live query path (the report functions
print() to stdout by design).

Parses rag_query_decisions rows, aggregates calibration data, and provides
debugging tools to accelerate empirical priors derivation (Week 1-3).

Schema: (depth_bucket, strategy, outcome) tuples extracted from rag_query_decisions.
Success is judged against each row's own persisted `confidence_bar` — rows
missing it are EXCLUDED, never guessed at.
"""

import logging
from typing import Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class CalibrationTuple:
    """Single (depth_bucket, strategy, outcome) measurement."""
    depth_bucket: int
    strategy_id: str
    success: bool  # did strategy meet confidence bar?
    confidence_achieved: float  # actual confidence Observer measured
    accuracy_achieved: float  # actual accuracy (if measured)
    latency_ms: int
    cost: float
    query_id: str  # for debugging/tracing


class CalibrationDataParser:
    """Parse rag_query_decisions rows into calibration tuples."""

    @staticmethod
    def row_to_tuple(row: dict[str, Any]) -> Optional[list[CalibrationTuple]]:
        """Extract calibration tuples from single rag_query_decisions row.

        Returns list of tuples (one per strategy actually executed).
        """
        if not row.get("is_calibration"):
            return None  # only calibration rows

        depth_bucket = row.get("depth_bucket", 0)
        strategy_sequence = row.get("strategy_sequence", [])
        if isinstance(strategy_sequence, str):
            try:
                strategy_sequence = json.loads(strategy_sequence)
            except (json.JSONDecodeError, TypeError):
                return None

        confidence = row.get("confidence", 0.0)
        accuracy = row.get("accuracy_estimate", 0.0)
        latency = row.get("total_ms", 0)
        cost = row.get("cost", 0.0)
        query_id = row.get("id", "unknown")

        # Success is judged against THIS ROW's confidence_bar (varies by
        # caller_mode/posture) — never a hardcoded threshold. A hardcoded 0.80
        # here would silently corrupt every derived prior (Retriever's
        # data-integrity catch, fixed 2026-07-23). confidence_bar is a
        # persisted column as of the 24-column schema widening.
        bar = row.get("confidence_bar")
        if bar is None:
            logger.warning(
                "row %s missing confidence_bar — excluding from calibration "
                "rather than mislabeling success against a guessed threshold",
                query_id,
            )
            return None

        tuples = []
        for strategy_id in strategy_sequence:
            # Assume each strategy in sequence was attempted
            # (real data will show which actually executed via Observer logs)
            tuples.append(
                CalibrationTuple(
                    depth_bucket=depth_bucket,
                    strategy_id=strategy_id,
                    success=confidence >= float(bar),
                    confidence_achieved=confidence,
                    accuracy_achieved=accuracy,
                    latency_ms=latency,
                    cost=cost,
                    query_id=query_id,
                )
            )
        return tuples


class CalibrationAggregator:
    """Aggregate calibration tuples into empirical strategy profiles."""

    def __init__(self):
        self.data: dict[tuple[int, str], list[CalibrationTuple]] = defaultdict(list)

    def add_tuple(self, t: CalibrationTuple):
        """Add calibration tuple to aggregation."""
        key = (t.depth_bucket, t.strategy_id)
        self.data[key].append(t)

    def add_tuples(self, tuples: list[CalibrationTuple]):
        """Add multiple tuples."""
        for t in tuples:
            self.add_tuple(t)

    def get_stats(self, depth_bucket: int, strategy_id: str) -> dict[str, Any]:
        """Compute aggregated statistics for (depth_bucket, strategy)."""
        key = (depth_bucket, strategy_id)
        tuples = self.data[key]

        if not tuples:
            return {"count": 0, "insufficient_data": True}

        successes = sum(1 for t in tuples if t.success)
        confidences = [t.confidence_achieved for t in tuples]
        accuracies = [t.accuracy_achieved for t in tuples]
        latencies = [t.latency_ms for t in tuples]
        costs = [t.cost for t in tuples]

        return {
            "count": len(tuples),
            "success_rate": successes / len(tuples),
            "recall_lift": sum(confidences) / len(confidences),  # empirical recall_lift
            "accuracy_estimate": sum(accuracies) / len(accuracies),
            "latency_p50_ms": sorted(latencies)[len(latencies) // 2],
            "latency_mean_ms": sum(latencies) / len(latencies),
            "cost": sum(costs) / len(costs),
        }

    def get_all_cells(self) -> dict[tuple[int, str], dict[str, Any]]:
        """Get stats for all (depth, strategy) cells."""
        result = {}
        for (depth, strategy) in sorted(self.data.keys()):
            result[(depth, strategy)] = self.get_stats(depth, strategy)
        return result

    def coverage_report(self) -> dict[str, Any]:
        """Report on data coverage across depth × strategy cells."""
        cells_with_data = len(self.data)
        total_cells = 5 * 6  # 5 depths × 6 strategies

        samples_by_depth = defaultdict(int)
        samples_by_strategy = defaultdict(int)

        for (depth, strategy), tuples in self.data.items():
            count = len(tuples)
            samples_by_depth[depth] += count
            samples_by_strategy[strategy] += count

        return {
            "cells_with_data": cells_with_data,
            "total_cells": total_cells,
            "coverage_pct": 100 * cells_with_data / total_cells,
            "samples_by_depth": dict(samples_by_depth),
            "samples_by_strategy": dict(samples_by_strategy),
            "cells_ready_for_empirical": sum(
                1 for _, stats in self.get_all_cells().items()
                if stats.get("count", 0) >= 50
            ),
        }

    def cells_below_threshold(self, min_samples: int = 50) -> list[tuple[int, str]]:
        """List cells that need more data."""
        return [
            (d, s)
            for (d, s), tuples in self.data.items()
            if len(tuples) < min_samples
        ]

    def regression_check(
        self,
        seed_priors: dict[int, dict[str, Any]],
        threshold: float = 0.80,
    ) -> list[tuple[int, str, str]]:
        """Flag cells where empirical performance regressed from seeds.

        Returns list of (depth, strategy, reason) for cells that underperformed.
        """
        regressions = []
        for (depth, strategy), stats in self.get_all_cells().items():
            if stats.get("insufficient_data"):
                continue

            seed = seed_priors.get(depth, {}).get(strategy)
            if not seed:
                continue

            empirical_recall = stats.get("recall_lift", 0)
            seed_recall = seed.get("recall_lift", 0)

            if empirical_recall < seed_recall * threshold:
                regressions.append(
                    (depth, strategy, f"recall {empirical_recall:.3f} < seed {seed_recall:.3f}")
                )

        return regressions


class CalibrationDebugger:
    """Debugging tools for calibration analysis."""

    @staticmethod
    def print_coverage_report(agg: CalibrationAggregator):
        """Print human-readable coverage report."""
        report = agg.coverage_report()
        print("\n" + "="*80)
        print(" CALIBRATION COVERAGE REPORT")
        print("="*80)
        print(f"\nCells with data: {report['cells_with_data']}/{report['total_cells']} "
              f"({report['coverage_pct']:.1f}%)")
        print(f"Cells ready for empirical (N≥50): {report['cells_ready_for_empirical']}")

        print("\nSamples by depth:")
        for depth in sorted(report['samples_by_depth'].keys()):
            print(f"  Depth {depth}: {report['samples_by_depth'][depth]} samples")

        print("\nSamples by strategy:")
        for strategy in sorted(report['samples_by_strategy'].keys()):
            print(f"  {strategy}: {report['samples_by_strategy'][strategy]} samples")

        print("\nCells needing more data (< 50 samples):")
        cells_needed = agg.cells_below_threshold(50)
        if not cells_needed:
            print("  ✅ All cells have sufficient data!")
        else:
            for depth, strategy in cells_needed[:10]:
                count = len(agg.data[(depth, strategy)])
                print(f"  Depth {depth}, {strategy}: {count}/50 samples")
            if len(cells_needed) > 10:
                print(f"  ... and {len(cells_needed) - 10} more")

    @staticmethod
    def print_empirical_priors(agg: CalibrationAggregator):
        """Print table of empirical priors."""
        print("\n" + "="*80)
        print(" EMPIRICAL PRIORS (from calibration data)")
        print("="*80 + "\n")

        print(f"{'Depth':<8} {'Strategy':<10} {'Samples':<10} "
              f"{'Recall':<10} {'Accuracy':<10} {'Latency':<10}")
        print("-" * 80)

        for (depth, strategy), stats in sorted(agg.get_all_cells().items()):
            if stats.get("insufficient_data"):
                print(f"{depth:<8} {strategy:<10} {'--':<10} {'--':<10} "
                      f"{'--':<10} {'--':<10}")
            else:
                recall = stats.get("recall_lift", 0)
                accuracy = stats.get("accuracy_estimate", 0)
                latency = stats.get("latency_p50_ms", 0)
                samples = stats.get("count", 0)
                print(f"{depth:<8} {strategy:<10} {samples:<10} "
                      f"{recall:<10.3f} {accuracy:<10.3f} {latency:<10.0f}")

    @staticmethod
    def print_regressions(
        agg: CalibrationAggregator,
        seed_priors: dict[int, dict[str, Any]],
    ):
        """Print regression analysis."""
        regressions = agg.regression_check(seed_priors, threshold=0.80)

        print("\n" + "="*80)
        print(" REGRESSION ANALYSIS (empirical vs seeds)")
        print("="*80 + "\n")

        if not regressions:
            print("✅ No regressions detected!")
        else:
            print(f"⚠️  {len(regressions)} cells underperformed (empirical < 80% of seed):\n")
            for depth, strategy, reason in regressions:
                print(f"  Depth {depth}, {strategy}: {reason}")


# Schema documentation for Eval's parser
CALIBRATION_TUPLE_SCHEMA = """
CALIBRATION TUPLE SCHEMA (for Eval's empirical priors derivation)

Source: rag_query_decisions rows WHERE is_calibration=true

Per-tuple fields:
  depth_bucket: int (0-4)        -- corpus-depth signal from Pool
  strategy_id: str              -- which strategy (a, b, c, d, f, s)
  success: bool                 -- did strategy meet confidence bar?
  confidence_achieved: float    -- actual confidence Observer measured
  accuracy_achieved: float      -- actual accuracy (if measured)
  latency_ms: int               -- strategy execution time
  cost: float                   -- strategy cost (relative units)
  query_id: str                 -- for tracing/debugging

Aggregation (per depth_bucket × strategy cell):
  COUNT: number of samples
  SUCCESS_RATE: successes / total
  RECALL_LIFT: mean(confidence_achieved)  -- empirical recall_lift
  ACCURACY_ESTIMATE: mean(accuracy_achieved)
  LATENCY_P50_MS: median(latency_ms)
  COST: mean(cost)

Empirical priors ready when:
  - All 30 cells have N≥50 samples
  - Coverage looks stable across depth/strategy
  - No major regressions from seeds

Fallback if sparse:
  - Use qclass-based grouping (not depth_bucket)
  - Or continue with seeds until Week 3 convergence
"""
