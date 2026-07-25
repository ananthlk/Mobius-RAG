"""Structured decision tracing — every Router decision is replayable by hand.

The trace is the single source of truth for BOTH:
  - the structured telemetry emit (to_dict(), queryable/analyzable), and
  - the human-readable narration (router_narrate.narrate() renders from it).

Replayability guarantee: the trace records the depth-bucket inputs, every prior
actually looked up (the values, not just the tier), and the running confidence
arithmetic at each step — someone can recompute the exact decision from the
emitted data alone. test_tracing.py enforces this by literally recomputing.

PHI rule (same as Gate/Reformat): traces carry slot ids and numbers, never raw
query text. Narration is computed on demand and NEVER persisted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StrategyStep:
    """One strategy considered for a slot — taken or skipped, with the arithmetic."""
    strategy_id: str
    action: str                      # "added" | "skipped"
    skip_reason: str = ""            # for skipped: "zero_lift" | "over_latency_allowance" | "no_prior" | ...
    # Prior actually used (real values, so the step is replayable)
    prior_source: str = ""           # "depth_bucket" | "qclass_fallback"
    recall_lift: float = 0.0         # MEAN
    n: int = 0                       # sample count behind the prior
    lb_lift: float = 0.0             # Wilson lower bound of recall_lift at the policy level
    latency_p50_ms: int = 0
    cost: float = 0.0
    accuracy_estimate: float = 0.0
    # Running arithmetic (added steps only) — mean track AND lower-bound track
    conf_before: float = 0.0
    added_term: float = 0.0          # (1 - conf_before) * recall_lift
    conf_after: float = 0.0
    lb_before: float = 0.0
    lb_after: float = 0.0
    latency_before_ms: int = 0
    latency_after_ms: int = 0

    def arithmetic(self) -> str:
        """The cumulative-confidence calculation as a human-checkable string."""
        if self.action != "added":
            return ""
        return (
            f"{self.conf_before:.4f} + (1-{self.conf_before:.4f})*{self.recall_lift:.3f}"
            f" = {self.conf_after:.4f}"
        )


@dataclass
class SlotTrace:
    """Full reasoning trace for one slot."""
    slot_id: str
    priority: int
    required: bool = True            # False: supplementary slot — reported, never gated
    slot_semantics: str = "direct_answer"  # role — gates strategy eligibility
    # depth-bucket computation, inputs included
    pool_size: Any = None
    top_score_percentile: Any = None
    distinct_content_topk: Any = None  # diversity signal behind the bucket (may demote it)
    depth_bucket: int = 4
    phase: str = ""                  # "phase1" steps then possibly "topup" steps
    steps: list[StrategyStep] = field(default_factory=list)
    stop_reason: str = ""            # "bar_cleared" | "attempts_exhausted" | "budget_exhausted"
                                     # | "strategies_exhausted" | "solver_selected"
    final_chain: list[str] = field(default_factory=list)
    final_confidence: float = 0.0    # MEAN chain confidence (telemetry)
    final_lb: float = 0.0            # LOWER-BOUND chain confidence (the enforced quantity)
    final_latency_ms: int = 0
    payload_tokens_worst_case: Optional[int] = None  # max-rung slot payload (capacity × per-chunk)
    bar_cleared: bool = False        # final_lb vs adjusted bar (per-slot enforcement)
    status: str = ""                 # "CLEARED" | "UNDER_CONFIDENT" | "NO_VIABLE_STRATEGY" | "OPTIONAL"
    terminal_action: Optional[str] = None  # verdict-driven final leg: "q" | "e" | None (no priors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "priority": self.priority,
            "required": self.required,
            "slot_semantics": self.slot_semantics,
            "pool_size": self.pool_size,
            "top_score_percentile": self.top_score_percentile,
            "distinct_content_topk": self.distinct_content_topk,
            "depth_bucket": self.depth_bucket,
            "steps": [vars(s) | {"arithmetic": s.arithmetic()} for s in self.steps],
            "stop_reason": self.stop_reason,
            "final_chain": self.final_chain,
            "final_confidence": self.final_confidence,
            "final_lb": self.final_lb,
            "final_latency_ms": self.final_latency_ms,
            "payload_tokens_worst_case": self.payload_tokens_worst_case,
            "bar_cleared": self.bar_cleared,
            "status": self.status,
            "terminal_action": self.terminal_action,
        }


@dataclass
class DecisionTrace:
    """Full reasoning trace for one allocator's plan on one query.

    In production BOTH allocators produce a trace: one with role='executed'
    (its ladder is walked by Fillers) and one with role='shadow' (plan only,
    comparison against the executed one — never run)."""
    mode: str = ""                   # "forced" | "greedy" | "optimizer"
    role: str = "executed"           # "executed" | "shadow"
    mode_reason: str = ""            # incl. A/B draw arithmetic
    ab_split: Optional[float] = None
    draw: Optional[float] = None     # deterministic hash-based draw in [0,1)
    priors_version: str = ""
    priors_source: str = ""
    caller_mode: str = ""
    tolerance_pct: float = 0.0
    confidence_bar: float = 0.0
    adjusted_confidence_bar: float = 0.0
    speed_budget_ms: int = 0
    latency_allowance_ms: float = 0.0
    confidence_level: float = 0.95   # one-sided Wilson LB level in force
    slots: list[SlotTrace] = field(default_factory=list)
    aggregate_confidence: float = 0.0  # mean over slots — TELEMETRY ONLY, never a gate
    aggregate_arithmetic: str = ""
    outcome: str = ""                # "all_slots_cleared" | "partial_infeasible" | "no_slots" | "forced"
    helpers: list = field(default_factory=list)  # recall-failure helper plan (clarify/sitemap)
    feasible: bool = False
    infeasibility_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "role": self.role,
            "mode_reason": self.mode_reason,
            "ab_split": self.ab_split,
            "draw": self.draw,
            "priors_version": self.priors_version,
            "priors_source": self.priors_source,
            "caller_mode": self.caller_mode,
            "tolerance_pct": self.tolerance_pct,
            "confidence_bar": self.confidence_bar,
            "adjusted_confidence_bar": self.adjusted_confidence_bar,
            "speed_budget_ms": self.speed_budget_ms,
            "latency_allowance_ms": self.latency_allowance_ms,
            "slots": [s.to_dict() for s in self.slots],
            "confidence_level": self.confidence_level,
            "aggregate_confidence": self.aggregate_confidence,
            "aggregate_arithmetic": self.aggregate_arithmetic,
            "outcome": self.outcome,
            "helpers": self.helpers,
            "feasible": self.feasible,
            "infeasibility_reason": self.infeasibility_reason,
        }
