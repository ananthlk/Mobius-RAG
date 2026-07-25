"""BAYESIAN optimizer — third allocator (Ananth's directive 2026-07-23, additive).

Runs alongside greedy (allocation.py) and the Wilson-LB optimizer
(optimizer.py) — a THIRD option in the A/B/C comparison, not a replacement.
Purpose: produce real comparative data for the Observer-redesign question
(sequential Bayesian confidence) instead of deciding it on reasoning alone.

DESIGN: this module deliberately shares ALL machinery with optimizer.py —
subset enumeration, per-slot argmax, required/optional gating, semantics
eligibility, tracing — and differs in EXACTLY ONE thing: the per-strategy
lower-bound function is the Beta posterior quantile (priors.beta_lower_bound)
instead of the Wilson score bound. That isolation is intentional: any
divergence in the three-way comparison is attributable to the bound, nothing
else.

THE BOUND: Beta(alpha, beta) with alpha = recall_lift*n + 0.5,
beta = (1-recall_lift)*n + 0.5 (Jeffreys smoothing — required because several
seed cells have recall_lift exactly 0, where a raw Beta(0, n) is degenerate).
Lower bound = the (1 - confidence_level) posterior quantile, computed stdlib-
only (regularized incomplete beta via continued fraction + bisection).
Same (recall_lift, n) inputs as everything else — no new data.

Modeling choices flagged to Eval for ratification (their domain):
Jeffreys +0.5 smoothing; one-sided posterior quantile at the policy
confidence_level; conservative per-rung LB composition kept identical to the
Wilson allocators so the comparison stays clean.
"""

from __future__ import annotations

from typing import Any, Optional

from app.services.router.allocation import AnswerSlot, RoutingLadder
from app.services.router.optimizer import optimize_allocation
from app.services.router.priors import PriorsBundle, beta_lower_bound
from app.services.router.tracing import DecisionTrace


def optimize_allocation_bayesian(
    slots: list[AnswerSlot],
    pool_metadata: dict[str, dict],
    resource_posture: dict[str, Any],
    bundle: Optional[PriorsBundle] = None,
    trace: Optional[DecisionTrace] = None,
) -> RoutingLadder:
    """Beta-posterior-quantile allocation (see module docstring)."""
    return optimize_allocation(
        slots,
        pool_metadata,
        resource_posture,
        bundle=bundle,
        trace=trace,
        lb_fn=beta_lower_bound,
        allocator_name="bayesian",
    )
