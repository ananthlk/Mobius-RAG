"""Bayesian optimizer tests — the Beta math grounded independently, then the allocator.

The incomplete-beta implementation is verified against CLOSED-FORM identities
(not against itself): I_x(1,1)=x, I_x(a,1)=x^a, I_x(1,b)=1-(1-x)^b, symmetry
I_x(a,b)=1-I_{1-x}(b,a), and the Beta(2,2) median. The quantile then only
needs the bisection property CDF(quantile)=target.
"""

import pytest

from app.services.router.allocation import AnswerSlot
from app.services.router.bayesian_optimizer import optimize_allocation_bayesian
from app.services.router.optimizer import optimize_allocation
from app.services.router.priors import (
    PriorsBundle,
    StrategyProfile,
    beta_lower_bound,
    regularized_incomplete_beta,
    wilson_lower_bound,
)


POOL_DEPTH_2 = {"top_score_percentile": 0.60, "pool_size": 400}
POOL_DEPTH_3 = {"top_score_percentile": 0.30, "pool_size": 2000}


def _slot(slot_id="slot_0", priority=0, required=True,
          slot_semantics="direct_answer"):
    """REAL AnswerSlot (shape/slots.py). max_attempts is query-level (posture)."""
    return AnswerSlot(slot_id=slot_id, slot_semantics=slot_semantics, capacity=5,
                      rewritten_query="q", required=required, priority=priority)


def _posture(**overrides):
    base = {
        "speed_budget": "interactive",
        "confidence_bar": 0.85,
        "caller_mode": "chat.default",
        "max_attempts_per_slot": 6,
        # tests simulate PAYOR queries so tag-gated 's' stays eligible
        "gate_j_codes": ["payor.sunshine_health"],
    }
    base.update(overrides)
    return base


class TestIncompleteBetaAgainstClosedForms:
    @pytest.mark.parametrize("x", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    def test_uniform_identity(self, x):
        """I_x(1,1) = x (Beta(1,1) is uniform)."""
        assert regularized_incomplete_beta(1, 1, x) == pytest.approx(x, abs=1e-10)

    @pytest.mark.parametrize("a,x", [(2, 0.3), (3, 0.5), (5, 0.8), (2.5, 0.4)])
    def test_power_identity(self, a, x):
        """I_x(a,1) = x^a."""
        assert regularized_incomplete_beta(a, 1, x) == pytest.approx(x ** a, abs=1e-10)

    @pytest.mark.parametrize("b,x", [(2, 0.3), (4, 0.6), (3.5, 0.2)])
    def test_complement_power_identity(self, b, x):
        """I_x(1,b) = 1-(1-x)^b."""
        assert regularized_incomplete_beta(1, b, x) == pytest.approx(
            1 - (1 - x) ** b, abs=1e-10)

    @pytest.mark.parametrize("a,b,x", [(2, 3, 0.4), (4.34, 3.66, 0.55), (0.5, 8.5, 0.05)])
    def test_symmetry_identity(self, a, b, x):
        """I_x(a,b) = 1 - I_{1-x}(b,a)."""
        assert regularized_incomplete_beta(a, b, x) == pytest.approx(
            1 - regularized_incomplete_beta(b, a, 1 - x), abs=1e-9)

    def test_beta_2_2_median(self):
        assert regularized_incomplete_beta(2, 2, 0.5) == pytest.approx(0.5, abs=1e-10)


class TestBetaLowerBound:
    def test_quantile_property(self):
        """CDF(lower_bound) must equal the lower-tail mass (1 - level)."""
        for p, n, level in [(0.543, 8, 0.95), (0.15, 8, 0.95), (0.8, 200, 0.99)]:
            lb = beta_lower_bound(p, n, level)
            a = p * n + 0.5
            b = (1 - p) * n + 0.5
            assert regularized_incomplete_beta(a, b, lb) == pytest.approx(
                1 - level, abs=1e-9)

    def test_monotone_in_n(self):
        """More data → tighter bound → higher LB at fixed p."""
        assert beta_lower_bound(0.5, 200) > beta_lower_bound(0.5, 8) > beta_lower_bound(0.5, 2)

    def test_monotone_in_p(self):
        assert beta_lower_bound(0.8, 8) > beta_lower_bound(0.5, 8) > beta_lower_bound(0.2, 8)

    def test_zero_lift_cell_does_not_degenerate(self):
        """p̂=0 cells (c at depth_2/4) work via Jeffreys smoothing — a small
        positive bound region exists rather than a crash/degenerate 0/0."""
        lb = beta_lower_bound(0.0, 8)
        assert 0.0 <= lb < 0.05  # tiny but well-defined

    def test_n_zero_returns_zero(self):
        assert beta_lower_bound(0.5, 0) == 0.0

    def test_bounds_are_in_unit_interval_and_below_mean(self):
        for p in (0.1, 0.35, 0.543, 0.8):
            lb = beta_lower_bound(p, 8)
            assert 0.0 <= lb < p  # a 95% lower bound sits below the mean

    def test_beta_and_wilson_agree_asymptotically_differ_at_small_n(self):
        """At large n the two bounds converge; at n=8 they measurably differ —
        that difference IS the three-way experiment."""
        p = 0.543
        assert abs(beta_lower_bound(p, 5000) - wilson_lower_bound(p, 5000)) < 0.005
        assert abs(beta_lower_bound(p, 8) - wilson_lower_bound(p, 8)) > 0.005


class TestBayesianAllocator:
    def test_allocator_tag_and_determinism(self):
        l1 = optimize_allocation_bayesian([_slot()], {"slot_0": POOL_DEPTH_2}, _posture())
        l2 = optimize_allocation_bayesian([_slot()], {"slot_0": POOL_DEPTH_2}, _posture())
        assert l1.allocator == "bayesian"
        assert l1.per_slot == l2.per_slot
        assert l1.per_slot_lb == l2.per_slot_lb

    def test_differs_from_wilson_optimizer_only_in_bound(self):
        """Same inputs: the two optimizers may pick the same chain, but their
        LB numbers must come from different bound functions (measurably
        different at n=8) while the MEAN track is identical for equal chains."""
        w = optimize_allocation([_slot()], {"slot_0": POOL_DEPTH_2}, _posture())
        b = optimize_allocation_bayesian([_slot()], {"slot_0": POOL_DEPTH_2}, _posture())
        assert w.allocator == "optimizer" and b.allocator == "bayesian"
        assert b.per_slot_lb["slot_0"] != pytest.approx(w.per_slot_lb["slot_0"], abs=1e-4)
        if b.per_slot["slot_0"] == w.per_slot["slot_0"]:
            assert b.per_slot_confidence["slot_0"] == pytest.approx(
                w.per_slot_confidence["slot_0"], abs=1e-9)

    def test_inherits_per_question_gating_and_eligibility(self):
        """The shared machinery carries §2a + semantics gating unchanged."""
        slots = [
            _slot("question", priority=0),
            _slot("external_context", priority=1, required=False,
                  slot_semantics="external_context"),
        ]
        pool = {"question": POOL_DEPTH_3, "external_context": POOL_DEPTH_2}
        ladder = optimize_allocation_bayesian(
            slots, pool,
            _posture(caller_mode="batch", speed_budget="background",
                     confidence_bar=0.70),
        )
        assert ladder.per_slot["external_context"] == ["d"]  # eligibility + optional-cap
        assert ladder.per_slot_status["external_context"] == "OPTIONAL"
        assert ladder.per_slot_status["question"] in ("CLEARED", "UNDER_CONFIDENT")
        assert ladder.outcome in ("all_slots_cleared", "partial_infeasible")

    def test_trace_lb_track_uses_beta_bound(self):
        from app.services.router.tracing import DecisionTrace
        trace = DecisionTrace()
        optimize_allocation_bayesian([_slot()], {"slot_0": POOL_DEPTH_2}, _posture(),
                                     trace=trace)
        added = [s for s in trace.slots[0].steps if s.action == "added"]
        for step in added:
            expected = beta_lower_bound(step.recall_lift, step.n, 0.95)
            assert step.lb_lift == pytest.approx(expected, abs=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
