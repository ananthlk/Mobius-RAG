"""Optimizer allocator tests — the exact constrained solve + shadow A/B contract.

YAML depth values used in expectations:
  depth_1: s .350/100ms/0c, a .350/500/1, b .175/1500/1, c .300/2000/2, d .330/3000/3, f .280/2500/2
  depth_2: s .500/100/0,   a .543/500/1, b .286/1500/1, c .330/2000/2, d .450/3000/3, f .380/2500/2
"""

import textwrap

import pytest

from app.services.router.allocation import AnswerSlot, allocate_strategies
from app.services.router.optimizer import optimize_allocation
from app.services.router.priors import PriorsBundle, StrategyProfile, load_priors
from app.services.router.tracing import DecisionTrace


POOL_DEPTH_0 = {"top_score_percentile": 0.95, "pool_size": 30}
POOL_DEPTH_1 = {"top_score_percentile": 0.80, "pool_size": 150}
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


class TestSolverCorrectness:
    def test_single_slot_maximizes_lb_within_time_budget(self):
        """LB-MAXIMIZER contract (v3): spend the budget maximizing the Wilson
        LB. depth_2 LBs (n=8): s .2486, a .2815, b .1065, c .1327, d .2122.
        Best-LB subset fitting 5750ms is {s,a,c,d} = 5600ms:
          lb = 1-.7514*.7185*.8673*.7878 = .6311 ; mean = .9158.
        .6311 < adjusted bar .7225 → honest UNDER_CONFIDENT verdict (the old
        mean gate would have called this feasible)."""
        ladder = optimize_allocation([_slot()], {"slot_0": POOL_DEPTH_2}, _posture())
        assert ladder.per_slot["slot_0"] == ["s", "a", "c", "d"]
        assert ladder.per_slot_lb["slot_0"] == pytest.approx(0.6311, abs=1e-3)
        assert ladder.per_slot_confidence["slot_0"] == pytest.approx(0.9158, abs=1e-3)
        assert ladder.per_slot_latency_ms["slot_0"] == 5600  # within 5750 allowance
        assert ladder.per_slot_status["slot_0"] == "UNDER_CONFIDENT"
        assert ladder.outcome == "partial_infeasible"
        assert ladder.feasible is False

    def test_optimizer_confidence_never_below_greedy(self):
        """Maximizer property: greedy's chain is one of the enumerated subsets
        under identical constraints, so optimizer's aggregate confidence must be
        >= greedy's on the same inputs — always, feasible or not."""
        scenarios = [
            ({"slot_0": POOL_DEPTH_1}, _posture(confidence_bar=0.70)),
            ({"slot_0": POOL_DEPTH_2}, _posture()),
            ({"slot_0": POOL_DEPTH_3}, _posture(caller_mode="batch")),
            ({"slot_0": POOL_DEPTH_0}, _posture(speed_budget="real_time")),
            ({"a": POOL_DEPTH_1, "b": POOL_DEPTH_3},
             _posture(caller_mode="batch", confidence_bar=0.75)),
        ]
        for pool, posture in scenarios:
            slots = [_slot(sid, priority=i) for i, sid in enumerate(pool)]
            g = allocate_strategies(slots, pool, posture)
            o = optimize_allocation(slots, pool, posture)
            assert o.aggregate_confidence_estimate >= g.aggregate_confidence_estimate - 1e-9, \
                f"optimizer conf {o.aggregate_confidence_estimate} < greedy " \
                f"{g.aggregate_confidence_estimate} on {pool}"

    def test_lb_changes_the_answer_vs_mean_only(self):
        """THE uncertainty test (Eval's re-verify ask): the LB must actually
        change strategy selection, not just rescale numbers.

        One attempt allowed. Strategy X: mean .60 but n=4 (wide) → lb95 ≈ .246.
        Strategy Y: mean .55 but n=200 (tight) → lb95 ≈ .492.
        A mean-only allocator picks X (.60 > .55). The LB allocator must pick
        Y — the tighter-spread option is the safer bet (Ananth's exact ask:
        equal-ish means stop tying; spread decides)."""
        from app.services.router.priors import wilson_lower_bound
        bundle = PriorsBundle(
            by_depth={
                "a": {2: StrategyProfile(0.60, 500, 1, 0.5, n=4)},    # X: wide
                "b": {2: StrategyProfile(0.55, 1500, 1, 0.5, n=200)}, # Y: tight
            },
            version="test", source="test",
        )
        # sanity on the hand-computed bounds
        assert wilson_lower_bound(0.60, 4) == pytest.approx(0.246, abs=2e-3)
        assert wilson_lower_bound(0.55, 200) == pytest.approx(0.492, abs=2e-3)

        ladder = optimize_allocation(
            [_slot()], {"slot_0": POOL_DEPTH_2},
            _posture(confidence_bar=0.99, max_attempts_per_slot=1), bundle=bundle,
        )
        assert ladder.per_slot["slot_0"] == ["b"], \
            "LB allocator must prefer tight-n Y over wide-n X despite lower mean"
        # and the mean-only comparison would have gone the other way:
        assert 0.60 > 0.55

    def test_cmhc013_case_no_satisficing(self):
        """The exact bug Ananth/Eval caught: depth_3 slot, equal-cost options
        {a} (conf .80, 500ms) vs {s,a} (conf .92, 600ms) with ~5750ms allowance.
        Old objective picked {a} to save 100ms nobody needed. Maximizer must
        include BOTH s and a (plus whatever else fits) — conf >= .92."""
        ladder = optimize_allocation([_slot()], {"slot_0": POOL_DEPTH_3}, _posture())
        chain = ladder.per_slot["slot_0"]
        assert "s" in chain and "a" in chain
        assert ladder.per_slot_confidence["slot_0"] >= 0.92 - 1e-9
        assert ladder.per_slot_latency_ms["slot_0"] <= 5750

    def test_every_viable_slot_gets_nonempty_chain(self):
        """Optimizer may not silently drop a slot to save cost — every slot with
        viable strategies gets >= 1 rung even when the aggregate already clears."""
        slots = [_slot("easy", priority=0), _slot("also_easy", priority=1)]
        pool = {"easy": POOL_DEPTH_3, "also_easy": POOL_DEPTH_3}
        ladder = optimize_allocation(slots, pool,
                                     _posture(caller_mode="batch", confidence_bar=0.40))
        assert len(ladder.per_slot["easy"]) >= 1
        assert len(ladder.per_slot["also_easy"]) >= 1

    def test_respects_max_attempts(self):
        ladder = optimize_allocation([_slot()], {"slot_0": POOL_DEPTH_2},
                                     _posture(confidence_bar=0.99, max_attempts_per_slot=1))
        assert len(ladder.per_slot["slot_0"]) == 1
        assert ladder.feasible is False

    def test_respects_per_slot_latency_allowance(self):
        """real_time chat.default → 2300ms chain allowance: subsets summing over
        2300ms are excluded; chosen chain must fit."""
        ladder = optimize_allocation([_slot()], {"slot_0": POOL_DEPTH_2},
                                     _posture(speed_budget="real_time", confidence_bar=0.99))
        assert ladder.per_slot_latency_ms["slot_0"] <= 2300

    def test_infeasible_returns_best_effort_with_reason(self):
        """depth_0, bar unreachable: best-effort assignment + explicit verdict."""
        ladder = optimize_allocation([_slot()], {"slot_0": POOL_DEPTH_0},
                                     _posture(caller_mode="batch",
                                              speed_budget="background",
                                              confidence_bar=0.70))
        assert ladder.feasible is False
        assert ladder.outcome == "partial_infeasible"
        assert ladder.per_slot_status["slot_0"] == "UNDER_CONFIDENT"
        assert "LB bar" in ladder.infeasibility_reason
        assert len(ladder.per_slot["slot_0"]) >= 1  # still returns a plan

    def test_zero_slots_flagged(self):
        ladder = optimize_allocation([], {}, _posture())
        assert ladder.feasible is False
        assert ladder.infeasibility_reason == "no slots to allocate"

    def test_deterministic(self):
        l1 = optimize_allocation([_slot()], {"slot_0": POOL_DEPTH_2}, _posture())
        l2 = optimize_allocation([_slot()], {"slot_0": POOL_DEPTH_2}, _posture())
        assert l1.per_slot == l2.per_slot
        assert l1.total_estimated_cost == l2.total_estimated_cost

    def test_per_slot_verdicts_hard_and_easy_slot(self):
        """§2a: no cross-slot compensation. hard (depth_0) can't clear its LB
        bar even with every strategy → UNDER_CONFIDENT; easy (depth_3) clears
        on its own → CLEARED. Decision outcome = partial_infeasible, both
        slots still get non-empty best-effort chains."""
        slots = [_slot("hard", priority=0), _slot("easy", priority=1)]
        pool = {"hard": POOL_DEPTH_0, "easy": POOL_DEPTH_3}
        ladder = optimize_allocation(
            slots, pool,
            _posture(caller_mode="batch", speed_budget="background",
                     confidence_bar=0.70),
        )
        assert ladder.per_slot_status["hard"] == "UNDER_CONFIDENT"
        assert ladder.per_slot_status["easy"] == "CLEARED"
        assert ladder.outcome == "partial_infeasible"
        assert ladder.feasible is False
        assert len(ladder.per_slot["hard"]) >= 1
        assert len(ladder.per_slot["easy"]) >= 1


class TestPerQuestionEnforcementOptimizer:
    def test_under_confident_optional_slot_does_not_flip_outcome(self):
        """Same §2a per-question rule in the optimizer: hopeless optional slot
        (required=False) reports OPTIONAL, never gates; outcome tracks the
        required slot only."""
        slots = [
            _slot("question", priority=0),
            _slot("external_context", priority=1, required=False),
        ]
        pool = {"question": POOL_DEPTH_3, "external_context": POOL_DEPTH_0}
        ladder = optimize_allocation(
            slots, pool,
            _posture(caller_mode="batch", speed_budget="background",
                     confidence_bar=0.70),
        )
        assert ladder.per_slot_status["question"] == "CLEARED"
        assert ladder.per_slot_status["external_context"] == "OPTIONAL"
        assert len(ladder.per_slot["external_context"]) == 1  # single-attempt cap
        assert ladder.outcome == "all_slots_cleared"
        assert ladder.feasible is True

    def test_optional_slot_cheapest_rung_in_optimizer(self):
        """Eval's ruling: optional slots select CHEAPEST-FIRST (cost, latency),
        not best-LB — a telemetry-only slot must not pay wall-clock for a
        number nobody gates. At any depth the cheapest viable rung is 's'
        (cost 0, 100ms) — converging with greedy's natural behavior."""
        slots = [
            _slot("question", priority=0),
            _slot("external_context", priority=1, required=False),
        ]
        pool = {"question": POOL_DEPTH_3, "external_context": POOL_DEPTH_2}
        ladder = optimize_allocation(
            slots, pool,
            _posture(caller_mode="batch", speed_budget="background",
                     confidence_bar=0.70),
        )
        chain = ladder.per_slot["external_context"]
        assert chain == ["s"]  # cheapest, NOT the best-LB rung ('a')
        assert ladder.per_slot_latency_ms["external_context"] == 100
        # optional slot can no longer be the query's latency max
        assert ladder.total_estimated_ms == ladder.per_slot_latency_ms["question"]


class TestEligibilityOptimizer:
    def test_external_context_slot_only_considers_d(self):
        """Eligibility gate in the optimizer: external_context enumerates
        subsets over {d} only — required OR optional."""
        slots = [_slot("ext", slot_semantics="external_context")]
        ladder = optimize_allocation(
            slots, {"ext": POOL_DEPTH_2},
            _posture(caller_mode="batch", speed_budget="background",
                     confidence_bar=0.70),
        )
        assert ladder.per_slot["ext"] == ["d"]

        slots_opt = [
            _slot("question", priority=0),
            _slot("external_context", priority=1, required=False,
                  slot_semantics="external_context"),
        ]
        pool = {"question": POOL_DEPTH_3, "external_context": POOL_DEPTH_0}
        ladder2 = optimize_allocation(
            slots_opt, pool,
            _posture(caller_mode="batch", speed_budget="background",
                     confidence_bar=0.70),
        )
        assert ladder2.per_slot["external_context"] == ["d"]  # only eligible rung
        assert ladder2.outcome == "all_slots_cleared"


class TestOptimizerTrace:
    def test_trace_steps_replayable(self):
        trace = DecisionTrace()
        optimize_allocation([_slot()], {"slot_0": POOL_DEPTH_2}, _posture(), trace=trace)
        st = trace.slots[0]
        added = [s for s in st.steps if s.action == "added"]
        prev = 0.0
        for step in added:
            assert step.conf_after == pytest.approx(prev + (1 - prev) * step.recall_lift, abs=1e-9)
            prev = step.conf_after
        assert st.final_confidence == pytest.approx(prev, abs=1e-9)
        assert st.stop_reason == "solver_selected"

    def test_trace_reports_solver_stats(self):
        trace = DecisionTrace()
        optimize_allocation([_slot()], {"slot_0": POOL_DEPTH_2}, _posture(), trace=trace)
        assert "subset options" in trace.mode_reason


class TestAllocatorWeightsPolicy:
    def test_allocator_weights_parsed_and_normalized(self, tmp_path, monkeypatch):
        yaml_path = tmp_path / "priors.yaml"
        yaml_path.write_text(textwrap.dedent("""
            exploration_policy:
              allocator_weights: {greedy: 2, optimizer: 1, bayesian: 1}
              phase: bootstrap
            seed_priors:
              depth_2:
                s: {recall_lift: 0.5, latency_p50_ms: 100, cost_per_attempt: 0, accuracy_estimate: 0.5}
        """))
        monkeypatch.setenv("ROUTER_PRIORS_PATH", str(yaml_path))
        w = load_priors().exploration_policy["allocator_weights"]
        assert w == {"greedy": 0.5, "optimizer": 0.25, "bayesian": 0.25}

    def test_missing_policy_block_defaults_to_equal_thirds(self):
        bundle = load_priors(force_reload=True)  # real eval file has no block yet
        w = bundle.exploration_policy["allocator_weights"]
        assert w["greedy"] == pytest.approx(1 / 3)
        assert w["optimizer"] == pytest.approx(1 / 3)
        assert w["bayesian"] == pytest.approx(1 / 3)

    def test_legacy_ab_split_maps_to_two_way(self, tmp_path, monkeypatch):
        yaml_path = tmp_path / "priors.yaml"
        yaml_path.write_text(textwrap.dedent("""
            exploration_policy:
              ab_split_optimizer: 0.25
            seed_priors:
              depth_2:
                s: {recall_lift: 0.5, latency_p50_ms: 100, cost_per_attempt: 0, accuracy_estimate: 0.5}
        """))
        monkeypatch.setenv("ROUTER_PRIORS_PATH", str(yaml_path))
        w = load_priors().exploration_policy["allocator_weights"]
        assert w == {"greedy": 0.75, "optimizer": 0.25}  # no bayesian traffic

    def test_unknown_allocator_names_ignored(self, tmp_path, monkeypatch):
        yaml_path = tmp_path / "priors.yaml"
        yaml_path.write_text(textwrap.dedent("""
            exploration_policy:
              allocator_weights: {greedy: 1, quantum: 5}
            seed_priors:
              depth_2:
                s: {recall_lift: 0.5, latency_p50_ms: 100, cost_per_attempt: 0, accuracy_estimate: 0.5}
        """))
        monkeypatch.setenv("ROUTER_PRIORS_PATH", str(yaml_path))
        w = load_priors().exploration_policy["allocator_weights"]
        assert w == {"greedy": 1.0}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
