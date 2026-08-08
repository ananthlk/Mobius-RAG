"""Portfolio allocator tests — blend-model design doc §3 (signed 2026-07-24).

All hand-derivations against the frozen fixture (conftest ROUTER_PRIORS_PATH).
"""

import asyncio
import math
from itertools import product

import pytest

from app.services.router.allocation import AnswerSlot
from app.services.router.portfolio import (
    K0_NOMINAL,
    PortfolioPlan,  # noqa: F401 — public shape
    _solve_knapsack,
    allocate_portfolio,
    coverage,
    q_from_recall,
)

POOL_DEPTH_2 = {"top_score_percentile": 0.60, "pool_size": 400}


def make_slot(slot_id="slot_0", capacity=10, required=True, priority=0,
              slot_semantics="direct_answer"):
    return AnswerSlot(slot_id=slot_id, slot_semantics=slot_semantics,
                      capacity=capacity, rewritten_query="q",
                      required=required, priority=priority)


def posture(**overrides):
    base = {
        "speed_budget": "background",  # generous latency so token budget binds
        "confidence_bar": 0.85,
        "caller_mode": "chat.default",
        "max_attempts_per_slot": 6,
        "gate_j_codes": ["payor.sunshine_health"],
        "gate_d_codes": ["claims.timely_filing"],
        "token_allowance_per_slot": 3000,
    }
    base.update(overrides)
    return base


class TestTransform:
    def test_recovers_cell_value_at_k0(self):
        """P(satisfied | k=k₀) must equal the cell's recall_lift exactly."""
        for r in (0.1, 0.286, 0.543, 0.8):
            q = q_from_recall(r)
            assert coverage([(q, K0_NOMINAL)]) == pytest.approx(r, abs=1e-9)

    def test_diminishing_returns_shape(self):
        """Coverage rises with k, concave (each chunk buys less)."""
        q = q_from_recall(0.543)
        covs = [coverage([(q, k)]) for k in range(1, 16)]
        gains = [b - a for a, b in zip(covs, covs[1:])]
        assert all(x < y for x, y in zip(covs, covs[1:]))         # increasing
        assert all(g2 < g1 for g1, g2 in zip(gains, gains[1:]))   # concave

    def test_r_clamp_stability(self):
        assert 0.0 < q_from_recall(1.0) < 1.0  # no ln(0) downstream
        assert q_from_recall(0.0) == 0.0


class TestKnapsackExactness:
    def test_dp_matches_brute_force(self):
        """Exact solve: DP allocation value equals brute-force optimum over
        all feasible {k_i} on realistic candidate sets."""
        candidates = [  # (sid, v_mean, v_lb, tokens, cap)
            ("s", 0.05, 0.03, 150, 5),
            ("a", 0.08, 0.04, 250, 5),
            ("d", 0.06, 0.025, 500, 5),
        ]
        for budget in (700, 1250, 3000):
            alloc = _solve_knapsack(candidates, budget)
            dp_val = sum(k * next(c[2] for c in candidates if c[0] == sid)
                         for sid, k in alloc.items())
            dp_cost = sum(k * next(c[3] for c in candidates if c[0] == sid)
                          for sid, k in alloc.items())
            assert dp_cost <= budget
            best = 0.0
            for ks in product(*(range(c[4] + 1) for c in candidates)):
                cost = sum(k * c[3] for k, c in zip(ks, candidates))
                if cost <= budget:
                    best = max(best, sum(k * c[2] for k, c in zip(ks, candidates)))
            assert dp_val == pytest.approx(best, abs=1e-9), budget

    def test_zero_budget_empty(self):
        assert _solve_knapsack([("a", 0.1, 0.05, 250, 5)], 0) == {}


class TestAllocation:
    def test_blend_emerges_multiple_strategies_contribute(self):
        """depth_2, capacity 10, budget 3000: portfolio spreads across
        strategies (the '3 from a, 2 from b' shape) rather than single-winner."""
        ladder = allocate_portfolio([make_slot()], {"slot_0": POOL_DEPTH_2},
                                    posture())
        alloc = ladder.per_slot_portfolio["slot_0"]
        assert len(alloc) >= 2, alloc  # genuine blend, not winner-takes-all
        assert ladder.total_payload_tokens <= 3000  # SUM model, by construction
        assert sum(alloc.values()) > 0

    def test_budget_binds_payload_by_construction(self):
        for budget in (700, 1500, 3000, 8000):
            ladder = allocate_portfolio(
                [make_slot()], {"slot_0": POOL_DEPTH_2},
                posture(token_allowance_per_slot=budget))
            assert ladder.per_slot_payload_tokens["slot_0"] <= budget

    def test_caps_respected(self):
        ladder = allocate_portfolio(
            [make_slot(capacity=3)], {"slot_0": POOL_DEPTH_2},
            posture(token_allowance_per_slot=8000))
        assert all(k <= 3 for k in ladder.per_slot_portfolio["slot_0"].values())

    def test_eligibility_gates_respected(self):
        # no payor tag → s never contributes (tag gate, fail-closed)
        ladder = allocate_portfolio(
            [make_slot()], {"slot_0": POOL_DEPTH_2},
            posture(gate_j_codes=[]))
        assert "s" not in ladder.per_slot_portfolio["slot_0"]
        # citable_required → d never contributes to a required slot
        ladder = allocate_portfolio(
            [make_slot()], {"slot_0": POOL_DEPTH_2},
            posture(authority_requirement="citable_required"))
        assert "d" not in ladder.per_slot_portfolio["slot_0"]

    def test_latency_gate_parallel_model(self):
        """real_time allowance 2300ms: d (3000ms p50) cannot contribute; the
        rest still blend (parallel model — MAX not sum)."""
        ladder = allocate_portfolio(
            [make_slot()], {"slot_0": POOL_DEPTH_2},
            posture(speed_budget="real_time"))
        alloc = ladder.per_slot_portfolio["slot_0"]
        assert "d" not in alloc
        assert ladder.per_slot_latency_ms["slot_0"] <= 2300

    def test_optional_slot_single_cheapest_chunk(self):
        """Eval's optional-cheap ruling carries over: one chunk, cheapest."""
        slots = [make_slot("core", priority=0),
                 make_slot("opt", priority=1, required=False)]
        pool = {"core": POOL_DEPTH_2, "opt": POOL_DEPTH_2}
        ladder = allocate_portfolio(slots, pool, posture())
        assert sum(ladder.per_slot_portfolio["opt"].values()) == 1
        assert ladder.per_slot_status["opt"] == "OPTIONAL"

    def test_lb_track_enforced_not_mean(self):
        """Status keys off LB coverage vs adjusted bar; mean is telemetry."""
        ladder = allocate_portfolio([make_slot()], {"slot_0": POOL_DEPTH_2},
                                    posture())
        assert ladder.per_slot_lb["slot_0"] < ladder.per_slot_confidence["slot_0"]
        expected_status = ("CLEARED" if ladder.per_slot_lb["slot_0"] >=
                           ladder.adjusted_confidence_bar else "UNDER_CONFIDENT")
        assert ladder.per_slot_status["slot_0"] == expected_status

    def test_no_slots_outcome(self):
        ladder = allocate_portfolio([], {}, posture())
        assert ladder.outcome == "no_slots"

    def test_scheduling_order_matches_allocation(self):
        ladder = allocate_portfolio([make_slot()], {"slot_0": POOL_DEPTH_2},
                                    posture())
        alloc = ladder.per_slot_portfolio["slot_0"]
        order = ladder.per_slot["slot_0"]
        assert set(order) == set(alloc)
        ks = [alloc[s] for s in order]
        assert ks == sorted(ks, reverse=True)  # k desc (scheduling hint)


class TestShadowWiring:
    """Portfolio rides as the FOURTH allocator, shadow-only at bootstrap."""

    def test_dispatch_never_executes_portfolio_by_default_legacy_weighted_draw(self):
        """With authority-conditioned routing OFF (legacy path, e.g. Eval's
        kill switch), portfolio stays shadow-only until allocator_weights
        says otherwise -- the original bootstrap invariant."""
        from app.services.router.dispatch import dispatch
        seen = set()
        for i in range(60):
            dd = dispatch(is_calibration=False, forced_strategy=None,
                          allocator_weights=None, query_key=f"q{i}",
                          authority_conditioned_routing=False)
            seen.add(dd.path)
            assert dd.path != "portfolio"
            assert "portfolio" in dd.shadow_allocators
        assert seen == {"greedy", "optimizer", "bayesian"}

    def test_authority_conditioned_routing_sends_any_to_portfolio_always(self):
        """Ananth's rule (2026-08-08, default ON): authority_requirement
        'any' or unset executes portfolio unconditionally -- this REPLACES
        the old shadow-only-by-default invariant above for real traffic."""
        from app.services.router.dispatch import dispatch
        for authority in (None, "any"):
            for i in range(10):
                dd = dispatch(is_calibration=False, forced_strategy=None,
                              query_key=f"q{i}", authority_requirement=authority)
                assert dd.path == "portfolio"

    def test_authority_conditioned_routing_citable_call_number_split(self):
        """Ananth's rule: citable_required + call_number<2 -> greedy;
        call_number>=2 -> portfolio (deviates from Eval-RAG's unconditional
        citable floor -- flagged, revisit with production data)."""
        from app.services.router.dispatch import dispatch
        dd1 = dispatch(is_calibration=False, forced_strategy=None,
                       query_key="q", authority_requirement="citable_required",
                       call_number=1)
        assert dd1.path == "greedy"
        dd_none = dispatch(is_calibration=False, forced_strategy=None,
                           query_key="q", authority_requirement="citable_required",
                           call_number=None)
        assert dd_none.path == "greedy"  # None fails closed to call 1
        dd2 = dispatch(is_calibration=False, forced_strategy=None,
                       query_key="q", authority_requirement="citable_required",
                       call_number=2)
        assert dd2.path == "portfolio"
        dd3 = dispatch(is_calibration=False, forced_strategy=None,
                       query_key="q", authority_requirement="citable_required",
                       call_number=3)
        assert dd3.path == "portfolio"

    def test_weights_file_can_shift_traffic_to_portfolio(self):
        """Eval's code-free cutover: weights including portfolio route real
        traffic to it."""
        from app.services.router.dispatch import dispatch
        dd = dispatch(is_calibration=False, forced_strategy=None,
                      allocator_weights={"portfolio": 1.0}, query_key="any")
        assert dd.path == "portfolio"

    def test_route_end_to_end_portfolio_shadow_persisted(self):
        """Full route(): portfolio plan appears in the persisted shadow list
        with its {strategy: k} allocation."""
        import json
        from app.services.router.decision import ResourcePosture, RoutingContext
        from app.services.router.router import route
        from app.services.router.test_router import FakeSessionFactory

        factory = FakeSessionFactory()
        decision = asyncio.run(route(factory, RoutingContext(
            query="portfolio shadow test", agent_id="router-4c",
            allocator_override="greedy",
            resource_posture=ResourcePosture(max_attempts_per_slot=6,
                                             token_budget=3000),
            pool_metadata={"slot_0": {
                "top_score_percentile": 0.60, "pool_size": 400,
                "required": True, "priority": 0,
                "slot_semantics": "direct_answer", "capacity": 10,
            }},
            gate_j_codes=["payor.sunshine_health"],
        )))
        shadow_names = [s.allocator for s in decision.shadow_ladders]
        assert "portfolio" in shadow_names
        params = factory.db.calls[0][1]
        plans = json.loads(params["shadow_ladder"])["plans"]
        port = next(p for p in plans if p["allocator"] == "portfolio")
        assert port["per_slot_portfolio"]["slot_0"]  # real {strategy: k}
        # PARTIAL-FILL model (2026-07-24): chain allocators now ALSO carry
        # per-rung fills in per_slot_portfolio (budget-scoped, SUM model)
        chain = next(p for p in plans if p["allocator"] != "portfolio")
        assert chain["per_slot_portfolio"]["slot_0"]


class TestExplorationPolicyParser:
    def test_portfolio_weight_accepted_from_file(self, tmp_path, monkeypatch):
        import textwrap
        from app.services.router.priors import load_priors
        f = tmp_path / "p.yaml"
        f.write_text(textwrap.dedent("""
            seed_priors:
              depth_2:
                a: {recall_lift: 0.5, latency_p50_ms: 500, cost_per_attempt: 1,
                    accuracy_estimate: 0.5}
            exploration_policy:
              allocator_weights: {greedy: 0.25, optimizer: 0.25, bayesian: 0.25,
                                  portfolio: 0.25}
        """))
        bundle = load_priors(path=f, force_reload=True)
        w = bundle.exploration_policy["allocator_weights"]
        assert w["portfolio"] == pytest.approx(0.25)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestPerCellK0:
    """Eval's audit (2026-07-24): k₀ is per-(depth,strategy), NOT global —
    c/s calibrate at occupancy ~1 while a/b/d run ~10. The file carries k0
    per cell; the transform must consume it per cell."""

    def test_k0_parsed_from_file_and_defaults(self, tmp_path):
        import textwrap
        from app.services.router.priors import K0_NOMINAL, load_priors
        f = tmp_path / "p.yaml"
        f.write_text(textwrap.dedent("""
            seed_priors:
              depth_2:
                a: {recall_lift: 0.5, latency_p50_ms: 500, cost_per_attempt: 1,
                    accuracy_estimate: 0.5, n: 30, k0: 10}
                s: {recall_lift: 0.5, latency_p50_ms: 100, cost_per_attempt: 0,
                    accuracy_estimate: 0.5, n: 30, k0: 1}
                b: {recall_lift: 0.5, latency_p50_ms: 1500, cost_per_attempt: 1,
                    accuracy_estimate: 0.5}
        """))
        bundle = load_priors(path=f, force_reload=True)
        assert bundle.by_depth["a"][2].k0 == 10
        assert bundle.by_depth["s"][2].k0 == 1
        assert bundle.by_depth["b"][2].k0 == K0_NOMINAL  # absent → default

    def test_per_cell_k0_changes_transform(self):
        """Same recall_lift, different k₀ → very different per-chunk q:
        r=0.5 at k₀=1 means ONE chunk carries 0.5 (q=0.5); at k₀=10 the same
        r spreads over ten (q≈0.067). Misrebasing c/s was exactly this gap."""
        assert q_from_recall(0.5, 1) == pytest.approx(0.5)
        assert q_from_recall(0.5, 10) == pytest.approx(1 - 0.5 ** 0.1)
        assert q_from_recall(0.5, 1) > 5 * q_from_recall(0.5, 10)

    def test_trust_window_caps_allocation_at_2k0(self, tmp_path):
        """Eval's contingency made literal: a k₀=1 cell can never be allocated
        beyond k=2 (2·k₀), no matter the slot capacity or budget — the
        extrapolation the trust window forbids is impossible by construction."""
        import textwrap
        from app.services.router.priors import load_priors
        f = tmp_path / "p.yaml"
        f.write_text(textwrap.dedent("""
            seed_priors:
              depth_2:
                a: {recall_lift: 0.5, latency_p50_ms: 500, cost_per_attempt: 1,
                    accuracy_estimate: 0.5, n: 30, k0: 1}
        """))
        bundle = load_priors(path=f, force_reload=True)
        ladder = allocate_portfolio(
            [make_slot(capacity=10)], {"slot_0": POOL_DEPTH_2},
            posture(token_allowance_per_slot=8000), bundle=bundle)
        alloc = ladder.per_slot_portfolio["slot_0"]
        assert alloc.get("a", 0) <= 2  # 2·k₀, not capacity 10
