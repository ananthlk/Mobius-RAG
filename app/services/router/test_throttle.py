"""Data-collection throttle tests (Ananth's pullback, 2026-07-24).

Prod policy: 1-in-5 forced / 4-in-5 router models; offline matrix = 100%
caller-forced (existing path, untouched). Arms {a,b,c,d,s} only; s payor-
gated; prod arm set configurable (c exclusion = config flip).
"""

import asyncio
import json
import textwrap

import pytest

from app.services.router.dispatch import (
    THROTTLE_ARM_ROSTER,
    dispatch,
)


def _dd(query, **kw):
    base = dict(is_calibration=False, forced_strategy=None,
                query_key=query, phase="data_collection",
                forced_fraction=0.2, has_payor_context=True)
    base.update(kw)
    return dispatch(**base)


class TestThrottleDraw:
    def test_fraction_honored_deterministically(self):
        """~20% of distinct queries land on forced arms; same query always
        lands the same way (stateless stable_draw, replayable)."""
        forced = sum(1 for i in range(1000)
                     if _dd(f"query {i}").bypass_kind == "data_collection_throttle")
        assert 150 <= forced <= 250  # ~200 expected; deterministic, not flaky
        d1, d2 = _dd("same query"), _dd("same query")
        assert (d1.path, d1.forced_strategy) == (d2.path, d2.forced_strategy)

    def test_bootstrap_phase_never_throttles(self):
        for i in range(200):
            dd = dispatch(is_calibration=False, forced_strategy=None,
                          query_key=f"q{i}")  # default phase=bootstrap
            assert dd.bypass_kind != "data_collection_throttle"

    def test_arm_coverage_over_roster(self):
        arms = {_dd(f"q{i}").forced_strategy
                for i in range(2000)
                if _dd(f"q{i}").bypass_kind == "data_collection_throttle"}
        assert arms == set(THROTTLE_ARM_ROSTER)  # every arm gets traffic

    def test_s_excluded_without_payor_context(self):
        for i in range(2000):
            dd = _dd(f"q{i}", has_payor_context=False)
            if dd.bypass_kind == "data_collection_throttle":
                assert dd.forced_strategy != "s"

    def test_configurable_arm_exclusion_is_config_flip(self):
        """Ananth's pending forced-c-on-live decision: excluding c is a
        weights-dict edit, no code change."""
        no_c = {"a": 1.0, "b": 1.0, "d": 1.0, "s": 1.0}
        for i in range(2000):
            dd = _dd(f"q{i}", forced_arm_weights=no_c)
            if dd.bypass_kind == "data_collection_throttle":
                assert dd.forced_strategy != "c"

    def test_throttle_keeps_counterfactual_shadows(self):
        """Throttle-forced ≠ isolation-forced: all four allocators shadow."""
        for i in range(200):
            dd = _dd(f"q{i}")
            if dd.bypass_kind == "data_collection_throttle":
                assert set(dd.shadow_allocators) == {"greedy", "optimizer",
                                                     "bayesian", "portfolio"}
                assert dd.max_attempts == 1
                return
        pytest.fail("no throttle hit in 200 draws")

    def test_isolation_forced_paths_unchanged(self):
        """Calibration and caller-forced keep isolation semantics (no shadows) —
        Eval's measurement isolation is untouched by the throttle."""
        cal = dispatch(is_calibration=True, forced_strategy="a",
                       phase="data_collection", forced_fraction=1.0)
        assert cal.bypass_kind == "calibration" and cal.shadow_allocators == []
        forced = dispatch(is_calibration=False, forced_strategy="d",
                          phase="data_collection", forced_fraction=1.0)
        assert forced.bypass_kind == "forced_strategy" and forced.shadow_allocators == []

    def test_allocator_override_wins_over_throttle(self):
        dd = dispatch(is_calibration=False, forced_strategy=None,
                      query_key="q", allocator_override="greedy",
                      phase="data_collection", forced_fraction=1.0)
        assert dd.path == "greedy"
        assert dd.bypass_kind is None


class TestPolicyParsing:
    def _load(self, tmp_path, policy_yaml):
        from app.services.router.priors import load_priors
        f = tmp_path / "p.yaml"
        f.write_text(textwrap.dedent(f"""
            seed_priors:
              depth_2:
                a: {{recall_lift: 0.5, latency_p50_ms: 500, cost_per_attempt: 1,
                     accuracy_estimate: 0.5}}
            exploration_policy:
{textwrap.indent(textwrap.dedent(policy_yaml), '              ')}
        """))
        return load_priors(path=f, force_reload=True).exploration_policy

    def test_data_collection_policy_parsed(self, tmp_path):
        p = self._load(tmp_path, """
            phase: data_collection
            forced_fraction: 0.2
            forced_arms: [a, b, d, s]
        """)
        assert p["phase"] == "data_collection"
        assert p["forced_fraction"] == pytest.approx(0.2)
        assert p["forced_arm_weights"] == {"a": 1.0, "b": 1.0, "d": 1.0, "s": 1.0}

    def test_weighted_arms_and_roster_filter(self, tmp_path):
        p = self._load(tmp_path, """
            phase: data_collection
            forced_fraction: 1.5
            forced_arms: {a: 2.0, c: 1.0, e: 5.0, f: 3.0, zz: 1.0}
        """)
        assert p["forced_fraction"] == 1.0  # clamped
        # e (terminal), f (retired), zz (unknown) filtered by STRATEGY_IDS
        assert p["forced_arm_weights"] == {"a": 2.0, "c": 1.0}

    def test_absent_knobs_default_off(self, tmp_path):
        p = self._load(tmp_path, """
            phase: bootstrap
        """)
        assert p["forced_fraction"] == 0.0
        assert "forced_arm_weights" not in p


class TestThrottleEndToEnd:
    def test_route_throttle_forced_with_counterfactual_shadows_persisted(self):
        """Full route(): a throttle-hit query executes forced-X AND persists
        all four counterfactual shadow plans + bypass_kind (row hygiene)."""
        from app.services.router.decision import ResourcePosture, RoutingContext
        from app.services.router.priors import load_priors
        from app.services.router.router import route
        from app.services.router.test_router import FakeSessionFactory

        # find a query key that lands on the throttle (deterministic search)
        hit = next(q for q in (f"throttle e2e {i}" for i in range(200))
                   if dispatch(is_calibration=False, forced_strategy=None,
                               query_key=q, phase="data_collection",
                               forced_fraction=0.2, has_payor_context=True
                               ).bypass_kind == "data_collection_throttle")

        bundle = load_priors(force_reload=True)
        bundle.exploration_policy["phase"] = "data_collection"
        bundle.exploration_policy["forced_fraction"] = 0.2

        factory = FakeSessionFactory()
        decision = asyncio.run(route(factory, RoutingContext(
            query=hit, agent_id="router-4c",
            resource_posture=ResourcePosture(max_attempts_per_slot=6),
            pool_metadata={"slot_0": {
                "top_score_percentile": 0.60, "pool_size": 400,
                "required": True, "priority": 0,
                "slot_semantics": "direct_answer", "capacity": 10,
            }},
            gate_j_codes=["payor.sunshine_health"],
        )))
        assert decision.dispatch_path == "forced"
        assert decision.routing_ladder.per_slot["slot_0"]  # forced arm's chain
        assert {s.allocator for s in decision.shadow_ladders} == \
            {"greedy", "optimizer", "bayesian", "portfolio"}
        params = factory.db.calls[0][1]
        fv = json.loads(params["feature_vector"])
        assert fv["bypass_kind"] == "data_collection_throttle"
        plans = json.loads(params["shadow_ladder"])["plans"]
        assert {p["allocator"] for p in plans} == \
            {"greedy", "optimizer", "bayesian", "portfolio"}

    def test_isolation_forced_still_shadowless_end_to_end(self):
        """Caller-forced route(): no shadows, bypass_kind=forced_strategy —
        Eval's offline 100-question matrix path, byte-for-byte as before."""
        from app.services.router.decision import ResourcePosture, RoutingContext
        from app.services.router.router import route
        from app.services.router.test_router import FakeSessionFactory

        factory = FakeSessionFactory()
        decision = asyncio.run(route(factory, RoutingContext(
            query="offline matrix q", agent_id="eval-harness",
            forced_strategy="b",
            resource_posture=ResourcePosture(max_attempts_per_slot=6),
            pool_metadata={"slot_0": {"top_score_percentile": 0.60,
                                      "pool_size": 400}},
        )))
        assert decision.dispatch_path == "forced"
        assert decision.shadow_ladders == []
        fv = json.loads(factory.db.calls[0][1]["feature_vector"])
        assert fv["bypass_kind"] == "forced_strategy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
