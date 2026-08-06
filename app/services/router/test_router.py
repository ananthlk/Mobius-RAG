"""Router core tests — dispatch, depth-bucket boundaries, persistence, ONE-WRITER, integration.

Async tests use asyncio.run() directly (no pytest-asyncio dependency).
"""

import asyncio
import json
import re
from pathlib import Path

import pytest

from app.services.router.dispatch import dispatch
from app.services.router.priors import compute_depth_bucket
from app.services.router.persist import persist_decision
from app.services.router.decision import ResourcePosture, RouterDecision, RoutingContext
from app.services.router import router as router_module
from app.services.router.router import route


# ---------------------------------------------------------------------------
# Fakes (proper async context-manager protocol — no AsyncMock gymnastics)
# ---------------------------------------------------------------------------

class _FakeDB:
    def __init__(self):
        self.calls = []
        self.committed = 0

    async def execute(self, stmt, params):
        self.calls.append((str(stmt), params))

    async def commit(self):
        self.committed += 1


class FakeSessionFactory:
    """persist_decision does `async with db_session_factory() as db:` — this
    object is both the factory (callable) and the async context manager."""

    def __init__(self):
        self.db = _FakeDB()

    def __call__(self):
        return self

    async def __aenter__(self):
        return self.db

    async def __aexit__(self, *exc):
        return False


class ExplodingSessionFactory:
    def __call__(self):
        raise RuntimeError("db unavailable")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class TestDispatchThreeWay:
    def test_calibration_routes_to_forced(self):
        d = dispatch(is_calibration=True, forced_strategy=None)
        assert d.path == "forced"
        assert d.bypass_kind == "calibration"
        assert d.max_attempts == 1
        assert d.shadow_allocators == []  # no shadows in forced mode

    def test_forced_strategy_routes_to_forced(self):
        d = dispatch(is_calibration=False, forced_strategy="a")
        assert d.path == "forced"
        assert d.bypass_kind == "forced_strategy"
        assert d.forced_strategy == "a"
        assert d.max_attempts == 1

    def test_calibration_takes_precedence_over_forced_strategy(self):
        d = dispatch(is_calibration=True, forced_strategy="b")
        assert d.bypass_kind == "calibration"

    def test_weight_one_allocator_takes_all_traffic(self):
        for name in ("greedy", "optimizer", "bayesian"):
            weights = {name: 1.0}
            for q in ["query one", "query two", "query three"]:
                d = dispatch(is_calibration=False, forced_strategy=None,
                             allocator_weights=weights, query_key=q)
                assert d.path == name
                assert set(d.shadow_allocators) == ({"greedy", "optimizer", "bayesian", "portfolio"} - {name})
                assert d.draw is not None and 0.0 <= d.draw < 1.0

    def test_equal_thirds_covers_all_allocators(self):
        """Across many distinct queries, the equal-thirds draw actually lands
        on every allocator (deterministic per query, spread across queries)."""
        weights = {"greedy": 1/3, "optimizer": 1/3, "bayesian": 1/3}
        seen = set()
        for i in range(60):
            d = dispatch(is_calibration=False, forced_strategy=None,
                         allocator_weights=weights, query_key=f"query {i}")
            seen.add(d.path)
        assert seen == {"greedy", "optimizer", "bayesian"}

    def test_forced_wins_over_weights(self):
        d = dispatch(is_calibration=False, forced_strategy="c",
                     allocator_weights={"bayesian": 1.0}, query_key="q")
        assert d.path == "forced"

    def test_allocator_override_pins_executed_allocator(self):
        for name in ("greedy", "optimizer", "bayesian"):
            d = dispatch(is_calibration=False, forced_strategy=None,
                         allocator_weights={"greedy": 1.0}, query_key="q",
                         allocator_override=name)
            assert d.path == name
            assert set(d.shadow_allocators) == ({"greedy", "optimizer", "bayesian", "portfolio"} - {name})

    def test_draw_is_deterministic_and_replayable(self):
        from app.services.router.dispatch import stable_draw
        assert stable_draw("same query") == stable_draw("same query")
        assert stable_draw("query A") != stable_draw("query B")
        # reason string carries the draw + weights arithmetic for the narrate layer
        d = dispatch(is_calibration=False, forced_strategy=None,
                     query_key="query A")
        assert f"{d.draw:.4f}" in d.reason
        assert "weights" in d.reason


# ---------------------------------------------------------------------------
# Depth bucketing — exact boundary values, not just mid-bucket points
# ---------------------------------------------------------------------------

class TestDepthBucketBoundaries:
    @pytest.mark.parametrize("score,pool,expected", [
        # bucket 0 boundary: score >= 0.90 AND pool < 50
        (0.90, 49, 0),
        (0.90, 50, 1),      # pool at cap → falls to bucket 1 rule
        (0.899, 49, 1),     # score just under → bucket 1 rule
        # bucket 1 boundary: score >= 0.75 AND pool < 200
        (0.75, 199, 1),
        (0.75, 200, 2),
        (0.749, 100, 2),
        # bucket 2 boundary: score >= 0.50 AND pool < 500
        (0.50, 499, 2),
        (0.50, 500, 3),
        (0.499, 400, 3),
        # bucket 3 boundary: score >= 0.25 AND pool < 5000
        (0.25, 4999, 3),
        (0.25, 5000, 4),
        (0.249, 100, 4),
        # extremes
        (1.0, 1, 0),
        (0.0, 10**9, 4),
    ])
    def test_boundaries(self, score, pool, expected):
        assert compute_depth_bucket(
            {"top_score_percentile": score, "pool_size": pool}
        ) == expected

    def test_missing_and_none_fields_default_to_broad(self):
        assert compute_depth_bucket({}) == 4
        assert compute_depth_bucket({"top_score_percentile": None, "pool_size": None}) == 4


# ---------------------------------------------------------------------------
# Persistence (ONE-WRITER function itself)
# ---------------------------------------------------------------------------

# Was ALL_24_COLUMNS -- 2026-08-06, +correlation_id (Chat Master's grading-
# callback gap: legacy's correlation_id=caller_id pattern had no equivalent
# on this pipeline's persist path -- the DB column already existed, this
# pipeline just never wrote to it). 25 columns now.
ALL_25_COLUMNS = {
    "decision_id", "agent_id", "query", "correlation_id",
    "is_calibration", "is_prod", "eval_run_id",
    "depth_bucket", "strategy_chosen", "strategy_sequence",
    "executed_ladder", "shadow_ladder", "confidence_bar",
    "gate_contour", "gate_underspecified_kind", "reformat_posture", "reformat_fanout_n",
    "feature_vector", "strategy_scores", "priors_version",
    "confidence", "accuracy_estimate", "cost",
    "total_ms", "leaf_key",
}


class TestPersistence:
    def test_writes_all_25_parameters(self):
        factory = FakeSessionFactory()
        decision_id = asyncio.run(persist_decision(
            factory,
            agent_id="router-4c", query="q",
            is_calibration=False, is_prod=True, eval_run_id=None,
            depth_bucket=2, strategy_chosen="a", strategy_sequence=["a", "b"],
            executed_ladder={"allocator": "greedy", "per_slot": {"s0": ["a", "b"]}},
            shadow_ladder={"allocator": "optimizer", "per_slot": {"s0": ["s", "a"]}},
            confidence_bar=0.85,
            feature_vector={"k": "v"}, strategy_scores={"a": 0.5},
            priors_version="file:x@1", confidence=0.75, accuracy_estimate=0.4,
            cost=2.0, total_ms=1500, leaf_key="slot_0",
            gate_contour="core", gate_underspecified_kind=None,
            reformat_posture="PRECISE", reformat_fanout_n=None,
        ))
        assert decision_id
        assert len(factory.db.calls) == 1
        stmt, params = factory.db.calls[0]
        assert set(params.keys()) == ALL_25_COLUMNS
        assert json.loads(params["strategy_sequence"]) == ["a", "b"]
        assert json.loads(params["executed_ladder"])["allocator"] == "greedy"
        assert json.loads(params["shadow_ladder"])["allocator"] == "optimizer"
        assert params["confidence_bar"] == 0.85
        assert params["gate_contour"] == "core"
        assert params["reformat_posture"] == "PRECISE"
        assert factory.db.committed == 1

    def test_none_values_pass_through(self):
        factory = FakeSessionFactory()
        decision_id = asyncio.run(persist_decision(
            factory,
            agent_id="router-4c", query="q",
            depth_bucket=2, strategy_chosen="a", strategy_sequence=["a"],
            feature_vector=None, strategy_scores=None,
            total_ms=10, leaf_key="s0",
        ))
        assert decision_id
        _, params = factory.db.calls[0]
        assert params["feature_vector"] is None
        assert params["gate_contour"] is None

    def test_generates_uuid_when_not_provided(self):
        factory = FakeSessionFactory()
        d1 = asyncio.run(persist_decision(
            factory, agent_id="r", query="q", depth_bucket=0,
            strategy_chosen="s", strategy_sequence=["s"], total_ms=1, leaf_key="x",
        ))
        d2 = asyncio.run(persist_decision(
            factory, agent_id="r", query="q", depth_bucket=0,
            strategy_chosen="s", strategy_sequence=["s"], total_ms=1, leaf_key="x",
        ))
        assert d1 and d2 and d1 != d2

    def test_caller_provided_decision_id_used(self):
        factory = FakeSessionFactory()
        d = asyncio.run(persist_decision(
            factory, agent_id="r", query="q", depth_bucket=0,
            strategy_chosen="s", strategy_sequence=["s"], total_ms=1, leaf_key="x",
            decision_id="fixed-id-123",
        ))
        assert d == "fixed-id-123"
        assert factory.db.calls[0][1]["decision_id"] == "fixed-id-123"

    def test_db_failure_is_swallowed_not_raised(self):
        """Telemetry must never break the user's request."""
        d = asyncio.run(persist_decision(
            ExplodingSessionFactory(), agent_id="r", query="q", depth_bucket=0,
            strategy_chosen="s", strategy_sequence=["s"], total_ms=1, leaf_key="x",
        ))
        assert d  # id still returned; no exception propagated

    def test_non_list_strategy_sequence_coerced(self):
        factory = FakeSessionFactory()
        asyncio.run(persist_decision(
            factory, agent_id="r", query="q", depth_bucket=0,
            strategy_chosen="a", strategy_sequence="a",  # wrong type on purpose
            total_ms=1, leaf_key="x",
        ))
        _, params = factory.db.calls[0]
        assert json.loads(params["strategy_sequence"]) == ["a"]


# ---------------------------------------------------------------------------
# ONE-WRITER enforcement — proven, not just documented
# ---------------------------------------------------------------------------

class TestOneWriter:
    def test_only_persist_py_contains_the_insert(self):
        """Exactly one file in the router package writes rag_query_decisions."""
        pkg = Path(__file__).parent
        writers = []
        for py in pkg.glob("*.py"):
            text = py.read_text()
            if re.search(r"INSERT\s+INTO\s+rag_query_decisions", text, re.IGNORECASE):
                writers.append(py.name)
        assert writers == ["persist.py"]

    def test_all_three_route_paths_call_the_single_writer(self, monkeypatch):
        """Spy on persist_decision: forced, greedy-executed, and
        optimizer-executed paths each call it exactly once — one row per query
        even though production computes TWO ladders."""
        calls = []

        async def spy(*args, **kwargs):
            calls.append(kwargs)
            return "spy-id"

        monkeypatch.setattr(router_module, "persist_decision", spy)

        pool = {"s0": {"top_score_percentile": 0.6, "pool_size": 400,
                       "priority": 0, "query_class": "tight_pool"}}

        greedy = asyncio.run(route(FakeSessionFactory(), RoutingContext(
            query="q", agent_id="router-4c", allocator_override="greedy",
            resource_posture=ResourcePosture(), pool_metadata=pool,
        )))
        opt = asyncio.run(route(FakeSessionFactory(), RoutingContext(
            query="q", agent_id="router-4c", allocator_override="optimizer",
            resource_posture=ResourcePosture(), pool_metadata=pool,
        )))
        forced = asyncio.run(route(FakeSessionFactory(), RoutingContext(
            query="q", agent_id="router-4c", is_calibration=True,
            forced_strategy="b", resource_posture=ResourcePosture(),
            pool_metadata=pool,
        )))

        assert len(calls) == 3  # dual ladders, still ONE write per query
        assert {greedy.dispatch_path, opt.dispatch_path, forced.dispatch_path} == \
            {"greedy", "optimizer", "forced"}
        assert calls[0]["is_calibration"] is False and calls[0]["is_prod"] is True
        assert calls[2]["is_calibration"] is True and calls[2]["is_prod"] is False


# ---------------------------------------------------------------------------
# route() integration (fake DB, real allocation + real YAML priors)
# ---------------------------------------------------------------------------

class TestRouteIntegration:
    def _pool(self):
        return {
            "core": {"top_score_percentile": 0.60, "pool_size": 400,
                     "priority": 0, "query_class": "tight_pool"},
            "supporting": {"top_score_percentile": 0.30, "pool_size": 2000,
                           "priority": 1, "query_class": "wide_pool"},
        }

    def test_greedy_route_end_to_end(self):
        factory = FakeSessionFactory()
        decision = asyncio.run(route(factory, RoutingContext(
            query="What is the timely filing deadline?",
            agent_id="router-4c",
            allocator_override="greedy",
            resource_posture=ResourcePosture(speed_budget="interactive",
                                             confidence_bar=0.85,
                                             caller_mode="chat.default",
                                             max_attempts_per_slot=6),
            pool_metadata=self._pool(),
            gate_j_codes=["payor.sunshine_health"],  # payor query → s eligible
        )))
        assert isinstance(decision, RouterDecision)
        assert decision.dispatch_path == "greedy"
        # RE-DERIVED 2026-08-05 (best-LB-first, Ananth): depth_2 core chain
        # is now [a,s,d,c] (best-LB order a>s>d>c>b, not fixed s,a,b,c,d) —
        # chases the LB bar until b overflows the allowance, honest
        # UNDER_CONFIDENT verdict (same status as before, different order).
        assert decision.routing_ladder.per_slot["core"] == ["a", "s", "d", "c"]
        assert decision.routing_ladder.allocator == "greedy"
        assert decision.routing_ladder.per_slot_status["core"] == "UNDER_CONFIDENT"
        assert decision.routing_ladder.outcome == "partial_infeasible"
        assert decision.routing_ladder.feasible is False
        # shadow plans computed by BOTH other allocators, never executed
        assert {s.allocator for s in decision.shadow_ladders} == {"optimizer", "bayesian", "portfolio"}
        assert all(t.role == "shadow" for t in decision.shadow_traces)

        # Retriever 2026-08-05 (post-deploy trace catch): the earlier
        # per_slot_pool_metadata fix only reached persist_decision's DB
        # write (feature_vector) — this asserts the RETURNED RouterDecision
        # (feature_context, what actually flows to callers/routing_keys)
        # carries the same fields. This is the gap that shipped unnoticed:
        # the DB-side test below passed while the return value stayed None.
        assert "per_slot_pool_metadata" in decision.feature_context
        assert decision.feature_context["per_slot_pool_metadata"]["core"]["pool_size"] == 400
        assert decision.feature_context["per_slot_depth_buckets"]["core"] == 2

        _, params = factory.db.calls[0]
        assert params["depth_bucket"] == 2                      # core slot's bucket
        assert params["strategy_chosen"] == "a"                 # RE-DERIVED 2026-08-05: best-LB-first, a leads not s
        assert json.loads(params["strategy_sequence"]) == ["a", "s", "d", "c"]
        assert params["leaf_key"] == "core"
        assert params["priors_version"].startswith("file:")
        assert params["is_prod"] is True
        assert params["confidence_bar"] == 0.85
        executed = json.loads(params["executed_ladder"])
        shadow_plans = json.loads(params["shadow_ladder"])["plans"]
        assert executed["allocator"] == "greedy"
        assert {p["allocator"] for p in shadow_plans} == {"optimizer", "bayesian", "portfolio"}
        # §2a verdicts ride in the persisted ladders
        assert executed["outcome"] == "partial_infeasible"
        assert executed["per_slot_status"]["core"] == "UNDER_CONFIDENT"
        assert "per_slot_lb" in executed
        assert all("per_slot_lb" in p for p in shadow_plans)
        fv = json.loads(params["feature_vector"])
        assert fv["dispatch_mode"] == "greedy"
        assert set(fv["shadow_allocators"]) == {"optimizer", "bayesian", "portfolio"}
        assert fv["per_slot_depth_buckets"]["core"] == 2
        # raw pool signals behind the bucket, for Eval's stratified CIs
        # (2026-08-05) — must ride alongside the derived bucket, not replace it
        assert "top_score_percentile" in fv["per_slot_pool_metadata"]["core"]
        assert "pool_size" in fv["per_slot_pool_metadata"]["core"]

    def test_default_traffic_three_way_split_by_deterministic_draw(self):
        """No override → the sha256 draw over equal-thirds weights picks the
        executed allocator; the other TWO shadow. Expectation derived from the
        draw itself, not hardcoded."""
        from app.services.router.dispatch import _pick_allocator, stable_draw
        query = "any production query"
        weights = {"greedy": 1/3, "optimizer": 1/3, "bayesian": 1/3}
        expected = _pick_allocator(weights, stable_draw(query))
        others = {"greedy", "optimizer", "bayesian"} - {expected}

        factory = FakeSessionFactory()
        decision = asyncio.run(route(factory, RoutingContext(
            query=query, agent_id="router-4c",
            resource_posture=ResourcePosture(max_attempts_per_slot=6),
            pool_metadata=self._pool(),
        )))
        assert decision.dispatch_path == expected
        assert {s.allocator for s in decision.shadow_ladders} == others | {"portfolio"}
        # all three plans in the persisted row
        _, params = factory.db.calls[0]
        assert json.loads(params["executed_ladder"])["allocator"] == expected
        shadow_plans = json.loads(params["shadow_ladder"])["plans"]
        assert {p["allocator"] for p in shadow_plans} == others | {"portfolio"}

    def test_calibration_route_forces_single_strategy(self):
        factory = FakeSessionFactory()
        decision = asyncio.run(route(factory, RoutingContext(
            query="q", agent_id="router-4c",
            is_calibration=True, forced_strategy="b",
            resource_posture=ResourcePosture(),
            pool_metadata=self._pool(),
        )))
        assert decision.dispatch_path == "forced"
        assert all(chain == ["b"] for chain in decision.routing_ladder.per_slot.values())
        assert decision.shadow_ladders == []  # no shadows in forced/calibration mode
        _, params = factory.db.calls[0]
        assert params["is_calibration"] is True
        assert params["strategy_chosen"] == "b"
        assert params["shadow_ladder"] is None

    def test_upstream_diagnostics_flow_into_deferred_columns(self):
        factory = FakeSessionFactory()
        asyncio.run(route(factory, RoutingContext(
            query="q", agent_id="router-4c",
            resource_posture=ResourcePosture(),
            pool_metadata=self._pool(),
            upstream_diagnostics={
                "gate_contour": "core",
                "gate_underspecified_kind": "entity",
                "reformat_posture": "FAN_OUT",
                "reformat_fanout_n": 3,
            },
        )))
        _, params = factory.db.calls[0]
        assert params["gate_contour"] == "core"
        assert params["gate_underspecified_kind"] == "entity"
        assert params["reformat_posture"] == "FAN_OUT"
        assert params["reformat_fanout_n"] == 3

    def test_empty_pool_metadata_production_is_infeasible_but_persisted(self):
        factory = FakeSessionFactory()
        decision = asyncio.run(route(factory, RoutingContext(
            query="q", agent_id="router-4c",
            resource_posture=ResourcePosture(),
            pool_metadata={},
        )))
        assert decision.routing_ladder.feasible is False
        assert decision.routing_ladder.infeasibility_reason == "no slots to allocate"
        assert len(factory.db.calls) == 1  # still exactly one telemetry row


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestDiversityDemotion:
    """Junk-poisoning defense: Pool's distinct_content_topk demotes buckets
    whose high score is dominated by repeated content (Filler b's verified
    cluster: whole top-10 = one 23-char boilerplate at 0.797 cosine)."""

    def test_all_junk_top_forces_broad_bucket(self):
        """The exact failure: tight-looking score, distinct=1 → forced >= 3,
        so the tight-corpus priors row is NOT selected and d-escalation
        stays reachable."""
        meta = {"top_score_percentile": 0.95, "pool_size": 30,
                "distinct_content_topk": 1}
        assert compute_depth_bucket(meta) == 3  # was 0 blind

    def test_partial_duplication_moderate_demotion(self):
        meta = {"top_score_percentile": 0.95, "pool_size": 30,
                "distinct_content_topk": 4}
        assert compute_depth_bucket(meta) == 2  # was 0 blind

    def test_healthy_diversity_unchanged(self):
        """Live cmhc001 post-fix shape: score .7997, distinct=7 → no demotion."""
        meta = {"top_score_percentile": 0.7997, "pool_size": 150,
                "distinct_content_topk": 7}
        assert compute_depth_bucket(meta) == 1

    def test_absent_field_fail_open_unchanged(self):
        """Callers not yet supplying the field keep exact legacy behavior."""
        assert compute_depth_bucket(
            {"top_score_percentile": 0.95, "pool_size": 30}) == 0

    def test_demotion_never_promotes(self):
        """Diversity only ever demotes (max) — a broad bucket stays broad."""
        meta = {"top_score_percentile": 0.10, "pool_size": 50000,
                "distinct_content_topk": 10}
        assert compute_depth_bucket(meta) == 4


class TestPostureBridgeEndToEnd:
    """Structure's catch (2026-07-23): ResourcePosture (wire contract) and the
    internal posture dict are TWO representations bridged in route() — a knob
    that exists only on the dict side is unreachable from the live
    orchestrator path. These tests pin the bridge: every caller-facing knob
    must flow dataclass → dict → gate."""

    def _route(self, posture):
        factory = FakeSessionFactory()
        return asyncio.run(route(factory, RoutingContext(
            query="bridge test", agent_id="router-4c",
            allocator_override="greedy",
            resource_posture=posture,
            pool_metadata={"slot_0": {
                "top_score_percentile": 0.60, "pool_size": 400,
                "required": True, "priority": 0,
                "slot_semantics": "direct_answer", "capacity": 5,
            }},
            gate_j_codes=["payor.sunshine_health"],
        )))

    def test_token_budget_flows_from_dataclass_to_gate(self):
        """PARTIAL-FILL model: budget 1000 through the REAL route() path
        seats the chain at scoped fills (floor 1 each, 150+250×3=900 ≤ 1000)
        instead of gating whole rungs — payload bound by construction proves
        the dataclass→bridge→assignment flow."""
        decision = self._route(ResourcePosture(
            speed_budget="interactive", max_attempts_per_slot=6,
            token_budget=1000,
        ))
        fills = decision.routing_ladder.per_slot_portfolio["slot_0"]
        assert fills and all(k >= 1 for k in fills.values())
        assert decision.routing_ladder.per_slot_payload_tokens["slot_0"] <= 1000
        # default-budget run fills far past 1000 — proves 1000 actually bound
        loose = self._route(ResourcePosture(
            speed_budget="interactive", max_attempts_per_slot=6,
        ))
        assert loose.routing_ladder.per_slot_payload_tokens["slot_0"] > 1000

    def test_token_budget_none_uses_default(self):
        decision = self._route(ResourcePosture(
            speed_budget="interactive", max_attempts_per_slot=6,
        ))
        assert "a" in decision.routing_ladder.per_slot["slot_0"]  # default 8000: not gated

    def test_authority_requirement_flows_from_dataclass_to_gate(self):
        decision = self._route(ResourcePosture(
            speed_budget="background", max_attempts_per_slot=6,
            confidence_bar=0.99, authority_requirement="citable_required",
        ))
        assert "d" not in decision.routing_ladder.per_slot["slot_0"]
        relaxed = self._route(ResourcePosture(
            speed_budget="background", max_attempts_per_slot=6,
            confidence_bar=0.99,
        ))
        assert "d" in relaxed.routing_ladder.per_slot["slot_0"]


class TestRetryBackReference:
    """Whole-loop technical retry (Eval-ratified pairing): the retry row must
    carry retry_of_decision in feature_vector so attempt-0 can be excluded by
    EXACT id pairing — fuzzy time-window pairing false-excludes real
    observations on identical-text queries."""

    def _route(self, upstream):
        factory = FakeSessionFactory()
        decision = asyncio.run(route(factory, RoutingContext(
            query="retry pairing test", agent_id="retriever-orchestrator-retry1",
            allocator_override="greedy",
            resource_posture=ResourcePosture(max_attempts_per_slot=6),
            pool_metadata={"slot_0": {
                "top_score_percentile": 0.60, "pool_size": 400,
                "required": True, "priority": 0,
                "slot_semantics": "direct_answer", "capacity": 5,
            }},
            upstream_diagnostics=upstream,
        )))
        assert len(factory.db.calls) == 1
        return factory.db.calls[0][1]  # persisted params

    def test_retry_row_carries_back_reference(self):
        import json
        params = self._route({"retry_of_decision": "dec-attempt0-abc123"})
        fv = json.loads(params["feature_vector"])
        assert fv["retry_of_decision"] == "dec-attempt0-abc123"

    def test_non_retry_row_carries_none(self):
        import json
        params = self._route({})
        fv = json.loads(params["feature_vector"])
        assert fv["retry_of_decision"] is None
