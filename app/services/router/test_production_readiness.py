"""Production-readiness tests mapped to TECH's checklist items.

Each test names the checklist section it verifies. These are the items that
weren't already covered by the existing suites.
"""

import asyncio
import json

import pytest

from app.services.router.allocation import AnswerSlot, allocate_strategies
from app.services.router.optimizer import optimize_allocation
from app.services.router.decision import ResourcePosture, RoutingContext
from app.services.router import router as router_module
from app.services.router.router import route


POOL = {
    "core": {"top_score_percentile": 0.80, "pool_size": 150,
             "priority": 0, "query_class": "tight_pool"},
    "supporting": {"top_score_percentile": 0.55, "pool_size": 420,
                   "priority": 1, "query_class": "wide_pool"},
}


class FakeSessionFactory:
    def __init__(self):
        self.calls = []

    def __call__(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, stmt, params):
        self.calls.append((str(stmt), params))

    async def commit(self):
        pass


def _slots():
    return [
        AnswerSlot("core", "direct_answer", 5, "q", True, 0),
        AnswerSlot("supporting", "direct_answer", 5, "q", True, 1),
    ]


def _posture():
    return {
        "speed_budget": "interactive",
        "confidence_bar": 0.85,
        "caller_mode": "chat.default",
        "max_attempts_per_slot": 6,
    }


class TestCharacterization:
    """TECH §5: deterministic, byte-identical ladder for identical input."""

    def test_greedy_byte_identical_across_runs(self):
        dumps = set()
        for _ in range(3):
            ladder = allocate_strategies(_slots(), POOL, _posture())
            dumps.add(json.dumps({
                "per_slot": ladder.per_slot,
                "conf": ladder.aggregate_confidence_estimate,
                "cost": ladder.total_estimated_cost,
                "lat": ladder.total_estimated_ms,
                "feasible": ladder.feasible,
            }, sort_keys=True))
        assert len(dumps) == 1

    def test_optimizer_byte_identical_across_runs(self):
        dumps = set()
        for _ in range(3):
            ladder = optimize_allocation(_slots(), POOL, _posture())
            dumps.add(json.dumps({
                "per_slot": ladder.per_slot,
                "conf": ladder.aggregate_confidence_estimate,
                "cost": ladder.total_estimated_cost,
                "lat": ladder.total_estimated_ms,
                "feasible": ladder.feasible,
            }, sort_keys=True))
        assert len(dumps) == 1


class TestShadowResilience:
    """TECH §11: shadow failures must not crash the production path."""

    def test_shadow_allocator_exception_degrades_gracefully(self, monkeypatch):
        """One exploding shadow must neither block production NOR take down the
        other shadow — it is simply absent from the shadow list."""
        def exploding_optimizer(*args, **kwargs):
            raise RuntimeError("solver blew up")

        monkeypatch.setattr(router_module, "optimize_allocation", exploding_optimizer)

        factory = FakeSessionFactory()
        decision = asyncio.run(route(factory, RoutingContext(
            query="q", agent_id="router-4c",
            mode_override="greedy",  # executed=greedy; shadows=optimizer(explodes)+bayesian
            resource_posture=ResourcePosture(max_attempts_per_slot=6),
            pool_metadata=POOL,
        )))
        # production result intact
        assert decision.routing_ladder.per_slot["core"]
        assert decision.dispatch_path == "greedy"
        # exploded shadow absent; surviving shadow (bayesian) still present
        assert [s.allocator for s in decision.shadow_ladders] == ["bayesian", "portfolio"]
        _, params = factory.calls[0]
        import json as _json
        plans = _json.loads(params["shadow_ladder"])["plans"]
        assert [p["allocator"] for p in plans] == ["bayesian", "portfolio"]

    def test_executed_allocator_result_never_depends_on_shadow(self, monkeypatch):
        """Same executed plan whether the shadow succeeds or explodes."""
        factory1 = FakeSessionFactory()
        d1 = asyncio.run(route(factory1, RoutingContext(
            query="q", agent_id="router-4c", mode_override="greedy",
            resource_posture=ResourcePosture(max_attempts_per_slot=6),
            pool_metadata=POOL,
        )))

        def exploding_optimizer(*args, **kwargs):
            raise RuntimeError("boom")
        monkeypatch.setattr(router_module, "optimize_allocation", exploding_optimizer)

        factory2 = FakeSessionFactory()
        d2 = asyncio.run(route(factory2, RoutingContext(
            query="q", agent_id="router-4c", mode_override="greedy",
            resource_posture=ResourcePosture(max_attempts_per_slot=6),
            pool_metadata=POOL,
        )))
        assert d1.routing_ladder.per_slot == d2.routing_ladder.per_slot


class TestSegmentTimings:
    """TECH §2/§6: per-segment timings emitted (allocate/shadow in the row;
    persist in the log line)."""

    def test_allocate_and_shadow_ms_in_feature_vector(self):
        factory = FakeSessionFactory()
        asyncio.run(route(factory, RoutingContext(
            query="q", agent_id="router-4c",
            resource_posture=ResourcePosture(max_attempts_per_slot=6),
            pool_metadata=POOL,
        )))
        fv = json.loads(factory.calls[0][1]["feature_vector"])
        assert "router_allocate_ms" in fv and fv["router_allocate_ms"] >= 0
        assert "router_shadow_ms" in fv and fv["router_shadow_ms"] >= 0

    def test_route_summary_log_has_timings_and_no_query_text(self, caplog):
        import logging
        factory = FakeSessionFactory()
        with caplog.at_level(logging.INFO, logger="app.services.router.router"):
            asyncio.run(route(factory, RoutingContext(
                query="SECRET-PHI-QUERY-TEXT should never appear in logs",
                agent_id="router-4c",
                resource_posture=ResourcePosture(max_attempts_per_slot=6),
                pool_metadata=POOL,
            )))
        summary = [r for r in caplog.records if "router.route" in r.getMessage()]
        assert summary, "route must emit a timing summary log line"
        msg = summary[0].getMessage()
        assert "router_persist_ms=" in msg and "router_allocate_ms=" in msg
        # TECH §2/§10: no raw query text in any log line
        for record in caplog.records:
            assert "SECRET-PHI-QUERY-TEXT" not in record.getMessage()


class TestPerformanceBudget:
    """TECH §6: allocation completes well under 100ms."""

    def test_both_allocators_under_100ms(self):
        import time
        for fn in (allocate_strategies, optimize_allocation):
            t0 = time.monotonic()
            for _ in range(20):
                fn(_slots(), POOL, _posture())
            per_call_ms = (time.monotonic() - t0) * 1000 / 20
            assert per_call_ms < 100, f"{fn.__name__}: {per_call_ms:.1f}ms per call"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
