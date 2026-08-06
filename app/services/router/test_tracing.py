"""Tracing + narration tests — the "no black box" contract.

Three guarantees under test:
  1. REPLAYABLE: the emitted structured trace contains enough real values
     (priors, arithmetic steps) to recompute the exact decision by hand —
     enforced by literally recomputing every step.
  2. TRANSPARENT: narrate() states depth inputs, every strategy tried/skipped
     with its actual prior values, the running arithmetic, and the stop reason.
  3. PHI-SAFE: narration is never persisted; the persisted row never contains
     the prose or raw query-derived trace text.
"""

import asyncio
import json

import pytest

from app.services.router.allocation import AnswerSlot, allocate_strategies
from app.services.router.decision import ResourcePosture, RoutingContext
from app.services.router.router import route
from app.services.router.router_narrate import narrate
from app.services.router.tracing import DecisionTrace


POOL_DEPTH_1 = {"top_score_percentile": 0.80, "pool_size": 150}
POOL_DEPTH_2 = {"top_score_percentile": 0.60, "pool_size": 400}


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


class TestTraceReplayability:
    def test_every_added_step_recomputes_exactly(self):
        """Replay the trace by hand: conf_after must equal
        conf_before + (1-conf_before)*recall_lift for every step, and the
        slot's final confidence must equal the last step's conf_after."""
        trace = DecisionTrace()
        allocate_strategies([_slot()], {"slot_0": POOL_DEPTH_1}, _posture(), trace=trace)
        st = trace.slots[0]
        added = [s for s in st.steps if s.action == "added"]
        assert added, "trace must contain added steps"
        prev_conf = 0.0
        for step in added:
            assert step.conf_before == pytest.approx(prev_conf, abs=1e-9)
            expected = prev_conf + (1 - prev_conf) * step.recall_lift
            assert step.conf_after == pytest.approx(expected, abs=1e-9)
            assert step.added_term == pytest.approx((1 - prev_conf) * step.recall_lift, abs=1e-9)
            prev_conf = step.conf_after
        assert st.final_confidence == pytest.approx(prev_conf, abs=1e-9)

    def test_trace_records_depth_inputs_and_priors_provenance(self):
        trace = DecisionTrace()
        allocate_strategies([_slot()], {"slot_0": POOL_DEPTH_2}, _posture(), trace=trace)
        st = trace.slots[0]
        assert st.pool_size == 400
        assert st.top_score_percentile == 0.60
        assert st.depth_bucket == 2
        assert trace.priors_version.startswith("file:")
        assert trace.priors_source == "file"
        for step in st.steps:
            if step.action == "added":
                assert step.prior_source in ("depth_bucket", "qclass_fallback")
                assert step.recall_lift > 0

    def test_latency_arithmetic_in_steps(self):
        trace = DecisionTrace()
        allocate_strategies([_slot()], {"slot_0": POOL_DEPTH_2}, _posture(), trace=trace)
        st = trace.slots[0]
        added = [s for s in st.steps if s.action == "added"]
        running = 0
        for step in added:
            assert step.latency_before_ms == running
            running += step.latency_p50_ms
            assert step.latency_after_ms == running
        assert st.final_latency_ms == running

    def test_skipped_strategies_recorded_with_reason(self):
        """real_time budget: d skipped for latency — recorded, not silent."""
        trace = DecisionTrace()
        allocate_strategies(
            [_slot()], {"slot_0": POOL_DEPTH_2},
            _posture(speed_budget="real_time", confidence_bar=0.99),
            trace=trace,
        )
        skips = {s.strategy_id: s.skip_reason
                 for s in trace.slots[0].steps if s.action == "skipped"}
        assert skips.get("d") == "over_latency_allowance"

    def test_stop_reason_and_aggregate_arithmetic(self):
        trace = DecisionTrace()
        ladder = allocate_strategies([_slot()], {"slot_0": POOL_DEPTH_2}, _posture(),
                                     trace=trace)
        # depth_2 at bar .85: LB chase exhausts the budget → binding constraint
        assert trace.slots[0].stop_reason == "budget_exhausted"
        assert trace.slots[0].status == "UNDER_CONFIDENT"
        assert trace.aggregate_arithmetic.startswith("per-slot LBs [")
        assert "mean is telemetry only" in trace.aggregate_arithmetic
        assert trace.outcome == ladder.outcome == "partial_infeasible"

    def test_lb_track_replayable(self):
        """The LB arithmetic is replayable too: lb_after = lb_before + (1-lb_before)*lb_lift."""
        trace = DecisionTrace()
        allocate_strategies([_slot()], {"slot_0": POOL_DEPTH_2}, _posture(), trace=trace)
        prev = 0.0
        for step in [s for s in trace.slots[0].steps if s.action == "added"]:
            assert step.lb_before == pytest.approx(prev, abs=1e-9)
            assert step.lb_after == pytest.approx(prev + (1 - prev) * step.lb_lift, abs=1e-9)
            assert 0.0 <= step.lb_lift < step.recall_lift  # LB strictly under the mean here
            assert step.n == 8  # seed pseudo-count flows into the trace
            prev = step.lb_after
        assert trace.slots[0].final_lb == pytest.approx(prev, abs=1e-9)

    def test_trace_to_dict_is_json_serializable(self):
        trace = DecisionTrace()
        allocate_strategies([_slot()], {"slot_0": POOL_DEPTH_2}, _posture(), trace=trace)
        payload = json.dumps(trace.to_dict())
        assert '"arithmetic"' in payload


class TestNarration:
    def _trace(self, **posture_overrides):
        trace = DecisionTrace(mode="greedy", mode_reason="test run")
        allocate_strategies([_slot()], {"slot_0": POOL_DEPTH_1},
                            _posture(**posture_overrides), trace=trace)
        return trace

    def test_narration_contains_depth_inputs_and_bucket(self):
        text = narrate(self._trace())
        assert 'Slot "slot_0"' in text
        assert "depth_bucket=1" in text
        assert "pool_size=150" in text
        assert "top_score=0.8" in text

    def test_narration_shows_each_strategy_with_actual_prior_values(self):
        text = narrate(self._trace())
        # depth_1 YAML values must appear verbatim
        assert "Tried 's' — prior[depth_bucket]: recall_lift=0.350" in text
        assert "Tried 'a' — prior[depth_bucket]: recall_lift=0.350" in text

    def test_narration_shows_running_arithmetic_not_just_final(self):
        text = narrate(self._trace())
        # second rung at depth_1: 0.35 + (1-0.35)*0.35 = 0.5775
        assert "0.3500 + (1-0.3500)*0.350 = 0.5775" in text

    def test_narration_states_result_status_and_stop_reason(self):
        # low bar → depth_1 chain clears its LB bar → CLEARED / bar_cleared
        text_ok = narrate(self._trace(confidence_bar=0.30))
        assert "RESULT: sequence=[" in text_ok
        assert "→ CLEARED" in text_ok
        assert "stopped: bar_cleared" in text_ok
        # default bar .85 → LB unreachable → UNDER_CONFIDENT + binding constraint
        text_fail = narrate(self._trace())
        assert "→ UNDER_CONFIDENT" in text_fail
        assert "stopped: budget_exhausted" in text_fail

    def test_narration_shows_lb_and_n_alongside_mean(self):
        text = narrate(self._trace())
        assert "(n=8, lb95=" in text
        assert "| LB: " in text  # dual-track cumulative arithmetic
        assert "Wilson lower bound PER SLOT" in text  # header states the gate

    def test_narration_includes_outcome_line(self):
        text = narrate(self._trace(confidence_bar=0.30))
        assert "OUTCOME: ALL_SLOTS_CLEARED" in text
        assert "telemetry only" in text
        text_fail = narrate(self._trace())
        assert "OUTCOME: PARTIAL_INFEASIBLE" in text_fail

    def test_narration_includes_mode_role_and_tolerance_header(self):
        text = narrate(self._trace())
        assert text.startswith("ROUTER [GREEDY, EXECUTED plan]")
        assert "tolerance ±15%" in text


class TestPhiRule:
    def test_narration_is_never_persisted(self):
        """The persisted row must not contain the prose narration."""
        factory = FakeSessionFactory()
        decision = asyncio.run(route(factory, RoutingContext(
            query="What is the timely filing deadline?",
            agent_id="router-4c", allocator_override="greedy",
            resource_posture=ResourcePosture(max_attempts_per_slot=6),
            pool_metadata={"core": {**POOL_DEPTH_2, "priority": 0,
                                    "query_class": "tight_pool"}},
        )))
        prose = narrate(decision.trace)
        assert "RESULT: sequence=[" in prose  # narration works
        _, params = factory.calls[0]
        for value in params.values():
            if isinstance(value, str):
                assert "RESULT: sequence=[" not in value
                assert "Cumulative confidence" not in value

    def test_route_returns_trace_for_all_modes(self):
        pool = {"core": {**POOL_DEPTH_2, "priority": 0, "query_class": "tight_pool"}}
        for kwargs, expected_mode in [
            ({"allocator_override": "greedy"}, "greedy"),
            ({"allocator_override": "optimizer"}, "optimizer"),
            ({"allocator_override": "bayesian"}, "bayesian"),
            ({"is_calibration": True, "forced_strategy": "b"}, "forced"),
        ]:
            decision = asyncio.run(route(FakeSessionFactory(), RoutingContext(
                query="q", agent_id="router-4c",
                resource_posture=ResourcePosture(max_attempts_per_slot=6),
                pool_metadata=pool, **kwargs,
            )))
            assert decision.trace is not None
            assert decision.trace.mode == expected_mode
            assert decision.trace.role == "executed"
            narrate(decision.trace)  # must render without error for every mode
            for s_trace in decision.shadow_traces:
                assert s_trace.role == "shadow"
                narrate(s_trace)  # every shadow narrates too


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
