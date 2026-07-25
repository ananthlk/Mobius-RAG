"""Real-seam tests — Router runs against the ACTUAL Slots contract, not stand-ins.

P0 regression guard (Retriever's fleet inspection 2026-07-23): Router
previously defined a local AnswerSlot with fields the real contract doesn't
have (query_class, max_attempts), so 164 tests passed against a fake while the
real seam would have raised AttributeError on first contact. These tests make
that class of drift impossible to reintroduce silently:

  1. IDENTITY: Router's AnswerSlot IS shape.slots.AnswerSlot — same object.
  2. END-TO-END: a genuine AnswerShapeResult (built with the real dataclasses,
     including a FAN_OUT multi-slot shape with a required=False slot) flows
     through all three allocators and route() without touching any field the
     real contract doesn't define.
"""

import asyncio
import dataclasses

import pytest

from app.services.retriever.shape.slots import (
    AnswerShapeResult,
    AnswerSlot as RealAnswerSlot,
)
from app.services.router import allocation as allocation_module
from app.services.router.allocation import allocate_strategies
from app.services.router.bayesian_optimizer import optimize_allocation_bayesian
from app.services.router.optimizer import optimize_allocation
from app.services.router.decision import ResourcePosture, RoutingContext
from app.services.router.router import route


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


def _real_shape_result() -> AnswerShapeResult:
    """A FAN_OUT-style AnswerShapeResult built from the REAL dataclasses."""
    return AnswerShapeResult(
        query="What are Sunshine Health's behavioral-health billing rules?",
        slots=[
            RealAnswerSlot(
                slot_id="fanout_0", slot_semantics="thematic_exploration",
                capacity=4, rewritten_query="prior authorization rules",
                required=True, priority=0,
            ),
            RealAnswerSlot(
                slot_id="fanout_1", slot_semantics="thematic_exploration",
                capacity=4, rewritten_query="timely filing rules",
                required=True, priority=1,
            ),
            RealAnswerSlot(
                slot_id="external_context", slot_semantics="external_context",
                capacity=2, rewritten_query="",
                required=False, priority=2,
            ),
        ],
        reason="FAN_OUT: test shape",
    )


def _posture_dict():
    return {
        "speed_budget": "interactive",
        "confidence_bar": 0.85,
        "caller_mode": "chat.default",
        "max_attempts_per_slot": 6,
        "gate_j_codes": ["payor.sunshine_health"],
    }


def _pool_for(shape: AnswerShapeResult) -> dict:
    return {
        s.slot_id: {"top_score_percentile": 0.60, "pool_size": 400}
        for s in shape.slots
    }


class TestSeamIdentity:
    def test_router_answer_slot_is_the_real_class(self):
        """Not a copy, not a compatible stand-in — the SAME class object."""
        assert allocation_module.AnswerSlot is RealAnswerSlot

    def test_real_contract_has_no_stand_in_fields(self):
        """The two phantom fields stay dead: if someone adds them back to the
        real contract this test flags it for a deliberate re-review here."""
        fields = {f.name for f in dataclasses.fields(RealAnswerSlot)}
        assert fields == {"slot_id", "slot_semantics", "capacity",
                          "rewritten_query", "required", "priority"}


class TestRealShapeEndToEnd:
    def test_all_three_allocators_accept_real_slots(self):
        shape = _real_shape_result()
        pool = _pool_for(shape)
        for fn in (allocate_strategies, optimize_allocation, optimize_allocation_bayesian):
            ladder = fn(shape.slots, pool, _posture_dict())
            assert set(ladder.per_slot.keys()) == {"fanout_0", "fanout_1", "external_context"}
            # required fanout slots gated; external slot OPTIONAL + d-only + capped
            assert ladder.per_slot_status["fanout_0"] in ("CLEARED", "UNDER_CONFIDENT")
            assert ladder.per_slot_status["external_context"] == "OPTIONAL"
            assert ladder.per_slot["external_context"] == ["d"]

    def test_route_end_to_end_with_real_shape(self):
        """Full route() against real Slots output — the exact call that would
        have raised AttributeError before the fix."""
        shape = _real_shape_result()
        factory = FakeSessionFactory()
        decision = asyncio.run(route(factory, RoutingContext(
            query=shape.query,
            agent_id="router-4c",
            resource_posture=ResourcePosture(max_attempts_per_slot=6),
            pool_metadata={
                s.slot_id: {
                    "top_score_percentile": 0.60, "pool_size": 400,
                    "priority": s.priority, "required": s.required,
                    "slot_semantics": s.slot_semantics, "capacity": s.capacity,
                    "rewritten_query": s.rewritten_query,
                }
                for s in shape.slots
            },
            gate_j_codes=["payor.sunshine_health"],
        )))
        assert set(decision.routing_ladder.per_slot.keys()) == \
            {"fanout_0", "fanout_1", "external_context"}
        assert len(factory.calls) == 1  # ONE-WRITER intact through the real seam
        assert decision.routing_ladder.per_slot_status["external_context"] == "OPTIONAL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestVerbatimSlotsSeam:
    """ctx.slots: the orchestrator passes AnswerShapeResult.slots VERBATIM —
    no shim reconstruction (completes the seam Retriever caught as
    comment-without-code)."""

    def test_route_consumes_real_slots_verbatim(self):
        shape = _real_shape_result()
        factory = FakeSessionFactory()
        decision = asyncio.run(route(factory, RoutingContext(
            query=shape.query,
            agent_id="router-4c",
            resource_posture=ResourcePosture(max_attempts_per_slot=6),
            slots=shape.slots,  # REAL objects, verbatim
            pool_metadata=_pool_for(shape),  # depth signals only
            gate_j_codes=["payor.sunshine_health"],
            gate_d_codes=["claims.timely_filing"],
        )))
        assert set(decision.routing_ladder.per_slot.keys()) == \
            {"fanout_0", "fanout_1", "external_context"}
        assert decision.routing_ladder.per_slot_status["external_context"] == "OPTIONAL"
        assert len(factory.calls) == 1

    def test_verbatim_slots_win_over_pool_metadata_shim(self):
        """When both are supplied, ctx.slots is authoritative — pool_metadata
        contributes only depth signals, never slot construction."""
        shape = _real_shape_result()
        factory = FakeSessionFactory()
        decision = asyncio.run(route(factory, RoutingContext(
            query=shape.query, agent_id="router-4c",
            resource_posture=ResourcePosture(max_attempts_per_slot=6),
            slots=shape.slots,
            # metadata deliberately claims different required/semantics —
            # must be IGNORED for slot construction
            pool_metadata={
                s.slot_id: {"top_score_percentile": 0.60, "pool_size": 400,
                            "required": True, "slot_semantics": "direct_answer"}
                for s in shape.slots
            },
            gate_j_codes=["payor.sunshine_health"],
        )))
        # external_context slot kept its REAL required=False + semantics
        assert decision.routing_ladder.per_slot_status["external_context"] == "OPTIONAL"
        assert decision.routing_ladder.per_slot["external_context"] == ["d"]
