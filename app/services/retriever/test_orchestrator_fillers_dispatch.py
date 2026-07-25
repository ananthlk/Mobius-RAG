"""Unit tests for _run_fillers_simple's ladder-walk dispatch: advance to the
next implemented rung on ZERO occupancy only (2026-07-23, Router/Vector
Search finding -- Filler b's honest-empty fix on junk-poisoned pools made
"viable later rung sits unused behind an honestly-empty first rung" a real
case, not theoretical). Mocked fillers, no real DB/network -- this exercises
only the dispatch loop's own control flow, not any filler's real logic.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.services.retriever.orchestrator import _run_fillers_simple
from app.services.retriever.fillers.contracts import FilledChunk, FilledShape, FilledSlot
from app.services.retriever.pool.contracts import PoolResult
from app.services.retriever.shape.slots import AnswerSlot
from app.services.router.allocation import RoutingLadder
from app.services.router.decision import RouterDecision, ResourcePosture
from app.services.router.tracing import DecisionTrace
from app.services.retriever.shape.gate import GateResult


def _slot(slot_id="direct_answer", capacity=5, required=True):
    return AnswerSlot(
        slot_id=slot_id, slot_semantics="direct_answer", capacity=capacity,
        rewritten_query="", required=required, priority=0,
    )


def _empty_pool():
    return PoolResult(query="", candidates=[], pool_ms=0)


def _filled_slot(slot_id, occupancy):
    chunks = [
        FilledChunk(chunk_id=f"c{i}", document_id="d", text="real text", source_type="internal")
        for i in range(occupancy)
    ]
    return FilledSlot(
        slot_id=slot_id, slot_semantics="direct_answer", capacity=5, required=True,
        chunks=chunks, occupancy=occupancy, under_filled=occupancy < 5, over_filled=False,
    )


def _decision(per_slot: dict, latency_allowance_ms: float = 60_000, per_slot_portfolio: dict | None = None):
    """latency_allowance_ms defaults generously -- these tests exercise
    dispatch/verdict control flow, not Router's own budget arithmetic
    (that's continuation.py's own test suite's job)."""
    trace = DecisionTrace(mode="test", role="executed")
    trace.latency_allowance_ms = latency_allowance_ms
    return RouterDecision(
        routing_ladder=RoutingLadder(
            per_slot=per_slot, per_slot_portfolio=per_slot_portfolio or {},
            outcome="ok", feasible=True,
        ),
        decision_id="test", resource_posture=ResourcePosture(), trace=trace,
    )


def _gate_result():
    return GateResult(
        query="q", normalized="q", d_codes=[], j_codes=[], p_codes=[],
        expansion_phrases=[], probe=None, process_intent=False, contour="exact",
        reason="", gate_ms=0,
    )


@pytest.mark.asyncio
async def test_advances_to_next_rung_on_zero_occupancy():
    """b (empty) -> a (real) should return a's result, not b's empty one --
    the exact scenario Router/Vector Search flagged as now-real."""
    slot = _slot()
    decision = _decision({"direct_answer": ["b", "a"]})

    with patch("app.services.retriever.orchestrator.fill_shape_vector") as mock_b, \
         patch("app.services.retriever.orchestrator.fill_shape_bm25") as mock_a:
        mock_b.return_value = FilledShape(
            slots=[_filled_slot("direct_answer", 0)],
            total_chunks_assigned=0, filling_strategy="", emit={},
        )
        mock_a.return_value = FilledShape(
            slots=[_filled_slot("direct_answer", 3)],
            total_chunks_assigned=3, filling_strategy="", emit={},
        )

        result = await _run_fillers_simple(
            db=None, slots=[slot], pool_results=[_empty_pool()],
            router_decision=decision, raw_query="q", gate_result=_gate_result(),
        )

    assert result.slots[0].occupancy == 3
    mock_b.assert_called_once()
    mock_a.assert_called_once()


class TestFillDepthEmit:
    """Data-collection posture (2026-07-24, Ananth's pullback): every
    executed rung on the chain/forced path records {strategy, capacity,
    occupancy} -- occupancy@capacity is each calibration cell's k0 (Router's
    requirement). Without it, forced calibration observations re-create the
    silent-k0-rebase gap Eval's audit just closed."""

    @pytest.mark.asyncio
    async def test_forced_single_strategy_records_one_fill_depth_entry(self):
        """The offline calibration case: a forced single-strategy slot
        emits exactly one fill_depth entry -- the forced strategy's true
        fill depth at the requested capacity, uncontaminated by fallback."""
        slot = _slot(capacity=10)
        decision = _decision({"direct_answer": ["a"]})  # forced single strategy

        with patch("app.services.retriever.orchestrator.fill_shape_bm25") as mock_a:
            mock_a.return_value = FilledShape(
                slots=[_filled_slot("direct_answer", 7)],
                total_chunks_assigned=7, filling_strategy="", emit={},
            )
            result = await _run_fillers_simple(
                db=None, slots=[slot], pool_results=[_empty_pool()],
                router_decision=decision, raw_query="q", gate_result=_gate_result(),
            )

        fd = result.emit["fill_depth"]["direct_answer"]
        assert fd == [{"strategy": "a", "capacity": 10, "occupancy": 7}]

    @pytest.mark.asyncio
    async def test_fill_depth_records_every_executed_rung_in_order(self):
        """A fallback chain (b empty -> a real) records BOTH rungs' fill
        depth in execution order, so a strategy's true depth is observed
        even when it wasn't the first rung tried."""
        slot = _slot(capacity=5)
        decision = _decision({"direct_answer": ["b", "a"]})

        with patch("app.services.retriever.orchestrator.fill_shape_vector") as mock_b, \
             patch("app.services.retriever.orchestrator.fill_shape_bm25") as mock_a:
            mock_b.return_value = FilledShape(
                slots=[_filled_slot("direct_answer", 0)],
                total_chunks_assigned=0, filling_strategy="", emit={},
            )
            mock_a.return_value = FilledShape(
                slots=[_filled_slot("direct_answer", 3)],
                total_chunks_assigned=3, filling_strategy="", emit={},
            )
            result = await _run_fillers_simple(
                db=None, slots=[slot], pool_results=[_empty_pool()],
                router_decision=decision, raw_query="q", gate_result=_gate_result(),
            )

        fd = result.emit["fill_depth"]["direct_answer"]
        assert fd == [
            {"strategy": "b", "capacity": 5, "occupancy": 0},
            {"strategy": "a", "capacity": 5, "occupancy": 3},
        ]

    @pytest.mark.asyncio
    async def test_portfolio_slot_leaves_fill_depth_empty(self):
        """Portfolio slots use portfolio_fill, not fill_depth -- the two
        paths are mutually exclusive, no double-counting."""
        slot = _slot(capacity=10)
        decision = _decision({"direct_answer": []}, per_slot_portfolio={"direct_answer": {"a": 3}})

        with patch("app.services.retriever.orchestrator.fill_shape_bm25") as mock_a:
            mock_a.return_value = FilledShape(
                slots=[_filled_slot("direct_answer", 3)],
                total_chunks_assigned=3, filling_strategy="", emit={},
            )
            result = await _run_fillers_simple(
                db=None, slots=[slot], pool_results=[_empty_pool()],
                router_decision=decision, raw_query="q", gate_result=_gate_result(),
            )

        assert result.emit["fill_depth"]["direct_answer"] == []
        assert result.emit["portfolio_fill"]["direct_answer"] == {"a": {"k_planned": 3, "k_delivered": 3}}


@pytest.mark.asyncio
async def test_stops_at_first_nonempty_rung_does_not_call_later_ones():
    """a (real, first) should short-circuit -- b never gets called."""
    slot = _slot()
    decision = _decision({"direct_answer": ["a", "b"]})

    with patch("app.services.retriever.orchestrator.fill_shape_bm25") as mock_a, \
         patch("app.services.retriever.orchestrator.fill_shape_vector") as mock_b:
        mock_a.return_value = FilledShape(
            slots=[_filled_slot("direct_answer", 5)],
            total_chunks_assigned=5, filling_strategy="", emit={},
        )

        result = await _run_fillers_simple(
            db=None, slots=[slot], pool_results=[_empty_pool()],
            router_decision=decision, raw_query="q", gate_result=_gate_result(),
        )

    assert result.slots[0].occupancy == 5
    mock_a.assert_called_once()
    mock_b.assert_not_called()


@pytest.mark.asyncio
async def test_weak_but_nonempty_result_does_not_advance():
    """1 chunk (weak, but nonempty) must NOT trigger a fall-through to the
    next rung -- that's explicitly Observer's future job, not this loop's."""
    slot = _slot()
    decision = _decision({"direct_answer": ["b", "a"]})

    with patch("app.services.retriever.orchestrator.fill_shape_vector") as mock_b, \
         patch("app.services.retriever.orchestrator.fill_shape_bm25") as mock_a:
        mock_b.return_value = FilledShape(
            slots=[_filled_slot("direct_answer", 1)],
            total_chunks_assigned=1, filling_strategy="", emit={},
        )

        result = await _run_fillers_simple(
            db=None, slots=[slot], pool_results=[_empty_pool()],
            router_decision=decision, raw_query="q", gate_result=_gate_result(),
        )

    assert result.slots[0].occupancy == 1
    mock_b.assert_called_once()
    mock_a.assert_not_called()


@pytest.mark.asyncio
async def test_attempt_spans_recorded_per_executed_rung():
    """Timing (Step 7, module-gates.md §7): each rung actually run must
    leave a (t_attempt_start_ms, t_attempt_end_ms) span behind, in
    execution order -- the concrete gap the gate named by line number."""
    slot = _slot()
    decision = _decision({"direct_answer": ["b", "a"]})

    with patch("app.services.retriever.orchestrator.fill_shape_vector") as mock_b, \
         patch("app.services.retriever.orchestrator.fill_shape_bm25") as mock_a:
        mock_b.return_value = FilledShape(
            slots=[_filled_slot("direct_answer", 0)],
            total_chunks_assigned=0, filling_strategy="", emit={},
        )
        mock_a.return_value = FilledShape(
            slots=[_filled_slot("direct_answer", 3)],
            total_chunks_assigned=3, filling_strategy="", emit={},
        )

        result = await _run_fillers_simple(
            db=None, slots=[slot], pool_results=[_empty_pool()],
            router_decision=decision, raw_query="q", gate_result=_gate_result(),
        )

    spans = result.emit["attempt_spans"]["direct_answer"]
    assert [s["strategy"] for s in spans] == ["b", "a"]
    for s in spans:
        assert s["t_attempt_start_ms"] <= s["t_attempt_end_ms"]
        assert s["attempt_number"] >= 1


@pytest.mark.asyncio
async def test_all_rungs_empty_returns_empty_slot_no_crash():
    slot = _slot()
    decision = _decision({"direct_answer": ["b", "a"]})

    with patch("app.services.retriever.orchestrator.fill_shape_vector") as mock_b, \
         patch("app.services.retriever.orchestrator.fill_shape_bm25") as mock_a:
        mock_b.return_value = FilledShape(
            slots=[_filled_slot("direct_answer", 0)],
            total_chunks_assigned=0, filling_strategy="", emit={},
        )
        mock_a.return_value = FilledShape(
            slots=[_filled_slot("direct_answer", 0)],
            total_chunks_assigned=0, filling_strategy="", emit={},
        )

        result = await _run_fillers_simple(
            db=None, slots=[slot], pool_results=[_empty_pool()],
            router_decision=decision, raw_query="q", gate_result=_gate_result(),
        )

    assert result.slots[0].occupancy == 0
    assert result.slots[0].under_filled
    mock_b.assert_called_once()
    mock_a.assert_called_once()


@pytest.mark.asyncio
async def test_single_slot_error_gets_no_retry_turn():
    """Deliberate, documented Router behavior (continuation.py): ERROR is
    ride-along-only, it never self-justifies a turn. With only one required
    slot and no sibling to justify a turn, an errored slot gets no retry --
    this is a real design change from the pre-continuation advance-on-empty
    behavior (which retried unconditionally), not a regression."""
    slot = _slot()
    decision = _decision({"direct_answer": ["b", "a"]})

    with patch("app.services.retriever.orchestrator.fill_shape_vector") as mock_b, \
         patch("app.services.retriever.orchestrator.fill_shape_bm25") as mock_a:
        mock_b.side_effect = RuntimeError("boom")

        result = await _run_fillers_simple(
            db=None, slots=[slot], pool_results=[_empty_pool()],
            router_decision=decision, raw_query="q", gate_result=_gate_result(),
        )

    assert result.slots[0].occupancy == 0
    mock_a.assert_not_called()


@pytest.mark.asyncio
async def test_errored_slot_rides_along_when_a_sibling_justifies_a_turn():
    """A required sibling with a real WOULD_BENEFIT verdict justifies a
    turn; the errored slot rides along for free since its next rung (BM25,
    fast) fits inside the envelope set by the sibling's next rung (web
    search, slow) -- exactly the ride-along mechanism Router/Eval designed."""
    slot1 = _slot(slot_id="direct_answer")
    slot2 = _slot(slot_id="fanout_0")
    decision = _decision({"direct_answer": ["b", "a"], "fanout_0": ["c", "d"]})

    with patch("app.services.retriever.orchestrator.fill_shape_vector") as mock_b, \
         patch("app.services.retriever.orchestrator.fill_shape_bm25") as mock_a, \
         patch("app.services.retriever.orchestrator.fill_shape_llm_retrieval",
               new_callable=AsyncMock) as mock_c, \
         patch("app.services.retriever.orchestrator.fill_shape_external",
               new_callable=AsyncMock) as mock_d:
        mock_b.side_effect = RuntimeError("boom")
        mock_a.return_value = FilledShape(
            slots=[_filled_slot("direct_answer", 2)],
            total_chunks_assigned=2, filling_strategy="", emit={},
        )
        mock_c.return_value = FilledShape(
            slots=[_filled_slot("fanout_0", 0)],
            total_chunks_assigned=0, filling_strategy="", emit={},
        )
        mock_d.return_value = FilledShape(
            slots=[_filled_slot("fanout_0", 3)],
            total_chunks_assigned=3, filling_strategy="", emit={},
        )

        result = await _run_fillers_simple(
            db=None, slots=[slot1, slot2], pool_results=[_empty_pool()],
            router_decision=decision, raw_query="q", gate_result=_gate_result(),
        )

    by_id = {s.slot_id: s for s in result.slots}
    assert by_id["fanout_0"].occupancy == 3       # c empty -> WOULD_BENEFIT -> justifies turn 2 with d
    assert by_id["direct_answer"].occupancy == 2  # b errored -> rode along -> tried a, succeeded
    mock_a.assert_called_once()
    mock_d.assert_called_once()


@pytest.mark.asyncio
async def test_not_ready_d_gets_deferred_behind_a_ready_alternative():
    """Real ask (2026-07-24, Ananth via Web Search): Router shouldn't commit
    to "d" as a slot's next attempt while its speculative prescreen isn't
    ready -- prioritize an already-ready strategy instead. A not-done
    prescreen_search_task defers "d" behind "a" for THIS attempt, without
    permanently dropping "d" from the slot's ladder."""
    slot = _slot()
    decision = _decision({"direct_answer": ["d", "a"]})
    not_done_task = asyncio.get_event_loop().create_future()  # never resolved -- .done() is False

    with patch("app.services.retriever.orchestrator.fill_shape_external",
               new_callable=AsyncMock) as mock_d, \
         patch("app.services.retriever.orchestrator.fill_shape_bm25") as mock_a:
        mock_a.return_value = FilledShape(
            slots=[_filled_slot("direct_answer", 2)],
            total_chunks_assigned=2, filling_strategy="", emit={},
        )

        result = await _run_fillers_simple(
            db=None, slots=[slot], pool_results=[_empty_pool()],
            router_decision=decision, raw_query="q", gate_result=_gate_result(),
            prescreen_search_task=not_done_task,
        )

    assert result.slots[0].occupancy == 2
    mock_a.assert_called_once()
    mock_d.assert_not_called()
    not_done_task.cancel()


@pytest.mark.asyncio
async def test_ready_d_runs_normally_no_deferral():
    """Complementary check: an already-done prescreen_search_task must NOT
    trigger deferral -- "d" runs first exactly as the ladder specifies."""
    slot = _slot()
    decision = _decision({"direct_answer": ["d", "a"]})
    done_task = asyncio.get_event_loop().create_future()
    done_task.set_result(object())  # any real result -- .done() is True

    with patch("app.services.retriever.orchestrator.fill_shape_external",
               new_callable=AsyncMock) as mock_d, \
         patch("app.services.retriever.orchestrator.fill_shape_bm25") as mock_a:
        mock_d.return_value = FilledShape(
            slots=[_filled_slot("direct_answer", 5)],
            total_chunks_assigned=5, filling_strategy="", emit={},
        )

        result = await _run_fillers_simple(
            db=None, slots=[slot], pool_results=[_empty_pool()],
            router_decision=decision, raw_query="q", gate_result=_gate_result(),
            prescreen_search_task=done_task,
        )

    assert result.slots[0].occupancy == 5
    mock_d.assert_called_once()
    mock_a.assert_not_called()


@pytest.mark.asyncio
async def test_prescreen_not_ready_deferred_emitted_distinctly_when_d_never_runs():
    """Router's required distinction (router-build-spec.md Sec11 boundary 3):
    if "d" gets deferred for readiness and then never runs at all this
    query (a satisfies first), that must surface as a DISTINCT emit label
    -- not folded into final_verdicts/EXHAUSTED_ATTEMPTS -- so Eval's
    calibration cells for "d" don't absorb readiness noise as if it were
    the strategy's own performance."""
    slot = _slot()
    decision = _decision({"direct_answer": ["d", "a"]})
    not_done_task = asyncio.get_event_loop().create_future()

    with patch("app.services.retriever.orchestrator.fill_shape_external",
               new_callable=AsyncMock) as mock_d, \
         patch("app.services.retriever.orchestrator.fill_shape_bm25") as mock_a:
        mock_a.return_value = FilledShape(
            slots=[_filled_slot("direct_answer", 5)],
            total_chunks_assigned=5, filling_strategy="", emit={},
        )

        result = await _run_fillers_simple(
            db=None, slots=[slot], pool_results=[_empty_pool()],
            router_decision=decision, raw_query="q", gate_result=_gate_result(),
            prescreen_search_task=not_done_task,
        )

    assert result.emit["prescreen_not_ready_deferred_slots"] == ["direct_answer"]
    assert result.emit["final_verdicts"]["direct_answer"] == "SATISFIED"  # a's success, not a "d" failure
    assert result.emit["executed_order"]["direct_answer"] == ["a"]
    mock_d.assert_not_called()
    not_done_task.cancel()


class TestPortfolioExecution:
    """Blend model (2026-07-24, blend-model-design.md, Router-approved
    seam): a slot with a non-empty RoutingLadder.per_slot_portfolio runs
    ALL its strategies concurrently in turn 0, each capped to its own k_i,
    with everyone's chunks retained -- not the chain-mode single-rung-per-
    turn dance. Chain-mode slots (empty portfolio, today's default) must
    keep working exactly as before -- covered by every other test in this
    file, none of which set per_slot_portfolio.
    """

    @pytest.mark.asyncio
    async def test_portfolio_runs_all_strategies_concurrently_and_retains_both(self):
        slot = _slot(capacity=10)
        decision = _decision({"direct_answer": []}, per_slot_portfolio={"direct_answer": {"a": 3, "b": 2}})

        with patch("app.services.retriever.orchestrator.fill_shape_bm25") as mock_a, \
             patch("app.services.retriever.orchestrator.fill_shape_vector") as mock_b:
            mock_a.return_value = FilledShape(
                slots=[_filled_slot("direct_answer", 3)],
                total_chunks_assigned=3, filling_strategy="", emit={},
            )
            mock_b.return_value = FilledShape(
                slots=[_filled_slot("direct_answer", 2)],
                total_chunks_assigned=2, filling_strategy="", emit={},
            )

            result = await _run_fillers_simple(
                db=None, slots=[slot], pool_results=[_empty_pool()],
                router_decision=decision, raw_query="q", gate_result=_gate_result(),
            )

        mock_a.assert_called_once()
        mock_b.assert_called_once()
        # capacity-adjusted slot copies -- each filler sees its own k_i, not slot.capacity
        assert mock_a.call_args[0][1].slots[0].capacity == 3
        assert mock_b.call_args[0][1].slots[0].capacity == 2
        assert result.slots[0].occupancy == 5  # 3 + 2, both retained
        assert set(result.emit["executed_order"]["direct_answer"]) == {"a", "b"}
        assert result.emit["portfolio_fill"]["direct_answer"] == {
            "a": {"k_planned": 3, "k_delivered": 3},
            "b": {"k_planned": 2, "k_delivered": 2},
        }

    @pytest.mark.asyncio
    async def test_portfolio_verdict_measures_against_plan_not_slot_capacity(self):
        """Router's correction (2026-07-24): a budget-bound portfolio that
        delivers exactly what it planned must be SATISFIED even though
        slot.capacity is much larger -- capacity is the ceiling that shaped
        the plan, not the success bar."""
        slot = _slot(capacity=10)
        decision = _decision({"direct_answer": []}, per_slot_portfolio={"direct_answer": {"a": 2}})

        with patch("app.services.retriever.orchestrator.fill_shape_bm25") as mock_a:
            mock_a.return_value = FilledShape(
                slots=[_filled_slot("direct_answer", 2)],
                total_chunks_assigned=2, filling_strategy="", emit={},
            )
            result = await _run_fillers_simple(
                db=None, slots=[slot], pool_results=[_empty_pool()],
                router_decision=decision, raw_query="q", gate_result=_gate_result(),
            )

        assert result.emit["final_verdicts"]["direct_answer"] == "SATISFIED"
        assert result.slots[0].under_filled is True  # true vs slot.capacity=10, but that's not the verdict bar

    @pytest.mark.asyncio
    async def test_portfolio_no_expansion_even_when_underdelivered(self):
        """No multi-turn expansion for portfolio slots yet (deferred to
        Synthesis's CoverageDiagnostic) -- an under-delivered portfolio
        still resolves terminally in turn 0, not WOULD_BENEFIT."""
        slot = _slot(capacity=10)
        decision = _decision({"direct_answer": []}, per_slot_portfolio={"direct_answer": {"a": 5}})

        with patch("app.services.retriever.orchestrator.fill_shape_bm25") as mock_a:
            mock_a.return_value = FilledShape(
                slots=[_filled_slot("direct_answer", 2)],  # under-delivers its own plan of 5
                total_chunks_assigned=2, filling_strategy="", emit={},
            )
            result = await _run_fillers_simple(
                db=None, slots=[slot], pool_results=[_empty_pool()],
                router_decision=decision, raw_query="q", gate_result=_gate_result(),
            )

        assert result.emit["final_verdicts"]["direct_answer"] == "EXHAUSTED_ATTEMPTS"
        mock_a.assert_called_once()  # never retried/expanded
        assert result.emit["executed_order"]["direct_answer"] == ["a"]
