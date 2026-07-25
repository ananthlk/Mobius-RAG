"""Real end-to-end integration tests: real DB, real Vertex embeddings, real
Pool/Router/Fillers/Synthesis -- no mocks. Built 2026-07-24 in direct
response to two real production bugs this session's manual live-tracing
caught that NO existing test suite caught (both were unit/mocked-fixture
tests, none exercised the full live pipeline at the real production shape):

  1. Router's payload gate collapsing the ladder to a single cheap strategy
     (PAYLOAD_TOKENS_PER_CHUNK=1000 x capacity=10 > token_budget=3000,
     mathematically impossible for a/b/c to ever pass) -- fixed in Router.
  2. Observer's _evaluate_fact_store treating any fact-store hit as
     SATISFIED regardless of the slot's real capacity -- fixed in Observer.

Both bugs were deterministic (same result on every real query of a given
shape), yet neither was caught before real bank queries surfaced them --
because nothing exercised run_retriever_partial() against a real DB at the
real production ResourcePosture shape (capacity=10, token_budget=3000,
real_time speed_budget). These tests close that gap.

Requires a live DB connection (DATABASE_URL) and Vertex credentials (ADC) --
same requirements as running the app itself. Not mocked, not marked skip:
this IS the safety net that should have existed before those two bugs ever
shipped. Run with:
    .venv/bin/python -m pytest app/services/retriever/test_integration_production_shapes.py -v

Real wall-clock cost: each test makes real DB + embedding calls (~5-15s).
Not a fast unit test file -- deliberately so; that's what makes it able to
catch what these tests catch.
"""

import pytest

from app.database import AsyncSessionLocal
from app.services.retriever.orchestrator import run_retriever_partial
from app.services.retriever.shape.contracts import ReformatPosture

pytestmark = pytest.mark.integration


# Real bank queries (eval/queries_cmhc.yaml) with a real, well-populated
# corpus match -- chosen because both bugs above reproduced on them live.
_REAL_CONTENT_QUERY = (
    "Does Sunshine Health require prior authorization for residential "
    "substance use treatment under code H0019?"
)
_REAL_TIMELY_FILING_QUERY = "What is the timely filing deadline for Sunshine Health FL Medicaid claims?"

# A query Gate should route to OUT_OF_SCOPE/UNCLEAR -- exercises the
# no-retrieval path end-to-end (zero Pool/Router/Fillers cost).
_OUT_OF_SCOPE_QUERY = "What's the best pizza topping combination?"


class TestProductionShapeRegression:
    """The exact regression class both bugs belonged to: a REQUIRED,
    multi-capacity slot at real_time speed_budget must not silently
    collapse to a single cheap strategy when the corpus has real, strong
    matching content. This is the test that should have existed before
    PAYLOAD_TOKENS_PER_CHUNK and _evaluate_fact_store shipped."""

    @pytest.mark.asyncio
    async def test_real_content_query_does_not_collapse_to_single_strategy(self):
        async with AsyncSessionLocal() as db:
            result = await run_retriever_partial(db, _REAL_CONTENT_QUERY)

        assert result.router_decision is not None
        ladder = result.router_decision.routing_ladder.per_slot.get("direct_answer", [])
        # Regression guard for the exact bug: a real content query with a
        # strong pool (Pool's own candidates score >0.8 vector similarity
        # for this query, verified live 2026-07-24) must plan MORE than
        # just the cheapest strategy -- a single-strategy ladder here means
        # the payload/latency gates are mis-rejecting real content again.
        assert len(ladder) > 1, (
            f"ladder collapsed to a single strategy {ladder!r} -- this is the exact "
            "payload-gate/priors collapse this test exists to catch"
        )

    @pytest.mark.asyncio
    async def test_real_content_query_actually_executes_more_than_fact_store(self):
        """Planning alone isn't enough (this is exactly what the Observer
        capacity-1 bug hid): assert the EXECUTED chunks include real
        content, not just a single fact-store hit that satisfied the loop
        prematurely."""
        async with AsyncSessionLocal() as db:
            result = await run_retriever_partial(db, _REAL_CONTENT_QUERY)

        assert result.filled_shape is not None
        slot = result.filled_shape.slots[0]
        # Regression guard: a slot with real corpus content available must
        # not settle for a single chunk -- occupancy=1 here (with capacity
        # >1) is the fact-store-satisfies-alone bug reproducing.
        assert slot.occupancy > 1, (
            f"occupancy={slot.occupancy} on a capacity={slot.capacity} slot -- "
            "looks like the fact-store-satisfies-alone bug (Observer's "
            "_evaluate_fact_store capacity-1 assumption)"
        )
        source_types = {c.source_type for c in slot.chunks}
        assert source_types - {"fact_store"}, (
            "every executed chunk came from fact_store alone -- a/b/c/d never ran "
            "despite being planned (execution-time collapse, not a planning bug)"
        )

    @pytest.mark.asyncio
    async def test_synthesis_produces_real_citations_for_content_query(self):
        """Full pipeline coherence check: Synthesis's compiled output must
        reflect the real retrieved content, not an empty/degenerate result."""
        async with AsyncSessionLocal() as db:
            result = await run_retriever_partial(db, _REAL_CONTENT_QUERY)

        assert result.synthesis is not None
        assert len(result.synthesis.citations) > 1
        assert result.synthesis.telemetry.unverified_citations >= 0  # real telemetry, not a crash


class TestNoRetrievalPostures:
    """Zero-cost path: CLARIFY/DECLINE/CLARIFY_REPHRASE must genuinely
    skip Pool/Router/Fillers, not just return an empty result after paying
    the full retrieval cost anyway."""

    @pytest.mark.asyncio
    async def test_out_of_scope_query_skips_retrieval_entirely(self):
        async with AsyncSessionLocal() as db:
            result = await run_retriever_partial(db, _OUT_OF_SCOPE_QUERY)

        assert result.router_decision is None
        assert result.filled_shape is None
        assert result.synthesis is None
        assert result.pool_ms == 0
        assert result.router_ms == 0
        assert result.fillers_ms == 0


class TestFullPipelineCoherence:
    """A handful of distinct real bank queries, asserting the whole chain
    runs to completion without crashing and produces internally-consistent
    telemetry -- not testing quality/recall (that's Eval's calibration job),
    just that the pipeline doesn't silently break on real, varied input."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", [_REAL_CONTENT_QUERY, _REAL_TIMELY_FILING_QUERY])
    async def test_pipeline_completes_and_timing_is_internally_consistent(self, query):
        async with AsyncSessionLocal() as db:
            result = await run_retriever_partial(db, query)

        assert result.pipeline_complete is True
        # Segment times should roughly sum to (or be less than) total_ms --
        # a real internal-consistency check, not tautological: if any
        # segment silently double-counted or omitted work, this catches it.
        segment_sum = (
            result.gate_ms + result.reformat_ms + result.slots_ms
            + result.pool_ms + result.router_ms + result.fillers_ms + result.synthesis_ms
        )
        assert segment_sum <= result.total_ms + 50, (
            f"segments sum to {segment_sum}ms but total_ms={result.total_ms}ms -- "
            "timing bookkeeping is inconsistent"
        )

    @pytest.mark.asyncio
    async def test_decision_persists_without_schema_error(self):
        """Regression guard for the THREE separate persist schema bugs
        found and fixed this session (is_calibration missing, 14 columns
        missing, strategy_used NOT NULL) -- confirms a real decision row
        actually lands in rag_query_decisions with no swallowed exception."""
        from sqlalchemy import text

        async with AsyncSessionLocal() as db:
            result = await run_retriever_partial(db, _REAL_TIMELY_FILING_QUERY)
        decision_id = result.router_decision.decision_id

        async with AsyncSessionLocal() as db2:
            row = (await db2.execute(
                text("SELECT id FROM rag_query_decisions WHERE id = :id"), {"id": decision_id},
            )).fetchone()
        assert row is not None, (
            "decision row did not persist -- a persist_decision() schema error is "
            "being silently swallowed again (see persist_decision's own non-fatal "
            "try/except -- this test is the only thing that catches a REGRESSION there)"
        )


class TestAuthorityRequirementThreading:
    """Real gap found 2026-07-24 while building this test file: Chat's
    caller-declared authority_requirement ("any" | "citable_required") has
    NO path into run_retriever_partial() today -- Structure supports the
    param, but the orchestrator's top-level entry point never accepts or
    forwards it, so every live call silently defaults to "any" regardless
    of what a real caller would want. Logged, not silently fixed here --
    same class of gap as token_budget_for_retrieval before this session's
    fix, but not yet actioned. This test documents the CURRENT (gap) state
    honestly rather than asserting the desired behavior and failing
    confusingly."""

    @pytest.mark.asyncio
    async def test_authority_requirement_has_no_threading_path_today(self):
        async with AsyncSessionLocal() as db:
            result = await run_retriever_partial(db, _REAL_CONTENT_QUERY)
        # Documents the gap, not a desired end-state: remove/update this
        # test once run_retriever_partial exposes a real
        # authority_requirement param threaded from a caller.
        assert result.structure.resource_posture.authority_requirement == "any"
