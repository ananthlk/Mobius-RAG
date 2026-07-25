"""Unit tests for Filler s (Payor Platform Fact Store strategy)."""

import pytest

from app.services.retriever.fillers.filler_s import (
    _build_request,
    _gate_passes,
    _is_conceptual,
    _stable_fact_id,
    fill_shape_fact_store,
)
from app.services.retriever.fillers.payer_context import extract_payer_slug
from app.services.retriever.pool.contracts import PoolResult
from app.services.retriever.shape.slots import AnswerShapeResult, AnswerSlot


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class _FakeClient:
    """Records calls, returns a scripted response (or raises)."""

    def __init__(self, response: _FakeResponse | None = None, exc: Exception | None = None):
        self._response = response
        self._exc = exc
        self.calls: list[tuple[str, dict]] = []

    async def post(self, url: str, json: dict):
        self.calls.append((url, json))
        if self._exc is not None:
            raise self._exc
        return self._response


def _shape_with_direct_answer(capacity: int = 1) -> AnswerShapeResult:
    return AnswerShapeResult(
        slots=[
            AnswerSlot(
                slot_id="direct_answer",
                slot_semantics="direct_answer",
                capacity=capacity,
                required=True,
                priority=0,
            ),
            AnswerSlot(
                slot_id="fanout_0",
                slot_semantics="thematic_exploration",
                capacity=2,
                required=True,
                priority=1,
            ),
        ]
    )


def _empty_pool() -> PoolResult:
    return PoolResult(query="test", candidates=[], pool_ms=0)


HIT_RESPONSE = {
    "hit": True,
    "served": {
        "record_type": "atomic",
        "predicate": "phone",
        "answer_text": "1-844-477-8313",
        "value": "1-844-477-8313",
        "source_ref": {"doc_id": "doc_sunshine_1", "url": "https://sunshine.example/contact"},
        "authority_level": "payer_website",
        "scope": None,
        "score": 0.91,
    },
    "shortlist": [],
    "gate": {"payer_key": "Sunshine Health|FL|", "applied": True, "excluded_n": 3},
    "blend": {"alpha": 0.5, "beta": 0.5, "tau": 0.75, "version": "v1"},
    "verify": None,
    "telemetry_id": "abc-123",
}

MISS_RESPONSE = {
    "hit": False,
    "served": None,
    "shortlist": [],
    "gate": {"payer_key": "Sunshine Health|FL|", "applied": True, "excluded_n": 3},
    "blend": {"alpha": 0.5, "beta": 0.5, "tau": 0.75, "version": "v1"},
    "verify": None,
    "telemetry_id": "def-456",
}


class TestGateCondition:
    def test_has_payor_tag_via_shared_payer_context_helper(self):
        """Filler s reuses the fleet-shared extract_payer_slug (payer_context.py,
        built anticipating c/d/f/s) rather than re-deriving this inline."""
        assert extract_payer_slug(["j:payor.sunshine_health", "d:prior_auth"]) == "sunshine_health"
        assert extract_payer_slug(["d:prior_auth", "p:process"]) is None

    def test_is_conceptual(self):
        assert _is_conceptual("What is the philosophy behind prior auth?")
        assert _is_conceptual("Explain the credentialing process")
        assert not _is_conceptual("What is the phone number for Sunshine Health?")

    def test_gate_requires_direct_answer_semantics(self):
        thematic_slot = AnswerSlot(
            slot_id="fanout_0", slot_semantics="thematic_exploration", capacity=2
        )
        assert not _gate_passes(
            "phone number for sunshine health", ["j:payor.sunshine_health"], thematic_slot
        )

    def test_gate_requires_payor_tag(self):
        direct_slot = AnswerSlot(slot_id="direct_answer", slot_semantics="direct_answer", capacity=1)
        assert not _gate_passes("phone number for sunshine health", ["d:contact"], direct_slot)

    def test_gate_rejects_conceptual_query(self):
        direct_slot = AnswerSlot(slot_id="direct_answer", slot_semantics="direct_answer", capacity=1)
        assert not _gate_passes(
            "philosophy of sunshine health prior auth",
            ["j:payor.sunshine_health"],
            direct_slot,
        )

    def test_gate_passes_when_all_conditions_met(self):
        direct_slot = AnswerSlot(slot_id="direct_answer", slot_semantics="direct_answer", capacity=1)
        assert _gate_passes(
            "phone number for sunshine health", ["j:payor.sunshine_health"], direct_slot
        )


class TestRequestPayload:
    def test_tags_only_v1_no_embedding_key(self):
        """CRITICAL: v1 must never send `embedding` -- see module spec §6
        (sending it rescales the payor blend formula and risks regressing
        currently-good serves; deferred to a bundled fast-follow)."""
        payload = _build_request(
            "phone number", ["d:contact", "p:phone", "j:payor.sunshine_health"]
        )
        assert "embedding" not in payload
        assert payload["query"] == "phone number"
        assert payload["d_tags"] == ["d:contact"]
        assert payload["p_tags"] == ["p:phone"]
        assert payload["j_tags"] == ["j:payor.sunshine_health"]
        assert payload["intent_scope"] is None
        assert payload["k"] == 5


class TestFillShapeFactStore:
    @pytest.mark.asyncio
    async def test_hit_fills_direct_answer_slot(self):
        client = _FakeClient(response=_FakeResponse(200, HIT_RESPONSE))
        shape = _shape_with_direct_answer()

        filled = await fill_shape_fact_store(
            _empty_pool(),
            shape,
            "phone number for sunshine health",
            tag_matches=["j:payor.sunshine_health", "d:contact"],
            http_client=client,
        )

        direct_slot = filled.slots[0]
        assert direct_slot.occupancy == 1
        chunk = direct_slot.chunks[0]
        assert chunk.chunk_id == _stable_fact_id(HIT_RESPONSE["served"])
        assert chunk.document_id == "doc_sunshine_1"
        assert chunk.text == "1-844-477-8313"
        assert chunk.source_type == "fact_store"
        assert chunk.original_score == 0.91
        assert chunk.assignment_reason == "fact_store_hit"
        assert chunk.is_neighbor is False

        # thematic slot untouched -- no natural per-theme analog for a fact-store hit.
        thematic_slot = filled.slots[1]
        assert thematic_slot.occupancy == 0

        assert len(client.calls) == 1
        assert client.calls[0][0].endswith("/api/skills/v1/fact_query")

    @pytest.mark.asyncio
    async def test_miss_leaves_slot_empty_no_special_casing(self):
        client = _FakeClient(response=_FakeResponse(200, MISS_RESPONSE))
        shape = _shape_with_direct_answer()

        filled = await fill_shape_fact_store(
            _empty_pool(),
            shape,
            "phone number for sunshine health",
            tag_matches=["j:payor.sunshine_health"],
            http_client=client,
        )

        direct_slot = filled.slots[0]
        assert direct_slot.occupancy == 0
        assert direct_slot.under_filled

    @pytest.mark.asyncio
    async def test_gate_fail_makes_no_http_call(self):
        client = _FakeClient(response=_FakeResponse(200, HIT_RESPONSE))
        shape = _shape_with_direct_answer()

        filled = await fill_shape_fact_store(
            _empty_pool(),
            shape,
            "philosophy of sunshine health prior auth",  # conceptual -> gate fails
            tag_matches=["j:payor.sunshine_health"],
            http_client=client,
        )

        assert filled.slots[0].occupancy == 0
        assert len(client.calls) == 0  # no HTTP call made at all

    @pytest.mark.asyncio
    async def test_network_error_is_clean_miss_not_a_crash(self):
        client = _FakeClient(exc=ConnectionError("boom"))
        shape = _shape_with_direct_answer()

        filled = await fill_shape_fact_store(
            _empty_pool(),
            shape,
            "phone number for sunshine health",
            tag_matches=["j:payor.sunshine_health"],
            http_client=client,
        )

        assert filled.slots[0].occupancy == 0
        assert filled.slots[0].under_filled

    @pytest.mark.asyncio
    async def test_non_200_status_is_clean_miss(self):
        client = _FakeClient(response=_FakeResponse(500, {}))
        shape = _shape_with_direct_answer()

        filled = await fill_shape_fact_store(
            _empty_pool(),
            shape,
            "phone number for sunshine health",
            tag_matches=["j:payor.sunshine_health"],
            http_client=client,
        )

        assert filled.slots[0].occupancy == 0

    @pytest.mark.asyncio
    async def test_missing_source_ref_falls_back_to_synthetic_document_id(self):
        response = {
            **HIT_RESPONSE,
            "served": {**HIT_RESPONSE["served"], "source_ref": {}},
        }
        client = _FakeClient(response=_FakeResponse(200, response))
        shape = _shape_with_direct_answer()

        filled = await fill_shape_fact_store(
            _empty_pool(),
            shape,
            "phone number for sunshine health",
            tag_matches=["j:payor.sunshine_health"],
            http_client=client,
        )

        chunk = filled.slots[0].chunks[0]
        assert chunk.document_id == chunk.chunk_id == _stable_fact_id(response["served"])

    @pytest.mark.asyncio
    async def test_emit_diagnostics(self):
        client = _FakeClient(response=_FakeResponse(200, HIT_RESPONSE))
        shape = _shape_with_direct_answer()

        filled = await fill_shape_fact_store(
            _empty_pool(),
            shape,
            "phone number for sunshine health",
            tag_matches=["j:payor.sunshine_health"],
            http_client=client,
        )

        assert filled.filling_strategy == "fact_store"
        assert filled.emit["fillers_decision"] == "fact_store_query"
        assert filled.emit["slots_filled"] == 1
        assert filled.emit["total_chunks_assigned"] == 1
        per_slot = filled.emit["per_slot_details"]
        assert per_slot[0]["gate_passed"] is True
        assert per_slot[0]["hit"] is True
        assert per_slot[0]["telemetry_id"] == "abc-123"
        assert per_slot[1]["gate_passed"] is False  # thematic slot never gated-in
        assert per_slot[1]["hit"] is None

    @pytest.mark.asyncio
    async def test_deterministic(self):
        shape = _shape_with_direct_answer()

        filled_1 = await fill_shape_fact_store(
            _empty_pool(),
            shape,
            "phone number for sunshine health",
            tag_matches=["j:payor.sunshine_health"],
            http_client=_FakeClient(response=_FakeResponse(200, HIT_RESPONSE)),
        )
        filled_2 = await fill_shape_fact_store(
            _empty_pool(),
            shape,
            "phone number for sunshine health",
            tag_matches=["j:payor.sunshine_health"],
            http_client=_FakeClient(response=_FakeResponse(200, HIT_RESPONSE)),
        )

        assert [c.chunk_id for c in filled_1.slots[0].chunks] == [
            c.chunk_id for c in filled_2.slots[0].chunks
        ]

    @pytest.mark.asyncio
    async def test_chunk_id_stable_across_different_telemetry_ids(self):
        """Regression test for the 2026-07-23 determinism bug: the fact-store
        serves a fresh `telemetry_id` per call even when it's the exact same
        fact, and chunk_id/document_id used to be derived from telemetry_id --
        so the same query, run twice, got a different chunk identity every
        time (caught via a live 3x-repeat determinism check). chunk_id must
        be derived from the fact's own content instead, so identical facts
        served under different telemetry_ids still produce the same id."""
        shape = _shape_with_direct_answer()
        response_call_1 = {**HIT_RESPONSE, "telemetry_id": "telemetry-run-1"}
        response_call_2 = {**HIT_RESPONSE, "telemetry_id": "telemetry-run-2-totally-different"}

        filled_1 = await fill_shape_fact_store(
            _empty_pool(), shape, "phone number for sunshine health",
            tag_matches=["j:payor.sunshine_health"],
            http_client=_FakeClient(response=_FakeResponse(200, response_call_1)),
        )
        filled_2 = await fill_shape_fact_store(
            _empty_pool(), shape, "phone number for sunshine health",
            tag_matches=["j:payor.sunshine_health"],
            http_client=_FakeClient(response=_FakeResponse(200, response_call_2)),
        )

        chunk_1 = filled_1.slots[0].chunks[0]
        chunk_2 = filled_2.slots[0].chunks[0]
        assert chunk_1.chunk_id == chunk_2.chunk_id
        assert chunk_1.chunk_id == _stable_fact_id(HIT_RESPONSE["served"])

    def test_stable_fact_id_changes_when_content_changes(self):
        """Complementary check: the hash is content-derived, not a constant --
        a genuinely different fact must still get a different id."""
        served_a = HIT_RESPONSE["served"]
        served_b = {**served_a, "answer_text": "1-800-000-0000", "value": "1-800-000-0000"}
        assert _stable_fact_id(served_a) != _stable_fact_id(served_b)
