"""Unit tests for Contract (Step 6) -- the one-emitter 12-field envelope.
Builds real dataclass instances (no mocked framework internals) and asserts
on `build_contract()`'s output; no DB/network involved, same posture as
`test_orchestrator_fillers_dispatch.py`.
"""

from app.services.retriever.contract import build_contract
from app.services.retriever.orchestrator import RetrieverPartialResult
from app.services.retriever.shape.contracts import GateResult, ReformatResult, ReformatPosture
from app.services.retriever.synthesis_contracts import (
    CompiledCitation, CompiledSlot, SynthesisResult, SynthesisTelemetry,
)
from app.services.router.decision import RouterDecision, ResourcePosture
from app.services.router.allocation import RoutingLadder


def _partial_result(**overrides) -> RetrieverPartialResult:
    defaults = dict(
        query="what is the timely filing limit",
        gate=GateResult(query="q", normalized="q", contour="PRECISE", reason="matched d+j"),
        reformat=ReformatResult(query="q", posture=ReformatPosture.PRECISE),
        slots=_answer_shape_result(),
        router_decision=RouterDecision(
            routing_ladder=RoutingLadder(
                per_slot={"direct_answer": ["a", "b"]}, outcome="all_slots_cleared", feasible=True,
                terminal_action=None, helpers=["e"],
                per_slot_status={"direct_answer": "CLEARED"},
                per_slot_lb={"direct_answer": 0.9},
                per_slot_terminal={"direct_answer": None},
                per_slot_helpers={"direct_answer": []},
                adjusted_confidence_bar=0.7,
            ),
            decision_id="dec-1", resource_posture=ResourcePosture(confidence_bar=0.85, authority_requirement="any"),
            dispatch_path="greedy",
        ),
        filled_shape=_filled_shape(),
        gate_ms=10, reformat_ms=5, slots_ms=2, pool_ms=50, router_ms=8, fillers_ms=120, total_ms=200,
        narrative="Matched eligibility + Sunshine Health.",
    )
    defaults.update(overrides)
    return RetrieverPartialResult(**defaults)


def _filled_shape():
    from app.services.retriever.fillers.contracts import FilledShape
    return FilledShape(
        slots=[], total_chunks_assigned=1, filling_strategy="multi_turn_continuation_stopgap_verdict",
        emit={"executed_order": {"direct_answer": ["a"]}},
    )


def _answer_shape_result():
    from app.services.retriever.shape.slots import AnswerSlot, AnswerShapeResult
    return AnswerShapeResult(
        query="q", posture=None,
        slots=[AnswerSlot(slot_id="direct_answer", slot_semantics="direct_answer", capacity=5, rewritten_query="q", required=True, priority=0)],
        reason="", slots_ms=0,
    )


def _citation(index, slot_id="direct_answer", score=0.9, verified=True, status="live"):
    return CompiledCitation(
        index=index, chunk_id=f"c{index}", document_name="Provider Manual",
        text="timely filing is 90 days", source_type="internal", document_status=status,
        verified=verified, original_score=score, slot_id=slot_id, slot_semantics="direct_answer",
    )


def _synthesis_result(citations=None, unverified=0, under_filled_flags=None, compile_ms=15, model_trace=None):
    citations = citations if citations is not None else [_citation(1)]
    under_filled_flags = under_filled_flags or {}
    slot = CompiledSlot(
        slot_id="direct_answer", slot_semantics="direct_answer", capacity=5, required=True,
        citations=citations, occupancy=len(citations),
        under_filled=under_filled_flags.get("direct_answer", False),
        model_trace=model_trace or {},
    )
    return SynthesisResult(
        query="q", slots=[slot], citations=citations,
        telemetry=SynthesisTelemetry(unverified_citations=unverified, compile_ms=compile_ms),
    )


class TestBuildContractHappyPath:
    def test_all_twelve_fields_present(self):
        envelope = build_contract(_partial_result(), _synthesis_result())
        d = envelope.to_dict()
        assert set(d.keys()) == {
            "query", "chosen_slot", "score", "chunks", "answer_text", "thinking",
            "traces", "routing_keys", "grounding_markers", "latency_ms",
            "attempt_count", "status",
        }

    def test_chosen_slot_and_score_from_best_citation(self):
        envelope = build_contract(_partial_result(), _synthesis_result())
        assert envelope.chosen_slot == "direct_answer"
        assert envelope.score == 0.9

    def test_chunks_reflect_citations(self):
        envelope = build_contract(_partial_result(), _synthesis_result())
        assert len(envelope.chunks) == 1
        assert envelope.chunks[0]["chunk_id"] == "c1"
        assert envelope.chunks[0]["document_name"] == "Provider Manual"

    def test_chunks_carry_authority_for_grounding_badge(self):
        """Chat's Gap 1 (2026-07-24): authority was dropped from chunks[]
        serialization despite being a real CompiledCitation field -- Chat
        needs it for the grounding badge's all-authoritative check."""
        citation = _citation(1)
        citation.authority = "authoritative"
        envelope = build_contract(_partial_result(), _synthesis_result(citations=[citation]))
        assert envelope.chunks[0]["authority"] == "authoritative"

    def test_status_ok_when_fully_verified_and_filled(self):
        envelope = build_contract(_partial_result(), _synthesis_result())
        assert envelope.status == "ok"

    def test_status_partial_when_unverified_present(self):
        envelope = build_contract(_partial_result(), _synthesis_result(unverified=1))
        assert envelope.status == "partial"

    def test_status_partial_when_under_filled(self):
        result = _synthesis_result(under_filled_flags={"direct_answer": True})
        envelope = build_contract(_partial_result(), result)
        assert envelope.status == "partial"

    def test_attempt_count_sums_executed_order(self):
        envelope = build_contract(_partial_result(), _synthesis_result())
        assert envelope.attempt_count == 1

    def test_latency_ms_carries_all_segments(self):
        envelope = build_contract(_partial_result(), _synthesis_result())
        assert envelope.latency_ms["total_ms"] == 200
        assert envelope.latency_ms["fillers_ms"] == 120
        assert envelope.latency_ms["synthesis_ms"] == 15

    def test_routing_keys_carries_decision_id_and_ladder(self):
        envelope = build_contract(_partial_result(), _synthesis_result())
        assert envelope.routing_keys["decision_id"] == "dec-1"
        assert envelope.routing_keys["dispatch_path"] == "greedy"
        assert envelope.routing_keys["routing_ladder_per_slot"] == {"direct_answer": ["a", "b"]}

    def test_answer_text_and_thinking_optional_and_threaded(self):
        envelope = build_contract(
            _partial_result(), _synthesis_result(),
            answer_text="Timely filing is 90 days.", thinking="Cited provider manual.",
        )
        assert envelope.answer_text == "Timely filing is 90 days."
        assert envelope.thinking == "Cited provider manual."

    def test_answer_text_defaults_none_when_not_supplied(self):
        envelope = build_contract(_partial_result(), _synthesis_result())
        assert envelope.answer_text is None
        assert envelope.thinking is None

    def test_routing_verdict_assembled_from_real_ladder_fields(self):
        envelope = build_contract(_partial_result(), _synthesis_result())
        rv = envelope.routing_keys["routing_verdict"]
        assert rv["outcome"] == "all_slots_cleared"
        assert rv["terminal_action"] is None
        assert rv["helpers"] == ["e"]
        assert rv["confidence_bar"] == 0.85
        assert rv["adjusted_bar"] == 0.7
        assert rv["slots"]["direct_answer"] == {
            "status": "CLEARED", "lb": 0.9, "terminal": None,
            "required": True, "helpers": [],
        }

    def test_terminal_action_and_authority_requirement_surfaced(self):
        envelope = build_contract(_partial_result(), _synthesis_result())
        assert envelope.routing_keys["terminal_action"] is None
        assert envelope.routing_keys["authority_requirement"] == "any"

    def test_model_trace_passed_through_with_real_field_names(self):
        result = _synthesis_result(model_trace={"stage": "c", "model_used": "gpt-x", "llm_call_id": "call-1"})
        envelope = build_contract(_partial_result(), result)
        assert envelope.routing_keys["model_trace"] == [
            {"slot_id": "direct_answer", "stage": "c", "model_id": "gpt-x", "call_id": "call-1"},
        ]

    def test_model_trace_empty_when_no_llm_call_was_made(self):
        envelope = build_contract(_partial_result(), _synthesis_result())
        assert envelope.routing_keys["model_trace"] == []


class TestBuildContractDegradedPaths:
    def test_no_synthesis_result_degrades_honestly(self):
        envelope = build_contract(_partial_result(), None)
        assert envelope.chunks == []
        assert envelope.chosen_slot is None
        assert envelope.score is None
        assert envelope.status == "filled_no_synthesis"
        assert envelope.grounding_markers == {}

    def test_no_filled_shape_reports_no_retrieval(self):
        pr = _partial_result(filled_shape=None)
        envelope = build_contract(pr, None)
        assert envelope.status == "no_retrieval"
        assert envelope.attempt_count == 0

    def test_empty_citations_reports_empty_status(self):
        envelope = build_contract(_partial_result(), _synthesis_result(citations=[]))
        assert envelope.status == "empty"
        assert envelope.chosen_slot == "direct_answer"  # honest "tried, got nothing"
        assert envelope.score is None
