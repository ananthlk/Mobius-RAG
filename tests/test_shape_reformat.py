"""Tests for Step 1b SHAPE reformat (app/services/retriever/shape/reformat.py
+ reformat_narrate.py).

Same two-layer structure as test_shape_gate.py:

  1. Pure unit tests — no DB. Covers `_dispatch`'s non-DB-dependent branches
     (PRECISE, RELY_ON_EXTERNAL x2, DECLINE, CLARIFY_REPHRASE) with synthetic
     GateResult fixtures, plus the pure math/logic helpers (`_cosine`,
     `_agglomerative_cluster`, `_union_prevalence`) and narration helpers
     (`_compose`, `_soften_theme_label`, `_found_path`).
  2. DB-backed integration tests — real `run_gate()` + `run_reformat()`
     against live data, pinned assertions (not just the eval-bank runner
     script, which reports pass/fail but isn't pytest-discoverable or part
     of CI). Mirrors `queries_reformat_postures.yaml`'s cases but as
     deterministic pinned tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.database import AsyncSessionLocal
from app.services.retriever.shape.contracts import (
    Contour,
    CorpusProbe,
    GateResult,
    ReformatPosture,
)
from app.services.retriever.shape.gate import run_gate
from app.services.retriever.shape.reformat import (
    _agglomerative_cluster,
    _cosine,
    _dispatch,
    _union_prevalence,
    run_reformat,
)
from app.services.retriever.shape.reformat_narrate import (
    _compose,
    _found_path,
    _soften_theme_label,
    narrate,
    narrate_full,
)


def _gate(**kwargs) -> GateResult:
    r = GateResult(**{k: v for k, v in kwargs.items() if k != "probe"})
    r.probe = kwargs.get("probe", CorpusProbe())
    return r


# ---------------------------------------------------------------------------
# Pure unit tests — dispatch, no DB
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDispatchPureUnit:
    """Only the branches of _dispatch that never touch the DB."""

    async def test_exact_gives_precise_passthrough(self):
        gate = _gate(query="How do I confirm eligibility", normalized="how do i confirm eligibility", contour=Contour.EXACT)
        result = await _dispatch(None, gate)
        assert result.posture == ReformatPosture.PRECISE
        assert result.rewritten_queries == ["how do i confirm eligibility"]

    async def test_vicinity_gives_rely_on_external_with_reason(self):
        gate = _gate(query="q", contour=Contour.VICINITY)
        result = await _dispatch(None, gate)
        assert result.posture == ReformatPosture.RELY_ON_EXTERNAL
        assert result.external_reason == "vicinity"

    async def test_corpus_gap_gives_rely_on_external_with_reason(self):
        gate = _gate(query="q", contour=Contour.CORPUS_GAP)
        result = await _dispatch(None, gate)
        assert result.posture == ReformatPosture.RELY_ON_EXTERNAL
        assert result.external_reason == "corpus_gap"

    async def test_out_of_scope_gives_decline(self):
        gate = _gate(query="q", contour=Contour.OUT_OF_SCOPE)
        result = await _dispatch(None, gate)
        assert result.posture == ReformatPosture.DECLINE
        assert result.decline_reason == "out_of_scope"

    async def test_unclear_gives_tentative_clarify_rephrase(self):
        gate = _gate(query="q", contour=Contour.UNCLEAR)
        result = await _dispatch(None, gate)
        assert result.posture == ReformatPosture.CLARIFY_REPHRASE
        assert "NOT yet confirmed by Ananth" in result.reason


# ---------------------------------------------------------------------------
# Pure unit tests — math/logic helpers
# ---------------------------------------------------------------------------


class TestCosine:
    def test_identical_vectors_give_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_give_zero(self):
        assert _cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)

    def test_zero_vector_degrades_gracefully_not_nan(self):
        # No divide-by-zero — a zero-norm vector must return 0.0, not NaN,
        # per the graceful-degradation design principle (schematic spec §3).
        result = _cosine(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        assert result == 0.0
        assert not np.isnan(result)


class TestAgglomerativeCluster:
    def test_fewer_points_than_clusters_returns_singletons(self):
        vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = _agglomerative_cluster(vecs, n_clusters=3)
        assert labels == [0, 1]

    def test_two_well_separated_blobs_cluster_correctly(self):
        rng = np.random.default_rng(42)
        blob_a = rng.normal(loc=[10, 0], scale=0.1, size=(15, 2))
        blob_b = rng.normal(loc=[0, 10], scale=0.1, size=(15, 2))
        vecs = np.vstack([blob_a, blob_b])
        labels = _agglomerative_cluster(vecs, n_clusters=2)
        # Every point in blob_a shares a label, every point in blob_b shares
        # the other label — the two blobs must not be mixed.
        assert len(set(labels[:15])) == 1
        assert len(set(labels[15:])) == 1
        assert labels[0] != labels[15]

    def test_deterministic_with_fixed_seed(self):
        rng = np.random.default_rng(7)
        vecs = rng.normal(size=(30, 8))
        a = _agglomerative_cluster(vecs, n_clusters=3, seed=1)
        b = _agglomerative_cluster(vecs, n_clusters=3, seed=1)
        assert a == b

    def test_no_empty_clusters_produced(self):
        # Regression guard for the real bug caught live 2026-07-23:
        # average-linkage agglomerative collapsed 78/80 real codes into one
        # cluster. k-means must not reproduce that — every requested
        # cluster gets at least one member.
        rng = np.random.default_rng(3)
        vecs = rng.normal(size=(80, 16))
        labels = _agglomerative_cluster(vecs, n_clusters=3)
        assert len(set(labels)) == 3
        for k in range(3):
            assert labels.count(k) > 0


class TestUnionPrevalence:
    def test_union_not_sum_across_overlapping_docs(self):
        # The exact bug the fix (2026-07-23) targets: a doc tagged with 2
        # codes from the same theme must count ONCE, not twice.
        doc_ids_by_code = {
            "a": {"doc1", "doc2"},
            "b": {"doc2", "doc3"},
        }
        assert _union_prevalence(doc_ids_by_code, ["a", "b"]) == 3  # not 4

    def test_missing_code_treated_as_empty(self):
        assert _union_prevalence({"a": {"doc1"}}, ["a", "nonexistent"]) == 1

    def test_empty_member_list_gives_zero(self):
        assert _union_prevalence({"a": {"doc1"}}, []) == 0


# ---------------------------------------------------------------------------
# Pure unit tests — narration
# ---------------------------------------------------------------------------


class TestSoftenThemeLabel:
    def test_strips_known_boilerplate_prefix(self):
        # UX's own worked example (2026-07-23 sign-off) — must match exactly.
        label = "Policies and criteria related to gross income for eligibility"
        assert _soften_theme_label(label) == "gross income for eligibility"

    def test_lowercases_first_letter(self):
        assert _soften_theme_label("Plan Assignment").startswith("p")

    def test_cuts_at_first_comma(self):
        assert _soften_theme_label("Foo, bar, baz") == "foo"

    def test_empty_input_has_fallback(self):
        assert _soften_theme_label("") == "another angle"


class TestFoundPath:
    def test_empty_codes_gives_empty_string(self):
        gate = _gate(query="q")
        assert _found_path(gate) == ""

    def test_d_and_j_both_present(self):
        gate = _gate(query="q", d_codes=["d:eligibility"], j_codes=["j:program.medicaid"])
        found = _found_path(gate)
        assert "eligibility" in found
        assert "medicaid" in found


class TestCompose:
    def test_with_found_path_states_it_first(self):
        result = _compose("about **x**", "this can be answered directly.")
        assert result.startswith("I see you're asking about **x**")

    def test_without_found_path_stands_alone_capitalized(self):
        # Regression guard for the real grammar bug caught live 2026-07-23:
        # "I see I want to make sure I get this right..." (double subject).
        result = _compose("", "i want to make sure i get this right.")
        assert result == "I want to make sure i get this right."
        assert not result.startswith("I see I")


# ---------------------------------------------------------------------------
# DB-backed integration tests — real run_gate() + run_reformat()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunReformatIntegration:
    """Live, pinned — mirrors eval/queries_reformat_postures.yaml but as
    deterministic pytest assertions rather than a report-only runner script.

    KNOWN FLAKINESS when run as a full class together, verified
    order-dependent 2026-07-23 (not a new bug — same pattern
    test_shape_gate.py's memory already documents): whichever test runs
    immediately after `test_explore_siblings_gives_fan_out_capped_at_max_themes`
    (the one FAN_OUT case, with heavy embedding/asyncio.to_thread work) can
    fail with "Event loop is closed" during connection teardown — a
    pytest-asyncio event-loop/connection-pool-scoping issue shared with
    other async DB tests in this repo, not a Reformat defect. Every test
    here passes individually and in isolation from that one. Run
    `-k "not fan_out"` plus the fan_out test separately if this flakes in CI."""

    async def test_exact_gives_precise(self):
        async with AsyncSessionLocal() as db:
            gate = await run_gate(db, "How do I confirm eligibility for Medicaid")
            result = await run_reformat(db, gate)
            assert result.posture == ReformatPosture.PRECISE
            assert len(result.rewritten_queries) == 1

    async def test_explore_siblings_gives_fan_out_capped_at_max_themes(self):
        async with AsyncSessionLocal() as db:
            gate = await run_gate(db, "Eligibility for Medicaid")
            assert gate.underspecified_kind == "explore_siblings"  # precondition
            result = await run_reformat(db, gate)
            assert result.posture == ReformatPosture.FAN_OUT
            assert len(result.fanout_themes) <= 4
            assert sum(1 for t in result.fanout_themes if t.is_catchall) == 1
            catchall = next(t for t in result.fanout_themes if t.is_catchall)
            assert catchall.member_codes == []  # deliberately not corpus-derived

    async def test_missing_jurisdiction_gives_clarify(self):
        async with AsyncSessionLocal() as db:
            gate = await run_gate(db, "How do I get hospice services")
            assert gate.underspecified_kind == "missing_jurisdiction"  # precondition, verified live 2026-07-23
            result = await run_reformat(db, gate)
            assert result.posture == ReformatPosture.CLARIFY
            assert len(result.clarify_questions) >= 1

    async def test_vicinity_gives_rely_on_external(self):
        async with AsyncSessionLocal() as db:
            gate = await run_gate(db, "What is the prior authorization process in Clarendon, AR")
            result = await run_reformat(db, gate)
            assert result.posture == ReformatPosture.RELY_ON_EXTERNAL
            assert result.external_reason == "vicinity"

    async def test_out_of_scope_gives_decline(self):
        async with AsyncSessionLocal() as db:
            gate = await run_gate(db, "What's the weather forecast for tomorrow?")
            result = await run_reformat(db, gate)
            assert result.posture == ReformatPosture.DECLINE

    async def test_reformat_ms_is_populated(self):
        async with AsyncSessionLocal() as db:
            gate = await run_gate(db, "What's the weather forecast for tomorrow?")
            result = await run_reformat(db, gate)
            assert result.reformat_ms >= 0

    async def test_narrate_and_narrate_full_do_not_raise_on_every_posture(self):
        # Cheap smoke coverage across postures — narration must never crash,
        # even for the low-DB-cost paths.
        queries = [
            "How do I confirm eligibility for Medicaid",
            "What documentation is required to enroll a new pediatric patient",
            "What is the prior authorization process in Clarendon, AR",
            "What's the weather forecast for tomorrow?",
            "asdkfjqwoeiru",
        ]
        async with AsyncSessionLocal() as db:
            for q in queries:
                gate = await run_gate(db, q)
                result = await run_reformat(db, gate)
                short = narrate(gate, result)
                full = narrate_full(gate, result)
                assert short and isinstance(short, str)
                assert full and isinstance(full, str)
                assert f'"{q}"' in full  # PHI-adjacent: full trace echoes the raw query, must never persist
