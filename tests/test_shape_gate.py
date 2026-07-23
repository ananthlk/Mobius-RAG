"""Tests for the Step 1 SHAPE gate (app/services/retriever/shape/gate.py).

Two layers, matching how the contour logic is actually exercised:

  1. Pure unit tests on ``_classify()`` with a synthetic GateResult/CorpusProbe
     — deterministic, no DB, and the only practical way to hit CORPUS_GAP
     (a real zero-doc lexicon code's phrase overlaps other codes that DO
     have docs, so no natural-language query cleanly isolates it).
  2. DB-backed integration tests confirming real queries land on the
     contour the unit tests predict — VICINITY and UNCLEAR both have clean
     real-query triggers; EXACT/UNDERSPECIFIED are covered by the cmhc
     22-query bank (see scripts/run_gate_on_cmhc.py, not duplicated here).
"""

from __future__ import annotations

import pytest

from app.database import AsyncSessionLocal
from app.services.retriever.shape.contracts import Contour, CorpusProbe, GateResult
from app.services.retriever.shape.gate import _classify, _is_general_only_match, run_gate


def _result(**kwargs) -> GateResult:
    r = GateResult(**{k: v for k, v in kwargs.items() if k != "probe"})
    r.probe = kwargs.get("probe", CorpusProbe())
    r.missing_kinds = [
        k for k, codes in (("d", r.d_codes), ("j", r.j_codes), ("p", r.p_codes)) if not codes
    ]
    return r


class TestClassifyPureUnit:
    """Direct tests on _classify() — no DB, synthetic probes."""

    def test_unclear_when_query_malformed(self):
        r = _result(normalized="")
        contour, reason = _classify(r, all_d_codes=set())
        assert contour == Contour.UNCLEAR
        assert "too short to parse" in reason

    def test_out_of_scope_when_wellformed_but_no_tags(self):
        r = _result(normalized="what is the weather forecast for tomorrow")
        contour, reason = _classify(r, all_d_codes=set())
        assert contour == Contour.OUT_OF_SCOPE
        assert "not this corpus's domain" in reason

    def test_corpus_gap_when_union_zero(self):
        # Tags matched (kinds_matched > 0) but the probe found zero documents
        # carrying them — the real-world case is a lexicon code whose phrase
        # can't be cleanly isolated (jurisdiction.west_virginia's "west
        # virginia" trigger overlaps state.west_virginia, which has docs),
        # so this is tested as a pure classify() case rather than end-to-end.
        r = _result(
            d_codes=["d:some_code_with_zero_docs"],
            j_codes=["j:jurisdiction.west_virginia"],
            probe=CorpusProbe(d_docs=0, j_docs=0, union_docs=0, intersection_docs=0),
        )
        contour, reason = _classify(r, all_d_codes={"some_code_with_zero_docs"})
        assert contour == Contour.CORPUS_GAP
        assert "zero documents" in reason

    def test_vicinity_when_dj_matched_but_no_overlap(self):
        # Confirmed live 2026-07-22: "prior authorization in Clarendon, AR"
        # — d:utilization_management.prior_authorization (1696 union docs)
        # + j:jurisdiction.clarendon_ar (1 doc) — that one doc isn't also
        # about prior auth, so intersection=0 but union=1696.
        r = _result(
            d_codes=["d:utilization_management.prior_authorization"],
            j_codes=["j:jurisdiction.clarendon_ar"],
            probe=CorpusProbe(d_docs=1696, j_docs=1, union_docs=1696, intersection_docs=0),
        )
        contour, reason = _classify(r, all_d_codes={"utilization_management.prior_authorization"})
        assert contour == Contour.VICINITY
        assert "no doc covers the combination" in reason

    def test_exact_when_dj_matched_and_intersect(self):
        r = _result(
            d_codes=["d:claims.timely_filing"],
            j_codes=["j:payor.sunshine_health"],
            probe=CorpusProbe(union_docs=100, intersection_docs=42),
        )
        contour, reason = _classify(r, all_d_codes={"claims.timely_filing", "claims.general"})
        assert contour == Contour.EXACT
        assert "42 docs" in reason

    def test_underspecified_when_j_missing_and_corpus_broad(self):
        r = _result(
            d_codes=["d:utilization_management.prior_authorization"],
            probe=CorpusProbe(union_docs=5000, intersection_docs=0),
        )
        contour, _ = _classify(r, all_d_codes={"utilization_management.prior_authorization"})
        assert contour == Contour.UNDERSPECIFIED

    def test_exact_when_missing_kind_but_corpus_already_narrow(self):
        # Missing J, but the matched D-code alone narrows to ≤ _BROAD_MIN_DOCS —
        # the corpus itself acts as the specifier.
        r = _result(
            d_codes=["d:disputes.grievance"],
            probe=CorpusProbe(union_docs=10, intersection_docs=0),
        )
        contour, reason = _classify(r, all_d_codes={"disputes.grievance"})
        assert contour == Contour.EXACT
        assert "corpus narrows" in reason

    def test_underspecified_when_d_general_only_no_p_no_process_intent(self):
        # "Eligibility for Medicaid" — matched only the general bucket,
        # dozens of specific siblings exist, no disambiguating signal. This
        # is the "explore_siblings" strategy: we KNOW D+J are valid and the
        # corpus has content, we just don't know which facet — a bounded,
        # enumerable fan-out list, not a dead end.
        r = _result(
            d_codes=["d:eligibility", "d:eligibility.general"],
            j_codes=["j:program.medicaid"],
            probe=CorpusProbe(union_docs=6309, intersection_docs=1914),
        )
        all_d = {"eligibility", "eligibility.general", "eligibility.verification", "eligibility.income_criteria"}
        contour, reason = _classify(r, all_d_codes=all_d)
        assert contour == Contour.UNDERSPECIFIED
        assert "general 'eligibility' bucket" in reason
        assert r.underspecified_kind == "explore_siblings"
        assert r.fanout_codes == ["eligibility.income_criteria", "eligibility.verification"]

    def test_underspecified_missing_domain_has_no_fanout_codes(self):
        # "How do I get credentialed" — D matched nothing at all, so there
        # is no root to enumerate siblings under. Not explorable; downstream
        # needs a different strategy (relax/escalate/lexicon-gap flag).
        r = _result(
            j_codes=["j:payor.sunshine_health"],
            probe=CorpusProbe(union_docs=559, intersection_docs=559),
        )
        contour, reason = _classify(r, all_d_codes=set())
        assert contour == Contour.UNDERSPECIFIED
        assert r.underspecified_kind == "missing_domain"
        assert r.fanout_codes == []
        assert "not explorable" in reason

    def test_exact_when_d_general_only_but_p_code_present(self):
        # "How do I verify eligibility for Medicaid" — lexicon alias hit.
        r = _result(
            d_codes=["d:eligibility", "d:eligibility.general"],
            j_codes=["j:program.medicaid"],
            p_codes=["p:verification.verify"],
            probe=CorpusProbe(union_docs=6726, intersection_docs=831),
        )
        all_d = {"eligibility", "eligibility.general", "eligibility.verification"}
        contour, reason = _classify(r, all_d_codes=all_d)
        assert contour == Contour.EXACT
        assert "p=yes" in reason

    def test_exact_when_d_general_only_but_process_intent_phrasing(self):
        # "How do I check eligibility for Medicaid" — "check" isn't a lexicon
        # alias, but the structural "how do I" phrasing resolves it anyway.
        r = _result(
            d_codes=["d:eligibility", "d:eligibility.general"],
            j_codes=["j:program.medicaid"],
            process_intent=True,
            probe=CorpusProbe(union_docs=6309, intersection_docs=1914),
        )
        all_d = {"eligibility", "eligibility.general", "eligibility.verification"}
        contour, reason = _classify(r, all_d_codes=all_d)
        assert contour == Contour.EXACT
        assert "process-intent phrasing" in reason

    def test_underspecified_when_criteria_phrasing_stays_ambiguous(self):
        # "What are the eligibility criteria" deliberately does NOT match
        # the process-intent regex — it's a different, still-ambiguous ask
        # (which facet's rules?), not a phrasing gap.
        r = _result(
            d_codes=["d:eligibility", "d:eligibility.general"],
            j_codes=["j:program.medicaid"],
            process_intent=False,
            probe=CorpusProbe(union_docs=6309, intersection_docs=1914),
        )
        all_d = {"eligibility", "eligibility.general", "eligibility.income_criteria"}
        contour, _ = _classify(r, all_d_codes=all_d)
        assert contour == Contour.UNDERSPECIFIED

    def test_is_general_only_match_false_on_multi_root(self):
        # Multiple distinct top-level domains matched — richly specified,
        # never "general-only" regardless of individual code names.
        is_general, _, _ = _is_general_only_match(
            ["d:claims.general", "d:disputes.appeal"], all_codes={"claims.general", "disputes.appeal"}
        )
        assert is_general is False

    def test_is_general_only_match_false_when_specific_leaf_present(self):
        is_general, _, _ = _is_general_only_match(
            ["d:claims.general", "d:claims.timely_filing"],
            all_codes={"claims.general", "claims.timely_filing"},
        )
        assert is_general is False

    def test_is_general_only_match_true_with_siblings(self):
        is_general, root, n = _is_general_only_match(
            ["d:eligibility", "d:eligibility.general"],
            all_codes={"eligibility", "eligibility.general", "eligibility.verification", "eligibility.income_criteria"},
        )
        assert is_general is True
        assert root == "eligibility"
        assert n == 2


class TestProcessIntentDetector:
    """The regex-based structural signal, independent of lexicon matching."""

    @pytest.mark.parametrize(
        "query",
        [
            "How do I check eligibility for Medicaid",
            "how can I verify this claim",
            "How to submit a grievance",
            "What is the process for filing an appeal",
            "What are the steps to enroll a new patient",
            "steps to credential a provider",
        ],
    )
    def test_fires_on_action_phrasing(self, query):
        from app.services.retriever.shape.gate import _detect_process_intent

        assert _detect_process_intent(query.lower()) is True

    @pytest.mark.parametrize(
        "query",
        [
            "eligibility for medicaid",
            "what are the eligibility criteria for medicaid",
            "does sunshine health cover dental services",
            "what is the timely filing deadline",
        ],
    )
    def test_silent_on_fact_lookups(self, query):
        from app.services.retriever.shape.gate import _detect_process_intent

        assert _detect_process_intent(query.lower()) is False


@pytest.mark.asyncio
class TestRunGateIntegration:
    """DB-backed: confirms real queries land on the contour the unit tests predict."""

    async def test_out_of_scope_on_wellformed_off_domain_query(self):
        async with AsyncSessionLocal() as db:
            r = await run_gate(db, "What's the weather forecast for tomorrow?")
            assert r.contour == Contour.OUT_OF_SCOPE
            assert r.d_codes == r.j_codes == r.p_codes == []

    async def test_unclear_on_single_garbled_token(self):
        # Multi-word "fake English" (e.g. "asdkfj qwoeiru xyz") is a known,
        # documented gap — it slips through as OUT_OF_SCOPE since telling it
        # apart from real off-domain English needs a dictionary/LLM check.
        # A single token / too-short fragment is the case this can catch cheaply.
        async with AsyncSessionLocal() as db:
            r = await run_gate(db, "asdkfjqwoeiru")
            assert r.contour == Contour.UNCLEAR

    async def test_vicinity_on_rare_jurisdiction_common_domain(self):
        # d:utilization_management.prior_authorization (broad, real docs)
        # + a jurisdiction with exactly one document that isn't about
        # prior auth — union > 0, intersection == 0.
        async with AsyncSessionLocal() as db:
            r = await run_gate(db, "What is the prior authorization process in Clarendon, AR?")
            assert r.contour == Contour.VICINITY
            assert r.probe.union_docs > 0
            assert r.probe.intersection_docs == 0

    async def test_underspecified_on_bare_eligibility(self):
        async with AsyncSessionLocal() as db:
            r = await run_gate(db, "Eligibility for Medicaid")
            assert r.contour == Contour.UNDERSPECIFIED

    async def test_exact_on_eligibility_with_verify_verb(self):
        async with AsyncSessionLocal() as db:
            r = await run_gate(db, "How do I verify eligibility for Medicaid")
            assert r.contour == Contour.EXACT

    async def test_exact_on_eligibility_with_process_intent_phrasing(self):
        async with AsyncSessionLocal() as db:
            r = await run_gate(db, "How do I check eligibility for Medicaid")
            assert r.contour == Contour.EXACT

    async def test_general_only_on_health_care_services_root_live(self):
        async with AsyncSessionLocal() as db:
            r = await run_gate(db, "Behavioral health services for Medicaid")
            # Eval's original hypothesis expected UNDERSPECIFIED here; empirically
            # "behavioral_health" matches its own specific leaf under
            # health_care_services, not the bare umbrella — see gate015's notes
            # in eval/queries_gate_contours.yaml. Pinning the TRUE live behavior.
            assert r.contour == Contour.EXACT


# --- Eval's drafted additions (docs/rag-agents/shape-gate-eval-qa-scenarios.md §5) ---
# Landed verbatim per TECH's 2026-07-22 review — these close Eval's own stated
# sign-off gate (§3.6: "≥2 distinct umbrella roots tested", 8 coverage gaps in §1).


class TestGeneralOnlyMatchAcrossRoots:
    """§1.1/§1.2: general-only-match must not be eligibility-shaped by accident."""

    def test_general_only_on_health_care_services_root(self):
        # health_care_services is the lexicon's actual largest umbrella (631
        # live siblings) — the real stress case the spec's "eligibility"
        # example was standing in for.
        is_general, root, n = _is_general_only_match(
            ["d:health_care_services", "d:health_care_services.general"],
            all_codes={"health_care_services", "health_care_services.general",
                       "health_care_services.dental", "health_care_services.behavioral_health"},
        )
        assert is_general is True
        assert root == "health_care_services"
        assert n == 2

    def test_general_only_false_across_two_different_umbrella_roots(self):
        # eligibility.general (bare) + claims.timely_filing (specific, different root) —
        # multi-root spans should never be flagged general-only, regardless of which
        # individual root looks bare.
        is_general, _, _ = _is_general_only_match(
            ["d:eligibility.general", "d:claims.timely_filing"],
            all_codes={"eligibility.general", "eligibility.verification",
                       "claims.timely_filing", "claims.general"},
        )
        assert is_general is False


class TestProcessOnlyQuery:
    """§1.3: a P-code match with zero D/J — currently unexercised."""

    def test_process_only_no_domain_or_jurisdiction(self):
        r = _result(
            p_codes=["p:submission.submit"],
            probe=CorpusProbe(union_docs=0, intersection_docs=0),
        )
        contour, _ = _classify(r, all_d_codes=set())
        # kinds_matched == 1 (P only) — should NOT hit the ==0 UNCLEAR/OUT_OF_SCOPE
        # branch; falls through to union_docs==0 → CORPUS_GAP under current logic.
        assert contour == Contour.CORPUS_GAP


class TestBroadMinDocsBoundary:
    """§1.5: pin the Eval-tunable _BROAD_MIN_DOCS=25 constant at its edges."""

    def test_exact_at_exactly_broad_min_docs(self):
        r = _result(d_codes=["d:disputes.grievance"], probe=CorpusProbe(union_docs=25, intersection_docs=0))
        contour, _ = _classify(r, all_d_codes={"disputes.grievance"})
        assert contour == Contour.EXACT

    def test_underspecified_one_doc_over_broad_min_docs(self):
        r = _result(d_codes=["d:disputes.grievance"], probe=CorpusProbe(union_docs=26, intersection_docs=0))
        contour, _ = _classify(r, all_d_codes={"disputes.grievance"})
        assert contour == Contour.UNDERSPECIFIED


class TestCorpusGapNotTrippedByPartialZero:
    """§1.6: union_docs==0 requires ALL matched codes to be zero-doc, not just one."""

    def test_not_corpus_gap_when_only_one_of_two_codes_is_zero_doc(self):
        r = _result(
            d_codes=["d:some_zero_doc_code"],
            j_codes=["j:jurisdiction.sunshine_health"],
            probe=CorpusProbe(union_docs=500, intersection_docs=0),
        )
        contour, _ = _classify(r, all_d_codes={"some_zero_doc_code"})
        assert contour != Contour.CORPUS_GAP


class TestProcessIntentVerbVariations:
    """§4a: alias-path vs regex-path vs false-positive-alias-word distinctions."""

    def test_exact_via_confirm_alias_not_regex(self):
        # "Can I confirm..." doesn't match the how-do-i/how-to regex family, but
        # "confirm" IS a verification.verify lexicon alias — should still resolve.
        r = _result(
            d_codes=["d:eligibility", "d:eligibility.general"],
            j_codes=["j:program.medicaid"],
            p_codes=["p:verification.verify"],
            process_intent=False,
            probe=CorpusProbe(union_docs=6309, intersection_docs=1914),
        )
        all_d = {"eligibility", "eligibility.general", "eligibility.verification"}
        contour, reason = _classify(r, all_d_codes=all_d)
        assert contour == Contour.EXACT
        assert "p=yes" in reason

    def test_underspecified_when_required_alias_fires_but_not_actually_process_intent(self):
        # "required" is a compliance_action.required P-tag alias, but "Is eligibility
        # required for Medicaid" is a fact lookup, not a how-do-I ask. If the P-code
        # fires, general-only-match treats it as resolved (p=yes) by design — this
        # test documents that behavior explicitly rather than leaving it implicit.
        r = _result(
            d_codes=["d:eligibility", "d:eligibility.general"],
            j_codes=["j:program.medicaid"],
            p_codes=["p:compliance_action.required"],
            process_intent=False,
            probe=CorpusProbe(union_docs=6309, intersection_docs=1914),
        )
        all_d = {"eligibility", "eligibility.general", "eligibility.verification"}
        contour, reason = _classify(r, all_d_codes=all_d)
        assert contour == Contour.EXACT  # documents current behavior — flag if undesired
        assert "p=yes" in reason
