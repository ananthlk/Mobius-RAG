"""Tests for the stemming fallback in corpus_search_lexicon.py's _match_entry
(2026-07-30). See the module-level comment above _stem_cache_lock for the
full design rationale and the verified collision-exclusion list.

Live-DB tests (require a real dev DB connection, same convention as the
rest of this repo's retriever tests) -- exercises the real
_load_lexicon_snapshot/_load_stem_index/_match_entry pipeline against the
actual lexicon, not mocked data, since the whole point is to catch real
corpus collisions before they ship.
"""
from __future__ import annotations

import pytest

from app.database import AsyncSessionLocal
from app.services.corpus_search_lexicon import (
    _match_entry,
    expand_query_via_lexicon,
    _load_lexicon_snapshot,
    _load_stem_index,
)


class TestMatchEntryStemmingFallback:
    """Pure-unit tests against _match_entry directly -- no DB needed, since
    query_stems/phrase_stem_lookup/unsafe_stems are passed in directly."""

    def test_exact_match_still_works_unchanged(self):
        assert _match_entry("i need dme equipment", ["durable medical equipment", "dme"]) == "dme"

    def test_stemming_fallback_fires_when_exact_match_fails(self):
        # "credentialed" is not a substring of "credentialing" -- only the
        # stemming fallback can bridge this.
        query_stems = frozenset({"credenti"})
        phrase_stem_lookup = {"credentialing": "credenti"}
        hit = _match_entry(
            "how do i get credentialed",
            ["credentialing"],
            query_stems=query_stems,
            phrase_stem_lookup=phrase_stem_lookup,
            unsafe_stems=frozenset(),
        )
        assert hit == "credentialing"

    def test_stemming_fallback_blocked_for_unsafe_stem(self):
        # Same setup, but the stem is in the exclusion set -- must NOT match,
        # this is the core safety guarantee the whole design rests on.
        query_stems = frozenset({"aid"})
        phrase_stem_lookup = {"aids": "aid"}
        hit = _match_entry(
            "home health aide services",
            ["aids"],
            query_stems=query_stems,
            phrase_stem_lookup=phrase_stem_lookup,
            unsafe_stems=frozenset({"aid"}),
        )
        assert hit is None

    def test_stemming_fallback_never_applies_to_multiword_phrases(self):
        # Multi-word phrases keep exact-substring-only behavior regardless
        # of stems -- the fallback is scoped to single-word phrases only.
        query_stems = frozenset({"prior", "author"})
        phrase_stem_lookup = {"prior authorization": "author"}  # contrived, shouldn't matter
        hit = _match_entry(
            "some unrelated query text",
            ["prior authorization"],
            query_stems=query_stems,
            phrase_stem_lookup=phrase_stem_lookup,
            unsafe_stems=frozenset(),
        )
        assert hit is None

    def test_no_stemming_params_behaves_exactly_like_before(self):
        # Backward-compat: omitting the new params entirely must reproduce
        # the exact prior behavior (no fallback, exact-match only).
        assert _match_entry("credentialed provider", ["credentialing"]) is None


@pytest.mark.asyncio
class TestStemIndexLiveCollisionSafety:
    """Live-DB tests against the real lexicon -- verifies the collision
    detector actually protects the known-risky pairs and doesn't
    over-exclude the benign general/specific-leaf pattern."""

    async def test_credentialing_not_excluded_single_word_form(self):
        """credentialing is the SAME literal phrase on multiple codes
        (d:credentialing, d:credentialing.general) -- not a genuine
        stemming risk, must not be excluded (this was a real bug caught
        live before shipping: the first version of the collision detector
        wrongly excluded this exact case)."""
        async with AsyncSessionLocal() as db:
            snapshot = await _load_lexicon_snapshot(db)
            phrase_stem_lookup, unsafe_stems = await _load_stem_index(db, snapshot)
        stem = phrase_stem_lookup.get("credentialing")
        assert stem is not None
        assert stem not in unsafe_stems

    async def test_known_risky_collision_is_excluded(self):
        """aide/aids is a real, verified cross-concept collision (home
        health aide the person vs AIDS the disease) -- must be excluded."""
        async with AsyncSessionLocal() as db:
            snapshot = await _load_lexicon_snapshot(db)
            phrase_stem_lookup, unsafe_stems = await _load_stem_index(db, snapshot)
        aids_stem = phrase_stem_lookup.get("aids")
        if aids_stem is not None:  # only assert if the lexicon still has this phrase
            assert aids_stem in unsafe_stems

    async def test_credentialed_query_now_matches_credentialing(self):
        """End-to-end: the real regression this fallback fixes (cmhc016)."""
        async with AsyncSessionLocal() as db:
            exp = await expand_query_via_lexicon(db, "How do I get credentialed with Sunshine Health?")
        assert any("credentialing" in c for c in exp.matched_codes)

    async def test_enroll_query_now_matches_enrollment(self):
        """End-to-end: the real regression this fallback fixes (cmhc021)."""
        async with AsyncSessionLocal() as db:
            exp = await expand_query_via_lexicon(
                db, "What documentation is required to enroll a new pediatric patient with Sunshine Health?"
            )
        assert any("enrollment" in c for c in exp.matched_codes)

    async def test_aide_query_does_not_spuriously_match_hiv_aids(self):
        """The exact false-positive risk this whole exclusion mechanism
        exists to prevent."""
        async with AsyncSessionLocal() as db:
            exp = await expand_query_via_lexicon(db, "How do I file a home health aide reimbursement claim?")
        assert not any("hiv_aids" in c for c in exp.matched_codes)


class TestMatchBudgetKindPriority:
    """Real regression (2026-08-08, ReAct's live-traffic report, Ananth's
    catch): _MAX_ENTRIES_PER_QUERY=12 caps the WHOLE match loop, and
    snapshot order used to be whatever _load_lexicon_snapshot returned --
    not kind-prioritized. A query with enough incidental overlap with
    generic ".general" domain phrases could exhaust the entire budget on
    low-value D matches before the loop ever reached J/P entries later in
    snapshot order -- confirmed live: jurisdiction/process came back
    completely empty (never evaluated), not "didn't match", on a query
    that plainly stated "Florida". Fix: J/P kinds now sort before D."""

    @pytest.mark.asyncio
    async def test_generic_word_overlap_does_not_starve_jurisdiction_match(self):
        """The exact real regression -- "general" and "requirements" fuzzy-
        match many .general domain phrases; jurisdiction must still surface."""
        async with AsyncSessionLocal() as db:
            exp = await expand_query_via_lexicon(
                db, "general eligibility requirements for Florida Medicaid"
            )
        assert any("florida" in c.lower() for c in exp.jurisdiction_tags), (
            f"jurisdiction starved out of the match budget by generic domain "
            f"matches -- jurisdiction_tags={exp.jurisdiction_tags}"
        )
        assert any("medicaid" in c.lower() for c in exp.jurisdiction_tags)
