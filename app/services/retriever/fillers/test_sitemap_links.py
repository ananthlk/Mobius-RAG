"""Tests for the sitemap-links suggested-link lookup.

Verifies the ported _lookup_sitemap_candidates logic and the new
title-derivation rule (Chat's ruling) behave correctly -- not a FilledShape,
not a FilledChunk. Not a strategy-lettered filler (Retriever corrected the
earlier "Filler f" label, 2026-07-23 -- Router's real "f" is a separate,
unbuilt, scored strategy); this module's output rides inert through the
pipeline, no slot competition.
"""

import pytest

from app.services.retriever.fillers.sitemap_links import (
    SuggestedLink,
    _derive_title_from_url,
    _keywords_for_tags,
    lookup_sitemap_links,
)


class _FakeRows:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows
        self.last_params = None

    async def execute(self, stmt, params=None):
        self.last_params = params
        return _FakeRows(self._rows)


class _ExplodingDB:
    async def execute(self, stmt, params=None):
        raise RuntimeError("db unavailable")


# ---------------------------------------------------------------------------
# Title derivation
# ---------------------------------------------------------------------------

class TestDeriveTitleFromUrl:
    def test_hyphenated_path_slug(self):
        assert _derive_title_from_url(
            "https://sunshinehealth.com/providers/prior-authorization-requirements"
        ) == "Prior Authorization Requirements"

    def test_underscored_path_slug(self):
        assert _derive_title_from_url(
            "https://payer.com/docs/timely_filing_policy"
        ) == "Timely Filing Policy"

    def test_strips_file_extension(self):
        assert _derive_title_from_url(
            "https://payer.com/forms/prior-auth-form.pdf"
        ) == "Prior Auth Form"

    def test_root_url_no_path_returns_none(self):
        assert _derive_title_from_url("https://payer.com/") is None
        assert _derive_title_from_url("https://payer.com") is None

    def test_trailing_slash_uses_last_real_segment(self):
        assert _derive_title_from_url(
            "https://payer.com/providers/eligibility/"
        ) == "Eligibility"


# ---------------------------------------------------------------------------
# d-tag -> keyword mapping
# ---------------------------------------------------------------------------

class TestKeywordsForTags:
    def test_maps_timely_filing(self):
        assert _keywords_for_tags(["d:claims.timely_filing"]) == ["timely", "filing"]

    def test_maps_prior_auth_leaf(self):
        assert _keywords_for_tags(["d:utilization_management.prior_authorization"]) == [
            "preauth", "prior-auth", "authorization",
        ]

    def test_utilization_management_other_leaf_falls_back_to_prefix(self):
        assert _keywords_for_tags(["d:utilization_management.concurrent_review"]) == [
            "preauth", "prior-auth", "authorization",
        ]

    def test_no_matching_tag_returns_empty(self):
        assert _keywords_for_tags(["d:general.info", "j:payor.aetna"]) == []

    def test_none_or_empty_tag_matches(self):
        assert _keywords_for_tags(None) == []
        assert _keywords_for_tags([]) == []

    def test_first_matching_prefix_wins(self):
        # claims.timely_filing should match the more specific entry, not
        # fall through to the generic "claims" entry.
        assert _keywords_for_tags(["d:claims.timely_filing"]) == ["timely", "filing"]


# ---------------------------------------------------------------------------
# lookup_sitemap_links -- gate conditions + row mapping
# ---------------------------------------------------------------------------

class TestLookupSitemapLinks:
    @pytest.mark.asyncio
    async def test_no_payer_display_name_short_circuits(self):
        db = _FakeDB(rows=[{"url": "https://payer.com/x"}])
        result = await lookup_sitemap_links(db, ["d:claims.timely_filing"], None)
        assert result == []

    @pytest.mark.asyncio
    async def test_no_matching_d_tag_short_circuits(self):
        db = _FakeDB(rows=[{"url": "https://payer.com/x"}])
        result = await lookup_sitemap_links(db, ["j:payor.aetna"], "Aetna")
        assert result == []

    @pytest.mark.asyncio
    async def test_maps_rows_to_suggested_links_with_derived_titles(self):
        db = _FakeDB(rows=[
            {"url": "https://sunshinehealth.com/providers/timely-filing-policy"},
            {"url": "https://sunshinehealth.com/claims/appeals"},
        ])
        result = await lookup_sitemap_links(
            db, ["d:claims.timely_filing"], "Sunshine Health",
        )
        assert result == [
            SuggestedLink(url="https://sunshinehealth.com/providers/timely-filing-policy",
                           title="Timely Filing Policy"),
            SuggestedLink(url="https://sunshinehealth.com/claims/appeals",
                           title="Appeals"),
        ]

    @pytest.mark.asyncio
    async def test_passes_payer_and_limit_params(self):
        db = _FakeDB(rows=[])
        await lookup_sitemap_links(
            db, ["d:pharmacy.formulary"], "Sunshine Health", limit=5,
        )
        assert db.last_params["payer"] == "Sunshine Health"
        assert db.last_params["limit"] == 5

    @pytest.mark.asyncio
    async def test_db_error_fails_closed_not_crashes(self):
        db = _ExplodingDB()
        result = await lookup_sitemap_links(
            db, ["d:claims.timely_filing"], "Sunshine Health",
        )
        assert result == []
