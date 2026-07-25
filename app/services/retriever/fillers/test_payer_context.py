"""Tests for the shared payer-context resolution (Filler c/d/f/s).

Verifies the ported logic behaves like the legacy
corpus_search_strategy_d.py._extract_payer_slug/_resolve_payer_context it was
freshly re-implemented from, not just "looks like the original."
"""

import sys

import pytest

from app.services.retriever.fillers import payer_context
from app.services.retriever.fillers.payer_context import (
    PayerContext,
    extract_payer_slug,
    resolve_payer_context,
)


@pytest.fixture(autouse=True)
def _no_real_payor_platform_call(monkeypatch):
    """resolve_payer_context tries a live HTTP call to the Payor Platform
    registry first. These tests exercise the discovered_sources FALLBACK
    layer only (layer 2) -- point layer 1 at a closed local port so it fails
    fast (connection refused) instead of a real network call or a 3s
    timeout, keeping these tests fast and hermetic."""
    monkeypatch.setattr(payer_context, "_PAYOR_PLATFORM_BASE", "http://127.0.0.1:1")


# ---------------------------------------------------------------------------
# Fakes (proper async context-manager-free db stub, matching test_router.py's
# "no AsyncMock gymnastics" convention).
# ---------------------------------------------------------------------------

class _FakeRows:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _FakeDB:
    """Returns whatever row set was configured, regardless of the query
    text, since these tests exercise resolve_payer_context's own filtering
    logic (dominant-host / min-rows), not SQL correctness."""

    def __init__(self, rows):
        self._rows = rows

    async def execute(self, stmt, params=None):
        return _FakeRows(self._rows)


class _ExplodingDB:
    async def execute(self, stmt, params=None):
        raise RuntimeError("db unavailable")


class _FakeRegistryResponse:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


class _FakeRegistryClient:
    """Stands in for httpx.AsyncClient -- async context manager whose .get()
    returns a canned registry response, so the tri-state crawlable branch
    (registry layer) is exercised without a real network call."""

    def __init__(self, response, *, raises=None):
        self._response = response
        self._raises = raises

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        if self._raises:
            raise self._raises
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, *args, **kwargs):
        return self._response


def _patch_registry(monkeypatch, *, status_code=200, data=None, raises=None):
    """Injects a fake httpx module so resolve_payer_context's local
    `import httpx` picks up our stub instead of hitting the network."""
    fake_httpx = type(sys)("httpx")
    fake_httpx.AsyncClient = _FakeRegistryClient(
        _FakeRegistryResponse(status_code, data or {}), raises=raises,
    )
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)


# ---------------------------------------------------------------------------
# extract_payer_slug
# ---------------------------------------------------------------------------

class TestExtractPayerSlug:
    def test_finds_payor_tag(self):
        assert extract_payer_slug(["d:claims.timely_filing", "j:payor.sunshine_health"]) == "sunshine_health"

    def test_no_payor_tag_returns_none(self):
        assert extract_payer_slug(["d:claims.timely_filing", "j:regulatory_authority.ahca"]) is None

    def test_empty_or_none_tag_matches(self):
        assert extract_payer_slug([]) is None
        assert extract_payer_slug(None) is None

    def test_first_matching_tag_wins(self):
        assert extract_payer_slug(["j:payor.aetna", "j:payor.humana"]) == "aetna"


# ---------------------------------------------------------------------------
# resolve_payer_context — discovered_sources fallback layer
# (Payor Platform HTTP layer isn't exercised here — httpx call against a
# real/fake service is out of scope for a unit test; these confirm the
# fallback's own filtering logic, which is what Filler f actually depends on
# for payers the registry has no opinion on yet.)
# ---------------------------------------------------------------------------

class TestResolvePayerContextFallback:
    @pytest.mark.asyncio
    async def test_no_slug_returns_all_none(self):
        db = _FakeDB(rows=[])
        result = await resolve_payer_context(db, None)
        assert result == PayerContext(slug=None, display_name=None, site_domain=None, crawlable=None)
        assert result.source == "none"

    @pytest.mark.asyncio
    async def test_below_min_rows_returns_all_none_but_keeps_slug(self):
        # _MIN_PAYER_ROWS = 3 -- two rows on the same host isn't enough.
        db = _FakeDB(rows=[
            {"payer": "Sunshine Health", "url": "https://sunshinehealth.com/a"},
            {"payer": "Sunshine Health", "url": "https://sunshinehealth.com/b"},
        ])
        result = await resolve_payer_context(db, "sunshine_health")
        assert result == PayerContext(slug="sunshine_health", display_name=None, site_domain=None, crawlable=None)
        assert result.source == "none"

    @pytest.mark.asyncio
    async def test_dominant_host_wins_with_enough_rows(self):
        db = _FakeDB(rows=[
            {"payer": "Sunshine Health", "url": "https://www.sunshinehealth.com/a"},
            {"payer": "Sunshine Health", "url": "https://sunshinehealth.com/b"},
            {"payer": "Sunshine Health", "url": "https://sunshinehealth.com/c"},
        ])
        result = await resolve_payer_context(db, "sunshine_health")
        # www. stripped, so both hosts collapse to the same dominant host.
        # crawlable stays None -- the fallback layer has no robots signal --
        # but .source derives to "crawl_history", not a registry verdict.
        assert result == PayerContext(
            slug="sunshine_health", display_name="Sunshine Health",
            site_domain="sunshinehealth.com", crawlable=None,
        )
        assert result.source == "crawl_history"

    @pytest.mark.asyncio
    async def test_mixed_stray_hosts_below_min_returns_all_none(self):
        # 3 total rows but split across hosts -- no single host clears
        # _MIN_PAYER_ROWS, matching legacy's "can't manufacture confidence
        # from stray/mixed rows" guard.
        db = _FakeDB(rows=[
            {"payer": "Aetna", "url": "https://aetnabetterhealth.com/a"},
            {"payer": "Aetna", "url": "https://aetna.com/b"},
            {"payer": "Aetna", "url": "https://someaggregator.com/c"},
        ])
        result = await resolve_payer_context(db, "aetna")
        assert result == PayerContext(slug="aetna", display_name=None, site_domain=None, crawlable=None)
        assert result.source == "none"

    @pytest.mark.asyncio
    async def test_db_error_fails_closed_not_crashes(self):
        db = _ExplodingDB()
        result = await resolve_payer_context(db, "sunshine_health")
        assert result == PayerContext(slug="sunshine_health", display_name=None, site_domain=None, crawlable=None)
        assert result.source == "none"


# ---------------------------------------------------------------------------
# resolve_payer_context — Payor Platform registry layer, tri-state crawlable
# (the actual gap Filler d found: crawlable=False must stay distinguishable
# from "no opinion", since Router's crawl-gate fails open on None but
# disqualifies strategy d on an explicit False.)
# ---------------------------------------------------------------------------

class TestResolvePayerContextRegistryTriState:
    @pytest.mark.asyncio
    async def test_registry_crawlable_true_returns_host_and_true(self, monkeypatch):
        _patch_registry(monkeypatch, data={"crawlable": True, "host": "sunshinehealth.com"})
        db = _FakeDB(rows=[])  # should never be reached -- registry answers first
        result = await resolve_payer_context(db, "sunshine_health")
        assert result == PayerContext(
            slug="sunshine_health", display_name="Sunshine Health",
            site_domain="sunshinehealth.com", crawlable=True,
        )
        assert result.source == "metafact"

    @pytest.mark.asyncio
    async def test_registry_crawlable_false_preserves_display_name_not_just_none_none(self, monkeypatch):
        # The real fix Filler d's finding required: False must stay False,
        # AND display_name (independent of crawlability) must survive --
        # unlike the pre-fix behavior that discarded it too.
        _patch_registry(monkeypatch, data={"crawlable": False, "host": "aetnabetterhealth.com"})
        db = _FakeDB(rows=[])
        result = await resolve_payer_context(db, "aetna")
        assert result == PayerContext(
            slug="aetna", display_name="Aetna", site_domain=None, crawlable=False,
        )
        assert result.source == "metafact"

    @pytest.mark.asyncio
    async def test_registry_null_crawlable_falls_through_to_discovered_sources(self, monkeypatch):
        _patch_registry(monkeypatch, data={"crawlable": None})
        db = _FakeDB(rows=[
            {"payer": "Sunshine Health", "url": "https://sunshinehealth.com/a"},
            {"payer": "Sunshine Health", "url": "https://sunshinehealth.com/b"},
            {"payer": "Sunshine Health", "url": "https://sunshinehealth.com/c"},
        ])
        result = await resolve_payer_context(db, "sunshine_health")
        # Fell through to the crawl-history layer -- crawlable stays None
        # (no robots signal there), .source derives to "crawl_history".
        assert result == PayerContext(
            slug="sunshine_health", display_name="Sunshine Health",
            site_domain="sunshinehealth.com", crawlable=None,
        )
        assert result.source == "crawl_history"

    @pytest.mark.asyncio
    async def test_registry_non_200_falls_through_crawlable_stays_none(self, monkeypatch):
        _patch_registry(monkeypatch, status_code=503, data={})
        db = _FakeDB(rows=[])
        result = await resolve_payer_context(db, "sunshine_health")
        assert result == PayerContext(
            slug="sunshine_health", display_name=None, site_domain=None, crawlable=None,
        )
        assert result.source == "none"


class TestSourceDerivation:
    """Direct unit tests of the derived .source property, independent of
    resolve_payer_context's own branches -- confirms the derivation rule
    itself (Router's reasoning for withdrawing the stored field) rather
    than just the specific cases resolve_payer_context happens to hit."""

    def test_crawlable_true_is_metafact_regardless_of_site_domain(self):
        assert PayerContext(slug="x", display_name="X", site_domain=None, crawlable=True).source == "metafact"

    def test_crawlable_false_is_metafact_even_without_site_domain(self):
        assert PayerContext(slug="x", display_name="X", site_domain=None, crawlable=False).source == "metafact"

    def test_crawlable_none_with_site_domain_is_crawl_history(self):
        assert PayerContext(slug="x", display_name="X", site_domain="x.com", crawlable=None).source == "crawl_history"

    def test_crawlable_none_without_site_domain_is_none(self):
        assert PayerContext(slug="x", display_name=None, site_domain=None, crawlable=None).source == "none"
