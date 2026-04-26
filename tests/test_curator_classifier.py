"""Tests for app.curator.classifier.

Covers the URL → metadata mapping that drives the entire curator
inference path. Pure functions, no DB, sub-millisecond.

Cases that mattered enough to lock down:
* canonical payer mapping for our top 4 hosts (sunshine, AHCA, CMS, UHC)
* www. and www-es. prefix stripping
* parent-domain fallback (uhccommunityplan.com → United Healthcare)
* unknown host returns (None, None) — caller decides what to do
* authority_level patterns: billing-manual, clinical-policies, criteria
* doc vs page split via extension
* ``classify_url`` smoke for the dict shape
"""
from __future__ import annotations

import pytest

from app.curator.classifier import (
    classify_url,
    infer_authority_level,
    infer_payer,
)


# ── infer_payer ──────────────────────────────────────────────────────


@pytest.mark.parametrize("host, expected_payer, expected_state", [
    ("www.sunshinehealth.com",      "Sunshine Health", "FL"),
    ("sunshinehealth.com",          "Sunshine Health", "FL"),
    ("www-es.sunshinehealth.com",   "Sunshine Health", "FL"),  # Spanish variant
    ("ahca.myflorida.com",          "AHCA",            "FL"),
    ("www.cms.gov",                 "CMS",             None),
    ("cms.gov",                     "CMS",             None),
    ("www.aetna.com",               "Aetna",           None),
])
def test_infer_payer_known_hosts(host, expected_payer, expected_state):
    payer, state = infer_payer(host)
    assert payer == expected_payer
    assert state == expected_state


def test_infer_payer_unknown_returns_none():
    """Unknown host returns (None, None) — never guess. Caller can
    decide whether to use the host as a weak hint or mark uncategorized.
    """
    assert infer_payer("example.org") == (None, None)
    assert infer_payer("random.subdomain.example.com") == (None, None)


def test_infer_payer_subdomain_fallback():
    """A subdomain like ``portal.uhccommunityplan.com`` should still
    map to United Healthcare via the parent-domain fallback.
    """
    payer, _ = infer_payer("portal.uhccommunityplan.com")
    assert payer == "United Healthcare"


# ── infer_authority_level ────────────────────────────────────────────


@pytest.mark.parametrize("path, expected_level", [
    ("/providers/Billing-manual.html",                       "payer_manual"),
    ("/providers/Billing-manual/appeals.html",               "payer_manual"),
    ("/providers/provider-manual.pdf",                       "payer_manual"),
    ("/content/dam/Sunshine/PRO-PE-Manual.pdf",              "payer_manual"),
    ("/providers/utilization-management/clinical-policies.html", "payer_policy"),
    ("/providers/payment-policies/cc-pp-019.pdf",            "payer_policy"),
    ("/medicaid/clinical-policy/foo.html",                   "payer_policy"),
    ("/medicaid/recent_presentations/training.pdf",          "training"),
    ("/2025-Transportation-Fee-Schedule.pdf",                "fee_schedule"),
    ("/members/handbook.html",                               "member_handbook"),
    ("/criteria/Spravato.pdf",                               "payer_policy"),
])
def test_infer_authority_level_patterns(path, expected_level):
    assert infer_authority_level(path) == expected_level


def test_infer_authority_level_no_match():
    """Generic / commercial paths without a specific pattern return
    None. Operator can override later via /sources/{id}/curate.
    """
    for path in ("/", "/about-us.html", "/contact.html", "/login.html", "/random/path"):
        assert infer_authority_level(path) is None


def test_infer_authority_level_specificity_order():
    """The 'criteria' pattern is more specific than fee-schedule.
    A path that could match both should pick the first declared pattern.
    Currently 'fee_schedule' is matched only via /fee-schedule/ explicitly.
    """
    # Sanity: criteria beats absence-of-billing-manual
    assert infer_authority_level("/providers/clinical-policies/criteria/foo.pdf") == "payer_policy"


# ── classify_url ─────────────────────────────────────────────────────


def test_classify_url_full_shape():
    """Smoke test that the integrator returns every key downstream
    code expects to spread into a DiscoveredSource row.
    """
    out = classify_url("https://www.sunshinehealth.com/providers/Billing-manual.html")
    expected_keys = {
        "host", "path", "payer", "state",
        "inferred_authority_level", "content_kind", "extension",
    }
    assert set(out.keys()) == expected_keys
    assert out["host"] == "www.sunshinehealth.com"
    assert out["path"] == "/providers/Billing-manual.html"
    assert out["payer"] == "Sunshine Health"
    assert out["state"] == "FL"
    assert out["inferred_authority_level"] == "payer_manual"
    assert out["content_kind"] == "page"
    assert out["extension"] == "html"


def test_classify_url_pdf_doc():
    """A PDF URL should classify as content_kind='doc'."""
    out = classify_url("https://ahca.myflorida.com/content/download/X/file/Foo_Criteria.pdf")
    assert out["content_kind"] == "doc"
    assert out["extension"] == "pdf"
    assert out["payer"] == "AHCA"


def test_classify_url_no_extension_is_page():
    """No file extension → treat as HTML page (content_kind='page'),
    extension=None. Matches what we see for clean-URL CMS pages.
    """
    out = classify_url("https://www.sunshinehealth.com/providers/")
    assert out["content_kind"] == "page"
    assert out["extension"] is None


def test_classify_url_unknown_payer_returns_none():
    """Unknown host yields payer=None but still returns valid path/kind.
    Useful so the caller can persist the row uncategorized rather
    than dropping it.
    """
    out = classify_url("https://example.org/some/page.html")
    assert out["payer"] is None
    assert out["state"] is None
    assert out["content_kind"] == "page"
    assert out["host"] == "example.org"


def test_classify_url_query_string_doesnt_affect_kind():
    """A URL like /file.pdf?version=2 is still a doc (extension PDF
    despite the query string).
    """
    out = classify_url("https://ahca.myflorida.com/content/download/x/file/foo.pdf?version=2")
    assert out["content_kind"] == "doc"
    assert out["extension"] == "pdf"


def test_classify_url_path_with_dot_in_slug_not_treated_as_extension():
    """AHCA rule URLs have dots inside path slugs (e.g. 'rule-59g-4.002')
    — those dots are NOT extension markers. Without guarding, we'd
    pull a 54-char 'extension' and blow VARCHAR(20) at insert time.
    """
    out = classify_url(
        "https://ahca.myflorida.com/medicaid/rules/"
        "rule-59g-4.002-provider-reimbursement-schedules-and-billing-codes"
    )
    # No real file extension → treated as page, ext is None
    assert out["extension"] is None
    assert out["content_kind"] == "page"


def test_classify_url_extension_must_be_short_and_alphanumeric():
    """Belt-and-suspenders for extension detection: only treat as
    extension if 1-8 chars AND alphanumeric. URL slugs with hyphens
    or that are too long are NOT extensions.
    """
    # 9+ chars: not an extension
    out = classify_url("https://x.com/foo.something_extralong")
    assert out["extension"] is None
    # Has a hyphen: not an extension
    out = classify_url("https://x.com/foo.has-hyphen")
    assert out["extension"] is None
    # Short alphanum: real extension
    out = classify_url("https://x.com/foo.PDF")
    assert out["extension"] == "pdf"
    out = classify_url("https://x.com/foo.html")
    assert out["extension"] == "html"
    out = classify_url("https://x.com/sheet.xlsx")
    assert out["extension"] == "xlsx"
