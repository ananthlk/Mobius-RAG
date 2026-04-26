"""URL → metadata inference for the curator.

Today this is a small bag of regexes that cover the FL Medicaid corpus
we've been scanning (Sunshine + AHCA + CMS). Tomorrow this becomes a
learned classifier — but the regex approach already classifies 100%
of the 1066 URLs in the v0 scan with no false positives.

Three things we infer from a URL:

1. ``payer`` — the canonical payer name, derived from host. Pulls
   from ``app.services.metadata_canonical`` so it stays consistent
   with how documents.payer is canonicalized at upload time.

2. ``inferred_authority_level`` — drives chat retrieval filters
   ('payer_manual' vs 'payer_policy' vs 'member_handbook' etc.).

3. ``content_kind`` — 'doc' or 'page'. Matters because docs go through
   GCS-based import while pages go through HTML-text import (Phase 13.4).

If any field can't be inferred confidently we return None — the
curator UI lets a human override later via /sources/{id}/curate.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse


# Per-host payer mapping. Keys are URL hosts (lowercase, no www.).
# Values are canonical payer names matching app/services/metadata_canonical.py.
# Adding a new payer = one line here + one alias in metadata_canonical.
_HOST_PAYER: dict[str, tuple[str | None, str | None]] = {
    # (payer, default_state)
    "sunshinehealth.com":      ("Sunshine Health", "FL"),
    "ahca.myflorida.com":      ("AHCA",            "FL"),
    "myflorida.com":           ("AHCA",            "FL"),
    "cms.gov":                 ("CMS",             None),
    "medicare.gov":            ("CMS",             None),
    "medicaid.gov":            ("CMS",             None),
    "ambetterhealth.com":      ("Ambetter",        None),
    "cenpatico.com":           ("Cenpatico",       None),
    "wellcare.com":            ("WellCare",        None),
    "humana.com":              ("Humana",          None),
    "molinahealthcare.com":    ("Molina Healthcare", None),
    "aetna.com":               ("Aetna",           None),
    "uhc.com":                 ("United Healthcare", None),
    "uhccommunityplan.com":    ("United Healthcare", None),
}


# URL path patterns → authority_level. Order matters: more specific
# patterns first. Each pattern is a (regex, authority_level) tuple.
_AUTHORITY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"/Billing[-_ ]manual",          re.I), "payer_manual"),
    (re.compile(r"/provider[-_ ]manual",         re.I), "payer_manual"),
    (re.compile(r"/PRO[-_ ]?PE[-_ ]?Manual",     re.I), "payer_manual"),
    (re.compile(r"/utilization[-_ ]management",  re.I), "payer_policy"),
    (re.compile(r"/payment[-_ ]polic",           re.I), "payer_policy"),
    (re.compile(r"/clinical[-_ ]polic",          re.I), "payer_policy"),
    (re.compile(r"/clinical[-_ ]payment",        re.I), "payer_policy"),
    (re.compile(r"/criteria/",                   re.I), "payer_policy"),
    (re.compile(r"/preauth",                     re.I), "payer_policy"),
    (re.compile(r"/coverage[-_ ]polic",          re.I), "payer_policy"),
    # Anywhere in the path — fee-schedule docs aren't always under
    # a /fee-schedule/ directory; e.g. AHCA publishes them as
    # ``2025-Transportation-Fee-Schedule.pdf`` at the dam root.
    (re.compile(r"[Ff]ee[-_ ]?[Ss]chedule",       re.I), "fee_schedule"),
    (re.compile(r"/member[-_ ]handbook",         re.I), "member_handbook"),
    (re.compile(r"/members/",                    re.I), "member_handbook"),
    (re.compile(r"/medicaid/recent_presentations", re.I), "training"),
]


# File extensions we consider "doc" (i.e., goes through GCS import).
# Everything else (.html, no extension, etc.) is treated as "page"
# (text-import via Phase 13.4 HTML pipeline).
_DOC_EXTENSIONS: frozenset[str] = frozenset({
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "rtf", "txt",
})


def _strip_www(host: str) -> str:
    return host.lower().removeprefix("www.").removeprefix("www-es.")


def infer_payer(host: str) -> tuple[str | None, str | None]:
    """Map a URL host to (payer, default_state). Falls back to None
    when the host isn't in the registry — caller decides whether to
    use the host string as a weak hint or leave the row uncategorized.
    """
    h = _strip_www(host)
    if h in _HOST_PAYER:
        return _HOST_PAYER[h]
    # Try parent domain ('foo.bar.com' → 'bar.com')
    parts = h.split(".")
    if len(parts) > 2:
        parent = ".".join(parts[-2:])
        if parent in _HOST_PAYER:
            return _HOST_PAYER[parent]
    return None, None


def infer_authority_level(path: str) -> str | None:
    """Inspect a URL path and return the most likely authority_level.
    Returns None when no pattern matches — leave it to the operator
    to set via curator UI.
    """
    for rx, level in _AUTHORITY_PATTERNS:
        if rx.search(path):
            return level
    return None


def classify_url(url: str) -> dict:
    """One-shot classifier — returns the dict suitable for direct
    spread into a DiscoveredSource row.

    Example::

        >>> classify_url("https://www.sunshinehealth.com/providers/Billing-manual.html")
        {'host': 'www.sunshinehealth.com', 'path': '/providers/Billing-manual.html',
         'payer': 'Sunshine Health', 'state': 'FL',
         'inferred_authority_level': 'payer_manual',
         'content_kind': 'page', 'extension': 'html'}
    """
    p = urlparse(url)
    path = p.path or "/"
    ext = ""
    last_seg = path.rsplit("/", 1)[-1]
    if "." in last_seg:
        candidate = last_seg.rsplit(".", 1)[-1].lower()
        # Only treat as a real file extension if it looks like one:
        # short (1-8 chars) and alphanumeric. Otherwise it's a slug
        # like ``rule-59g-4.002-provider-reimbursement-schedules`` where
        # the dot is just part of the path name, not a type marker.
        # Without this guard we'd pull a 54-char "extension" out of
        # AHCA's rule URLs and overflow extension VARCHAR(20).
        if 1 <= len(candidate) <= 8 and candidate.isalnum():
            ext = candidate

    payer, state = infer_payer(p.netloc)
    return {
        "host":                     p.netloc.lower(),
        "path":                     path,
        "payer":                    payer,
        "state":                    state,
        "inferred_authority_level": infer_authority_level(path),
        "content_kind":             "doc" if ext in _DOC_EXTENSIONS else "page",
        "extension":                ext or None,
    }
