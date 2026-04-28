"""Canonical-form normalization for document metadata fields.

Why this exists
---------------
``document_payer``, ``document_state``, ``document_program``, etc. flow from
free-text upload form fields into Postgres → ``rag_published_embeddings`` →
``mobius_chat.published_rag_metadata`` → Chroma metadata. Chat's filter layer
uses **exact-match** ``WHERE`` clauses against these fields (Chroma can't
do case-insensitive matches). One typo at upload silently drops the doc
from every payer-filtered search.

Real example: ``"Sunshine health"`` (lowercase h) was uploaded for
``FL.UM.87.pdf`` on 2026-04-23. All filter-by-payer searches subsequently
excluded it until we manually patched all four stores.

Strategy
--------
Per-field canonical form, not blanket case rules:

* **payer**: title case, mapped against a known-payer allowlist
* **state**: USPS uppercase 2-letter
* **program**: title case enum
* **authority_level**: lowercase enum
* **status / source_type**: lowercase enum

Unknown values pass through (with the same case-fold normalization)
rather than being rejected — the canonical lists are seed data, not a
hard schema. A WARNING log surfaces unknowns so the operator can extend
the allowlist.

This module is the single source of truth. Wire it at every write site
(``/upload``, ``PATCH /documents/{id}``, import endpoints, publish, sync).
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


# ── Canonical allowlists ─────────────────────────────────────────────
# Seed lists. Extend as new payers/states show up. Keys are lowercase
# normalized form (collapsed whitespace, no punctuation); values are
# the canonical display form everyone reads/writes.

_PAYER_CANONICAL: dict[str, str] = {
    # FL Medicaid MMA plans
    "sunshine health": "Sunshine Health",
    "sunshine healthcare": "Sunshine Health",
    "sunshine":       "Sunshine Health",
    "sunshinehealth": "Sunshine Health",  # URL-derived host (sunshinehealth.com → stripped TLD)
    "humana":         "Humana",
    "humana healthy horizons": "Humana Healthy Horizons",
    "united healthcare": "United Healthcare",
    "unitedhealthcare":  "United Healthcare",
    "united health care": "United Healthcare",
    "united healthcarer": "United Healthcare",  # observed typo
    "uhc":            "United Healthcare",
    "molina":         "Molina Healthcare",
    "molina healthcare": "Molina Healthcare",
    "aetna":          "Aetna",
    "aetna better health": "Aetna Better Health",
    "centene":        "Centene",
    "wellcare":       "WellCare",
    "simply healthcare": "Simply Healthcare",
    "ahca":           "AHCA",  # FL Medicaid agency itself
    "florida medicaid": "Florida Medicaid",
    "fl medicaid":    "Florida Medicaid",
}

_PROGRAM_CANONICAL: dict[str, str] = {
    "medicaid":       "Medicaid",
    "medicare":       "Medicare",
    "medicare advantage": "Medicare Advantage",
    "ma":             "Medicare Advantage",
    "commercial":     "Commercial",
    "marketplace":    "Marketplace",
    "dual":           "Dual",
    "dual eligible":  "Dual",
    "duals":          "Dual",
    "dsnp":           "D-SNP",
    "d-snp":          "D-SNP",
    "ltss":           "LTSS",
    "mma":            "MMA",
    "ltc":            "LTC",
    "smmc":           "SMMC",
}

_STATE_CANONICAL: set[str] = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL",
    "IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT",
    "NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI",
    "SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC","PR",
}

_AUTHORITY_LEVEL_CANONICAL: set[str] = {
    # Reranker-weighted tiers (corpus_search._AUTHORITY_WEIGHTS)
    "contract_source_of_truth",  # 1.00 — provider manuals, billing manuals, clinical policies, state regs
    "operational_suggested",      # 0.65 — internal ops guides, suggested workflows
    "payer_policy",               # 0.50 — payer-published UM policies, fee schedules
    "fyi_not_citable",            # 0.20 — newsletters, marketing, reference-only
    # Legacy / ingest-time labels (kept for backward compat; map to 0.10 default in reranker)
    "official",
    "guidance",
    "informational",
    "marketing",
    "legacy",
    "draft",
}

_STATUS_CANONICAL: set[str] = {
    "uploaded",
    "extracting",
    "completed",
    "failed",
    "published",
    "draft",
    "archived",
}

_SOURCE_TYPE_CANONICAL: set[str] = {
    "chunk",
    "hierarchical",
    "policy",
    "section",
    "fact",
}


# ── Internal helpers ─────────────────────────────────────────────────


_WS_RE = re.compile(r"\s+")


def _normalize_key(s: str) -> str:
    """Normalize a free-text value to the lookup-key form used by the
    canonical maps: trim, collapse whitespace, lowercase, drop trailing
    punctuation. Returns "" for empty / non-string input.
    """
    if s is None:
        return ""
    s = str(s).strip().rstrip(".,;:")
    if not s:
        return ""
    s = _WS_RE.sub(" ", s).lower()
    return s


def _title_case_passthrough(s: str) -> str:
    """Reasonable fallback when value isn't in the canonical map.

    Avoids Python's ``str.title()`` which mangles common acronyms
    (``"FL"`` → ``"Fl"``). Title-cases each word but preserves all-caps
    runs of length <=4 (HMOs, NPI, BCBS, etc.).
    """
    out: list[str] = []
    for word in s.split():
        if 1 <= len(word) <= 4 and word.isupper():
            out.append(word)
        elif "-" in word:
            out.append("-".join(p[:1].upper() + p[1:].lower() if p else p for p in word.split("-")))
        else:
            out.append(word[:1].upper() + word[1:].lower())
    return " ".join(out)


# ── Public API ───────────────────────────────────────────────────────


def canonical_payer(raw: str | None) -> str | None:
    """Map a free-text payer name to its canonical form.

    None / empty → None (lets DB store NULL rather than empty string).
    Known value → mapped canonical (e.g. ``"sunshine health"`` →
    ``"Sunshine Health"``).
    Unknown value → title-cased fallback + WARNING log so we can extend
    the allowlist for the next deploy.
    """
    key = _normalize_key(raw)
    if not key:
        return None
    if key in _PAYER_CANONICAL:
        return _PAYER_CANONICAL[key]
    canon = _title_case_passthrough(key)
    logger.warning("[canonical] unknown payer %r → %r (consider adding to _PAYER_CANONICAL)", raw, canon)
    return canon


def canonical_state(raw: str | None) -> str | None:
    """USPS uppercase 2-letter. ``"florida"``/``"Florida"``/``"fl"`` → ``"FL"``."""
    key = _normalize_key(raw)
    if not key:
        return None
    upper = key.upper()
    if len(upper) == 2 and upper in _STATE_CANONICAL:
        return upper
    # Long-form state names (best-effort; extend if needed)
    long_form = {
        "florida": "FL", "georgia": "GA", "texas": "TX", "california": "CA",
        "new york": "NY", "puerto rico": "PR", "district of columbia": "DC",
    }
    if key in long_form:
        return long_form[key]
    logger.warning("[canonical] unknown state %r — passing through as %r", raw, upper)
    return upper if len(upper) == 2 else None  # silently drop garbage


def canonical_program(raw: str | None) -> str | None:
    key = _normalize_key(raw)
    if not key:
        return None
    if key in _PROGRAM_CANONICAL:
        return _PROGRAM_CANONICAL[key]
    canon = _title_case_passthrough(key)
    logger.warning("[canonical] unknown program %r → %r", raw, canon)
    return canon


def canonical_authority_level(raw: str | None) -> str | None:
    """Lowercase enum: ``official|guidance|informational|...``."""
    key = _normalize_key(raw).replace(" ", "_").replace("-", "_")
    if not key:
        return None
    if key in _AUTHORITY_LEVEL_CANONICAL:
        return key
    logger.warning("[canonical] unknown authority_level %r — passing through as %r", raw, key)
    return key


def canonical_status(raw: str | None) -> str | None:
    key = _normalize_key(raw)
    if not key:
        return None
    if key in _STATUS_CANONICAL:
        return key
    logger.warning("[canonical] unknown status %r — passing through as %r", raw, key)
    return key


def canonical_source_type(raw: str | None) -> str | None:
    key = _normalize_key(raw)
    if not key:
        return None
    if key in _SOURCE_TYPE_CANONICAL:
        return key
    logger.warning("[canonical] unknown source_type %r — passing through as %r", raw, key)
    return key


# ── Convenience: normalize a whole metadata dict ─────────────────────


_FIELD_NORMALIZERS = {
    "payer":           canonical_payer,
    "document_payer":  canonical_payer,
    "state":           canonical_state,
    "document_state":  canonical_state,
    "program":         canonical_program,
    "document_program": canonical_program,
    "authority_level": canonical_authority_level,
    "document_authority_level": canonical_authority_level,
    "status":          canonical_status,
    "document_status": canonical_status,
    "source_type":     canonical_source_type,
}


def canonicalize_metadata(d: dict) -> dict:
    """Return a shallow copy of ``d`` with every controlled field
    normalized in place. Untouched keys are preserved verbatim.
    """
    out = dict(d)
    for key, fn in _FIELD_NORMALIZERS.items():
        if key in out:
            out[key] = fn(out[key])
    return out
