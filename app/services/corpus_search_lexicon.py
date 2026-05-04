"""Lexicon-driven query expansion for BM25.

Matches the user's natural-language query against the curated
policy_lexicon_entries table (231 hand-approved tags × strong_phrases
+ aliases). Returns expansion phrases that are OR-joined with the
original query tokens, so:

  user query:  "DME prior auth"
  lexicon hit: d:benefits.dme               (strong_phrases: durable medical equipment, hme, ...)
               d:utilization_management.prior_authorization (strong_phrases: prior authorization, PA, preauth, ...)
  expanded:    'DME OR prior OR auth OR (durable medical equipment) OR hme OR PA OR preauth OR ...'

Cached in-process with a 5-minute TTL since lexicon changes are rare
and lookup is per-query (cold path uncached = ~30 ms across 231 rows).

The expansion bag is consumed by ``corpus_search._bm25_arm``, which
joins it with the original query tokens via OR so brand names not in
the lexicon (e.g. "Express Scripts") still match.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re as _re
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# In-process cache TTL for the lexicon snapshot.  Lexicon edits are rare
# (manual curation), and a stale 5-minute window is acceptable in exchange
# for skipping a 30 ms DB round-trip on every search.
_CACHE_TTL_SECONDS = 5 * 60

# Cap on number of lexicon entries that contribute to one query.  Without
# this a generic word like "claim" could light up a dozen entries and
# explode the tsquery.
_MAX_ENTRIES_PER_QUERY = 12


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LexiconExpansion:
    """Result of expanding one user query through the lexicon.

    Attributes
    ----------
    matched_codes:
        Full codes of matched entries, e.g. ``["d:benefits.dme",
        "d:utilization_management.prior_authorization"]``.
    expansion_phrases:
        Deduplicated bag of phrases (strong_phrases ∪ aliases ∪ leaf names)
        from all matched entries.  Caller OR-joins these with raw tokens.
    domain_tags / jurisdiction_tags / process_tags:
        Codes split by kind for downstream filtering / analytics.
    log:
        Human-readable trace lines like
        ``"matched 'DME' → d:benefits.dme"`` for the pipeline_trace UI.
    """

    matched_codes: list[str] = field(default_factory=list)
    expansion_phrases: list[str] = field(default_factory=list)
    domain_tags: list[str] = field(default_factory=list)
    jurisdiction_tags: list[str] = field(default_factory=list)
    process_tags: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Lexicon snapshot cache (in-process, 5-minute TTL)
# ---------------------------------------------------------------------------

# Each cached entry is the minimal projection we need at match time:
#   {"kind": "d", "code": "benefits.dme", "phrases": ["dme", "durable medical equipment", ...]}
# The phrases list is normalized lowercase + deduplicated.

_cache_lock = asyncio.Lock()
_cache_payload: list[dict[str, Any]] | None = None
_cache_loaded_at: float = 0.0


def _normalize_phrase(p: Any) -> str:
    if not isinstance(p, str):
        return ""
    return p.strip().lower()


def _extract_phrases(spec: Any) -> list[str]:
    """Pull phrases out of a lexicon entry's spec JSON.

    Accepts ``strong_phrases``, ``aliases``, and ``phrases`` keys (all
    optional).  Returns a deduplicated lowercase list.
    """
    if not isinstance(spec, dict):
        return []
    bag: list[str] = []
    for key in ("strong_phrases", "aliases", "phrases"):
        v = spec.get(key)
        if isinstance(v, list):
            bag.extend(_normalize_phrase(x) for x in v if x)
        elif isinstance(v, str):
            bag.append(_normalize_phrase(v))
    # dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in bag:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _leaf_name(code: str) -> str:
    """``benefits.dme`` -> ``dme``;  ``utilization_management.prior_authorization`` -> ``prior authorization``."""
    leaf = (code or "").split(".")[-1]
    return leaf.replace("_", " ").strip().lower()


async def _load_lexicon_snapshot(db: AsyncSession) -> list[dict[str, Any]]:
    """Return cached lexicon entries; refresh if stale."""
    global _cache_payload, _cache_loaded_at

    now = time.monotonic()
    if _cache_payload is not None and (now - _cache_loaded_at) < _CACHE_TTL_SECONDS:
        return _cache_payload

    async with _cache_lock:
        # Re-check after acquiring lock — another coroutine may have refreshed.
        now = time.monotonic()
        if _cache_payload is not None and (now - _cache_loaded_at) < _CACHE_TTL_SECONDS:
            return _cache_payload

        try:
            result = await db.execute(
                text(
                    "SELECT kind, code, spec FROM policy_lexicon_entries "
                    "WHERE active = true"
                )
            )
            rows = result.mappings().all()
        except Exception as exc:
            logger.warning(
                "corpus_search_lexicon: failed to load policy_lexicon_entries: %s "
                "(falling back to empty lexicon)",
                exc,
            )
            _cache_payload = []
            _cache_loaded_at = now
            return _cache_payload

        snapshot: list[dict[str, Any]] = []
        for row in rows:
            kind = (row["kind"] or "").strip().lower()
            code = (row["code"] or "").strip()
            if not kind or not code:
                continue
            phrases = _extract_phrases(row["spec"])
            leaf = _leaf_name(code)
            if leaf and leaf not in phrases:
                phrases.append(leaf)
            if not phrases:
                continue
            snapshot.append(
                {
                    "kind": kind,
                    "code": code,
                    "full_code": f"{kind}:{code}",
                    "phrases": phrases,
                }
            )

        _cache_payload = snapshot
        _cache_loaded_at = now
        logger.info(
            "corpus_search_lexicon: refreshed lexicon snapshot  entries=%d",
            len(snapshot),
        )
        return _cache_payload


# ---------------------------------------------------------------------------
# Precision-filtered expansion (experimental, env-gated)
# ---------------------------------------------------------------------------
#
# When ``LEXICON_PRECISION_CSV`` is set to a path of the diagnostic CSV
# produced by ``scripts/compute_lexicon_phrase_precision.py``, the
# expansion will FILTER out phrases marked DROP_NOISY / DROP_RARE /
# DROP_DUPE. This is the prototype for the eventual ``query_rewrite``
# JSONB column. Before committing to a schema change we use this path
# to verify the hypothesis on real queries:
#
#   * does the BM25 tsquery shrink?
#   * does latency drop on the slow queries (Q1, Q5, Q6, Q7, Q16)?
#   * do recall (n_chunks) and precision (top-doc relevance) hold?
#
# When the env var is unset, behaviour is unchanged — every phrase from
# every matched entry contributes to expansion_phrases as before.

import csv as _csv

_APPROVED_PHRASES_CACHE: dict[str, set[str]] | None = None
_APPROVED_PHRASES_CSV_PATH: str | None = None


def _load_approved_phrases_from_csv() -> dict[str, set[str]]:
    """Load the precision CSV once and cache. Returns {tag_code: {approved_phrases}}.

    Each CSV row has columns: tag_code, phrase, src, df, df_tagged,
    precision, verdict, is_canonical. We retain phrases whose verdict
    is KEEP or KEEP_CANONICAL.
    """
    global _APPROVED_PHRASES_CACHE, _APPROVED_PHRASES_CSV_PATH

    csv_path = os.environ.get("LEXICON_PRECISION_CSV", "").strip()
    if not csv_path:
        return {}
    if (
        _APPROVED_PHRASES_CACHE is not None
        and _APPROVED_PHRASES_CSV_PATH == csv_path
    ):
        return _APPROVED_PHRASES_CACHE

    approved: dict[str, set[str]] = {}
    try:
        with open(csv_path, newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                if row.get("verdict") in ("KEEP", "KEEP_CANONICAL"):
                    code = (row.get("tag_code") or "").strip()
                    phrase = (row.get("phrase") or "").strip().lower()
                    if code and phrase:
                        approved.setdefault(code, set()).add(phrase)
        _APPROVED_PHRASES_CACHE = approved
        _APPROVED_PHRASES_CSV_PATH = csv_path
        logger.info(
            "lexicon: loaded precision-approved phrases for %d tag codes "
            "from %s",
            len(approved), csv_path,
        )
    except Exception as exc:
        logger.warning(
            "lexicon: failed to load precision CSV %r — running with "
            "unfiltered expansion: %s",
            csv_path, exc,
        )
        _APPROVED_PHRASES_CACHE = {}
        _APPROVED_PHRASES_CSV_PATH = csv_path

    return _APPROVED_PHRASES_CACHE


def invalidate_cache() -> None:
    """Force the next call to reload the lexicon from DB.

    Useful after operator edits to policy_lexicon_entries.
    """
    global _cache_payload, _cache_loaded_at
    _cache_payload = None
    _cache_loaded_at = 0.0


async def list_active_d_tag_codes(db: AsyncSession) -> list[str]:
    """Return sorted active d-tag codes (e.g., ``utilization_management.prior_authorization``).

    Used by the Fail Fast gate to populate ``options`` when refusing
    with response_mode=reframe — gives the user the actual scope list
    rather than a hand-maintained string.
    """
    snap = await _load_lexicon_snapshot(db)
    return sorted({e["code"] for e in snap if e["kind"] == "d"})


# ---------------------------------------------------------------------------
# Match logic
# ---------------------------------------------------------------------------

def _match_entry(query_lower: str, phrases: list[str]) -> str | None:
    """Return the first phrase in *phrases* that appears in *query_lower*
    with word-aligned boundaries (already lowercase).  None if no match.

    Word-boundary matching is essential. Plain substring matching causes
    false positives like "oral health" matching inside
    "behavi**oral health** providers" — observed 2026-04-30 firing the
    dental classification on every behavioral-health query.

    We normalize both the query and each phrase by replacing any run of
    non-alphanumerics with a single space, then check space-padded
    containment. This handles punctuation around phrases too:

      "Notice of Meeting (PCTAP)"  → "notice of meeting pctap"
      phrase "pctap"               → matches " pctap " ✓

      "behavioral health providers" → "behavioral health providers"
      phrase "oral health"          → " oral health " ∉ "...behavioral health..." ✓
    """
    # Pre-normalize the query once
    q_norm = _re.sub(r"[^a-z0-9]+", " ", query_lower).strip()
    padded_q = f" {q_norm} "
    for p in phrases:
        if not p:
            continue
        p_norm = _re.sub(r"[^a-z0-9]+", " ", p.lower()).strip()
        if not p_norm:
            continue
        # Bigram-or-longer requirement for non-acronym phrases.
        # Single-word phrases like "provider", "rules", "general"
        # are too generic and over-classify queries (observed
        # 2026-04-30: "providers" → 5 different matches, exploding
        # the lexicon expansion to 22 phrases of which most were
        # unrelated to the query). Allow short ALL-CAPS-style codes
        # like "HCPCS", "NPI", "DME" through (4 chars or less,
        # source phrase was uppercase / acronym-like) since those
        # ARE meaningful single tokens.
        word_count = len(p_norm.split())
        if word_count == 1:
            # Allow short acronyms / codes (≤4 chars); reject longer
            # single words that are too generic.
            if len(p_norm) > 4:
                continue
        if f" {p_norm} " in padded_q:
            return p
    return None


async def expand_query_via_lexicon(
    db: AsyncSession,
    raw_query: str,
) -> LexiconExpansion:
    """Match *raw_query* against the lexicon and return its expansion bag.

    Behaviour:
      - Lowercases the query and substring-matches each active entry's
        strong_phrases, aliases, and leaf-name.
      - First phrase-hit per entry wins (no double-counting).
      - Caps at ``_MAX_ENTRIES_PER_QUERY`` entries to avoid query bloat.
      - On any DB error or empty lexicon: returns an empty expansion.
        Caller is responsible for falling back to raw tokens.
    """
    expansion = LexiconExpansion()
    if not raw_query or not raw_query.strip():
        return expansion

    query_lower = raw_query.lower()
    snapshot = await _load_lexicon_snapshot(db)
    if not snapshot:
        return expansion

    # Optional precision-filtered expansion. Empty dict = behaviour unchanged.
    approved_per_tag = _load_approved_phrases_from_csv()
    use_precision_filter = bool(approved_per_tag)

    phrase_seen: set[str] = set()
    n_phrases_before_filter = 0
    n_phrases_after_filter = 0

    for entry in snapshot:
        if len(expansion.matched_codes) >= _MAX_ENTRIES_PER_QUERY:
            break
        hit = _match_entry(query_lower, entry["phrases"])
        if not hit:
            continue

        full_code = entry["full_code"]
        expansion.matched_codes.append(full_code)
        kind = entry["kind"]
        if kind == "d":
            expansion.domain_tags.append(full_code)
        elif kind == "j":
            expansion.jurisdiction_tags.append(full_code)
        elif kind == "p":
            expansion.process_tags.append(full_code)

        # Precision filter: keep only phrases approved for this tag (or
        # all phrases when filter disabled / tag missing from CSV).
        if use_precision_filter and full_code in approved_per_tag:
            allowed = approved_per_tag[full_code]
            entry_phrases = [p for p in entry["phrases"] if p and p.lower() in allowed]
        else:
            entry_phrases = entry["phrases"]

        n_phrases_before_filter += len(entry["phrases"])
        n_phrases_after_filter += len(entry_phrases)

        for p in entry_phrases:
            if p and p not in phrase_seen:
                phrase_seen.add(p)
                expansion.expansion_phrases.append(p)

        expansion.log.append(f"matched '{hit}' → {full_code}")

    if use_precision_filter and n_phrases_before_filter > 0:
        expansion.log.append(
            f"precision_filter: {n_phrases_before_filter} → "
            f"{n_phrases_after_filter} phrases"
        )

    return expansion
