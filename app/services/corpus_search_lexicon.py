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


def invalidate_cache() -> None:
    """Force the next call to reload the lexicon from DB.

    Useful after operator edits to policy_lexicon_entries.
    """
    global _cache_payload, _cache_loaded_at
    _cache_payload = None
    _cache_loaded_at = 0.0


# ---------------------------------------------------------------------------
# Match logic
# ---------------------------------------------------------------------------

def _match_entry(query_lower: str, phrases: list[str]) -> str | None:
    """Return the first phrase in *phrases* that appears as a substring of
    *query_lower* (which is already lowercase).  None if no match.

    We require word-ish boundaries for very short phrases (≤ 3 chars) so
    "pa" doesn't match "appeal".  Longer phrases use plain substring
    match, which is sufficient for multi-word strong_phrases.
    """
    for p in phrases:
        if not p:
            continue
        if len(p) <= 3:
            # Boundary-aware match to avoid spurious hits inside longer words.
            # We test against space-padded query so "dme" matches "DME prior"
            # but not "edmer".
            padded = f" {query_lower} "
            if f" {p} " in padded:
                return p
        else:
            if p in query_lower:
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

    phrase_seen: set[str] = set()

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

        for p in entry["phrases"]:
            if p and p not in phrase_seen:
                phrase_seen.add(p)
                expansion.expansion_phrases.append(p)

        expansion.log.append(f"matched '{hit}' → {full_code}")

    return expansion
