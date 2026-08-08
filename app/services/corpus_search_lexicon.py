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

# Single-word phrases longer than 4 chars are rejected if and only if they are
# in this stoplist.  The old heuristic (reject ALL len>4 single words) wrongly
# dropped domain terms like "aetna", "florida", "telehealth", "pharmacy",
# "credentialing" — causing strategy e / no_domain_match even when the corpus
# had the answer.  These are genuinely generic words that add no retrieval
# signal when they appear alone; multi-word phrases containing them still match.
_SINGLE_WORD_STOPLIST: frozenset[str] = frozenset({
    "provider", "providers", "policy", "policies",
    "rule", "rules", "requirement", "requirements",
    "information", "info", "details", "general", "specific",
    "covered", "coverage", "applies", "apply",
    "process", "guideline", "guidelines",
    "service", "services", "plan", "plans",
    "member", "members", "patient", "patients",
    "client", "clients", "notice", "section",
    "program", "programs", "benefit", "benefits",
    "criteria", "procedure", "procedures",
    "standard", "standards", "update", "updates",
})


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

# Stemming fallback (2026-07-30, Ananth's ask): _match_entry's exact
# word-boundary substring match misses genuine morphological variants
# ("credentialed" never matches the stored phrase "credentialing";
# "enroll" never matches "enrollment") -- a real, confirmed cause of Gate
# returning zero d_codes and falling to UNDERSPECIFIED/CLARIFY (cmhc016,
# cmhc021). Fixed with Postgres's own English stemmer (to_tsvector) --
# already trusted in this codebase for BM25 -- rather than a hand-rolled
# regex/suffix-stripper, which was tested first and rejected: it fixed the
# target case but was WRONG on common words (care/cared -> car/care
# mismatch, English's silent-e rule) and missed enroll/enrollment
# entirely. Scoped conservatively to SINGLE-WORD phrases only, same
# principle _SINGLE_WORD_STOPLIST already uses -- multi-word phrase
# matching is unchanged.
#
# Verified corpus-wide before shipping (per this session's established
# discipline: heuristics that look safe on a couple of cases have
# repeatedly hidden false-positive classes at scale today): of 1,714
# unique single-word phrases across the whole lexicon, 18 stems collide
# across DIFFERENT tag codes. ~8 are benign general/specific-leaf overlap
# within the same domain (benefit/benefits, claim/claims). ~9 are genuine,
# concerning collisions -- almost all domain-specific ACRONYMS that
# Postgres's generic English stemmer doesn't understand as abbreviations:
#   aid/aids       -- home-health aide (a person) vs AIDS the disease
#   sar/sars       -- an annual-return-report abbreviation vs SARS the disease
#   chap/chaps     -- CHAP (regulatory body) vs CAHPS (quality survey)
#   recon/recons   -- disputes/appeal reconsideration vs plastic reconstructive surgery
#   em/ems         -- Evaluation-and-Management billing codes vs Emergency care
#   pec/pecs       -- PECS (communication system) vs PPEC (pediatric extended care)
#   aspir(e)       -- "Aspire Health" (a company name) vs medical aspiration procedure
#   type/typing    -- HLA genetic typing vs provider network type (high blast radius, common word)
#   resid          -- medical residency training vs residents (people living somewhere)
# _compute_stem_safety_exclusions() recomputes this exact collision set
# from the live lexicon (not a hardcoded list, so it stays correct as the
# lexicon changes) and any colliding stem is permanently blocked from
# ever triggering a stem-based match -- exact substring matching on those
# phrases is unaffected, only the NEW fallback path is restricted.
_stem_cache_lock = asyncio.Lock()
_phrase_stem_cache: dict[str, str] | None = None
_stem_safety_exclusions_cache: frozenset[str] | None = None
_stem_cache_loaded_at: float = 0.0


async def _compute_phrase_stems(db: AsyncSession, phrases: list[str]) -> dict[str, str]:
    """Bulk-stem a list of single-word phrases in ONE query (not one per
    phrase) via Postgres's English tsvector stemmer. Returns
    {phrase: stem}; a phrase that stems to nothing (pure stopword/empty)
    is omitted."""
    if not phrases:
        return {}
    rows = (await db.execute(
        text(
            "SELECT phrase, tsvector_to_array(to_tsvector('english', phrase)) AS stems "
            "FROM unnest(CAST(:phrases AS text[])) AS phrase"
        ),
        {"phrases": phrases},
    )).fetchall()
    out: dict[str, str] = {}
    for r in rows:
        if r.stems:
            out[r.phrase] = r.stems[0]
    return out


async def _load_stem_index(
    db: AsyncSession, snapshot: list[dict[str, Any]]
) -> tuple[dict[str, str], frozenset[str]]:
    """Returns (phrase_to_stem, unsafe_stems) for the stemming fallback,
    cached alongside the lexicon snapshot's own 5-minute TTL (stems for a
    fixed phrase set never change between refreshes). unsafe_stems is the
    EXCLUSION set -- any stem shared by phrases belonging to DIFFERENT
    lexicon codes (see module-level comment above for the concrete
    collision list found verifying this live). Callers must SKIP a stem
    match when the stem is IN this set, not the other way round."""
    global _phrase_stem_cache, _stem_safety_exclusions_cache, _stem_cache_loaded_at

    now = time.monotonic()
    if (
        _phrase_stem_cache is not None
        and (now - _stem_cache_loaded_at) < _CACHE_TTL_SECONDS
    ):
        return _phrase_stem_cache, _stem_safety_exclusions_cache or frozenset()

    async with _stem_cache_lock:
        now = time.monotonic()
        if (
            _phrase_stem_cache is not None
            and (now - _stem_cache_loaded_at) < _CACHE_TTL_SECONDS
        ):
            return _phrase_stem_cache, _stem_safety_exclusions_cache or frozenset()

        phrase_to_codes: dict[str, set[str]] = {}
        for entry in snapshot:
            for p in entry["phrases"]:
                p_norm = (p or "").strip().lower()
                if p_norm and len(p_norm.split()) == 1 and p_norm.isalpha():
                    phrase_to_codes.setdefault(p_norm, set()).add(entry["full_code"])

        try:
            phrase_to_stem = await _compute_phrase_stems(db, list(phrase_to_codes))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "corpus_search_lexicon: stem computation failed, stemming "
                "fallback disabled this cycle: %s", exc,
            )
            _phrase_stem_cache = {}
            _stem_safety_exclusions_cache = frozenset()
            _stem_cache_loaded_at = now
            return {}, frozenset()

        # Real bug caught live (2026-07-30, before this ever shipped): the
        # FIRST version of this check flagged a stem as unsafe whenever
        # multiple CODES shared it, even when it was the SAME literal
        # phrase attached to more than one code (e.g. "credentialing" on
        # both d:credentialing and d:credentialing.general -- an
        # intentional general/specific-leaf pairing, already handled by
        # EXACT substring matching regardless of stemming, not a stemming
        # risk at all). That wrongly blocked the credentialed/credentialing
        # target case this whole fallback was built for. The real risk is
        # DISTINCT phrases/words sharing a stem across different codes
        # (aide vs aids, recon vs reconstructive) -- so collect the set of
        # distinct WORDS per stem first, and only flag it unsafe if 2+
        # distinct words map to codes that aren't all the same.
        stem_to_words: dict[str, set[str]] = {}
        for phrase, stem in phrase_to_stem.items():
            stem_to_words.setdefault(stem, set()).add(phrase)

        unsafe_stems_set: set[str] = set()
        for stem, words in stem_to_words.items():
            if len(words) < 2:
                continue
            codes: set[str] = set()
            for w in words:
                codes |= phrase_to_codes[w]
            if len(codes) > 1:
                unsafe_stems_set.add(stem)
        unsafe_stems = frozenset(unsafe_stems_set)

        _phrase_stem_cache = phrase_to_stem
        _stem_safety_exclusions_cache = unsafe_stems
        _stem_cache_loaded_at = now
        logger.info(
            "corpus_search_lexicon: stem index refreshed  phrases=%d  "
            "excluded_collision_stems=%d",
            len(phrase_to_stem), len(unsafe_stems),
        )
        return phrase_to_stem, unsafe_stems


def _normalize_phrase(p: Any) -> str:
    if not isinstance(p, str):
        return ""
    return p.strip().lower()


def _extract_phrases(spec: Any) -> list[str]:
    """Pull phrases out of a lexicon entry's spec JSON.

    Accepts ``strong_phrases``, ``aliases``, ``phrases``, and
    ``query_expansion_phrases`` keys (all optional).  Returns a
    deduplicated lowercase list.

    ``query_expansion_phrases`` (2026-08-08, docs/rag-agents/query-
    expansion-phrases-spec.md, Lexicon-owned field/content, this read side
    is mine): QUERY-SIDE ONLY -- this function is exclusively used to build
    the phrase bag THIS module (corpus_search_lexicon.py / Gate) matches
    queries against. Doc-tagging (policy_path_b.py's own separate
    get_phrase_to_tag_map) reads only strong_phrases + weak_keywords and
    deliberately does NOT read this key -- resolves the two-faces tension
    where a phrase good for expanding queries (generic, e.g. "how to
    apply") would be toxic for tagging documents (over-broad, would pollute
    unrelated docs on the next retag). Safe to add here without any
    doc-tagging-side change.
    """
    if not isinstance(spec, dict):
        return []
    bag: list[str] = []
    for key in ("strong_phrases", "aliases", "phrases", "query_expansion_phrases"):
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
                    "WHERE active = true ORDER BY kind, code"
                )
            )
            rows = result.mappings().all()
        except Exception as exc:
            # Real bug found live, 2026-08-04 (Ananth's catch): this used to
            # cache [] with a FRESH _cache_loaded_at on failure -- since this
            # cache is PROCESS-GLOBAL (shared across every concurrent
            # request/query on the instance, not per-call), one transient
            # failure (e.g. this query hitting a shared session whose
            # transaction was aborted by an unrelated earlier statement)
            # poisoned Gate's lexicon matching to "zero tags, ever" for every
            # query on the WHOLE instance for the next _CACHE_TTL_SECONDS (5
            # min) -- confirmed live: 16/22 bank-run queries silently got
            # zero d/j/p codes, zero pool candidates, zero chunks, with no
            # error surfaced (Gate's own query succeeded fine, it just had
            # nothing to match against). Fix: on failure, keep serving the
            # last GOOD snapshot if one exists (stale-but-real beats empty),
            # and do NOT stamp _cache_loaded_at -- next call retries
            # immediately instead of being locked into the failure for the
            # full TTL. Only fall back to [] when there has never been a
            # successful load at all (fresh-process cold start).
            logger.warning(
                "corpus_search_lexicon: failed to load policy_lexicon_entries: %s "
                "(%s)",
                exc,
                "keeping last good snapshot" if _cache_payload is not None else "no prior snapshot, returning empty (not cached)",
            )
            return _cache_payload if _cache_payload is not None else []

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

def _match_entry(
    query_lower: str,
    phrases: list[str],
    query_stems: frozenset[str] | None = None,
    phrase_stem_lookup: dict[str, str] | None = None,
    unsafe_stems: frozenset[str] | None = None,
) -> str | None:
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
            # Short acronyms/codes (≤4 chars: NPI, DME, HCPCS) always pass.
            # Longer single words pass UNLESS they are provably generic —
            # the old len>4 heuristic wrongly rejected "aetna", "florida",
            # "telehealth", "pharmacy" etc. causing strategy e / no_domain_match
            # even when the corpus had the answer. Use an explicit stoplist
            # instead so domain terms (payors, states, services) get through.
            if len(p_norm) > 4 and p_norm in _SINGLE_WORD_STOPLIST:
                continue
        if f" {p_norm} " in padded_q:
            return p

    # Stemming fallback (2026-07-30) -- only reached when no phrase matched
    # by exact substring above. Scoped to single-word phrases whose stem
    # isn't in the collision-exclusion set (see module docstring on
    # _load_stem_index for the concrete risky-stem list this blocks).
    if query_stems and phrase_stem_lookup:
        for p in phrases:
            if not p:
                continue
            p_norm = _re.sub(r"[^a-z0-9]+", " ", p.lower()).strip()
            if not p_norm or len(p_norm.split()) != 1:
                continue
            stem = phrase_stem_lookup.get(p_norm)
            if not stem or (unsafe_stems is not None and stem in unsafe_stems):
                continue
            if stem in query_stems:
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

    # Stemming fallback inputs (2026-07-30) -- see _load_stem_index's
    # docstring for the collision-safety design. Best-effort: any failure
    # here (e.g. a transient DB blip) degrades to exact-match-only
    # behavior, never blocks the whole expansion.
    query_stems: frozenset[str] = frozenset()
    phrase_stem_lookup: dict[str, str] = {}
    unsafe_stems: frozenset[str] = frozenset()
    try:
        phrase_stem_lookup, unsafe_stems = await _load_stem_index(db, snapshot)
        if phrase_stem_lookup:
            row = (await db.execute(
                text("SELECT tsvector_to_array(to_tsvector('english', :q)) AS stems"),
                {"q": query_lower},
            )).fetchone()
            query_stems = frozenset(row.stems or []) if row else frozenset()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "corpus_search_lexicon: query stem computation failed, "
            "stemming fallback skipped this query: %s", exc,
        )

    # Optional precision-filtered expansion. Empty dict = behaviour unchanged.
    approved_per_tag = _load_approved_phrases_from_csv()
    use_precision_filter = bool(approved_per_tag)

    phrase_seen: set[str] = set()
    n_phrases_before_filter = 0
    n_phrases_after_filter = 0

    # Real bug found live, 2026-08-08 (Ananth's catch via ReAct's reported
    # false-positive CLARIFY-ask): _MAX_ENTRIES_PER_QUERY caps the whole
    # loop at 12 total matches, but snapshot order was whatever
    # _load_lexicon_snapshot happened to return -- NOT prioritized by kind.
    # A query with enough incidental overlap with generic ".general" domain
    # phrases (e.g. "general eligibility REQUIREMENTS for Florida Medicaid"
    # matching benefits.general/billing_codes.general/claims.general/etc.)
    # could exhaust the entire 12-entry budget on low-value D matches
    # BEFORE the loop ever reached the j/p entries later in snapshot order
    # -- confirmed live: j_codes=[] and p_codes=[] came back completely
    # EMPTY, not "didn't match", but "never evaluated". Gate then correctly
    # (given that starved input) couldn't find jurisdiction and asked to
    # clarify on a query that plainly stated "Florida".
    #
    # Fix: evaluate j/p-kind entries before d-kind entries, stable within
    # each group (preserves whatever relevance ordering the snapshot
    # already carries). j/p are the load-bearing "who/what process" axes
    # Gate's own contour logic treats as a hard requirement (missing_kinds
    # check) -- they must never lose the match-budget race to generic
    # domain umbrella phrases just because of snapshot iteration order.
    _kind_priority = {"j": 0, "p": 0, "d": 1}
    snapshot = sorted(snapshot, key=lambda e: _kind_priority.get(e["kind"], 1))

    for entry in snapshot:
        if len(expansion.matched_codes) >= _MAX_ENTRIES_PER_QUERY:
            break
        hit = _match_entry(query_lower, entry["phrases"], query_stems, phrase_stem_lookup, unsafe_stems)
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
