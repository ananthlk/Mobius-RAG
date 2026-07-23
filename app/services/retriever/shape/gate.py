"""Gate — step 1a of the shape module: "can / should we answer?"

Grounded contour classification:

  1. expand the query through the lexicon → J/P/D tag codes
  2. probe document_tags for the union and intersection of documents
     carrying those codes (doc-grain — cheap, never touches the chunk index)
  3. classify the contour from slot completeness + the counts

Contour rules (v1, constants below are Eval-tunable). Only D and J are
required slots — P is enrichment, EXCEPT when D matched only the bare
umbrella/``.general`` bucket for a domain that has specific siblings in the
lexicon (e.g. "eligibility" has ~90 leaves: age bands, income tests,
verification, ...): there P becomes the disambiguator instead.

  no codes matched, malformed query   → UNCLEAR        (can't parse it at all)
  no codes matched, well-formed query → OUT_OF_SCOPE   (understood it, wrong domain)
  union == 0                          → CORPUS_GAP     (tags exist, no docs carry them)
  D+J matched, D general-only, no P   → UNDERSPECIFIED (which facet of this broad domain?)
  D+J matched (P resolves if needed)  → EXACT          (docs cover the full combination)
  D+J matched, intersect==0           → VICINITY       (right area, combo not covered)
  D or J missing, corpus small        → EXACT          (corpus itself narrows the answer)
  D or J missing, corpus broad        → UNDERSPECIFIED (fan-out / clarify on missing axis)
"""

from __future__ import annotations

import logging
import re
import time

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.corpus_search_lexicon import expand_query_via_lexicon, list_active_d_tag_codes

from .contracts import Contour, CorpusProbe, GateResult

logger = logging.getLogger(__name__)

# A missing slot only makes the query "underspecified" when the corpus is
# too broad to answer without it. Below this many union docs, the corpus
# itself acts as the specifier and we can proceed exact.
_BROAD_MIN_DOCS = 25

# Structural process-intent phrasing: "how do I check eligibility" carries
# the same disambiguating signal as "how do I verify eligibility" even
# though the lexicon only aliases "verify". This catches the *shape* of an
# action request generically, instead of chasing an ever-growing list of
# specific verb synonyms as lexicon p-tag aliases. Deliberately narrow —
# "what are the eligibility criteria" does NOT match (that's a genuinely
# different, still-ambiguous ask: which facet's rules?), only match on
# explicit how-to framing.
_PROCESS_INTENT_RE = re.compile(
    r"\b(how do i|how can i|how does one|how to|"
    r"what('?s| is) the process (for|to)|"
    r"what are the steps (for|to)|steps to|procedure for)\b"
)


def _detect_process_intent(normalized_query: str) -> bool:
    return bool(_PROCESS_INTENT_RE.search(normalized_query))


def _is_malformed(normalized_query: str) -> bool:
    """Cheap, deterministic, no DB/LLM: true gibberish/fragment vs a
    well-formed-but-off-domain question. "What's the weather tomorrow?" has
    real words and reads as a question — that's OUT_OF_SCOPE, not UNCLEAR.
    A single garbled token or empty string has neither — that's UNCLEAR.
    Zero lexicon tag matches alone can't tell these apart; this can, for the
    cheap cases.

    Known limitation, accepted rather than hidden: this only catches
    structural malformation (too short, mostly non-alphabetic). Multi-word
    strings that are alphabetic but meaningless ("asdkfj qwoeiru xyz")
    still pass the word-count check and fall through to OUT_OF_SCOPE, not
    UNCLEAR — telling "real English, wrong domain" apart from "fake English"
    needs a dictionary/LLM check, which shape deliberately doesn't do.
    Real-world traffic skews heavily toward legitimate off-topic questions
    over multi-word gibberish, so this is an acceptable default, not a
    silently-swept defect.
    """
    if not normalized_query:
        return True
    words = [w for w in normalized_query.split() if w.isalpha()]
    if len(words) < 2:
        return True
    # Mostly-non-alphabetic content (e.g. stray codes/numbers with no real words).
    if len(words) < len(normalized_query.split()) / 2:
        return True
    return False


def _strip_kind(code: str) -> str:
    """``d:benefits.dme`` → ``benefits.dme`` (document_tags keys carry no kind prefix)."""
    return code.split(":", 1)[1] if ":" in code else code


async def _probe_corpus(
    db: AsyncSession,
    d_codes: list[str],
    j_codes: list[str],
    p_codes: list[str],
) -> CorpusProbe:
    """One SELECT over document_tags: per-kind, union, and intersection doc counts.

    document_tags HAS a GIN index per kind (ix_document_tags_{d,j,p}_tags_gin)
    — unlike the chunk-tag arm. Two things are required to hit it at all:

      1. No ``coalesce()`` wrapper around the column — it turns ``col ? key``
         into a function-of-column expression the planner won't index.
      2. A restricting ``WHERE`` clause (the union condition) so Postgres
         narrows via BitmapOr *before* computing the FILTER aggregates —
         without it, COUNT(*) FILTER (...) with no WHERE still evaluates
         every predicate on every row regardless of indexes (verified: 868ms
         unconditional seq-scan on 9.5k rows).

    DB-verified perf, CORRECTED 2026-07-22 (an earlier same-day pass reported
    ~630-715ms server-side for the typical case and blamed "inherent per-row
    JSONB evaluation cost" — that was wrong, kept as a record of the mistake,
    not as fact):
      - Isolated re-measurement (EXPLAIN ANALYZE, BUFFERS, run right after a
        fresh cloud-sql-proxy restart, with raw SELECT-1 ping round-trips
        measured alongside for comparison) found TRUE server-side execution
        for the typical 2-3 code case is **~52ms** — genuinely fast, matching
        what the GIN index migration's own benchmark implied. The earlier
        630-1500ms app-layer numbers were dominated by the LOCAL DEV
        cloud-sql-proxy tunnel (observed ping round-trips: 80-180ms EACH; the
        proxy is separately known to degrade badly after long uptime — see
        the project's dev-db-proxy-degradation note). This overhead will not
        reproduce in production (Cloud Run co-located with Cloud SQL,
        single-digit-ms network) — the query itself has large latency
        headroom against the <500ms shape-phase target.
      - Seq-scan fallback near _MAX_ENTRIES_PER_QUERY (12) with several broad
        ``*.general`` lexicon tags (combined selectivity crossing ~50-60%) is
        a real, correct planner decision, unaffected by the above correction
        — only the "how fast is the common case" number was wrong.

    NULL d/j/p_tags are excluded by ``col ? key`` naturally (NULL, not TRUE).
    """
    t0 = time.monotonic()
    probe = CorpusProbe()

    kinds = [(k, codes) for k, codes in (("d", d_codes), ("j", j_codes), ("p", p_codes)) if codes]
    if not kinds:
        return probe

    params: dict[str, str] = {}
    kind_exprs: dict[str, str] = {}
    for kind, codes in kinds:
        key_params = []
        for i, code in enumerate(codes):
            pname = f"{kind}_{i}"
            params[pname] = _strip_kind(code)
            key_params.append(pname)
        kind_exprs[kind] = "(" + " OR ".join(f"{kind}_tags ? :{p}" for p in key_params) + ")"

    union_expr = " OR ".join(kind_exprs.values())
    intersection_expr = " AND ".join(kind_exprs.values())
    count_cols = ", ".join(
        f"COUNT(*) FILTER (WHERE {expr}) AS {kind}_docs" for kind, expr in kind_exprs.items()
    )

    # WHERE = union_expr restricts the scan to a GIN bitmap lookup; the
    # FILTER aggregates (including union/intersection) then run over just
    # that narrowed row set instead of the full table.
    sql = text(
        f"SELECT {count_cols}, "
        f"COUNT(*) AS union_docs, "
        f"COUNT(*) FILTER (WHERE {intersection_expr}) AS intersection_docs "
        f"FROM document_tags WHERE {union_expr}"
    )

    row = (await db.execute(sql, params)).mappings().one()
    probe.d_docs = row.get("d_docs", 0) or 0
    probe.j_docs = row.get("j_docs", 0) or 0
    probe.p_docs = row.get("p_docs", 0) or 0
    probe.union_docs = row["union_docs"] or 0
    probe.intersection_docs = row["intersection_docs"] or 0
    probe.probe_ms = int((time.monotonic() - t0) * 1000)
    return probe


def _is_general_only_match(codes: list[str], all_codes: set[str]) -> tuple[bool, str, int]:
    """True if *codes* only hit the bare root / ``.general`` catch-all for one
    top-level domain, while the lexicon has more specific siblings under that
    same root that did NOT match.

    This is the signal for "eligibility for Medicaid" vs "how do I verify
    eligibility for Medicaid": both match only ``eligibility``/
    ``eligibility.general`` unless an action word narrows it (P), but the
    lexicon has ~90 specific ``eligibility.*`` leaves (age bands, income
    tests, immigration status, verification, work requirements, ...). Doc
    counts don't distinguish "many docs echo one fact" (timely filing
    deadline) from "this topic genuinely branches into several different
    answers" (eligibility) — sibling-code presence does.

    Returns (is_general_only, root, sibling_count) for tracing.
    """
    stripped = {_strip_kind(c) for c in codes}
    if not stripped:
        return False, "", 0
    roots = {c.split(".")[0] for c in stripped}
    if len(roots) != 1:
        return False, "", 0  # spans multiple distinct top-level domains — richly specified
    root = next(iter(roots))
    if not all(c == root or c == f"{root}.general" for c in stripped):
        return False, root, 0  # at least one specific leaf matched
    siblings = {c for c in all_codes if c.startswith(f"{root}.") and c != f"{root}.general"}
    return bool(siblings), root, len(siblings)


def _classify(result: GateResult, all_d_codes: set[str]) -> tuple[Contour, str]:
    """Contour from slot completeness + probe counts. Pure logic, unit-testable.

    Only D (domain) and J (jurisdiction) are required slots. P (process) is
    enrichment: if matched it narrows the probe further (folded into
    intersection_docs automatically, since _probe_corpus only ANDs kinds
    that had codes), but a missing P never triggers underspecified on its
    own — the majority of real questions ("what is the deadline", "does X
    require Y") are fact lookups with no process verb to match, not
    genuinely incomplete queries. Verified against the cmhc 22-query bank
    2026-07-22: requiring P pushed 17/22 to underspecified; dropping it to
    enrichment-only correctly reads 20/22 as exact, leaving only the 2
    queries with a real missing D-tag (lexicon coverage gaps, not a shape
    classification issue: "credentialed", "enroll a pediatric patient").

    Exception: if D matched only the bare umbrella/``.general`` bucket for
    its domain AND the lexicon has specific siblings that didn't fire (see
    ``_is_general_only_match``), the domain itself is unresolved — a raw
    "eligibility for Medicaid" doesn't say which of ~90 eligibility facets
    is meant. There P stops being pure enrichment and becomes the
    disambiguator — satisfied by EITHER a matched lexicon p-code OR the
    structural ``process_intent`` signal ("how do I check eligibility" asks
    for the same facet as "how do I verify eligibility" even though only
    "verify" is a lexicon alias). Absent both → genuinely underspecified,
    not a doc-count question at all.
    """
    probe = result.probe

    if result.kinds_matched == 0:
        if _is_malformed(result.normalized):
            return Contour.UNCLEAR, "no real words / too short to parse"
        return Contour.OUT_OF_SCOPE, (
            "well-formed question, but no J/P/D tags matched — not this corpus's domain"
        )

    if probe.union_docs == 0:
        return Contour.CORPUS_GAP, "tags matched but zero documents carry them"

    missing_required = [k for k in ("d", "j") if k in result.missing_kinds]

    if not missing_required:
        d_general_only, root, n_siblings = _is_general_only_match(result.d_codes, all_d_codes)
        if d_general_only and not result.p_codes and not result.process_intent:
            result.underspecified_kind = "explore_siblings"
            result.fanout_codes = sorted(
                c for c in all_d_codes if c.startswith(f"{root}.") and c != f"{root}.general"
            )
            return Contour.UNDERSPECIFIED, (
                f"D matched only the general '{root}' bucket ({n_siblings} more specific "
                f"facets exist in the lexicon) — known, enumerable siblings to explore, "
                f"not a dead end; no P and no process-intent phrasing narrowed it upfront"
            )
        if d_general_only and not result.p_codes and result.process_intent:
            return Contour.EXACT, (
                f"D matched only the general '{root}' bucket, but process-intent phrasing "
                f"('how do I...') resolves it structurally without a lexicon p-tag "
                f"({probe.intersection_docs} docs cover the combination)"
            )
        p_note = "p=yes" if result.p_codes else "p=no"
        if probe.intersection_docs > 0:
            return Contour.EXACT, (
                f"D+J matched ({p_note}), {probe.intersection_docs} docs cover the combination"
            )
        return Contour.VICINITY, (
            f"D+J matched ({p_note}) but no doc covers the combination (union={probe.union_docs})"
        )

    # D or J is missing. If the matched tags already narrow the corpus to a
    # small set, the corpus specifies for us; otherwise we need the missing axis.
    anchor = probe.intersection_docs if result.kinds_matched > 1 else probe.union_docs
    if anchor and anchor <= _BROAD_MIN_DOCS:
        return Contour.EXACT, (
            f"missing {missing_required} but corpus narrows to {anchor} docs"
        )
    # No sibling set to enumerate here — D or J matched nothing, so there is
    # no root to fan out under. Not explorable the way "explore_siblings" is;
    # downstream needs a different strategy (relax scope, escalate, or flag
    # as a lexicon coverage gap), not a proactive fan-out.
    result.underspecified_kind = "missing_domain" if "d" in missing_required else "missing_jurisdiction"
    return Contour.UNDERSPECIFIED, (
        f"missing {missing_required}, corpus too broad "
        f"(union={probe.union_docs}, intersection={probe.intersection_docs}) — "
        f"no sibling set to fan out under, not explorable"
    )


async def run_gate(db: AsyncSession, query: str) -> GateResult:
    """Classify one query. Lexicon lookup + one doc-grain probe; no chunk index access."""
    t0 = time.monotonic()
    result = GateResult(query=query, normalized=" ".join((query or "").lower().split()))

    expansion = await expand_query_via_lexicon(db, query)
    result.d_codes = list(expansion.domain_tags)
    result.j_codes = list(expansion.jurisdiction_tags)
    result.p_codes = list(expansion.process_tags)
    result.expansion_phrases = list(expansion.expansion_phrases)
    result.missing_kinds = [
        k for k, codes in (("d", result.d_codes), ("j", result.j_codes), ("p", result.p_codes))
        if not codes
    ]
    result.process_intent = _detect_process_intent(result.normalized)

    result.probe = await _probe_corpus(db, result.d_codes, result.j_codes, result.p_codes)

    # Cached 5-min TTL inside the lexicon module itself (same cache expand_query_via_lexicon
    # just populated) — this is a free in-memory lookup, not a second DB round-trip in practice.
    all_d_codes = set(await list_active_d_tag_codes(db)) if result.d_codes else set()
    result.contour, result.reason = _classify(result, all_d_codes)
    result.gate_ms = int((time.monotonic() - t0) * 1000)
    return result
