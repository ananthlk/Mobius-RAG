"""
Path B policy pipeline: build policy_paragraphs + policy_lines from chunked text, apply lexicon tags, extract candidates.
"""
from __future__ import annotations

import re
import logging
from uuid import uuid4
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# Weight for terms that appear in section headings (vs body). Terms in headings count more toward threshold and ranking.
HEADING_WEIGHT = 2

# N-grams that are entirely these words are skipped (no content word).
STOPWORDS = frozenset({
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "shall", "can", "the", "a", "an",
    "and", "or", "but", "if", "then", "else", "when", "what", "which", "who", "whom", "whose", "where", "why", "how",
    "that", "this", "these", "those", "it", "its", "as", "at", "by", "for", "with", "to", "from", "of", "in", "on",
    "into", "through", "during", "before", "after", "above", "below", "between", "under", "again", "further",
    "each", "both", "few", "more", "most", "other", "some", "such", "no", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "also", "please", "thank", "you", "we", "they", "our", "your", "their",
})

# Exact-match normalized phrases we never propose as candidates.
COMMON_PHRASES = frozenset({
    "thank you", "please contact", "as follows", "for more information", "in order to", "prior to",
    "in accordance with", "with respect to", "due to", "as well as", "such as", "as of", "in addition to",
    "in the event", "as required", "as applicable", "if applicable", "as needed", "at least", "at the time",
    "a copy of", "a member", "a provider", "a copy", "and provider", "and providers", "and services",
    "are required", "are required to", "in order", "of the", "to the", "for the", "with the",
    "access to", "accordance with", "along with", "and appeals", "and behavioral", "and behavioral health",
})

# All-caps tokens we do not propose as abbreviation candidates (generic or roman numerals).
COMMON_ABBREVIATIONS = frozenset({
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "id", "pdf", "url", "faq", "usa", "etc",
    "inc", "llc", "ltd", "tel", "fax", "www", "http", "https", "org", "com", "gov", "edu",
})

# Regex patterns for junk candidates that should never be proposed.
_JUNK_PATTERNS = re.compile(
    r"^(?:"
    r"©.*"                       # copyright lines
    r"|\d+\s*(?:a\.m\.|p\.m\.)"  # time patterns: "8 a.m.", "5 p.m."
    r"|\d+\s+days?"              # "30 days", "90 day"
    r"|(?:and|or|the|of|to|for|with|from|by|in|on|at|as|if|but|an?)\s" # starts with conjunction/preposition
    r"|.*\d{4,}"                 # contains long numbers (years, phone fragments)
    r"|\d+\s*$"                  # pure numbers
    r")",
    re.IGNORECASE,
)


def _normalize_phrase(s: str) -> str:
    """Normalize for phrase matching: lowercase, collapse whitespace."""
    if not s or not isinstance(s, str):
        return ""
    return " ".join(s.split()).strip().lower()


# Threshold for word-boundary matching: phrases this short (in chars) must
# match as whole words to avoid false positives (e.g. "er" matching "provider").
_SHORT_PHRASE_BOUNDARY_LEN = 4

# Pre-compiled word-boundary patterns for short phrases (cached for perf).
_boundary_cache: dict[str, "re.Pattern[str]"] = {}


def _phrase_in_text(phrase: str, text: str) -> bool:
    """Check if phrase appears in text, using word boundaries for short phrases.

    Short phrases (<= _SHORT_PHRASE_BOUNDARY_LEN chars) use \\b word-boundary
    regex to prevent substring false positives. Longer phrases use fast
    substring matching.
    """
    if not phrase or not text:
        return False
    if len(phrase) <= _SHORT_PHRASE_BOUNDARY_LEN:
        pat = _boundary_cache.get(phrase)
        if pat is None:
            pat = re.compile(r"\b" + re.escape(phrase) + r"\b")
            _boundary_cache[phrase] = pat
        return pat.search(text) is not None
    return phrase in text


def _is_only_stopwords(ngram: str, stopwords: frozenset | set) -> bool:
    """True if every word in the n-gram is in the stopword set."""
    if not ngram or not stopwords:
        return True
    words = ngram.strip().lower().split()
    return all(w in stopwords for w in words)


def _split_paragraph_into_lines(text: str) -> list[str]:
    """Split paragraph text into lines (by newline, then trim)."""
    if not text or not isinstance(text, str):
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines if lines else [text.strip()] if text.strip() else []



# Type alias for the refuted-words map: (kind, code) -> set of normalized refuted phrases
RefutedMap = dict[tuple[str, str], set[str]]


def get_phrase_to_tag_map(
    lexicon_snapshot,
) -> tuple[dict[str, tuple[str, str, float]], RefutedMap]:
    """Build phrase map and refuted-words map from a lexicon snapshot.

    Returns (phrase_map, refuted_map) where:
      - phrase_map: normalized_phrase -> (kind, code, score)
      - refuted_map: (kind, code) -> {normalized refuted words}

    Reads from multiple spec keys for backward/forward compatibility:
      - spec.phrases          (legacy RAG format)
      - spec.strong_phrases   (QA Lexicon Maintenance UI format)
      - spec.aliases          (alternative names, same weight as strong_phrases)
      - spec.description      (fallback when no phrases at all)
      - spec.weak_keywords.any_of  (weak matches at lower score)
      - spec.refuted_words    (words that suppress a tag match when present)

    Domain containers (no dot in code) are skipped for matching -- only leaf tags
    (with dot) carry matchable phrases.

    The tag code itself (e.g. "member_services" -> "member services") is also
    registered so candidates with the same normalized text are auto-suppressed.
    """
    out: dict[str, tuple[str, str, float]] = {}
    refuted_map: RefutedMap = {}

    for kind, root in (("p", getattr(lexicon_snapshot, "p_tags", None)), ("d", getattr(lexicon_snapshot, "d_tags", None)), ("j", getattr(lexicon_snapshot, "j_tags", None))):
        if not isinstance(root, dict):
            continue
        for code, spec in root.items():
            if not isinstance(spec, dict):
                continue
            code_str = str(code)

            # Skip domain containers (no dot) -- they carry no matching metadata
            is_domain_container = "." not in code_str
            if is_domain_container:
                # Still register the code itself to suppress it as a candidate
                code_key = _normalize_phrase(code_str.replace("_", " "))
                if code_key:
                    out[code_key] = (kind, code_str, 1.0)
                continue

            # Tag code as known (e.g. "member_services") so we don't propose it as a candidate
            code_key = _normalize_phrase(code_str.replace("_", " "))
            if code_key:
                out[code_key] = (kind, code_str, 1.0)

            # Collect strong phrases from both legacy and QA-UI keys, plus aliases
            phrases: list[str] = []
            for key in ("phrases", "strong_phrases", "aliases"):
                val = spec.get(key)
                if isinstance(val, list):
                    phrases.extend(str(p) for p in val if p and isinstance(p, str))
            if not phrases and isinstance(spec.get("description"), str):
                phrases = [spec["description"]]

            for phrase in phrases:
                norm = _normalize_phrase(phrase)
                if norm:
                    out[norm] = (kind, code_str, 1.0)

            # Weak keywords at a lower score (0.6) so they tag but don't dominate
            weak = spec.get("weak_keywords")
            if isinstance(weak, dict):
                any_of = weak.get("any_of")
                if isinstance(any_of, list):
                    for phrase in any_of:
                        if phrase and isinstance(phrase, str):
                            norm = _normalize_phrase(phrase)
                            if norm and norm not in out:
                                out[norm] = (kind, code_str, 0.6)

            # Refuted words: if any appear in a line, suppress the tag match
            refuted = spec.get("refuted_words")
            if isinstance(refuted, list):
                for rw in refuted:
                    if rw and isinstance(rw, str):
                        norm_rw = _normalize_phrase(str(rw))
                        if norm_rw:
                            refuted_map.setdefault((kind, code_str), set()).add(norm_rw)

    return out, refuted_map


def _apply_tags_to_line_text(
    line_text: str,
    phrase_map: dict[str, tuple[str, str, float]],
    refuted_map: RefutedMap | None = None,
) -> tuple[dict, dict, dict]:
    """Compute p_tags, d_tags, j_tags for a line by matching phrases. Returns (p_tags, d_tags, j_tags) as {code: score}.

    phrase_map values are (kind, code, score) tuples -- strong phrases have score=1.0, weak phrases 0.6.
    When a tag is matched by both strong and weak, the higher score wins.

    refuted_map: if a tag's refuted words appear in the line, the match is suppressed entirely.
    This prevents false positives (e.g., "approved" suppresses claims.denial).
    """
    p_tags: dict[str, float] = {}
    d_tags: dict[str, float] = {}
    j_tags: dict[str, float] = {}
    normalized_line = _normalize_phrase(line_text)
    if not normalized_line:
        return p_tags, d_tags, j_tags
    for phrase, (kind, code, score) in phrase_map.items():
        if not phrase or not _phrase_in_text(phrase, normalized_line):
            continue
        # Check refuted words: if any refuted word for this tag is in the line, skip the match
        if refuted_map:
            refuted_words = refuted_map.get((kind, code))
            if refuted_words and any(_phrase_in_text(rw, normalized_line) for rw in refuted_words):
                continue
        target = p_tags if kind == "p" else (d_tags if kind == "d" else j_tags)
        # Keep the highest score if the same code is matched multiple times
        if code not in target or score > target[code]:
            target[code] = score
    return p_tags, d_tags, j_tags


async def build_paragraph_and_lines(
    db: AsyncSession,
    document_id,
    page_number: int,
    order_index: int,
    heading_path: str | None,
    text: str,
) -> tuple:
    """
    Create one PolicyParagraph and its PolicyLines from paragraph text.
    Returns (policy_paragraph, list of policy_line ORM objects). Caller must flush to get paragraph.id for lines.
    """
    from app.models import PolicyParagraph, PolicyLine

    lines_text = _split_paragraph_into_lines(text)
    # One-line section titles (when heading_path is set) are treated as headings for weighting in extraction.
    is_heading_para = bool(
        heading_path and lines_text and len(lines_text) == 1 and len(lines_text[0].split()) <= 15
    )
    paragraph_type = "heading" if is_heading_para else "body"

    # DB column is JSONB: store string as one-element list so type matches
    heading_path_json = [heading_path] if (heading_path and heading_path.strip()) else None
    para = PolicyParagraph(
        document_id=document_id,
        page_number=page_number,
        order_index=order_index,
        paragraph_type=paragraph_type,
        heading_path=heading_path_json,
        text=text or "",
    )
    db.add(para)
    await db.flush()

    lines = []
    for line_idx, line_text in enumerate(lines_text):
        line = PolicyLine(
            document_id=document_id,
            page_number=page_number,
            paragraph_id=para.id,
            order_index=line_idx,
            text=line_text,
            line_type="sentence",
            is_atomic=True,
        )
        db.add(line)
        lines.append(line)
    await db.flush()
    return para, lines


async def apply_lexicon_to_lines(
    lines: list,
    phrase_map: dict[str, tuple[str, str, float]],
    refuted_map: RefutedMap | None = None,
) -> int:
    """Set p_tags, d_tags, j_tags on each line in place. Caller should commit. Returns count of lines that got at least one tag."""
    n_with_tags = 0
    for line in lines:
        p_tags, d_tags, j_tags = _apply_tags_to_line_text(line.text or "", phrase_map, refuted_map)
        line.p_tags = p_tags if p_tags else None
        line.d_tags = d_tags if d_tags else None
        line.j_tags = j_tags if j_tags else None
        if p_tags or d_tags or j_tags:
            n_with_tags += 1
    return n_with_tags


# Regex for all-caps words (2–15 chars, optional digits) for abbreviation detection.
_ABBREV_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]{1,14}\b")


def _is_junk_ngram(ngram: str) -> bool:
    """Return True if the n-gram matches any junk pattern and should be rejected."""
    return bool(_JUNK_PATTERNS.match(ngram))


def _has_single_char_word(ngram: str) -> bool:
    """Return True if any word in the n-gram is a single character (e.g., 'a copy')."""
    return any(len(w) == 1 for w in ngram.split())


def _is_substring_of_known_tag(ngram: str, phrase_map: dict) -> bool:
    """Return True if the ngram is a substring of an existing tag phrase or vice versa.
    This catches things like 'behavioral health' when 'health_care_services.behavioral_health' exists.
    """
    for known_phrase in phrase_map:
        if not known_phrase:
            continue
        # ngram is contained in an existing phrase, or existing phrase is contained in ngram
        if ngram in known_phrase or known_phrase in ngram:
            return True
    return False


async def extract_candidates_for_document(
    db: AsyncSession,
    document_id,
    run_id=None,
    phrase_map: dict | None = None,
    min_occurrences: int = 3,
):
    """
    Find phrases in policy_lines that are not in the lexicon and insert PolicyLexiconCandidate rows.

    Guardrails (anti-explosion):
      - Excludes approved lexicon phrases, rejected catalog, common phrases, stopword-only n-grams
      - Skips junk patterns (copyright, time, conjunctions, numbers)
      - Skips n-grams where any word is a single character ("a copy", "a member")
      - Skips n-grams that are substrings of existing tags (prevents re-proposing known concepts)
      - min_occurrences=3 (was 2) -- phrase must appear 3+ times to be considered
      - cap_ngrams=100 (was 200) -- max candidates per document
      - cap_abbrevs=30 (was 50) -- max abbreviation candidates per document
    """
    from app.models import PolicyLine, PolicyParagraph, PolicyLexiconCandidate, PolicyLexiconCandidateCatalog

    if not phrase_map:
        phrase_map = {}
    known = set(phrase_map.keys())
    known.update(COMMON_PHRASES)

    # Also suppress candidates whose code form matches an existing tag code
    known_codes = set()
    for _phrase, val in phrase_map.items():
        code = val[1] if len(val) > 1 else ""
        if code:
            known_codes.add(_normalize_phrase(code.replace("_", " ")))

    # Load decided phrases from candidate catalog (approved + rejected) so we don't re-propose them.
    # These are global decisions that persist across extraction runs.
    try:
        catalog_result = await db.execute(
            select(PolicyLexiconCandidateCatalog.normalized_key).where(
                PolicyLexiconCandidateCatalog.state.in_(["approved", "rejected"]),
                PolicyLexiconCandidateCatalog.normalized_key.isnot(None),
            )
        )
        for nk in catalog_result.scalars().all():
            # scalars().all() returns plain strings for single-column select
            if nk and isinstance(nk, str):
                known.add(nk.strip().lower())
    except Exception as e:
        logger.debug("Could not load decided catalog for candidate extraction: %s", e)

    # Also suppress candidates already decided in policy_lexicon_candidates itself
    # (catches cases where catalog doesn't have them but the candidate table does).
    try:
        decided_cands = await db.execute(
            select(PolicyLexiconCandidate.normalized).where(
                PolicyLexiconCandidate.state.in_(["approved", "rejected"]),
                PolicyLexiconCandidate.normalized.isnot(None),
            ).distinct()
        )
        for nk in decided_cands.scalars().all():
            if nk and isinstance(nk, str):
                known.add(nk.strip().lower())
    except Exception as e:
        logger.debug("Could not load decided candidates for suppression: %s", e)

    # Get (line text, paragraph_type) via join so we can weight heading lines
    result = await db.execute(
        select(PolicyLine.text, PolicyParagraph.paragraph_type)
        .select_from(PolicyLine)
        .join(PolicyParagraph, PolicyParagraph.id == PolicyLine.paragraph_id)
        .where(
            PolicyLine.document_id == document_id,
            PolicyLine.is_atomic.is_(True),
        )
    )
    rows = result.fetchall()
    line_data = [(r[0], (r[1] or "").strip().lower() == "heading") for r in rows if r[0]]

    # N-grams: weighted counts (body + HEADING_WEIGHT * heading)
    ngram_weighted: dict[str, tuple[int, int]] = {}  # ngram -> (count_body, count_heading)
    for text, is_heading in line_data:
        normalized = _normalize_phrase(text)
        words = normalized.split()
        for n in (2, 3, 4):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i : i + n])
                if ngram in known or len(ngram) < 4:
                    continue
                if _is_only_stopwords(ngram, STOPWORDS):
                    continue
                # Guardrail: skip single-character words and junk patterns
                if _has_single_char_word(ngram):
                    continue
                if _is_junk_ngram(ngram):
                    continue
                prev = ngram_weighted.get(ngram, (0, 0))
                if is_heading:
                    ngram_weighted[ngram] = (prev[0], prev[1] + 1)
                else:
                    ngram_weighted[ngram] = (prev[0] + 1, prev[1])

    run_uuid = run_id or uuid4()
    inserted = 0
    cap_ngrams = 100

    # Sort by weighted count descending, then take up to cap_ngrams
    def weighted(ngram: str) -> float:
        b, h = ngram_weighted[ngram]
        return b + HEADING_WEIGHT * h

    sorted_ngrams = sorted(ngram_weighted.keys(), key=weighted, reverse=True)
    for ngram in sorted_ngrams:
        if inserted >= cap_ngrams:
            break
        count_body, count_heading = ngram_weighted[ngram]
        w = count_body + HEADING_WEIGHT * count_heading
        if w < min_occurrences:
            continue
        if len(ngram) > 500:
            continue
        # Guardrail: skip if substring of an existing approved tag phrase
        if _is_substring_of_known_tag(ngram, phrase_map):
            continue
        # Guardrail: skip if code form matches an existing tag
        code_form = _normalize_phrase(ngram.replace(" ", "_"))
        if code_form in known_codes:
            continue
        base_conf = min(0.9, 0.5 + 0.1 * w)
        confidence = min(0.9, base_conf + 0.1) if count_heading > 0 else base_conf
        c = PolicyLexiconCandidate(
            document_id=document_id,
            run_id=run_uuid,
            candidate_type="d",
            normalized=ngram[:500],
            proposed_tag=ngram[:500].replace(" ", "_").lower(),
            confidence=confidence,
            source="path_b_ngram",
            occurrences=count_body + count_heading,
            state="proposed",
        )
        db.add(c)
        inserted += 1

    # Abbreviations: all-caps tokens from original line text, same heading weighting
    abbrev_weighted: dict[str, tuple[int, int]] = {}  # normalized -> (count_body, count_heading)
    for text, is_heading in line_data:
        for m in _ABBREV_PATTERN.findall(text):
            norm = m.lower()
            if norm in known or norm in COMMON_ABBREVIATIONS or len(norm) < 2:
                continue
            prev = abbrev_weighted.get(norm, (0, 0))
            if is_heading:
                abbrev_weighted[norm] = (prev[0], prev[1] + 1)
            else:
                abbrev_weighted[norm] = (prev[0] + 1, prev[1])

    cap_abbrevs = 30
    def abbrev_weight(norm: str) -> float:
        b, h = abbrev_weighted[norm]
        return b + HEADING_WEIGHT * h

    sorted_abbrevs = sorted(abbrev_weighted.keys(), key=abbrev_weight, reverse=True)
    for norm in sorted_abbrevs:
        if inserted >= cap_ngrams + cap_abbrevs:
            break
        count_body, count_heading = abbrev_weighted[norm]
        w = count_body + HEADING_WEIGHT * count_heading
        if w < min_occurrences:
            continue
        base_conf = min(0.9, 0.5 + 0.1 * w)
        confidence = min(0.9, base_conf + 0.1) if count_heading > 0 else base_conf
        c = PolicyLexiconCandidate(
            document_id=document_id,
            run_id=run_uuid,
            candidate_type="alias",
            normalized=norm[:500],
            proposed_tag=norm[:500],
            confidence=confidence,
            source="path_b_abbreviation",
            occurrences=count_body + count_heading,
            state="proposed",
        )
        db.add(c)
        inserted += 1

    if inserted:
        logger.info(f"[Path B] Extracted {inserted} lexicon candidates for document {document_id}")
    return inserted


# ---------------------------------------------------------------------------
# Tag propagation (forward: line → paragraph → document)
# ---------------------------------------------------------------------------

def _merge_tag_counts(
    existing: dict | None,
    new_tags: dict | None,
) -> dict:
    """Merge JSONB tag dicts (tag_code -> count or detail)."""
    merged: dict = dict(existing or {})
    for code, value in (new_tags or {}).items():
        if code in merged:
            if isinstance(merged[code], (int, float)) and isinstance(value, (int, float)):
                merged[code] = merged[code] + value
            elif isinstance(merged[code], dict) and isinstance(value, dict):
                # nested: sum numeric fields
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        merged[code][k] = merged[code].get(k, 0) + v
                    else:
                        merged[code][k] = v
            # else: overwrite
            else:
                merged[code] = value
        else:
            merged[code] = value
    return merged


def _count_tags(tag_dict: dict | None) -> dict[str, int]:
    """Convert ``{code: <anything>}`` to ``{code: 1}`` for simple counting."""
    if not tag_dict:
        return {}
    return {code: 1 for code in tag_dict}


async def aggregate_line_tags_to_paragraph(
    db: AsyncSession,
    paragraph_id,
) -> None:
    """Forward-propagate: aggregate all child PolicyLine tags into the parent
    PolicyParagraph (p_tags, d_tags, j_tags).  Caller should commit.
    """
    from app.models import PolicyParagraph, PolicyLine

    para = await db.get(PolicyParagraph, paragraph_id)
    if para is None:
        return

    lines_q = await db.execute(
        select(PolicyLine).where(PolicyLine.paragraph_id == paragraph_id)
    )
    lines = lines_q.scalars().all()

    agg_p: dict = {}
    agg_d: dict = {}
    agg_j: dict = {}
    for ln in lines:
        agg_p = _merge_tag_counts(agg_p, _count_tags(ln.p_tags))
        agg_d = _merge_tag_counts(agg_d, _count_tags(ln.d_tags))
        agg_j = _merge_tag_counts(agg_j, _count_tags(ln.j_tags))

    para.p_tags = agg_p or None
    para.d_tags = agg_d or None
    para.j_tags = agg_j or None


async def aggregate_paragraph_tags_to_document(
    db: AsyncSession,
    document_id,
) -> None:
    """Forward-propagate: aggregate all PolicyParagraph tags into
    DocumentTags (one row per document).  Caller should commit.
    """
    from app.models import PolicyParagraph, DocumentTags

    paras_q = await db.execute(
        select(PolicyParagraph).where(PolicyParagraph.document_id == document_id)
    )
    paras = paras_q.scalars().all()

    agg_p: dict = {}
    agg_d: dict = {}
    agg_j: dict = {}
    for p in paras:
        agg_p = _merge_tag_counts(agg_p, _count_tags(p.p_tags))
        agg_d = _merge_tag_counts(agg_d, _count_tags(p.d_tags))
        agg_j = _merge_tag_counts(agg_j, _count_tags(p.j_tags))

    doc_tags_q = await db.execute(
        select(DocumentTags).where(DocumentTags.document_id == document_id)
    )
    doc_tags = doc_tags_q.scalar_one_or_none()
    if doc_tags is None:
        doc_tags = DocumentTags(document_id=document_id)
        db.add(doc_tags)

    doc_tags.p_tags = agg_p or None
    doc_tags.d_tags = agg_d or None
    doc_tags.j_tags = agg_j or None


def compute_effective_line_tags(
    line_tags: dict | None,
    paragraph_tags: dict | None,
    document_tags: dict | None,
    *,
    w_line: float = 1.0,
    w_paragraph: float = 0.3,
    w_document: float = 0.1,
) -> dict[str, float]:
    """Backward-propagate: compute effective line-level tag scores as a
    weighted combination of line, paragraph, and document tag presence.

    Returns ``{tag_code: effective_score}`` for all tags found at any level.
    Called at query time (not persisted by default).
    """
    all_codes: set[str] = set()
    all_codes.update(line_tags or {})
    all_codes.update(paragraph_tags or {})
    all_codes.update(document_tags or {})

    effective: dict[str, float] = {}
    for code in all_codes:
        l_val = 1.0 if code in (line_tags or {}) else 0.0
        p_val = 1.0 if code in (paragraph_tags or {}) else 0.0
        d_val = 1.0 if code in (document_tags or {}) else 0.0
        score = l_val * w_line + p_val * w_paragraph + d_val * w_document
        effective[code] = round(score, 4)
    return effective
