"""Unit tests for Path B policy pipeline: normalization, stopwords, tag mapping, and candidate extraction logic."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.services.policy_path_b import (
    COMMON_ABBREVIATIONS,
    COMMON_PHRASES,
    HEADING_WEIGHT,
    STOPWORDS,
    _apply_tags_to_line_text,
    _is_only_stopwords,
    _normalize_phrase,
    _split_paragraph_into_lines,
    build_paragraph_and_lines,
    extract_candidates_for_document,
    get_phrase_to_tag_map,
)


# --- _normalize_phrase ---


def test_normalize_phrase_lowercase_collapse_whitespace():
    assert _normalize_phrase("  Member   Eligibility  ") == "member eligibility"
    assert _normalize_phrase("Prior Authorization") == "prior authorization"


def test_normalize_phrase_empty_and_non_string():
    assert _normalize_phrase("") == ""
    assert _normalize_phrase(None) == ""
    assert _normalize_phrase(123) == ""


# --- _is_only_stopwords ---


def test_is_only_stopwords_all_stopwords():
    assert _is_only_stopwords("is the", STOPWORDS) is True
    assert _is_only_stopwords("what is", STOPWORDS) is True
    assert _is_only_stopwords("and or but", STOPWORDS) is True


def test_is_only_stopwords_has_content_word():
    assert _is_only_stopwords("the member", STOPWORDS) is False
    assert _is_only_stopwords("eligibility criteria", STOPWORDS) is False
    assert _is_only_stopwords("prior authorization", STOPWORDS) is False


def test_is_only_stopwords_empty_or_no_stopwords_set():
    assert _is_only_stopwords("", STOPWORDS) is True
    assert _is_only_stopwords("hello", frozenset()) is True


# --- Constants ---


def test_stopwords_contain_common_function_words():
    assert "the" in STOPWORDS
    assert "is" in STOPWORDS
    assert "what" in STOPWORDS
    assert "thank" in STOPWORDS
    assert "and" in STOPWORDS


def test_common_phrases_contain_boilerplate():
    assert "thank you" in COMMON_PHRASES
    assert "for more information" in COMMON_PHRASES
    assert "in accordance with" in COMMON_PHRASES


def test_common_abbreviations_contain_generic_acronyms():
    assert "pdf" in COMMON_ABBREVIATIONS
    assert "url" in COMMON_ABBREVIATIONS
    assert "usa" in COMMON_ABBREVIATIONS
    assert "ii" in COMMON_ABBREVIATIONS


def test_heading_weight_is_positive():
    assert HEADING_WEIGHT >= 1


# --- _split_paragraph_into_lines ---


def test_split_paragraph_into_lines_by_newline():
    text = "First line.\n\nSecond line."
    assert _split_paragraph_into_lines(text) == ["First line.", "Second line."]


def test_split_paragraph_into_lines_single_line():
    assert _split_paragraph_into_lines("Only one line") == ["Only one line"]


def test_split_paragraph_into_lines_empty():
    assert _split_paragraph_into_lines("") == []
    assert _split_paragraph_into_lines("   \n\n  ") == []


# --- get_phrase_to_tag_map ---


def test_get_phrase_to_tag_map_builds_normalized_phrase_to_kind_code():
    lex = SimpleNamespace(
        p_tags={"eligibility.general": {"phrases": ["Member Eligibility", "prior auth"]}},
        d_tags={"coverage.general": {"phrases": ["covered services"]}},
        j_tags={},
    )
    m, refuted = get_phrase_to_tag_map(lex)
    assert m["member eligibility"] == ("p", "eligibility.general", 1.0)
    assert m["prior auth"] == ("p", "eligibility.general", 1.0)
    assert m["covered services"] == ("d", "coverage.general", 1.0)
    assert refuted == {}


def test_get_phrase_to_tag_map_empty_snapshot():
    lex = SimpleNamespace(p_tags={}, d_tags={}, j_tags={})
    m, refuted = get_phrase_to_tag_map(lex)
    assert m == {}
    assert refuted == {}


def test_get_phrase_to_tag_map_refuted_words():
    lex = SimpleNamespace(
        p_tags={},
        d_tags={"claims.denial": {
            "strong_phrases": ["denial", "denied claim"],
            "refuted_words": ["approval", "approved"],
        }},
        j_tags={},
    )
    m, refuted = get_phrase_to_tag_map(lex)
    assert m["denial"] == ("d", "claims.denial", 1.0)
    assert ("d", "claims.denial") in refuted
    assert "approval" in refuted[("d", "claims.denial")]
    assert "approved" in refuted[("d", "claims.denial")]


# --- _apply_tags_to_line_text ---


def test_apply_tags_to_line_text_matches_phrases():
    phrase_map = {
        "member eligibility": ("p", "elig", 1.0),
        "prior authorization": ("d", "pa", 1.0),
    }
    p, d, j = _apply_tags_to_line_text("The member eligibility criteria require prior authorization.", phrase_map)
    assert p == {"elig": 1.0}
    assert d == {"pa": 1.0}
    assert j == {}


def test_apply_tags_to_line_text_no_match():
    phrase_map = {"member eligibility": ("p", "elig", 1.0)}
    p, d, j = _apply_tags_to_line_text("Nothing relevant here.", phrase_map)
    assert p == {}
    assert d == {}
    assert j == {}


def test_apply_tags_to_line_text_normalizes_line():
    phrase_map = {"prior authorization": ("d", "pa", 1.0)}
    p, d, j = _apply_tags_to_line_text("  Prior   Authorization  required.  ", phrase_map)
    assert d == {"pa": 1.0}


def test_apply_tags_to_line_text_refuted_words_suppress_match():
    phrase_map = {
        "denial": ("d", "claims.denial", 1.0),
        "claim": ("d", "claims.general", 1.0),
    }
    refuted = {("d", "claims.denial"): {"approval", "approved"}}
    # "denial" matches but "approved" refutes it -- should suppress claims.denial
    p, d, j = _apply_tags_to_line_text("The claim denial was approved.", phrase_map, refuted)
    assert "claims.denial" not in d  # suppressed by refuted word "approved"
    assert d == {"claims.general": 1.0}  # claims.general is NOT refuted


def test_apply_tags_to_line_text_refuted_words_no_refute():
    phrase_map = {"denial": ("d", "claims.denial", 1.0)}
    refuted = {("d", "claims.denial"): {"approval", "approved"}}
    # No refuted word present -- match should proceed
    p, d, j = _apply_tags_to_line_text("The claim denial was rejected.", phrase_map, refuted)
    assert d == {"claims.denial": 1.0}


# --- Heading vs body paragraph_type logic (no DB) ---


def test_heading_paragraph_detection_single_short_line_with_heading_path():
    """Logic: single short line + heading_path => heading; else body."""
    from app.services.policy_path_b import _split_paragraph_into_lines

    lines = _split_paragraph_into_lines("Eligibility")
    is_heading = bool(
        "Eligibility" and lines and len(lines) == 1 and len(lines[0].split()) <= 15
    )
    assert is_heading is True

    lines_multi = _split_paragraph_into_lines("First line.\n\nSecond line.")
    is_heading_multi = bool(
        lines_multi and len(lines_multi) == 1 and len(lines_multi[0].split()) <= 15
    )
    assert is_heading_multi is False

    # Exactly 16 words => not a "short" heading per our rule (<= 15)
    lines_long = _split_paragraph_into_lines(
        "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen."
    )
    is_heading_long = bool(
        lines_long and len(lines_long) == 1 and len(lines_long[0].split()) <= 15
    )
    assert is_heading_long is False


# --- build_paragraph_and_lines: paragraph_type heading vs body ---
# Integration tests (require DB with policy_paragraphs table; skip by default to avoid schema/loop issues.)


@pytest.mark.asyncio
@pytest.mark.skip(reason="Integration test: requires DB with policy_paragraphs schema matching app.models")
async def test_build_paragraph_and_lines_single_short_line_with_heading_path_sets_heading_type():
    """When heading_path is set and paragraph is one short line, paragraph_type should be 'heading'."""
    from app.database import AsyncSessionLocal
    from app.models import Document, DocumentPage

    async with AsyncSessionLocal() as db:
        doc = Document(
            id=None,
            filename="test.pdf",
            status="pending",
            file_hash="test_hash_para_heading",
            file_path="gs://bucket/test.pdf",
        )
        db.add(doc)
        await db.flush()
        page = DocumentPage(document_id=doc.id, page_number=1, text="", text_markdown="")
        db.add(page)
        await db.flush()

        para, lines = await build_paragraph_and_lines(
            db, doc.id, page_number=1, order_index=0, heading_path="Eligibility", text="Eligibility"
        )
        await db.rollback()
    assert para.paragraph_type == "heading"
    assert len(lines) == 1
    assert lines[0].text == "Eligibility"


@pytest.mark.asyncio
@pytest.mark.skip(reason="Integration test: requires DB with policy_paragraphs schema matching app.models")
async def test_build_paragraph_and_lines_body_paragraph_sets_body_type():
    """Multi-line or no heading_path should yield paragraph_type 'body'."""
    from app.database import AsyncSessionLocal
    from app.models import Document, DocumentPage

    async with AsyncSessionLocal() as db:
        doc = Document(
            id=None,
            filename="test2.pdf",
            status="pending",
            file_hash="test_hash_para_body",
            file_path="gs://bucket/test2.pdf",
        )
        db.add(doc)
        await db.flush()
        page = DocumentPage(document_id=doc.id, page_number=1, text="", text_markdown="")
        db.add(page)
        await db.flush()

        para, lines = await build_paragraph_and_lines(
            db, doc.id, page_number=1, order_index=0, heading_path=None, text="First line.\n\nSecond line."
        )
        await db.rollback()
    assert para.paragraph_type == "body"
    assert len(lines) == 2


# --- extract_candidates_for_document (mocked DB) ---


@pytest.mark.asyncio
async def test_extract_candidates_excludes_stopword_only_ngrams_and_common_phrases():
    """Extraction should not propose n-grams that are only stopwords or in COMMON_PHRASES."""
    mock_db = AsyncMock()
    # Rejected catalog: .scalars().all() returns []
    mock_rejected_result = MagicMock()
    mock_rejected_result.scalars.return_value.all.return_value = []

    # Line data: body line with "is the" and "member eligibility" so we get mixed n-grams
    mock_lines_result = MagicMock()
    mock_lines_result.fetchall.return_value = [
        ("Member eligibility is the key criterion.", "body"),
        ("Member eligibility applies to all.", "body"),
    ]

    call_count = [0]
    def capture_execute(*args, **kwargs):
        call_count[0] += 1
        if "PolicyLexiconCandidateCatalog" in str(args[0]):
            return mock_rejected_result
        return mock_lines_result

    mock_db.execute = AsyncMock(side_effect=capture_execute)
    mock_db.flush = AsyncMock()
    added = []

    def capture_add(obj):
        added.append(obj)

    mock_db.add = capture_add

    from uuid import uuid4
    doc_id = uuid4()
    await extract_candidates_for_document(mock_db, doc_id, phrase_map={}, min_occurrences=2)

    # Should have proposed "member eligibility" (2 occurrences) but NOT "is the" (stopwords)
    ngram_candidates = [a for a in added if getattr(a, "source", None) == "path_b_ngram"]
    normalized_ngrams = [c.normalized for c in ngram_candidates]
    assert "is the" not in normalized_ngrams
    assert "member eligibility" in normalized_ngrams


@pytest.mark.asyncio
async def test_extract_candidates_weights_heading_lines():
    """Terms appearing in heading lines should get higher weighted count and confidence boost."""
    mock_db = AsyncMock()
    mock_rejected_result = MagicMock()
    mock_rejected_result.scalars.return_value.all.return_value = []

    # One body line and one heading line with same term so heading-weighted term wins
    mock_lines_result = MagicMock()
    mock_lines_result.fetchall.return_value = [
        ("Prior authorization required.", "body"),
        ("Prior Authorization", "heading"),  # same phrase in heading
    ]

    def capture_execute(*args, **kwargs):
        if "PolicyLexiconCandidateCatalog" in str(args[0]):
            return mock_rejected_result
        return mock_lines_result

    mock_db.execute = AsyncMock(side_effect=capture_execute)
    mock_db.flush = AsyncMock()
    added = []

    def capture_add(obj):
        added.append(obj)

    mock_db.add = capture_add

    from uuid import uuid4
    doc_id = uuid4()
    await extract_candidates_for_document(mock_db, doc_id, phrase_map={}, min_occurrences=1)

    ngram_candidates = [a for a in added if getattr(a, "source", None) == "path_b_ngram"]
    # "prior authorization" appears in both body and heading -> should have confidence boost if count_heading > 0
    pa = next((c for c in ngram_candidates if "prior authorization" in c.normalized), None)
    assert pa is not None
    assert pa.confidence >= 0.5
    assert pa.occurrences == 2


@pytest.mark.asyncio
async def test_extract_candidates_detects_abbreviations_with_separate_source():
    """All-caps tokens should be proposed as candidates with source path_b_abbreviation and candidate_type alias."""
    mock_db = AsyncMock()
    mock_rejected_result = MagicMock()
    mock_rejected_result.scalars.return_value.all.return_value = []

    mock_lines_result = MagicMock()
    mock_lines_result.fetchall.return_value = [
        ("LTC and MMA benefits apply here.", "body"),
        ("See LTC handbook.", "body"),
    ]

    def capture_execute(*args, **kwargs):
        if "PolicyLexiconCandidateCatalog" in str(args[0]):
            return mock_rejected_result
        return mock_lines_result

    mock_db.execute = AsyncMock(side_effect=capture_execute)
    mock_db.flush = AsyncMock()
    added = []

    def capture_add(obj):
        added.append(obj)

    mock_db.add = capture_add

    from uuid import uuid4
    doc_id = uuid4()
    await extract_candidates_for_document(mock_db, doc_id, phrase_map={}, min_occurrences=1)

    abbrev_candidates = [a for a in added if getattr(a, "source", None) == "path_b_abbreviation"]
    assert len(abbrev_candidates) >= 1
    ltc = next((c for c in abbrev_candidates if c.normalized == "ltc"), None)
    assert ltc is not None
    assert ltc.candidate_type == "alias"
    assert ltc.source == "path_b_abbreviation"


@pytest.mark.asyncio
async def test_extract_candidates_excludes_common_abbreviations():
    """Tokens in COMMON_ABBREVIATIONS (e.g. PDF, USA) should not be proposed."""
    mock_db = AsyncMock()
    mock_rejected_result = MagicMock()
    mock_rejected_result.scalars.return_value.all.return_value = []

    mock_lines_result = MagicMock()
    mock_lines_result.fetchall.return_value = [
        ("See PDF for details. PDF attached.", "body"),
    ]

    def capture_execute(*args, **kwargs):
        if "PolicyLexiconCandidateCatalog" in str(args[0]):
            return mock_rejected_result
        return mock_lines_result

    mock_db.execute = AsyncMock(side_effect=capture_execute)
    mock_db.flush = AsyncMock()
    added = []

    def capture_add(obj):
        added.append(obj)

    mock_db.add = capture_add

    from uuid import uuid4
    doc_id = uuid4()
    await extract_candidates_for_document(mock_db, doc_id, phrase_map={}, min_occurrences=1)

    abbrev_candidates = [a for a in added if getattr(a, "source", None) == "path_b_abbreviation"]
    pdf_candidates = [c for c in abbrev_candidates if c.normalized == "pdf"]
    assert len(pdf_candidates) == 0
