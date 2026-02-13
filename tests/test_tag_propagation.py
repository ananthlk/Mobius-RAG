"""Unit tests for tag propagation functions in app.services.policy_path_b."""
from __future__ import annotations

import pytest

from app.services.policy_path_b import (
    _count_tags,
    _merge_tag_counts,
    compute_effective_line_tags,
)


# ---------------------------------------------------------------------------
# _count_tags
# ---------------------------------------------------------------------------

def test_count_tags_empty():
    assert _count_tags(None) == {}
    assert _count_tags({}) == {}


def test_count_tags_converts_to_ones():
    tags = {"eligibility": {"phrases": ["x"]}, "coverage": 3}
    result = _count_tags(tags)
    assert result == {"eligibility": 1, "coverage": 1}


# ---------------------------------------------------------------------------
# _merge_tag_counts
# ---------------------------------------------------------------------------

def test_merge_tag_counts_both_empty():
    assert _merge_tag_counts(None, None) == {}
    assert _merge_tag_counts({}, {}) == {}


def test_merge_tag_counts_one_empty():
    assert _merge_tag_counts({"a": 1}, None) == {"a": 1}
    assert _merge_tag_counts(None, {"b": 2}) == {"b": 2}


def test_merge_tag_counts_sums_numbers():
    merged = _merge_tag_counts({"a": 1, "b": 2}, {"a": 3, "c": 1})
    assert merged["a"] == 4
    assert merged["b"] == 2
    assert merged["c"] == 1


def test_merge_tag_counts_nested_dicts():
    merged = _merge_tag_counts(
        {"a": {"count": 2, "weight": 0.5}},
        {"a": {"count": 1, "weight": 0.3}},
    )
    assert merged["a"]["count"] == 3
    assert merged["a"]["weight"] == pytest.approx(0.8)


def test_merge_tag_counts_mixed_types_overwrite():
    # If one side is int and other is dict, last wins
    merged = _merge_tag_counts({"a": 1}, {"a": {"complex": True}})
    assert merged["a"] == {"complex": True}


# ---------------------------------------------------------------------------
# compute_effective_line_tags
# ---------------------------------------------------------------------------

def test_effective_all_levels():
    eff = compute_effective_line_tags(
        {"eligibility": 1},   # line
        {"eligibility": 1, "coverage": 1},  # paragraph
        {"eligibility": 1, "coverage": 1, "payment": 1},  # document
    )
    # eligibility: 1*1.0 + 1*0.3 + 1*0.1 = 1.4
    assert eff["eligibility"] == pytest.approx(1.4)
    # coverage: 0*1.0 + 1*0.3 + 1*0.1 = 0.4
    assert eff["coverage"] == pytest.approx(0.4)
    # payment: 0 + 0 + 0.1 = 0.1
    assert eff["payment"] == pytest.approx(0.1)


def test_effective_line_only():
    eff = compute_effective_line_tags({"a": 1}, None, None)
    assert eff["a"] == pytest.approx(1.0)


def test_effective_custom_weights():
    eff = compute_effective_line_tags(
        {"x": 1}, {"x": 1}, {"x": 1},
        w_line=1.0, w_paragraph=0.5, w_document=0.25,
    )
    assert eff["x"] == pytest.approx(1.75)


def test_effective_empty_all():
    eff = compute_effective_line_tags(None, None, None)
    assert eff == {}


def test_effective_document_only():
    eff = compute_effective_line_tags(None, None, {"doc_tag": 1})
    assert eff["doc_tag"] == pytest.approx(0.1)
