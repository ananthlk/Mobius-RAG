"""Unit tests for app.services.chunking — paragraph splitting and the
code-list / changelog row-gluing pass.

The gluing pass exists because PDF→markdown conversion of code tables and
revision logs inserts a blank line inside a single logical entry, so the
blank-line paragraph split severs a "code + descriptor + date" row into an
orphaned <40-char fragment (e.g. "and G0659.", "(Genemarkers): 81418",
"Added HCPCS code [Q5129].  04.17.23"). Those fragments carry an answer-bearing
HCPCS/CPT code but no queryable context. We glue them back onto the preceding
paragraph of the same section. These tests pin both the positive cases (real
severed fragments observed in rag_published_embeddings) and the precision
guardrails (ordinary short paragraphs must not be merged).
"""
from __future__ import annotations

import pytest

from app.services.chunking import (
    _is_code_list_fragment,
    split_paragraphs_from_markdown,
)


# --- _is_code_list_fragment: positive cases (real severed fragments) ---


@pytest.mark.parametrize(
    "text",
    [
        "and G0659.",                          # wrapped code-list tail (CC.PP.056)
        "(Genemarkers): 81418",                # wrapped CPT table row
        "Added HCPCS code [Q5129].  04.17.23",  # changelog row (CP.PHAR.93)
        "Removed HCPCS code [J9259].  11.06.24",  # changelog row (CP.PHAR.176)
        "care \n99503",                        # wrapped descriptor + CPT
        "up to 4 months \n(N0925)",            # wrapped descriptor + HCPCS
        "and older \nT1005",                   # wrapped descriptor + HCPCS
    ],
)
def test_is_code_list_fragment_positive(text: str) -> None:
    assert _is_code_list_fragment(text) is True


# --- _is_code_list_fragment: negatives (must NOT be treated as fragments) ---


@pytest.mark.parametrize(
    "text",
    [
        # Pure numbers / page artifacts.
        "76,352",
        "4,841,152",
        "100778",          # 6 digits, not a 5-digit CPT
        "01202\n58.54",
        "-",
        # Headings / labels with no code token.
        "GOVERNOR",
        "Depression",
        "About SMI and SED",
        "Youth and Family Webinars",
        # Uppercase-initial self-contained lines that merely contain a 5-digit
        # ZIP or number — these begin their own entry, they are not continuations.
        "Tallahassee, FL 32308",
        "WASHINGTON, D.C. 20201",
        "Jacksonville, FL 32209",
        "Deny CON #10715 and CON #10716.",
        "CMS-10398",
        "Iowa's Medicaid Program or CHIP",
        # A bare standalone HCPCS code row is a complete entry, not a fragment.
        "S5100",
        "H2019",
        # Long enough to stand on its own even with a code.
        "Updated Appendix E to include Oklahoma.  06.07.23",
    ],
)
def test_is_code_list_fragment_negative(text: str) -> None:
    assert _is_code_list_fragment(text) is False


# --- split_paragraphs_from_markdown: end-to-end gluing behaviour ---


def test_glues_wrapped_code_list_tail() -> None:
    md = (
        "Coding Implications: the applicable codes are G0479, G0480,\nG0481"
        "\n\nand G0659."
    )
    paras = split_paragraphs_from_markdown(md)
    assert len(paras) == 1
    assert paras[0]["text"].endswith("and G0659.")
    assert "G0481" in paras[0]["text"]
    # Offsets span the whole merged range in the original markdown.
    assert paras[0]["start_offset"] == 0
    assert paras[0]["end_offset"] == len(md)


def test_glues_short_changelog_row_onto_preceding_row() -> None:
    md = (
        "4Q 2022 annual review: added indications for cancer X and\n"
        "reclassified anaplastic tumors per NCCN guidance here.\n\n"
        "Added HCPCS code [Q5129].\n04.17.23\n\n"
        "Updated Appendix E to include Oklahoma.\n06.07.23"
    )
    paras = split_paragraphs_from_markdown(md)
    # The short changelog row folds into the long review row; the 48-char
    # "Updated Appendix E..." row stays standalone (>= 40 chars, no code token).
    assert len(paras) == 2
    assert "Q5129" in paras[0]["text"]
    assert paras[0]["text"].startswith("4Q 2022 annual review")
    assert paras[1]["text"].startswith("Updated Appendix E")


def test_runs_of_fragments_all_fold_into_one() -> None:
    md = (
        "Applicable codes list follows below for reference here now:\n\n"
        "and G0659.\n\nand J9259."
    )
    paras = split_paragraphs_from_markdown(md)
    assert len(paras) == 1
    assert "G0659" in paras[0]["text"] and "J9259" in paras[0]["text"]


def test_never_glues_across_a_section_header() -> None:
    md = (
        "## Codes\n\nSome intro paragraph long enough to be standalone text.\n\n"
        "## Notes\n\nand G0659."
    )
    paras = split_paragraphs_from_markdown(md)
    # The fragment's only predecessor is in a different section, so it is left
    # as its own paragraph rather than merged across the heading boundary.
    assert len(paras) == 2
    assert paras[0]["section_path"] == "Codes"
    assert paras[1]["section_path"] == "Notes"
    assert paras[1]["text"] == "and G0659."


def test_paragraph_index_is_contiguous_after_gluing() -> None:
    md = (
        "First real paragraph with enough text to stand on its own here.\n\n"
        "and G0659.\n\n"
        "Second real paragraph, also long enough to stand on its own here."
    )
    paras = split_paragraphs_from_markdown(md)
    assert [p["paragraph_index"] for p in paras] == list(range(len(paras)))


def test_ordinary_short_paragraph_is_untouched() -> None:
    # A short heading-like paragraph carrying no code token is preserved as its
    # own chunk (regression guard against over-eager merging).
    md = "Bevacizumab and Biosimilars\n\nA following paragraph of real length here."
    paras = split_paragraphs_from_markdown(md)
    assert len(paras) == 2
    assert paras[0]["text"] == "Bevacizumab and Biosimilars"
