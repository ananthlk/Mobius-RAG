"""HTML must produce paragraph breaks, and must not shatter into fragments.

Found 2026-08-20 on a real AHCA fetch. `html_to_plain_text` joined every line
with a SINGLE newline and dropped empties, so its output could not contain a
blank line by construction. The chunker splits on `\\n\\s*\\n+`, so an entire page
became ONE chunk: a 14,226-character page produced exactly 1 chunk of 14,226
characters.

Two failure modes bracket the fix, and both are real:
  TOO FEW  — one mega-chunk per page. Retrieval cannot cite a passage, and the
             chunk exceeds the embedder's input limit, so the document embeds to
             NOTHING while the embedding job still reports "completed".
  TOO MANY — breaking at every block tag turns a nav menu into one chunk per
             link. The first fix produced 535 chunks with a MEAN of 25
             characters, which retrieves just as badly for the opposite reason.
"""
import sys
import pytest
sys.path.insert(0, ".")

from app.services.extract_text import html_to_plain_text, _MIN_BLOCK_CHARS
from app.services.chunking import split_paragraphs_from_markdown


LONG = "This is a substantive policy paragraph with enough text to stand on its own as a retrievable unit. " * 5


def test_output_contains_blank_lines():
    """The chunker splits on blank lines; without them nothing can split."""
    html = f"<html><body><p>{LONG}</p><p>{LONG}</p></body></html>"
    assert "\n\n" in html_to_plain_text(html)


def test_a_long_page_does_not_collapse_to_one_chunk():
    html = "<html><body>" + "".join(f"<p>{LONG}</p>" for _ in range(6)) + "</body></html>"
    chunks = split_paragraphs_from_markdown(html_to_plain_text(html))
    assert len(chunks) > 1, "a multi-paragraph page must not become a single chunk"


def test_a_link_list_does_not_become_one_chunk_per_link():
    """The over-correction: 535 chunks with a 25-character mean."""
    html = "<html><body><ul>" + "".join(
        f'<li><a href="/x{i}">Link {i}</a></li>' for i in range(60)) + "</ul></body></html>"
    chunks = split_paragraphs_from_markdown(html_to_plain_text(html))
    assert len(chunks) < 20, f"link list shattered into {len(chunks)} chunks"
    mean = sum(len(c["text"]) for c in chunks) / len(chunks)
    assert mean > 100, f"mean chunk {mean:.0f} chars — fragments, not passages"


def test_substantial_paragraphs_still_separate():
    """Coalescing must not silently re-create the mega-chunk."""
    html = "<html><body>" + "".join(f"<p>{LONG}</p>" for _ in range(4)) + "</body></html>"
    chunks = split_paragraphs_from_markdown(html_to_plain_text(html))
    assert len(chunks) >= 3, "distinct long paragraphs must remain distinct"
    assert max(len(c["text"]) for c in chunks) < 4000


def test_scripts_and_styles_are_still_stripped():
    html = "<html><body><script>var x=1;</script><style>p{}</style>" \
           f"<p>{LONG}</p></body></html>"
    out = html_to_plain_text(html)
    assert "var x" not in out and "p{}" not in out


def test_empty_and_blank_input():
    assert html_to_plain_text("") == ""
    assert html_to_plain_text("   ") == ""


def test_threshold_is_a_named_constant_not_a_literal():
    assert isinstance(_MIN_BLOCK_CHARS, int) and _MIN_BLOCK_CHARS > 0
