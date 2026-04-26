"""HTML → structured page sections for the rag pipeline.

The web-scraper already strips boilerplate (nav/footer/script) and
returns visible text — that's good for display but loses the heading
structure that makes good chunks. This module re-extracts from raw
HTML so we can split per ``<h1>``/``<h2>``/``<h3>`` section, which
maps cleanly onto the chunking pipeline's "page" concept.

Why per-section instead of per-document:
* Sunshine's billing-manual sub-pages are 4-5KB of text each but
  cover 3-5 sub-topics under different headings (e.g. "Standard
  appeal", "Expedited appeal", "External review"). One chunk per
  sub-topic = better retrieval relevance than one chunk per page.
* Real PDF pages give the chunker a natural unit; HTML has no pages,
  so headings substitute.

Output shape mirrors what extract_text_from_gcs returns:
    [
        {"page_number": 1, "text": "...", "text_length": N,
         "extraction_status": "success", "section_title": "..."}
        ...
    ]
so the existing import path can consume it without changes.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict
from typing import Iterable

from bs4 import BeautifulSoup, NavigableString, Tag

logger = logging.getLogger(__name__)


# Tags whose content is presentation/navigation, not policy text.
# Stripped before extraction. Order matters only for readability.
_BOILERPLATE_TAGS = (
    "script", "style", "nav", "header", "footer",
    "aside", "noscript", "form", "button",
)

# Heading tags that we treat as section boundaries. h1 + h2 are major
# splits; h3 is a soft hint that we keep within the parent h2 section
# (avoiding a fan-out into too-small chunks).
_MAJOR_HEADINGS = ("h1", "h2")


@dataclass
class HtmlSection:
    """One heading-bounded section of an HTML doc."""
    page_number: int           # 1-indexed, like PDF pages
    section_title: str         # the heading text; "(intro)" before first heading
    text: str                  # plain text under this section, paragraph-separated
    text_length: int           # convenience for callers / DB
    extraction_status: str = "success"
    extraction_error: str | None = None


# ── Public API ───────────────────────────────────────────────────────


def extract_sections(html: str, *, source_url: str | None = None) -> list[dict]:
    """Parse HTML, strip boilerplate, split by major headings.

    Returns a list of dicts compatible with what extract_text_from_gcs
    yields — caller can pass straight into DocumentPage row creation.

    Edge cases handled:
    * No headings at all → returns one section with the whole body
    * All-boilerplate (CMS shell with no real content) → returns one
      section with empty text and extraction_status='empty'
    * Unparseable HTML → returns one section with extraction_status='failed'
    """
    if not html or not html.strip():
        return [_empty_section(reason="HTML body was empty")]

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as exc:  # pragma: no cover — bs4 is very tolerant
        logger.warning("html_extractor: BeautifulSoup raised on %s: %s", source_url, exc)
        return [_failed_section(str(exc))]

    # Strip boilerplate before traversal — keeps section text clean.
    for tag_name in _BOILERPLATE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    body = soup.body or soup
    sections = list(_split_by_major_headings(body))

    # Drop empty-content sections (heading with no body — common for
    # table-of-contents pages).
    sections = [s for s in sections if s.text.strip() or s.section_title.strip() != "(intro)"]

    # Drop short ``(intro)`` sections when there's also a real
    # heading-bounded section. Pre-h1 content on Sunshine/AHCA pages is
    # typically the language picker + breadcrumb (200-300 chars of
    # link-list noise), not real policy content. Real intros — when
    # they exist — usually run 500+ chars before the first h2.
    has_real_section = any(s.section_title != "(intro)" for s in sections)
    if has_real_section:
        sections = [
            s for s in sections
            if not (s.section_title == "(intro)" and s.text_length < 500)
        ]
        # Re-number the remaining sections so page_number stays
        # contiguous from 1 (chunking pipeline expects no gaps).
        for i, s in enumerate(sections, 1):
            s.page_number = i

    if not sections:
        return [_empty_section(reason="no extractable content after stripping boilerplate")]

    return [asdict(s) for s in sections]


def derive_title(html: str, fallback: str = "Untitled HTML page") -> str:
    """Extract a useful Document.filename for an HTML import.

    Order of preference:
      1. <h1> on the page
      2. <title> tag
      3. fallback (caller-provided, usually the URL)
    """
    if not html:
        return fallback
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return fallback
    h1 = soup.find("h1")
    if h1:
        t = _normalize_whitespace(h1.get_text(strip=True))
        if t:
            return t[:240]
    title = soup.title
    if title and title.string:
        t = _normalize_whitespace(title.string.strip())
        if t:
            return t[:240]
    return fallback


# ── Internals ────────────────────────────────────────────────────────


def _split_by_major_headings(root: Tag) -> Iterable[HtmlSection]:
    """Walk the body in document order, accumulating text until we
    hit an h1/h2, then emit a new section.

    We deliberately don't use .find_all(_MAJOR_HEADINGS) + slicing —
    that loses content between sections. Document-order traversal
    keeps the ordering and ensures every text node lands somewhere.
    """
    page_number = 1
    current_title = "(intro)"
    current_buf: list[str] = []

    def _flush() -> HtmlSection | None:
        nonlocal page_number
        text = _join_paragraphs(current_buf)
        if not text and current_title == "(intro)":
            return None  # nothing before the first heading; skip
        sec = HtmlSection(
            page_number=page_number,
            section_title=current_title,
            text=text,
            text_length=len(text),
        )
        page_number += 1
        return sec

    # Track tables we've already consumed wholesale via _format_table
    # so the per-cell branch below doesn't double-emit each <td>/<th>.
    consumed_tables: set[int] = set()

    # Iterate descendants in document order. Skip nested matches by
    # tracking when we're inside a heading we've already consumed.
    for elem in root.descendants:
        if not isinstance(elem, Tag):
            continue
        if elem.name in _MAJOR_HEADINGS:
            sec = _flush()
            if sec is not None:
                yield sec
            current_title = _normalize_whitespace(elem.get_text(" ", strip=True)) or "(unnamed section)"
            current_buf = []
            continue

        # Tables: emit ONE block per row with header context preserved
        # (instead of fanning every <td>/<th> into its own paragraph,
        # which destroys the row semantics — see Phase 13.4 dental-plan
        # bug where 'REGION', 'COUNTIES', '1', 'Escambia...' all became
        # separate trivial chunks). Real-world tables (fee schedules,
        # transition dates, criteria matrices) only make sense row-by-row.
        if elem.name == "table" and id(elem) not in consumed_tables:
            consumed_tables.add(id(elem))
            for row_text in _format_table(elem):
                if row_text:
                    current_buf.append(row_text)
            continue

        # If we're inside an already-consumed table, skip.
        in_consumed_table = False
        for ancestor in elem.parents:
            if id(ancestor) in consumed_tables:
                in_consumed_table = True
                break
        if in_consumed_table:
            continue

        # Paragraph-y tags whose direct text we capture. We don't
        # capture nested <p> twice because we check `.parent.name` —
        # only the outermost paragraph contributes text. <li> we
        # handle similarly. <td>/<th> are NOT in this list anymore —
        # tables are handled wholesale above.
        if elem.name in ("p", "li", "blockquote", "pre", "h3", "h4", "h5", "h6"):
            t = _normalize_whitespace(elem.get_text(" ", strip=True))
            if t and (not elem.parent or elem.parent.name not in ("p", "li")):
                # h3+ headings get inlined as bold-ish prefix so they're
                # not lost — they signal sub-topics inside an h2 section.
                if elem.name in ("h3", "h4", "h5", "h6"):
                    current_buf.append(f"## {t}")
                else:
                    current_buf.append(t)

    last = _flush()
    if last is not None:
        yield last


def _format_table(table: Tag) -> list[str]:
    """Render a <table> as a list of row-paragraphs with header context.

    Strategy:
    * First <tr> with <th> cells → headers
    * Each subsequent <tr> → one paragraph: "header1: cell1 | header2: cell2"
    * If a row has more cells than headers, extras are appended without keys
    * If no <th> headers exist, fall back to "cell1 | cell2 | ..." per row

    Why row-paragraphs rather than one giant table-paragraph: chunker
    splits at paragraph boundaries, so each row becomes one chunk.
    Row chunks carry enough context (header keys) to be useful in
    isolation — perfect for "what's the rate for Region 5?" queries.
    """
    rows = table.find_all("tr")
    if not rows:
        return []

    headers: list[str] = []
    out: list[str] = []
    for ri, row in enumerate(rows):
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        cell_texts = [
            _normalize_whitespace(c.get_text(" ", strip=True))
            for c in cells
        ]
        # Detect header row: ALL cells are <th>
        is_header_row = all(c.name == "th" for c in cells)
        if is_header_row and not headers:
            headers = cell_texts
            continue
        # Body row — render with header context if available. Skip
        # any cell whose value is empty/whitespace-only — those add
        # no information and produce ugly 'Note: |' fragments that
        # confuse downstream chunkers.
        if headers:
            pairs = []
            for i, val in enumerate(cell_texts):
                if not val or not val.strip():
                    continue
                if i < len(headers) and headers[i]:
                    pairs.append(f"{headers[i]}: {val}")
                else:
                    pairs.append(val)
            line = " | ".join(pairs)
        else:
            line = " | ".join(c for c in cell_texts if c and c.strip())
        if line:
            out.append(line)
    return out


def _join_paragraphs(parts: list[str]) -> str:
    """Join with double-newline so chunker sees real paragraph breaks."""
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p in seen:
            continue  # dedupe (CMS templates often repeat the breadcrumb)
        seen.add(p)
        out.append(p)
    return "\n\n".join(out).strip()


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_whitespace(s: str) -> str:
    return _WHITESPACE_RE.sub(" ", s or "").strip()


def _empty_section(*, reason: str) -> dict:
    return asdict(HtmlSection(
        page_number=1,
        section_title="(empty)",
        text="",
        text_length=0,
        extraction_status="empty",
        extraction_error=reason,
    ))


def _failed_section(err: str) -> dict:
    return asdict(HtmlSection(
        page_number=1,
        section_title="(parse failed)",
        text="",
        text_length=0,
        extraction_status="failed",
        extraction_error=err[:500],
    ))
