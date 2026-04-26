"""Tests for app.services.html_extractor — the HTML → sections splitter
that feeds the rag pipeline.

These cases come from real Sunshine + AHCA pages we've scraped — they're
the actual edge cases we'll see in production, not synthetic minimums.
"""
from __future__ import annotations

import pytest

from app.services.html_extractor import (
    derive_title,
    extract_sections,
)


# ── derive_title ─────────────────────────────────────────────────────


def test_derive_title_prefers_h1():
    html = "<html><head><title>SEO title</title></head><body><h1>Real Page Title</h1></body></html>"
    assert derive_title(html, fallback="ignored") == "Real Page Title"


def test_derive_title_falls_back_to_title_tag():
    html = "<html><head><title>The Title</title></head><body>no h1 here</body></html>"
    assert derive_title(html, fallback="ignored") == "The Title"


def test_derive_title_ultimate_fallback():
    """No h1, no <title> → caller-provided fallback (typically the URL)."""
    html = "<html><body>no headings</body></html>"
    assert derive_title(html, fallback="https://x.com/page") == "https://x.com/page"


def test_derive_title_handles_empty_input():
    assert derive_title("", fallback="default") == "default"
    assert derive_title(None, fallback="default") == "default"


def test_derive_title_normalizes_whitespace():
    """A multi-line h1 with extra whitespace should collapse cleanly."""
    html = "<h1>\n  Provider Appeals   Process\n</h1>"
    assert derive_title(html, fallback="x") == "Provider Appeals Process"


def test_derive_title_caps_long_titles():
    """Pathological 1000-char h1 doesn't blow up Document.filename(255)."""
    html = "<h1>" + ("x " * 500) + "</h1>"
    out = derive_title(html, fallback="x")
    assert len(out) <= 240


# ── extract_sections — happy paths ───────────────────────────────────


def test_extracts_single_section_when_no_headings():
    html = "<body><p>Just one paragraph of policy text here.</p></body>"
    sections = extract_sections(html)
    assert len(sections) == 1
    assert "Just one paragraph" in sections[0]["text"]
    assert sections[0]["page_number"] == 1
    assert sections[0]["extraction_status"] == "success"


def test_splits_by_h1_and_h2():
    """The 'standard appeal vs expedited appeal' shape — most billing
    manual sub-pages look like this. Short ``(intro)`` content is
    dropped when there are real heading-bounded sections (filters out
    nav/breadcrumb junk that sits before the first h1)."""
    html = """
    <body>
      <p>Short intro paragraph before any headings.</p>
      <h1>Standard appeal</h1>
      <p>You have 30 days to file.</p>
      <p>Submit via fax to 1-800-FOO.</p>
      <h2>Expedited appeal</h2>
      <p>Available for urgent care decisions.</p>
      <h2>External review</h2>
      <p>Request after exhausting internal appeals.</p>
    </body>
    """
    sections = extract_sections(html)
    titles = [s["section_title"] for s in sections]
    # Short intro (< 500 chars) is dropped when real headings exist.
    assert titles == ["Standard appeal", "Expedited appeal", "External review"]
    # Page numbers stay contiguous from 1 after the drop.
    assert [s["page_number"] for s in sections] == [1, 2, 3]
    # Each section gets ONLY its own text — no bleed-over.
    standard = next(s for s in sections if s["section_title"] == "Standard appeal")
    assert "30 days" in standard["text"]
    assert "Expedited" not in standard["text"]
    assert "External" not in standard["text"]


def test_keeps_long_intro_section():
    """A real intro paragraph (500+ chars) is preserved even when
    other heading-bounded sections exist — it's substantive lead text,
    not nav noise.
    """
    long_intro = ("Substantive intro paragraph. " * 30)  # ~840 chars
    html = f"""
    <body>
      <p>{long_intro}</p>
      <h1>First section</h1>
      <p>section content</p>
    </body>
    """
    sections = extract_sections(html)
    titles = [s["section_title"] for s in sections]
    assert "(intro)" in titles
    assert "First section" in titles


def test_keeps_intro_when_only_section():
    """If a page has NO headings at all, the intro is the only
    content we have — must keep it regardless of length.
    """
    html = "<body><p>Short page with no headings at all.</p></body>"
    sections = extract_sections(html)
    assert len(sections) == 1
    assert sections[0]["section_title"] == "(intro)"
    assert "Short page" in sections[0]["text"]


def test_strips_boilerplate_tags():
    """Sunshine's CMS shell wraps content in <nav>/<footer>/<script>.
    None of that should leak into the section text.
    """
    html = """
    <html>
      <head><script>var nav = {};</script><style>body { color: red; }</style></head>
      <body>
        <nav>Home | About | Contact</nav>
        <header>Sunshine Health logo</header>
        <h1>Real Content</h1>
        <p>This is the actual policy text.</p>
        <footer>© 2026 Sunshine Health. <a href="#">Login</a></footer>
        <script>analytics();</script>
      </body>
    </html>
    """
    sections = extract_sections(html)
    body_text = " ".join(s["text"] for s in sections)
    assert "actual policy text" in body_text
    assert "Home | About" not in body_text
    assert "Sunshine Health logo" not in body_text
    assert "© 2026" not in body_text
    assert "var nav" not in body_text
    assert "color: red" not in body_text


def test_h3_inlined_into_parent_section():
    """h3+ are sub-topics within an h2 section — they shouldn't fan
    out into their own pages but they shouldn't be lost either.
    Inlined as ``## heading text`` markdown-ish prefix.
    """
    html = """
    <body>
      <h2>Documentation requirements</h2>
      <h3>For inpatient claims</h3>
      <p>Include itemized bill.</p>
      <h3>For outpatient claims</h3>
      <p>Include CPT codes.</p>
    </body>
    """
    sections = extract_sections(html)
    assert len(sections) == 1
    text = sections[0]["text"]
    # The h3 text appears with the markdown prefix — chunker sees it
    # as paragraph context.
    assert "## For inpatient claims" in text
    assert "## For outpatient claims" in text
    assert "itemized bill" in text
    assert "CPT codes" in text


def test_list_items_captured():
    html = """
    <body>
      <h1>Required forms</h1>
      <ul>
        <li>Form A</li>
        <li>Form B</li>
        <li>Form C</li>
      </ul>
    </body>
    """
    sections = extract_sections(html)
    text = sections[0]["text"]
    assert "Form A" in text
    assert "Form B" in text
    assert "Form C" in text


def test_table_cells_captured():
    """Fee schedule tables — common in AHCA pages."""
    html = """
    <body>
      <h1>Reimbursement rates</h1>
      <table>
        <tr><th>Service</th><th>Rate</th></tr>
        <tr><td>Office visit</td><td>$75.00</td></tr>
        <tr><td>Lab panel</td><td>$120.00</td></tr>
      </table>
    </body>
    """
    sections = extract_sections(html)
    text = sections[0]["text"]
    assert "Office visit" in text
    assert "$75.00" in text
    assert "Lab panel" in text


def test_dedupes_repeated_breadcrumb_text():
    """Sunshine's CMS repeats the breadcrumb path in the body —
    extract_sections should dedupe identical paragraph snippets.
    """
    html = """
    <body>
      <h1>Appeals</h1>
      <p>Home > Providers > Billing > Appeals</p>
      <p>Important content.</p>
      <p>Home > Providers > Billing > Appeals</p>
    </body>
    """
    sections = extract_sections(html)
    text = sections[0]["text"]
    # Breadcrumb appears once, not twice.
    assert text.count("Home > Providers > Billing > Appeals") == 1
    assert "Important content" in text


# ── extract_sections — edge cases ────────────────────────────────────


def test_empty_html_input():
    assert extract_sections("")[0]["extraction_status"] == "empty"
    assert extract_sections("   ")[0]["extraction_status"] == "empty"


def test_all_boilerplate_yields_empty():
    """A CMS shell with nothing real underneath returns one 'empty'
    section so the upstream import path knows to mark the doc as
    empty rather than as failed.
    """
    html = """
    <html><body>
      <nav>menu</nav>
      <script>x=1;</script>
      <footer>copyright</footer>
    </body></html>
    """
    sections = extract_sections(html)
    assert len(sections) == 1
    assert sections[0]["extraction_status"] == "empty"
    assert sections[0]["text"] == ""


def test_sections_have_text_length_field():
    """text_length is used by DB storage; needs to match len(text)."""
    html = "<body><p>Hello world</p></body>"
    sections = extract_sections(html)
    assert sections[0]["text_length"] == len(sections[0]["text"])
    assert sections[0]["text_length"] > 0


def test_section_count_for_realistic_billing_subpage():
    """Approximation of /providers/Billing-manual/appeals.html shape.
    The h1 lead paragraph (under 500 chars) is captured under the h1
    section, not as a separate intro — short pre-h2 content stays
    with its parent heading."""
    html = """
    <html><body>
      <nav>Provider | Member | About</nav>
      <h1>Medicaid Member and Provider Appeals Processes</h1>
      <p>At Sunshine Health, both members and providers have the right to appeal.</p>
      <h2>Member appeals</h2>
      <p>Members can appeal within 60 days of denial notice.</p>
      <h2>Provider appeals</h2>
      <p>Providers must submit Form FL-PAF-0676.</p>
      <h2>External review</h2>
      <p>Available after internal appeals exhausted.</p>
      <footer>© Sunshine</footer>
    </body></html>
    """
    sections = extract_sections(html)
    titles = [s["section_title"] for s in sections]
    assert titles == [
        "Medicaid Member and Provider Appeals Processes",
        "Member appeals",
        "Provider appeals",
        "External review",
    ]
    assert all(s["text"].strip() for s in sections)
    # Content under the h1 (the "right to appeal" line) lives in the
    # h1's own section (it's the text between h1 and the first h2).
    h1_section = next(s for s in sections if s["section_title"] == "Medicaid Member and Provider Appeals Processes")
    assert "right to appeal" in h1_section["text"]
