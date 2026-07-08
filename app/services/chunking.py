"""Hierarchical chunking service for splitting text into paragraphs.

Why we chunk (and don't send the whole page):
- Whole-page text is too long for typical LLM context and mixes many topics.
- Paragraph-sized chunks keep extraction focused and within token limits.
- We preserve structure by detecting section headers and passing them as context.

How we chunk:
- Per-page: each page's text is split by double newlines into candidate paragraphs.
- Section detection: if the first line looks like a header (short, ends with colon,
  or numbered like "3. Eligibility"), we store it as section_path and use the rest
  as the paragraph body. Standalone headers (no body) are skipped.
- Each unit sent to extraction is one paragraph, optionally with section_path so
  the extractor sees "## Section title\n\nparagraph body" (markdown-style structure).
- Optional: build full-page or full-doc markdown (see page_to_markdown_blocks) for
  export or for sending larger structured blocks later.
"""
import re
from typing import List, Dict, Any


# --- Code-list / changelog row gluing -------------------------------------
# PDF→markdown conversion of code tables and revision logs frequently inserts a
# blank line *inside* one logical entry, so the blank-line paragraph split below
# severs a single "code + descriptor + date" row into an orphaned <40-char
# fragment — e.g. "and G0659.", "(Genemarkers): 81418",
# "Added HCPCS code [Q5129].  04.17.23", "Removed HCPCS code [J9259].  11.06.24".
# Such a fragment carries an answer-bearing HCPCS/CPT code but no queryable
# context, which hurts retrieval and forces a keep-list exception in the
# publish-time short-fragment filter. We glue each such fragment back onto the
# preceding paragraph of the same section so the code stays with its row.
#
# The gate is deliberately tight (short AND carries a code token AND reads like a
# continuation or a changelog row) so ordinary short paragraphs — headings, page
# numbers, bare figures — are left untouched.

# HCPCS (e.g. G0659, J9259, Q5129) or a bare 5-digit CPT (e.g. 81418).
_CODE_TOKEN_RE = re.compile(r'\b(?:[A-Z]\d{4}|\d{5})\b')
# Revision-log verb paired with a code reference ("Added HCPCS code ...").
_CHANGELOG_RE = re.compile(
    r'(?i)\b(?:add(?:ed)?|remov(?:ed)?|updat(?:ed)?|revis(?:ed)?|'
    r'delet(?:ed)?|replac(?:ed)?|correct(?:ed)?)\b[^.\n]*\b(?:hcpcs|cpt|code)\b'
)
# Continuation-style opening: a mid-sentence *lowercase* start, an opening
# bracket/paren, or a lowercase connector word — i.e. text that plainly began on
# the previous line (a wrapped "...and G0659.", "(Genemarkers): 81418",
# "care  99503"). The bare-letter branch is intentionally case-SENSITIVE:
# uppercase-initial short lines are self-contained entries (addresses, ZIP codes,
# "Deny CON #10715.", bare "S5100" codes), not severed continuations.
_CONTINUATION_START_RE = re.compile(r'^(?:[a-z(\[]|(?:and|or|to|through|thru)\b)')

# A stripped paragraph shorter than this (chars) is a candidate for gluing.
_GLUE_MAX_FRAGMENT_LEN = 40


def _is_code_list_fragment(text: str) -> bool:
    """True if ``text`` is a short, code-bearing continuation/changelog fragment
    that the blank-line paragraph split severed from its row."""
    t = (text or "").strip()
    if not t or len(t) >= _GLUE_MAX_FRAGMENT_LEN:
        return False
    if not _CODE_TOKEN_RE.search(t):
        return False
    return bool(_CHANGELOG_RE.search(t)) or bool(_CONTINUATION_START_RE.match(t))


def _glue_code_list_fragments(
    paragraphs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge short code/changelog fragments back onto the preceding paragraph of
    the same section, keeping the code glued to its row.

    Offsets are preserved: the merged paragraph keeps the first paragraph's
    ``start_offset`` and takes the fragment's ``end_offset`` (a superset span in
    the original markdown). ``paragraph_index`` is re-numbered after merging.

    Only merges *backward* into an existing same-section predecessor; a fragment
    with no such predecessor (rare) is left as its own paragraph.
    """
    merged: List[Dict[str, Any]] = []
    for para in paragraphs:
        if (
            merged
            and _is_code_list_fragment(para["text"])
            and merged[-1]["section_path"] == para["section_path"]
        ):
            prev = merged[-1]
            prev["text"] = f"{prev['text']}\n{para['text']}"
            prev["end_offset"] = para["end_offset"]
        else:
            merged.append(para)
    for idx, para in enumerate(merged):
        para["paragraph_index"] = idx
    return merged


def split_paragraphs_from_markdown(md: str) -> List[Dict[str, Any]]:
    """
    Split markdown string into paragraphs with section context.
    Offsets are character indices in the original markdown string (body start/end).

    Returns:
        List of dicts with:
        - text: paragraph body text (sent to LLM)
        - paragraph_index: index within the page
        - section_path: section title (e.g. "Contact information") or None
        - start_offset: character index in md where this paragraph's body starts
        - end_offset: character index in md after the last character of this paragraph
    """
    if not md or not md.strip():
        return []
    cleaned_paragraphs: List[Dict[str, Any]] = []
    # Split by ## at line start (capturing group so we get alternating: content, "## X", content, ...)
    section_parts = re.split(r'(?m)^(## .+)$', md)
    current_pos = 0
    section_path: str | None = None
    for i, part in enumerate(section_parts):
        part_stripped = part.strip()
        if part_stripped.startswith('## '):
            section_path = part_stripped[3:].strip()
            current_pos += len(part)
            continue
        block = part
        if not block.strip():
            current_pos += len(part)
            continue
        boundaries = list(re.finditer(r'\n\s*\n+', block))
        starts = [0] + [m.end() for m in boundaries]
        ends = [m.start() for m in boundaries] + [len(block)]
        for k in range(len(starts)):
            segment = block[starts[k]:ends[k]]
            para = segment.strip()
            if not para:
                continue
            body_start_in_block = len(segment) - len(segment.lstrip())
            start_offset = current_pos + starts[k] + body_start_in_block
            end_offset = start_offset + len(para)
            cleaned_paragraphs.append({
                "text": para,
                "paragraph_index": len(cleaned_paragraphs),
                "section_path": section_path,
                "start_offset": start_offset,
                "end_offset": end_offset,
            })
        current_pos += len(part)
    return _glue_code_list_fragments(cleaned_paragraphs)


def page_to_markdown_blocks(text: str) -> List[Dict[str, str]]:
    """
    Split page text into markdown-style blocks (section + body) for structure-preserving
    extraction or export. Each block has "markdown" (## Section\\n\\nbody) and "section_path".
    """
    paras = split_paragraphs(text)
    blocks = []
    for p in paras:
        section = p.get("section_path") or ""
        body = p.get("text") or ""
        md = f"## {section}\n\n{body}" if section else body
        blocks.append({"markdown": md, "section_path": section, "text": body})
    return blocks


def split_paragraphs(text: str) -> List[Dict[str, any]]:
    """
    Split text into paragraphs, preserving structure.
    Records character offsets in the original text for each paragraph (and body).

    Returns:
        List of dicts with:
        - text: paragraph body text (sent to LLM)
        - paragraph_index: index within the page
        - section_path: detected section path (if any)
        - start_offset: character index in original text where this paragraph's body starts
        - end_offset: character index in original text after the last character of this paragraph
    """
    if not text:
        return []
    text_stripped = text.strip()
    if not text_stripped:
        return []
    # Leading offset: where stripped content starts in original text
    leading_skip = len(text) - len(text.lstrip())
    # Work in stripped space for boundary finding; we'll map back to original
    boundaries = list(re.finditer(r'\n\s*\n+', text_stripped))
    starts = [0]
    for m in boundaries:
        starts.append(m.end())
    ends = [m.start() for m in boundaries] + [len(text_stripped)]

    cleaned_paragraphs = []
    for i in range(len(starts)):
        block_start_s = starts[i]
        block_end_s = ends[i]
        raw_block = text_stripped[block_start_s:block_end_s]
        para = raw_block.strip()
        if not para:
            continue
        block_start_orig = leading_skip + block_start_s
        block_end_orig = leading_skip + block_end_s
        body_start_in_block = len(raw_block) - len(raw_block.lstrip())

        section_path = None
        lines = para.split('\n')
        if len(lines) > 0:
            first_line = lines[0].strip()
            if (len(first_line) < 100 and
                (first_line.endswith(':') or
                 re.match(r'^[A-Z][A-Z\s]+:', first_line) or
                 re.match(r'^\d+\.\s+[A-Z]', first_line))):
                section_path = first_line
                para = '\n'.join(lines[1:]).strip()

        if not para:
            continue
        if len(para) < 80 and '\n' not in para:
            stripped = para.strip()
            if not stripped.endswith('.') and not stripped.endswith(':'):
                if re.match(r'^[\d.]*\s*[A-Z][a-z]*(?:\s+[a-z]+)*\s*$', stripped) or len(stripped.split()) <= 6:
                    continue
        # Body start in original: start of block in original + offset of body within block
        if section_path and lines:
            body_start_in_block += len(lines[0]) + 1
        start_offset = block_start_orig + body_start_in_block
        # End of body in block: body is para, so end of body in block = body_start_in_block + len(para)
        end_in_block = body_start_in_block + len(para)
        end_offset = block_start_orig + end_in_block

        cleaned_paragraphs.append({
            "text": para,
            "paragraph_index": len(cleaned_paragraphs),
            "section_path": section_path,
            "start_offset": start_offset,
            "end_offset": end_offset,
        })
    return cleaned_paragraphs
