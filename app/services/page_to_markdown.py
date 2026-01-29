"""
Convert raw page text (from PDF extraction) to markdown for storage and display.

When a document is uploaded we keep the raw text in document_pages.text and
also produce a structured markdown version in document_pages.text_markdown
so the reader can render headings, sections, and paragraphs properly.
"""
import re
from typing import List, Tuple


def _detect_section_header(first_line: str) -> bool:
    """True if the line looks like a section header (colon, numbered, or short title-like line)."""
    if not first_line or len(first_line) > 120:
        return False
    line = first_line.strip()
    if line.endswith(":"):
        return True
    if re.match(r"^[A-Z][A-Z\s]+:", line):
        return True
    if re.match(r"^\d+\.\s+[A-Z]", line):
        return True
    if re.match(r"^\d+\.\d+\s+", line):
        return True
    # Short title-like line (e.g. "Contact information", "Provider services")
    if len(line) <= 60 and not line.endswith(".") and line and line[0].isupper():
        word_count = len(line.split())
        if 1 <= word_count <= 8:
            return True
    return False


def _blocks_from_raw(raw_text: str) -> List[Tuple[str | None, str]]:
    """
    Split raw page text into (section_header, body) blocks.
    section_header is None for blocks with no detected header.
    Standalone headers get body "" so we can still render them as ## in markdown.
    """
    if not raw_text or not raw_text.strip():
        return []
    paragraphs = re.split(r"\n\s*\n+", raw_text.strip())
    blocks: List[Tuple[str | None, str]] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        lines = para.split("\n")
        first_line = lines[0].strip() if lines else ""
        if _detect_section_header(first_line):
            header = first_line.rstrip(":")
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
            blocks.append((header, body))
        else:
            blocks.append((None, para))
    return blocks


def raw_page_to_markdown(raw_text: str) -> str:
    """
    Convert raw page text to a single markdown string for storage/display.

    - Section headers become ## Header
    - Paragraphs with a detected header become ## Header\\n\\nbody
    - Plain paragraphs are left as-is
    - Standalone headers (e.g. "Benefits and covered services") become ## Title
    """
    blocks = _blocks_from_raw(raw_text or "")
    if not blocks:
        return (raw_text or "").strip()
    out: List[str] = []
    for header, body in blocks:
        if header:
            out.append(f"## {header}")
            if body:
                out.append("")
                out.append(body)
            out.append("")
        else:
            if body:
                out.append(body)
                out.append("")
    return "\n".join(out).strip()
