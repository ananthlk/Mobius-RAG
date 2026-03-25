"""Render consolidated page markdown into a simple PDF for users who need a file, not .md."""

from __future__ import annotations

import html
import logging
import re
from io import BytesIO

logger = logging.getLogger(__name__)

# Avoid huge memory use on pathological documents
_MAX_MARKDOWN_CHARS = 450_000


def markdown_to_pdf_bytes(markdown_body: str, *, title: str = "Document") -> bytes:
    """
    Convert GitHub-flavored markdown (tables, fenced code) to a letter-size PDF.

    Raises RuntimeError if the PDF engine fails (caller maps to HTTP 500/503).
    """
    import markdown as md_lib
    from xhtml2pdf import pisa

    raw = (markdown_body or "").strip()
    if len(raw) > _MAX_MARKDOWN_CHARS:
        logger.warning("markdown_to_pdf: truncating body from %d to %d chars", len(raw), _MAX_MARKDOWN_CHARS)
        raw = raw[:_MAX_MARKDOWN_CHARS] + "\n\n… _(truncated for PDF generation)_\n"

    safe_title = html.escape((title or "Document").strip() or "Document", quote=True)
    html_fragment = md_lib.markdown(
        raw,
        extensions=["extra", "tables", "nl2br"],
    )

    # xhtml2pdf expects a full XHTML document
    doc = f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
  <meta charset="utf-8"/>
  <style type="text/css">
    @page {{ size: letter; margin: 1.5cm; }}
    body {{
      font-family: Helvetica, Arial, sans-serif;
      font-size: 10pt;
      line-height: 1.35;
      color: #111827;
    }}
    h1 {{ font-size: 16pt; margin: 0 0 0.6em 0; border-bottom: 1px solid #e5e7eb; padding-bottom: 0.25em; }}
    h2 {{ font-size: 13pt; margin: 1em 0 0.4em 0; }}
    h3 {{ font-size: 11pt; margin: 0.8em 0 0.3em 0; }}
    p {{ margin: 0.35em 0; }}
    ul, ol {{ margin: 0.35em 0 0.35em 1.2em; padding-left: 0.5em; }}
    pre, code {{
      font-family: Courier, monospace;
      font-size: 8.5pt;
      background-color: #f3f4f6;
    }}
    pre {{ padding: 6px; white-space: pre-wrap; word-wrap: break-word; }}
    table {{ border-collapse: collapse; width: 100%; margin: 0.5em 0; }}
    th, td {{ border: 1px solid #d1d5db; padding: 4px 6px; vertical-align: top; }}
    th {{ background-color: #f9fafb; font-weight: bold; }}
    hr {{ border: none; border-top: 1px solid #e5e7eb; margin: 1em 0; }}
    blockquote {{ margin: 0.5em 0; padding-left: 0.75em; border-left: 3px solid #d1d5db; color: #374151; }}
  </style>
</head>
<body>
  <h1>{safe_title}</h1>
  {html_fragment}
</body>
</html>"""

    buf = BytesIO()
    status = pisa.CreatePDF(doc, dest=buf, encoding="utf-8")
    if status.err:
        logger.error("xhtml2pdf failed: err=%s", status.err)
        raise RuntimeError("PDF generation failed")
    out = buf.getvalue()
    if not out:
        raise RuntimeError("PDF generation produced empty output")
    return out


def safe_pdf_filename(name: str) -> str:
    base = (name or "document").strip().replace(" ", "_")
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base).strip("._") or "document"
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    return base[:180]
