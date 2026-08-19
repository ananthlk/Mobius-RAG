import re
import fitz  # PyMuPDF
from google.cloud import storage
from bs4 import BeautifulSoup
from app.config import GCS_BUCKET


def html_to_plain_text(html: str) -> str:
    """
    Convert HTML to plain text (strip scripts/styles, get body text).
    Used when importing scraped pages that have only html and no text.
    """
    if not html or not html.strip():
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return "\n".join(chunk for chunk in chunks if chunk)


def extract_text_from_bytes(content: bytes, ext: str) -> str:
    """Extract plain text from raw file bytes. Supports PDF, HTML, TXT.

    Returns a single string (pages joined with double-newline for PDFs).
    Raises ValueError if the format is unsupported or extraction fails.
    """
    ext = (ext or "").lower().strip(".")
    if ext == "pdf":
        doc = fitz.open(stream=content, filetype="pdf")
        try:
            parts = []
            for page in doc:
                t = page.get_text()
                if t.strip():
                    parts.append(t)
            return "\n\n".join(parts)
        finally:
            doc.close()
    elif ext in ("html", "htm"):
        return html_to_plain_text(content.decode("utf-8", errors="replace"))
    elif ext in ("txt", "md", "csv"):
        return content.decode("utf-8", errors="replace")
    else:
        # Some files arrive mislabelled, so a PDF sniff is still worth one try.
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            try:
                parts = [page.get_text() for page in doc if page.get_text().strip()]
                if parts:
                    return "\n\n".join(parts)
            finally:
                doc.close()
        except Exception:
            pass
        # NO blind UTF-8 decode. The old fallback turned every binary .xls into
        # mojibake and called it text — 149 documents whose real problem was a
        # missing parser, reported as content. Say what is true instead.
        raise ValueError(f"unsupported_format: no parser for .{ext}")




# ── Ingest failure classification ────────────────────────────────────────────
#
# Every failure gets a TECHNICAL reason: a fact about the file, decided here,
# independent of who sent it or whether anyone wants it. The keep/discard policy
# sits on top as a separate decision.
#
# This exists because 161 documents reached status='failed' with zero rows in
# processing_errors and no error column on `documents` — the system recorded THAT
# they failed and never WHY. Nothing could retry them, because a retry policy
# needs a reason to condition on. The pile turned out to be 149 legacy .xls files
# with no parser: a PARSER gap presenting as a retry gap, invisible for exactly as
# long as the reason went unrecorded.

# Retrying these is free information — the next attempt may well succeed.
RETRYABLE_REASONS = {"fetch_timeout", "upstream_error", "parser_crashed"}
# Retrying these produces the identical failure and hides the real fix.
TERMINAL_REASONS = {"unsupported_format", "no_text_layer", "encrypted",
                    "empty_file", "text_below_threshold", "bad_storage_path"}
MAX_INGEST_ATTEMPTS = 3

# Formats with a real parser above. Anything else is unsupported_format — said
# plainly rather than UTF-8 decoded into mojibake, which is what the old
# catch-all did to every .xls file in the corpus.
SUPPORTED_EXTS = {"pdf", "html", "htm", "txt", "md", "csv"}
MIN_USEFUL_CHARS = 200
EMPTY_FILE_BYTES = 2048


def classify_ingest_failure(content: bytes | None, ext: str,
                            text: str | None, error: BaseException | None = None
                            ) -> tuple[str | None, str | None]:
    """Return (reason, message), or (None, None) when the extraction is usable.

    Order matters: transport failures first (we never saw the bytes), then
    format, then content. A file we could not fetch must not be classified by
    guessing at its type.
    """
    ext_l = (ext or "").lower().strip(".")
    if error is not None:
        msg = f"{type(error).__name__}: {error}"[:500]
        low = msg.lower()
        # Transport first — a file we never received cannot be judged by type.
        if any(k in low for k in ("timeout", "timed out", "deadline", "connection reset")):
            return "fetch_timeout", msg
        # A parse error on a format we have no parser for is NOT a crash, and the
        # difference decides whether we retry. The first run of this classifier
        # put 158 .xls files in `parser_crashed` because the extractor's own
        # "unsupported_format" ValueError fell through to the generic branch —
        # which would have retried 149 unparseable files three times each.
        if "unsupported_format" in low or (content is not None and ext_l not in SUPPORTED_EXTS):
            return "unsupported_format", msg
        if any(k in low for k in ("503", "502", "500", "rate limit", "unavailable",
                                  "quota", "too many requests")):
            return "upstream_error", msg
        # A storage path we cannot even parse is a DATA defect, not a transient
        # one. Found by reading the first apply back: nine documents classified
        # `parser_crashed` (retryable) all said "Bucket names must start and end
        # with a number or letter" — a malformed file_path. Retrying deterministic
        # bad data is the same waste as retrying an unsupported format.
        if any(k in low for k in ("bucket name", "invalid bucket", "not found: gs://",
                                  "no such object", "blob", "404")):
            return "bad_storage_path", msg
        if any(k in low for k in ("password", "encrypted", "cannot authenticate")):
            return "encrypted", msg
        return "parser_crashed", msg

    ext = (ext or "").lower().strip(".")
    if content is not None and len(content) < EMPTY_FILE_BYTES and not (text or "").strip():
        return "empty_file", f"{len(content)} bytes, no extractable content"
    if ext not in SUPPORTED_EXTS:
        return "unsupported_format", f"no parser for .{ext}"

    body = (text or "").strip()
    if not body:
        # A PDF that parsed cleanly and yielded nothing is a scan. Distinguishing
        # it from a broken file matters: one needs OCR, the other needs a re-fetch.
        if ext == "pdf" and content:
            try:
                import fitz
                doc = fitz.open(stream=content, filetype="pdf")
                try:
                    if any(page.get_images() for page in doc):
                        return "no_text_layer", f"{doc.page_count} pages, images only, no text"
                finally:
                    doc.close()
            except Exception:
                pass
        return "no_text_layer", "parsed without error but produced no text"
    if len(body) < MIN_USEFUL_CHARS:
        return "text_below_threshold", f"only {len(body)} characters extracted"
    return None, None


def should_retry(reason: str | None, attempts: int) -> bool:
    """Retry only what a retry can fix, and only to the cap.

    Blind retry would burn three attempts on every unsupported format in the
    corpus and still not surface the missing parser.
    """
    if reason is None or reason in TERMINAL_REASONS:
        return False
    return reason in RETRYABLE_REASONS and attempts < MAX_INGEST_ATTEMPTS


def split_into_paragraphs(text: str) -> list[dict]:
    """Split text into paragraph dicts (text, paragraph_index, page_number, section_path).

    Splits on double-newline boundaries; trims noise.  Used by org-doc ingest
    where full Path-B hierarchical chunking is not needed.
    """
    raw = re.split(r"\n\s*\n+", text.strip())
    paras = []
    for i, p in enumerate(raw):
        p = p.strip()
        if len(p) < 20:
            continue
        paras.append({
            "text": p,
            "paragraph_index": i,
            "page_number": 0,
            "section_path": "",
        })
    return paras


async def extract_text_from_gcs(gcs_path: str) -> list[dict]:
    """
    Extract text from PDF stored in GCS, page by page.
    Returns list of {page_number, text, extraction_status, extraction_error, text_length} dicts.
    """
    # Download file from GCS to memory
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    
    # Extract blob path from gcs_path (gs://bucket/path or gs://bucket/path/to/file.pdf)
    prefix = f"gs://{GCS_BUCKET}/"
    if gcs_path.startswith(prefix):
        blob_path = gcs_path[len(prefix):].lstrip("/")
    else:
        blob_path = gcs_path.split("/")[-1]
    blob = bucket.blob(blob_path)
    
    # Download to memory
    pdf_bytes = blob.download_as_bytes()
    
    # Extract text page by page with error tracking
    pages = []
    doc = None
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            page_data = {
                "page_number": page_num + 1,  # 1-indexed
                "text": None,
                "extraction_status": "failed",
                "extraction_error": None,
                "text_length": 0,
            }
            
            try:
                page = doc[page_num]
                text = page.get_text()
                text_length = len(text.strip())
                
                if text_length == 0:
                    page_data["extraction_status"] = "empty"
                    page_data["extraction_error"] = "No text found on this page (may be image-only or blank)"
                else:
                    page_data["extraction_status"] = "success"
                    page_data["text"] = text
                    page_data["text_length"] = text_length
                    
            except Exception as e:
                page_data["extraction_error"] = f"Error extracting text: {str(e)}"
                page_data["extraction_status"] = "failed"
            
            pages.append(page_data)
    
    except Exception as e:
        # If we can't even open the PDF, return error for all pages
        raise Exception(f"Failed to open PDF: {str(e)}")
    
    finally:
        if doc:
            doc.close()
    
    return pages
