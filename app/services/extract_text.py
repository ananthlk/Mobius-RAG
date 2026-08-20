import re
import os
import fitz  # PyMuPDF
from google.cloud import storage
from bs4 import BeautifulSoup
from app.config import GCS_BUCKET


class ExtractionError(Exception):
    """Base for extraction failures. Typed so classify_ingest_failure keys on the
    class rather than pattern-matching an arbitrary message."""


class UnsupportedFormat(ExtractionError):
    pass


class BadStoragePath(ExtractionError):
    pass


class SourceNotStored(ExtractionError):
    """file_path points at a URL or local path — the bytes were never archived."""


class PdfOpenFailed(ExtractionError):
    pass


class FileTooLarge(ExtractionError):
    pass


# A single document should never be able to exhaust the worker. download_as_bytes
# pulls the whole object into memory with no ceiling; one 2 GB scan would take the
# process down and look like an unrelated crash.
# Table capture. OFF by default: it REWRITES page text (tables excised to a
# breadcrumb), so it changes what gets chunked, embedded and — because page text
# is the input to the normalized-md5 that duplicate determination rests on — what
# the dedup gate compares. Enabled per-run for the stage-2 milestone only.
# Minimum characters a block must reach before html_to_plain_text emits a
# paragraph break. Tuned on real AHCA pages: 400 keeps nav/link lists together
# while letting real prose paragraphs stand alone.
_MIN_BLOCK_CHARS = 400

TABLE_CAPTURE = os.getenv("TABLE_CAPTURE", "").lower() in ("1", "true", "on", "yes")

MAX_DOWNLOAD_BYTES = 300 * 1024 * 1024


def parse_gcs_path(gcs_path: str, bucket_name: str) -> str:
    """Blob name within `bucket_name`, or raise.

    STRICT on purpose. The previous fallback was
        blob_path = gcs_path.split("/")[-1]
    which silently mapped `gs://OTHER-BUCKET/secret/policy.pdf` to `policy.pdf` in
    OUR bucket — reading a different document's bytes and attaching its text to
    this row, with no error anywhere. It did the same to URLs
    (`https://example.com/page` -> blob `page`). A path we cannot resolve must
    fail loudly; guessing at it is how one document ends up holding another's
    content.
    """
    if not gcs_path or not gcs_path.strip():
        raise BadStoragePath("empty file_path")
    if not gcs_path.startswith("gs://"):
        raise SourceNotStored(f"file_path is not a GCS object: {gcs_path[:120]}")
    rest = gcs_path[len("gs://"):]
    if "/" not in rest:
        raise BadStoragePath(f"no object name in path: {gcs_path[:120]}")
    bucket, blob = rest.split("/", 1)
    if bucket != bucket_name:
        raise BadStoragePath(
            f"path points at bucket {bucket!r}, not {bucket_name!r}: {gcs_path[:120]}")
    if not blob.strip("/"):
        raise BadStoragePath(f"empty object name: {gcs_path[:120]}")
    return blob.lstrip("/")


def html_to_plain_text(html: str) -> str:
    """
    Convert HTML to plain text (strip scripts/styles, get body text).
    Used when importing scraped pages that have only html and no text.
    """
    if not html or not html.strip():
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:                    # malformed markup, recursion limits
        raise ExtractionError(f"html parse failed: {type(e).__name__}: {e}") from e
    for tag in soup(["script", "style"]):
        tag.decompose()

    # PARAGRAPH BREAKS ARE LOAD-BEARING, and this used to emit none.
    #
    # The old implementation joined every line with a SINGLE newline and dropped
    # empties, so the output could not contain a blank line by construction. The
    # chunker splits paragraphs on `\n\s*\n+` — blank lines — so an entire HTML
    # page collapsed into ONE chunk. Measured on a real AHCA fetch (2026-08-20):
    # a 14,226-character page produced exactly 1 chunk of 14,226 characters, and
    # 3 scraped pages produced 3 chunks between them. Corpus-wide the signature is
    # 797 chunks over 8,000 characters across 38 documents.
    #
    # Two consequences, and the second is why it was invisible: retrieval
    # granularity is destroyed (a whole page is one citation, so nothing can be
    # cited precisely), and the mega-chunk then exceeds the embedder's input
    # limit, so the document embeds to NOTHING and never reaches the index. The
    # embedding job still reports "completed".
    #
    # Fix: insert a real blank line at block-level boundaries, which is where a
    # paragraph break actually belongs in HTML. Inline tags are left alone so a
    # <b> or <a> mid-sentence does not fragment it.
    BLOCK = ("p", "div", "section", "article", "li", "tr", "br",
             "h1", "h2", "h3", "h4", "h5", "h6",
             "blockquote", "pre", "table", "ul", "ol", "header", "footer")
    for tag in soup.find_all(BLOCK):
        tag.append(soup.new_string("\n\n"))

    text = soup.get_text(separator="\n")
    # Normalise within a paragraph, but PRESERVE the blank lines between them.
    blocks, para = [], []
    for raw in text.splitlines():
        line = " ".join(raw.split())          # collapse runs of whitespace
        if line:
            para.append(line)
        elif para:
            blocks.append("\n".join(para)); para = []
    if para:
        blocks.append("\n".join(para))

    # COALESCE SHORT BLOCKS, or we trade one failure for its mirror image.
    # Breaking at every block tag turns a nav menu into one chunk per link: the
    # first version of this fix took a 14,226-character page from 1 chunk to 535
    # with a MEAN of 25 characters. A 25-character chunk retrieves as badly as a
    # 14,000-character one, just for the opposite reason — it carries no context
    # to match a query against.
    #
    # So a paragraph break is only emitted once enough text has accumulated to be
    # worth retrieving on its own. Short runs (list items, link lists, table
    # cells) join into one block; a genuinely substantial block still breaks
    # immediately after it.
    merged, buf = [], []
    for b in blocks:
        buf.append(b)
        if sum(len(x) for x in buf) >= _MIN_BLOCK_CHARS:
            merged.append("\n".join(buf)); buf = []
    if buf:
        merged.append("\n".join(buf))
    return "\n\n".join(merged)


# Site chrome. Stripping these is the difference between storing a policy and
# storing a navigation menu: on a content-heavy page the nav is a rounding error,
# but on a thin page it is 95% of the text. 161 scraped pages in this corpus
# stored ~2,800 characters each of which the real content was a title and a date —
# "Home / Vision and Shared Purpose / Board of Directors / Meet the Team / ..."
# repeated identically across every page of the site. Indexing that adds hundreds
# of near-identical documents that match nothing anyone asks for.
_CHROME_TAGS = ("nav", "header", "footer", "aside", "form", "noscript",
                "script", "style", "svg", "button")
_CHROME_HINT = re.compile(
    r"(^|[-_ ])(nav|menu|breadcrumb|sidebar|side-bar|footer|header|masthead|"
    r"cookie|banner|skip|social|share|subscribe|newsletter|pagination|widget)([-_ ]|$)",
    re.I)
MIN_MAIN_CONTENT_CHARS = 200


def extract_main_content(html: str) -> tuple[str, str]:
    """(main_text, whole_text). Main content with site chrome removed.

    Prefers an explicit <main>/<article> when the page declares one, since that is
    the page telling us where its content is. Otherwise strips chrome tags and any
    element whose class or id names itself as navigation.
    """
    if not html or not html.strip():
        return "", ""
    whole = html_to_plain_text(html)

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as e:
        raise ExtractionError(f"html parse failed: {type(e).__name__}: {e}") from e
    for tag in soup(list(_CHROME_TAGS)):
        tag.decompose()
    # decompose() detaches nodes, so a later match can hold a already-freed
    # element whose .attrs is gone. Snapshot the list and guard each access.
    for el in list(soup.find_all(True)):
        if el.decomposed or not getattr(el, "attrs", None):
            continue
        cls = " ".join(el.attrs.get("class") or [])
        eid = str(el.attrs.get("id") or "")
        role = str(el.attrs.get("role") or "").lower()
        if (_CHROME_HINT.search(cls) or _CHROME_HINT.search(eid)
                or role in ("navigation", "banner", "contentinfo", "search")):
            el.decompose()

    node = soup.find("main") or soup.find("article") or soup.body or soup
    text = node.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    main = "\n".join(ln for ln in lines if ln)
    return main, whole


def classify_html_content(html: str) -> tuple[str, str | None, str | None]:
    """(main_text, reason, message).

    `boilerplate_only` is its own category rather than being folded into
    text_below_threshold, because the cause and the fix differ: a stub PDF has
    nothing to recover, whereas a nav-only page means the SITE has nothing on it
    and no parser improvement will change that. Keeping them apart also stops
    them being re-fetched forever in the hope that the next attempt finds content.
    """
    main, whole = extract_main_content(html)
    body = main.strip()
    if len(body) >= MIN_MAIN_CONTENT_CHARS:
        return main, None, None
    if len(whole.strip()) >= MIN_MAIN_CONTENT_CHARS:
        return main, "boilerplate_only", (
            f"{len(whole.strip())} chars on the page but only {len(body)} outside "
            f"navigation, header and footer")
    return main, "text_below_threshold", f"only {len(body)} characters of content"


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
        raise UnsupportedFormat(f"unsupported_format: no parser for .{ext}")




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
# `no_stored_file` is retryable, but by a DIFFERENT mechanism: re-fetch the URL,
# not re-extract bytes we never had. Kept in this set so the sweep surfaces it as
# recoverable rather than terminal.
RETRYABLE_REASONS = {"fetch_timeout", "upstream_error", "parser_crashed", "no_stored_file",
                     # the classifier being down is a statement about US, not the
                     # document — it must clear itself when the service returns
                     "classifier_unavailable"}
# Retrying these produces the identical failure and hides the real fix.
TERMINAL_REASONS = {"unsupported_format", "no_text_layer", "encrypted",
                    "empty_file", "text_below_threshold", "bad_storage_path",
                    "boilerplate_only", "file_too_large", "corrupt_file",
                    "classifier_held"}
MAX_INGEST_ATTEMPTS = 3

# Formats with a real parser above. Anything else is unsupported_format — said
# plainly rather than UTF-8 decoded into mojibake, which is what the old
# catch-all did to every .xls file in the corpus.
SUPPORTED_EXTS = {"pdf", "html", "htm", "txt", "md", "csv"}
MIN_USEFUL_CHARS = 200
EMPTY_FILE_BYTES = 2048


def classify_ingest_failure(content: bytes | None, ext: str,
                            text: str | None, error: BaseException | None = None,
                            storage_path: str | None = None
                            ) -> tuple[str | None, str | None]:
    """Return (reason, message), or (None, None) when the extraction is usable.

    Order matters: transport failures first (we never saw the bytes), then
    format, then content. A file we could not fetch must not be classified by
    guessing at its type.
    """
    ext_l = (ext or "").lower().strip(".")
    # A document whose file_path is a URL was never archived: scraped pages put
    # their text straight into document_pages and keep the source URL here. That
    # is NOT a malformed path, and the difference decides the fix — a malformed
    # path needs repair, an un-archived page needs a RE-FETCH from the URL we
    # already hold. 175 documents were about to be filed as corrupt when they are
    # simply not stored.
    if storage_path and not storage_path.startswith("gs://"):
        if error is not None or not (text or "").strip():
            return "no_stored_file", f"file_path is not a GCS object: {storage_path[:120]}"

    if error is not None:
        msg = f"{type(error).__name__}: {error}"[:500]
        low = msg.lower()
        # TYPE first, string second. The string branches below are heuristics over
        # third-party messages and will always be approximate; our own errors carry
        # their meaning in the class, so read that where it exists. String-matching
        # our own exception is how `unsupported_format` was mis-filed as
        # `parser_crashed` and marked 149 unparseable files retryable.
        if isinstance(error, UnsupportedFormat):
            return "unsupported_format", msg
        if isinstance(error, SourceNotStored):
            return "no_stored_file", msg
        if isinstance(error, BadStoragePath):
            return "bad_storage_path", msg
        if isinstance(error, FileTooLarge):
            return "file_too_large", msg
        if isinstance(error, PdfOpenFailed):
            # An unopenable PDF is not retryable — the bytes will not change.
            return "corrupt_file", msg
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
    if not text or not text.strip():
        return []
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
    # Resolve the path STRICTLY — see parse_gcs_path. A path we cannot resolve
    # raises rather than falling back to a basename lookup in our own bucket,
    # which could return a different document's bytes.
    blob_path = parse_gcs_path(gcs_path, GCS_BUCKET)

    # Every step below was previously unguarded: a missing object, a permissions
    # failure or a transport error propagated as a raw google-cloud exception with
    # no classification, which is how failures reached `documents` with no reason
    # recorded against them.
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(blob_path)
        blob.reload()                              # size before we pull it into memory
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():
            raise BadStoragePath(f"object not found: {gcs_path[:120]}") from e
        raise ExtractionError(
            f"could not reach storage for {gcs_path[:120]}: {type(e).__name__}: {e}") from e

    size = blob.size or 0
    if size > MAX_DOWNLOAD_BYTES:
        raise FileTooLarge(
            f"{size:,} bytes exceeds the {MAX_DOWNLOAD_BYTES:,} byte ceiling — "
            f"downloading it would risk the worker rather than one document")

    try:
        pdf_bytes = blob.download_as_bytes()
    except Exception as e:
        raise ExtractionError(
            f"download failed for {gcs_path[:120]}: {type(e).__name__}: {e}") from e
    if not pdf_bytes:
        raise ExtractionError(f"object is empty: {gcs_path[:120]}")
    
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
                    # TABLE CAPTURE (Table Capture program, stage 1).
                    #
                    # THIS is the seam, not main.py. main.py builds DocumentPage in
                    # EIGHT places; six of them source pages from here, and only here
                    # do we still hold the live fitz page — lines, words, bboxes —
                    # which is what line-based table detection needs. By the time
                    # main.py sees a page it has only the string.
                    #
                    # capture_page_tables() is Sourcing's, and pure. The wrapper is
                    # mine and is the whole safety story: ANY failure inside it
                    # degrades to today's behaviour — original text, no tables. A
                    # document must never fail to ingest because a table was hard to
                    # read. Ingest availability outranks table quality.
                    if TABLE_CAPTURE:
                        try:
                            from app.services.table_capture import capture_page_tables
                            text, tables = capture_page_tables(page, text, page_num + 1)
                            page_data["tables"] = tables
                        except Exception as cap_err:
                            logger.warning(
                                "table_capture failed on page %s, falling back to raw text: %s",
                                page_num + 1, cap_err)
                            page_data["tables"] = []
                        # text may have been rewritten (tables excised -> breadcrumb),
                        # so length is recomputed rather than carried from before.
                        text_length = len(text.strip())

                    page_data["extraction_status"] = "success"
                    page_data["text"] = text
                    page_data["text_length"] = text_length
                    
            except Exception as e:
                page_data["extraction_error"] = f"Error extracting text: {str(e)}"
                page_data["extraction_status"] = "failed"
            
            pages.append(page_data)
    
    except Exception as e:
        # Typed, and chained. `raise Exception(...)` discarded both the class and
        # the traceback, so the classifier could only pattern-match a string and
        # every open failure looked the same from the outside.
        raise PdfOpenFailed(f"failed to open PDF {gcs_path[:120]}: "
                            f"{type(e).__name__}: {e}") from e
    
    finally:
        if doc:
            doc.close()
    
    return pages
