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
        # Attempt PDF heuristic, then UTF-8 decode
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            try:
                parts = [page.get_text() for page in doc if page.get_text().strip()]
                return "\n\n".join(parts)
            finally:
                doc.close()
        except Exception:
            return content.decode("utf-8", errors="replace")


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
