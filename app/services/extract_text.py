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
