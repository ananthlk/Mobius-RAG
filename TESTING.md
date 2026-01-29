# Testing Guide

## Prerequisites

1. **Install Python deps** (when network is available):
   ```bash
   cd "/Users/ananth/Mobius RAG"
   source .venv/bin/activate
   pip install pymupdf
   # For unit tests:
   pip install pytest pytest-asyncio httpx
   ```

2. **Migrations (create DB + tables)**:
   ```bash
   brew services start postgresql   # if needed
   cd "/Users/ananth/Mobius RAG"
   ./mragm
   ```
   Or: `createdb mobius_rag` then `python -m app.init_db`.

3. **Frontend test deps** (optional, for `npm test`):
   ```bash
   cd frontend && npm install
   ```

## Testing Steps

### 1. Start the Backend Server

```bash
cd "/Users/ananth/Mobius RAG"
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start the Frontend (in another terminal)

```bash
cd "/Users/ananth/Mobius RAG/frontend"
npm run dev
```

### 3. Test Upload and Extraction

1. Open http://localhost:5173 in your browser
2. Upload a PDF file
3. Watch the status updates:
   - "uploaded" → "extracting" → "completed"
4. Check the page statistics and problematic pages

### 4. Check Status via API

```bash
# Get document status (replace DOCUMENT_ID with actual ID from upload response)
curl http://localhost:8000/documents/DOCUMENT_ID/status | python -m json.tool
```

### 5. Test Extraction Directly

```bash
# Test extraction from a file already in GCS
python test_extraction.py gs://mobius-rag-uploads-mobiusos/your-file.pdf
```

### 6. Unit tests

**Backend** (from project root; install `pytest`, `pytest-asyncio`, `httpx` first; Postgres running for chunking/results test):

```bash
pip install pytest pytest-asyncio httpx
./mragt
# or: python -m pytest tests -v
```

**Frontend** (install deps first: `cd frontend && npm install`):

```bash
cd frontend && npm test
```

## What to Look For

### Successful Extraction
- Status shows "completed"
- Page statistics show successful pages
- No problematic pages listed

### Empty Pages
- Pages with no text (image-only or blank)
- Status: "empty"
- Error message: "No text found on this page (may be image-only or blank)"

### Failed Pages
- Pages that failed to extract
- Status: "failed"
- Error message with specific reason

## Troubleshooting

### Database Connection Issues
- Check `.env` file has correct DATABASE_URL
- Verify Postgres is running: `pg_isready`
- Check database exists: `psql -l | grep mobius_rag`

### Import Errors
- Make sure PyMuPDF is installed: `pip list | grep pymupdf`
- Verify all dependencies: `pip install -r requirements.txt` (if you create one)

### GCS Access Issues
- Verify GCS credentials are set up
- Check bucket name in `.env` matches actual bucket
- Test GCS access: `gcloud storage ls gs://mobius-rag-uploads-mobiusos/`
