# Installation and Testing Checklist

## Current Status

✅ **Code is complete and ready** - All functionality for Step 6 (PDF extraction with error tracking) is implemented.

## Required Installations (when network is available)

### 1. Install Python Dependencies

```bash
cd "/Users/ananth/Mobius/mobius-rag"
source .venv/bin/activate

# Install all required packages
pip install sqlalchemy asyncpg python-dotenv pymupdf
```

### 2. Set Up Database

```bash
# Start Postgres (if not running)
brew services start postgresql

# Create database
createdb mobius_rag

# Initialize tables (documents, document_pages, chunking_results, etc.)
python app/init_db.py
```

Or run **`./mragm`** (after `./setup-mrag-cli.sh`): creates DB if missing, then `init_db`.

**If `brew services start postgresql` fails with "Bootstrap failed: 5" / "Input/output error":**

1. **Check if Postgres is already running:**  
   `pg_isready -h localhost` or `psql -l`. If it works, you can skip `brew services` and run `./mragm` / `createdb` / `init_db` as needed.

2. **Restart the service** (often fixes a stuck launchd state):  
   `brew services restart postgresql@14`  
   (Use `postgresql@14` if that’s what you have; otherwise `postgresql`.)

3. **Stop, then start:**  
   `brew services stop postgresql@14`  
   `brew services start postgresql@14`

4. **Run Postgres manually** (no launchd):  
   `pg_ctl -D /opt/homebrew/var/postgresql@14 start`  
   (Paths differ: Apple Silicon often uses `/opt/homebrew/var/...`, Intel `/usr/local/var/...`.)

5. **Reinit data dir** (only if you don’t need existing DBs):  
   `rm -rf /opt/homebrew/var/postgresql@14`  
   `initdb -D /opt/homebrew/var/postgresql@14 --locale=C -E UTF8`  
   Then `brew services start postgresql@14` or `pg_ctl` as above.

### 3. Embedding worker (optional, after chunking)

The embedding worker runs after chunking completes and stores vectors for top-k search.

1. **Install pgvector** in PostgreSQL (required for embedding storage):
   - macOS: `brew install pgvector`
   - See: https://github.com/pgvector/pgvector#installation

2. **Run migration**:  
   `python -m app.migrations.add_embedding_tables`

3. **Configure embedding provider** (in `.env`):
   - OpenAI: `EMBEDDING_PROVIDER=openai`, `OPENAI_API_KEY=...`, `EMBEDDING_MODEL=text-embedding-3-small`
   - Vertex: `EMBEDDING_PROVIDER=vertex`, `VERTEX_PROJECT_ID=...`, `GOOGLE_APPLICATION_CREDENTIALS=...`

4. **Start embedding worker**:  
   `./mrage` (or `python -m app.embedding_worker`)

5. **Optional – Chroma** (vector DB for top-k across documents):  
   Set `CHROMA_HOST` and `CHROMA_PORT` to point at a Chroma server, or `CHROMA_PERSIST_DIR` for local persistence.

### 4. Ollama setup (for chunking & extraction)

Chunking and fact extraction use a local Ollama model by default.

1. Install and run [Ollama](https://ollama.com).
2. Pull the default model:  
   `ollama pull llama3.1:8b`
3. Optionally set in `.env`:  
   `OLLAMA_MODEL=llama3.1:8b`  
   `OLLAMA_NUM_PREDICT=8192`  
   Use a model name that matches what you pulled.

### 5. Verify Installation

```bash
# Check imports work
python -c "from app.models import Document, DocumentPage; print('✓ Models OK')"
python -c "from app.services.extract_text import extract_text_from_gcs; print('✓ Extraction service OK')"

# Ensure Ollama is running and model is available (e.g. llama3.1:8b)
ollama list
ollama run llama3.1:8b "Hi"  # optional quick sanity check
```

## Quick: Migrations, build & test

```bash
cd "/Users/ananth/Mobius/mobius-rag"

# 1. Migrations (Postgres must be running)
./mragm

# 2. Frontend build
cd frontend && npm run build && cd ..

# 3. Backend unit tests (pip install pytest pytest-asyncio httpx first)
./mragt

# 4. Frontend unit tests (npm install in frontend first)
cd frontend && npm test
```

## Quick start: mragb and mragf

From any directory you can run:

- **`mragb`** — start backend (http://localhost:8000)
- **`mragf`** — start frontend (http://localhost:5173)
- **`mragm`** — run migrations (create DB + init tables)
- **`mragt`** — run backend unit tests (pytest)

**One-time setup** (so these are on your PATH):

```bash
cd "/Users/ananth/Mobius/mobius-rag"
./setup-mrag-cli.sh
```

Then add `~/bin` to PATH if prompted (e.g. `export PATH="$HOME/bin:$PATH"` in `~/.zshrc` and `source ~/.zshrc`).

**Restart:** stop with Ctrl+C in each terminal, then run `mragb` or `mragf` again.

---

## Testing the Full Flow

### Step 1: Start Backend

```bash
mragb
```

Or manually:

```bash
cd "/Users/ananth/Mobius/mobius-rag"
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 2: Start Frontend (new terminal)

```bash
mragf
```

Or manually:

```bash
cd "/Users/ananth/Mobius/mobius-rag/frontend"
npm run dev
```

### Step 3: Test Upload

1. Open http://localhost:5173
2. Upload a PDF file
3. Watch status updates in real-time:
   - Status changes: `uploaded` → `extracting` → `completed`
   - Page statistics appear
   - Problematic pages (if any) are listed with details

### Step 4: Check API Response

After upload, check the status endpoint:

```bash
# Replace DOCUMENT_ID with the ID from upload response
curl http://localhost:8000/documents/DOCUMENT_ID/status | python -m json.tool
```

Expected response includes:
- `pages_summary`: total, successful, failed, empty counts
- `problematic_pages`: array of pages with issues, including:
  - `page_number`
  - `status` (failed/empty)
  - `error` (error message)
  - `text_length`

## What Was Implemented

### Backend Features
- ✅ PDF text extraction page-by-page using PyMuPDF
- ✅ Error tracking per page (success, failed, empty)
- ✅ Status updates during processing (uploaded → extracting → completed)
- ✅ Detailed error messages for problematic pages
- ✅ Database storage of extracted pages with error info

### Frontend Features
- ✅ Real-time status polling (every 2 seconds)
- ✅ Page statistics display
- ✅ Problematic pages list with error details
- ✅ Color-coded status indicators

### Error Tracking
- ✅ Tracks which pages extracted successfully
- ✅ Identifies empty pages (no text found)
- ✅ Captures and displays extraction errors
- ✅ Shows text length for each page

## Example Test Output

When you upload a PDF, you should see:

```
✓ Text extraction complete! 45 successful, 2 empty, 1 failed out of 48 pages.

⚠️ Pages with issues (3):
  • Page 23: empty
    No text found (may be image-only or blank page)
  • Page 47: failed
    Error extracting text: [specific error message]
```

## Next Steps After Testing

Once everything works:
- Step 7: Filter to only eligibility-relevant paragraphs
- Step 8: Extract ONE fact type (eligibility_verification_method)
- And continue with the plan...
