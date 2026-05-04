# Local probe of corpus_search_agent

How to run the 17-query smoke test against the new RAG-as-agent.

## What you're testing

Strategy 1 of the new architecture: classifier → selectivity scoring →
term partition (REQUIRED / BOOSTED / DROP) → candidate-pool intersection
→ adaptive sub-strategy loop. Each query trace shows what the agent
*decided* about each term, what pool it built, and which sub-strategies
ran.

## Local setup

You need the rag service pointed at the dev Cloud SQL postgres + dev
embeddings. One-time per machine:

```bash
gcloud auth application-default login
gcloud auth login
```

Two ways to point at the dev DB:

### Option A — Cloud SQL Auth Proxy (recommended)

```bash
# in a tmux pane / separate terminal
cloud-sql-proxy mobius-os-dev:us-central1:mobius-platform-dev-db --port 5433
```

Then in your rag-service shell:

```bash
cd ~/Mobius/mobius-rag
DB_PASSWORD=$(gcloud secrets versions access latest \
  --secret=db-password --project=mobius-os-dev | python3 -c \
  "import urllib.parse, sys; print(urllib.parse.quote(sys.stdin.read().strip()))")

export DATABASE_URL="postgresql+asyncpg://postgres:${DB_PASSWORD}@127.0.0.1:5433/mobius_rag"
export EMBEDDING_PROVIDER=vertex
export VERTEX_PROJECT_ID=mobius-os-dev
export VERTEX_LOCATION=us-central1
export VERTEX_MODEL=gemini-2.5-flash
export VECTOR_STORE=pgvector
export ENV=local

# Skip the chat-side stuff that the rag service tries to connect to
unset CHAT_DATABASE_URL
unset CHROMA_HOST

uvicorn app.main:app --reload --port 8001
```

### Option B — point straight at deployed dev mobius-rag

If you don't want a local rag, hit the deployed instance:

```bash
BASE_URL=https://mobius-rag-ortabkknqa-uc.a.run.app \
  python3 scripts/probe_search_agent.py
```

(The new endpoint is already deployed if you ran the latest deploy
script. If not, deploy first: `bash deploy/deploy_cloudrun_dev.sh`)

## Run the probe

```bash
cd ~/Mobius/mobius-rag
python3 scripts/probe_search_agent.py
```

Output:
- Per-query trace block on stdout (what was classified, which terms went
  into REQUIRED/BOOSTED/DROP, candidate pool size, each sub-strategy's
  outcome, top 3 returned chunks, confidence, hint)
- `/tmp/probe_search_agent_<ts>.csv` summary table for cross-query view

## What to look for in the trace

- `[trace:classify]` — does query_type match the user's intent? Did
  the lexicon catch the right tags? Were literal anchors (FL.UM.51,
  H0019, etc.) detected?
- `[trace:partition]` — REQUIRED bucket should have ONLY high-
  selectivity terms (j-tags, literal anchors). BOOSTED should have
  meaningful tags + content nouns. DROP should catch "providers,"
  "rules," etc.
- `[trace:pool]` — `pool_size` of 5–500 is the sweet spot. If it's 0
  or 5000, something's off (intersection too narrow or too broad).
  `relaxed=True` means we had to drop a required tag because the
  intersection was empty/tiny.
- `[trace:order]` — should match the QueryType policy:
  PRECISION_DOMINANT → phrase_strict first; CONCEPTUAL → hybrid first;
  VAGUE → vector_broad first.
- Per-strategy lines — succeeded / failed and the *note* explaining
  why. The note is what we'll calibrate thresholds against.

## When you spot issues

- **Wrong classification**: tweak coverage thresholds in
  `corpus_search_agent.classify_query`.
- **Term in DROP that shouldn't be**: extend `_VERY_GENERIC_UNTAGGED`
  set, or check why the lexicon didn't tag it.
- **Pool too narrow** (e.g., `j:` ∩ `d:` = 0 docs): bump
  `_SELECTIVITY_REQUIRED` so the d-tag gets BOOSTED instead of REQUIRED.
- **Pool too broad** (>1000 docs): bump `_SELECTIVITY_BOOST` so more
  tags are required.
- **Strategy success threshold off**: tweak `_HYBRID_RERANK_HIGH`,
  `_VECTOR_NEW_DOC_MIN`, `_PHRASE_TOKEN_COVERAGE_HIGH`.

After each tweak, rerun the probe and diff the CSV.
