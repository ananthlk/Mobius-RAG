#!/usr/bin/env bash
# Deploy Mobius RAG to Cloud Run in mobius-os-dev.
#
# What this script does (all idempotent):
#   1. Build rag Docker image with Cloud Build → Artifact Registry
#   2. Deploy mobius-rag API (uvicorn app.main:app)
#   3. Deploy mobius-rag-chunking-worker (uvicorn app.worker_server_chunking:app)
#   4. Deploy mobius-rag-embedding-worker (uvicorn app.worker_server_embedding:app)
#
# Prerequisites (one-time, already done as of 2026-04-23):
#   * Secret Manager contains mobius-skill-llm-internal-key (shared with chat)
#   * Secret Manager contains rag-admin-api-key
#   * Chat service is patched to MOBIUS_SKILL_LLM_INTERNAL_KEY=<secret>
#   * Runtime SA (mobius-platform-dev) has secretmanager.secretAccessor role
#
# Usage:
#   ./deploy/deploy_cloudrun_dev.sh            # tag with git sha
#   TAG=v7-smoke ./deploy/deploy_cloudrun_dev.sh
set -euo pipefail

PROJECT_ID="mobius-os-dev"
REGION="us-central1"
CLOUD_SQL_INSTANCE="mobius-platform-dev-db"
CLOUD_SQL_CONNECTION="${PROJECT_ID}:${REGION}:${CLOUD_SQL_INSTANCE}"

# Chat URL is the proxy endpoint for rag's LLM calls. Point at
# /internal/skill-llm — chat gates on MOBIUS_SKILL_LLM_INTERNAL_KEY.
CHAT_INTERNAL_LLM_URL="https://mobius-chat-ortabkknqa-uc.a.run.app/internal/skill-llm"

# lexicon-maintenance service — inline candidate cleanup after extraction.
LEXICON_MAINTENANCE_URL="https://mobius-lexicon-maintenance-ortabkknqa-uc.a.run.app"

# DB connection via Cloud SQL Unix socket (Cloud Run connector). Chat
# uses `postgres@` (no password) with the auth-proxy; rag historically
# uses `mobius_app` with a password. Keep the latter for parity with
# staging.
DB_USER="postgres"
DB_PASS_SECRET="db-password"  # Secret Manager secret that has the rag DB password
# Read the password so we can inline it (gcloud run deploy doesn't
# support --set-secrets for URL-embedded passwords).
DB_PASS=$(gcloud secrets versions access latest --secret="$DB_PASS_SECRET" --project="$PROJECT_ID" 2>/dev/null || echo "")
if [[ -z "$DB_PASS" ]]; then
  echo "ERROR: could not read secret $DB_PASS_SECRET. Either create it or edit this script."
  exit 1
fi
# urlencode the '$' in the password manually; we know the password shape.
# URL-encode the password for safe embedding in connection strings
# (the dev password contains a ``$`` that would otherwise break
# shell expansion on the other side of the wire).
DB_PASS_ENC=$(python3 -c "import urllib.parse, sys; print(urllib.parse.quote(sys.argv[1]))" "$DB_PASS")
DB_URL="postgresql+asyncpg://${DB_USER}:${DB_PASS_ENC}@/mobius_rag?host=%2Fcloudsql%2F${PROJECT_ID}%3A${REGION}%3A${CLOUD_SQL_INSTANCE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo "manual-$(date +%s)")}"
REPO="us-central1-docker.pkg.dev/${PROJECT_ID}/mobius-rag"
IMAGE="${REPO}/rag:${TAG}"

echo "=============================================================="
echo "Deploy Mobius RAG → ${PROJECT_ID} (tag=${TAG})"
echo "=============================================================="

# 1. Ensure Artifact Registry repo exists (idempotent)
gcloud artifacts repositories describe mobius-rag \
  --project="$PROJECT_ID" --location="$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create mobius-rag \
    --project="$PROJECT_ID" --location="$REGION" \
    --repository-format=docker --description="Mobius RAG container images" --quiet

# 2a. Frontend: rebuild dist before docker build. The Dockerfile expects
#     ``frontend/dist`` to be present (no node toolchain inside the
#     image to keep size down). Operators were forgetting this step
#     and shipping with stale UI bundles — the 'wizard not visible'
#     debug session on 2026-04-27 was exactly this.
if [[ -d frontend ]] && [[ -f frontend/package.json ]]; then
  echo "--- building frontend dist ---"
  # Vite reads VITE_* env vars at build time and bakes them into the
  # bundle. Without these the URL panel's Submit button posts to a
  # same-origin /scrape (which is the rag service, not the scraper),
  # producing "Method Not Allowed". Default to the dev URLs; operators
  # can override per-call by exporting these before invoking this script.
  : "${VITE_SCRAPER_API_BASE:=https://mobius-web-scraper-ortabkknqa-uc.a.run.app}"
  : "${VITE_API_BASE:=}"   # empty → same-origin (rag-self), correct for the deployed shell
  # Lexicon Maintenance lives on its own Cloud Run service; the header link
  # is the natural path from the RAG dashboard into tag curation.
  : "${VITE_LEXICON_URL:=https://mobius-lexicon-maintenance-ortabkknqa-uc.a.run.app}"
  # DEV ONLY: lets the app self-mint a platform token when a long session
  # outlives the launcher token (avoids /admin/* 401s). Points at a SAME-ORIGIN
  # RAG proxy (/dev/mint-token) that server-side calls chat's mint-dev-token —
  # avoids browser CORS. NEVER set in prod; the proxy is also ENV-gated off there.
  : "${VITE_DEV_MINT_URL:=/dev/mint-token}"
  export VITE_SCRAPER_API_BASE VITE_API_BASE VITE_LEXICON_URL VITE_DEV_MINT_URL
  echo "--- frontend env: VITE_SCRAPER_API_BASE=$VITE_SCRAPER_API_BASE  VITE_API_BASE=${VITE_API_BASE:-<same-origin>}  VITE_LEXICON_URL=$VITE_LEXICON_URL ---"
  (cd frontend && npm run build) || {
    echo "ERROR: frontend build failed; aborting deploy" >&2
    exit 1
  }
  echo "--- frontend dist hash: $(ls frontend/dist/assets/index-*.js 2>/dev/null | head -1) ---"
fi

# 2b. Build image via Cloud Build (uses mobius-rag/.gcloudignore to avoid
#     pushing 1GB of pycache/node_modules).
echo "--- building $IMAGE ---"
gcloud builds submit --project="$PROJECT_ID" --tag="$IMAGE" .

# Common env + flag shape shared by all three services
# NOTE: LLM_PROVIDER removed (Phase 1 gate rejects it as stale).
#
# Chroma + chat Postgres are wired so the publish endpoint can sync
# directly into chat's retrieval stores (see
# app/services/publish_sync.py). Without these, publish still
# succeeds but downstream chat retrieval never sees the doc.
# Contract reference: mobius-chat/docs/rag_population_agent_setup.md.
CHAT_DB_URL_FOR_RAG="postgresql+psycopg2://postgres:${DB_PASS_ENC}@/mobius_chat?host=%2Fcloudsql%2F${PROJECT_ID}%3A${REGION}%3A${CLOUD_SQL_INSTANCE}"

COMMON_ENV=(
  "ENV=staging"
  # DEV ONLY: enables /dev/mint-token so the SPA can self-mint a platform token
  # (long sessions outlive the launcher token → /admin/* 401 → UI blanks). This is
  # the dev deploy script; prod/staging deploys must NOT set this.
  "ALLOW_DEV_MINT=1"
  "DATABASE_URL=${DB_URL}"
  "GCS_BUCKET=mobius-rag-uploads-dev"
  "VERTEX_PROJECT_ID=${PROJECT_ID}"
  "VERTEX_LOCATION=${REGION}"
  "VERTEX_MODEL=gemini-2.5-flash"
  "EMBEDDING_PROVIDER=vertex"
  "CHAT_INTERNAL_LLM_URL=${CHAT_INTERNAL_LLM_URL}"
  "LEXICON_MAINTENANCE_URL=${LEXICON_MAINTENANCE_URL}"
  # Vector store — pgvector is the prod backend post-cutover (2026-04-27).
  # Without this, vector_store.get_vector_store() falls back to Chroma when
  # CHROMA_HOST is set, silently re-routing to the unstable Chroma VM.
  "VECTOR_STORE=pgvector"
  # Chroma kept as legacy fallback for ad-hoc admin /vector_search?store=chroma
  # comparisons; ignored at runtime when VECTOR_STORE=pgvector.
  "CHROMA_HOST=34.170.243.161"
  "CHROMA_PORT=8000"
  "CHROMA_SSL=0"
  # chat Postgres (same Cloud SQL instance, mobius_chat database)
  "CHAT_DATABASE_URL=${CHAT_DB_URL_FOR_RAG}"
  # Strategy (d) external web search — same service chat uses for
  # google_search. Required for the corpus_search_agent to fall back
  # to external sources when no corpus docs match the payer/query.
  "CHAT_SKILLS_GOOGLE_SEARCH_URL=https://mobius-google-search-ortabkknqa-uc.a.run.app/search"
  # Auto-publish on embed: when an embedding_job completes, the worker
  # immediately copies vectors into rag_published_embeddings + chat
  # Postgres so the doc is queryable end-to-end without a separate
  # "publish" admin call. Without this, embedded docs sit invisible to
  # chat retrieval forever — exactly the failure mode that stranded
  # 23 humana docs on 2026-04-27 after a deploy reset env vars.
  "AUTO_PUBLISH_ON_EMBED=1"
)

COMMON_SECRETS=(
  "MOBIUS_SKILL_LLM_INTERNAL_KEY=mobius-skill-llm-internal-key:latest"
  "ADMIN_API_KEY=rag-admin-api-key:latest"
  "CHROMA_AUTH_TOKEN=chroma-auth-token:latest"
  "JWT_SECRET=jwt-secret:latest"
)

join_with() { local IFS="$1"; shift; echo "$*"; }

deploy_service() {
  local name="$1"
  local command="$2"         # comma-separated CMD override
  local min_instances="$3"
  local max_instances="$4"
  local cpu_throttling="$5"  # --no-cpu-throttling vs default
  local memory="$6"

  echo ""
  echo "--- deploying ${name} ---"
  local flags=(
    --image="$IMAGE"
    --project="$PROJECT_ID"
    --region="$REGION"
    --platform=managed
    --allow-unauthenticated
    --memory="$memory"
    --cpu=2
    --timeout=3600
    --add-cloudsql-instances="$CLOUD_SQL_CONNECTION"
    --service-account="mobius-platform-dev@${PROJECT_ID}.iam.gserviceaccount.com"
    --set-env-vars="$(join_with ',' "${COMMON_ENV[@]}")"
    --set-secrets="$(join_with ',' "${COMMON_SECRETS[@]}")"
    --min-instances="$min_instances"
    --max-instances="$max_instances"
    --quiet
  )
  if [[ "$cpu_throttling" == "no" ]]; then
    flags+=(--no-cpu-throttling)
  fi
  if [[ -n "$command" ]]; then
    flags+=(--command="$command")
  fi

  gcloud run deploy "$name" "${flags[@]}"
}

# 3. API service. min=max=1 (single instance) — REQUIRED: in-process background
#    state lives in memory on ONE instance (the eval/calibration runner AND the
#    nightly orchestrator's live status). At max>1, Cloud Run can route the
#    status poll to a different instance than the one running the job (empty
#    status) and can kill the instance mid-eval when it scales down (the 13/110
#    stall we hit). no-cpu-throttling keeps the long background task alive.
#    Dev-scale only; a multi-instance prod needs DB-backed job state instead.
deploy_service "mobius-rag" "" 1 1 "no" "2Gi"   # 2Gi: publishing a giant doc (~9k embeddings) OOM'd at 1Gi

# 4. Chunking worker. Self-polling supervisor (FOR UPDATE SKIP LOCKED
#    handles dedup across instances), so Cloud Run autoscaling never
#    fires from HTTP load — we have to pin min=max=N to get N parallel
#    pollers. With single instance, queue p50 wait was 2h (2026-04-27
#    perf scan); 5 pollers brings throughput from ~32 docs/24h to
#    theoretical ~9k/day. Drop min back to 1 if cost matters more
#    than instant-rag SLA.
deploy_service "mobius-rag-chunking-worker" \
  "uvicorn,app.worker_server_chunking:app,--host,0.0.0.0,--port,8080" \
  12 12 "no" "2Gi"

# 5. Embedding worker (same shape, fewer resources since Vertex does
#    the heavy lifting remotely).
deploy_service "mobius-rag-embedding-worker" \
  "uvicorn,app.worker_server_embedding:app,--host,0.0.0.0,--port,8080" \
  1 1 "no" "2Gi"   # 2Gi: auto-publish-on-embed loads a giant's ~9k embeddings; OOM'd at 1Gi

# 6. Print URLs
echo ""
echo "=============================================================="
echo "Deploy complete. URLs:"
for s in mobius-rag mobius-rag-chunking-worker mobius-rag-embedding-worker; do
  url=$(gcloud run services describe "$s" --project="$PROJECT_ID" --region="$REGION" --format='value(status.url)')
  printf "  %-34s %s\n" "$s" "$url"
done
echo ""
echo "Smoke check: curl $(gcloud run services describe mobius-rag --project=${PROJECT_ID} --region=${REGION} --format='value(status.url)')/health/deep"
