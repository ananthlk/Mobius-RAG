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
DB_URL="postgresql+asyncpg://${DB_USER}:${DB_PASS}@/mobius_rag?host=%2Fcloudsql%2F${PROJECT_ID}%3A${REGION}%3A${CLOUD_SQL_INSTANCE}"

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

# 2. Build image via Cloud Build (uses mobius-rag/.gcloudignore to avoid
#    pushing 1GB of pycache/node_modules).
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
  "DATABASE_URL=${DB_URL}"
  "GCS_BUCKET=mobius-rag-uploads-dev"
  "VERTEX_PROJECT_ID=${PROJECT_ID}"
  "VERTEX_LOCATION=${REGION}"
  "VERTEX_MODEL=gemini-2.5-flash"
  "EMBEDDING_PROVIDER=vertex"
  "CHAT_INTERNAL_LLM_URL=${CHAT_INTERNAL_LLM_URL}"
  # Chroma (chat's vector store on the GCE VM at 34.170.243.161)
  "CHROMA_HOST=34.170.243.161"
  "CHROMA_PORT=8000"
  "CHROMA_SSL=0"
  # chat Postgres (same Cloud SQL instance, mobius_chat database)
  "CHAT_DATABASE_URL=${CHAT_DB_URL_FOR_RAG}"
)

COMMON_SECRETS=(
  "MOBIUS_SKILL_LLM_INTERNAL_KEY=mobius-skill-llm-internal-key:latest"
  "ADMIN_API_KEY=rag-admin-api-key:latest"
  "CHROMA_AUTH_TOKEN=chroma-auth-token:latest"
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

# 3. API service (autoscales, minScale=0)
deploy_service "mobius-rag" "" 0 10 "" "1Gi"

# 4. Chunking worker (always-on, minScale=1, no CPU throttling — needs
#    to poll the DB between bursty requests).
deploy_service "mobius-rag-chunking-worker" \
  "uvicorn,app.worker_server_chunking:app,--host,0.0.0.0,--port,8080" \
  1 1 "no" "2Gi"

# 5. Embedding worker (same shape, fewer resources since Vertex does
#    the heavy lifting remotely).
deploy_service "mobius-rag-embedding-worker" \
  "uvicorn,app.worker_server_embedding:app,--host,0.0.0.0,--port,8080" \
  1 1 "no" "1Gi"

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
