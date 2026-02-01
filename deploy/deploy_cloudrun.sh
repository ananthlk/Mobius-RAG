#!/usr/bin/env bash
# Deploy Mobius RAG to Cloud Run (backend + frontend).
# Requires: gcloud, Docker, frontend built (VITE_API_BASE= npm run build).
#
# Usage:
#   DATABASE_PASSWORD='your_password' ./deploy/deploy_cloudrun.sh
#
# The password must match the Cloud SQL postgres user. If you don't have it, set/reset it:
#   gcloud sql users set-password postgres --instance=mobius-platform-db --password=YOUR_PASSWORD --project=mobiusos-new
# See docs/GCP_DEPLOYMENT.md for details.
#
set -e

PROJECT_ID="${GCP_PROJECT_ID:-mobiusos-new}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="mobius-rag"
IMAGE_NAME="mobius-rag"
CLOUD_SQL_INSTANCE="mobius-platform-db"
CLOUD_SQL_CONNECTION="${PROJECT_ID}:${REGION}:${CLOUD_SQL_INSTANCE}"

if [[ -z "$DATABASE_PASSWORD" ]]; then
  echo "Set DATABASE_PASSWORD (must match Cloud SQL postgres user)."
  echo "To set/reset on Cloud SQL: gcloud sql users set-password postgres --instance=mobius-platform-db --password=YOUR_PASSWORD --project=mobiusos-new"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Deploy Mobius RAG to Cloud Run ==="
echo "Project: $PROJECT_ID | Region: $REGION | Service: $SERVICE_NAME"
echo ""

# Enable APIs if needed
gcloud services enable cloudbuild.googleapis.com run.googleapis.com --project="$PROJECT_ID" --quiet 2>/dev/null || true

# Build frontend (relative URLs for same-origin)
if [[ ! -d frontend/dist ]] || [[ -z "$SKIP_FRONTEND_BUILD" ]]; then
  echo "Building frontend..."
  (cd frontend && npm ci 2>/dev/null || npm install)
  VITE_API_BASE= npm run build --prefix frontend
fi

# Configure Docker for gcloud (Container Registry)
gcloud auth configure-docker gcr.io --quiet 2>/dev/null || true

# Build and push image (Cloud Build uses Dockerfile in project root)
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

echo "Building and pushing image..."
gcloud builds submit --tag "$IMAGE" --project="$PROJECT_ID" .

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."

# DATABASE_URL for Cloud SQL Unix socket (Cloud Run connector)
# Format: postgresql+asyncpg://user:pass@/dbname?host=/cloudsql/PROJECT:REGION:INSTANCE
DB_URL="postgresql+asyncpg://postgres:${DATABASE_PASSWORD}@/mobius_rag?host=%2Fcloudsql%2F${PROJECT_ID}%3A${REGION}%3A${CLOUD_SQL_INSTANCE}"

gcloud run deploy "$SERVICE_NAME" \
  --image="$IMAGE" \
  --region="$REGION" \
  --platform=managed \
  --allow-unauthenticated \
  --memory=1Gi \
  --cpu=1 \
  --add-cloudsql-instances="$CLOUD_SQL_CONNECTION" \
  --set-env-vars="ENV=prod" \
  --set-env-vars="DATABASE_URL=${DB_URL}" \
  --set-env-vars="GCS_BUCKET=mobius-uploads-mobiusos-new" \
  --set-env-vars="VERTEX_PROJECT_ID=${PROJECT_ID}" \
  --set-env-vars="VERTEX_LOCATION=${REGION}" \
  --set-env-vars="VERTEX_MODEL=gemini-1.5-pro" \
  --set-env-vars="LLM_PROVIDER=vertex" \
  --project="$PROJECT_ID" \
  --quiet

SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region="$REGION" --project="$PROJECT_ID" --format='value(status.url)')
echo ""
echo "=== Deploy complete ==="
echo "Mobius RAG: $SERVICE_URL"
echo ""
echo "Note: Chunking and embedding workers still run on the VM (or separately)."
echo "To run workers: gcloud compute ssh mobius-platform-vm --zone=${REGION}-a --project=$PROJECT_ID"
echo "  Then: sudo systemctl start mobius-rag-chunking-worker mobius-rag-embedding-worker"
