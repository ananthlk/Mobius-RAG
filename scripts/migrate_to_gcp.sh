#!/usr/bin/env bash
# Migrate Mobius RAG database (schema + data) from local PostgreSQL to GCP Cloud SQL.
# Prerequisites: pg_dump, gcloud CLI, local mobius_rag DB running.
#
# Usage:
#   ./scripts/migrate_to_gcp.sh                    # Use .env DATABASE_URL, create dump
#   ./scripts/migrate_to_gcp.sh backup.sql         # Use existing backup file
#   ./scripts/migrate_to_gcp.sh --import-only      # Skip dump, just import last uploaded file
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PROJECT_ID="${GCP_PROJECT_ID:-mobiusos-new}"
INSTANCE="${GCP_SQL_INSTANCE:-mobius-platform-db}"
DATABASE="mobius_rag"
BUCKET="mobius-uploads-${PROJECT_ID}"
IMPORT_ONLY=false

# Parse args
if [[ "$1" == "--import-only" ]]; then
  IMPORT_ONLY=true
  shift
fi

DUMP_FILE="${1:-mobius_rag_migration_$(date +%Y%m%d_%H%M%S).sql}"

echo "=== Mobius RAG DB Migration to GCP ==="
echo "Project: $PROJECT_ID"
echo "Instance: $INSTANCE"
echo "Database: $DATABASE"
echo ""

# Step 1: Dump from source (unless import-only)
if [[ "$IMPORT_ONLY" == true ]]; then
  echo "[1/5] Skipping dump (--import-only)"
  # Use most recent dump in project root if exists
  LATEST_DUMP=$(ls -t mobius_rag_migration_*.sql 2>/dev/null | head -1)
  if [[ -n "$LATEST_DUMP" ]]; then
    DUMP_FILE="$LATEST_DUMP"
    echo "      Using: $DUMP_FILE"
  else
    echo "      No dump file found. Run without --import-only first."
    exit 1
  fi
elif [[ -f "$DUMP_FILE" ]]; then
  echo "[1/5] Using existing dump: $DUMP_FILE"
else
  echo "[1/5] Dumping local database..."
  if [[ -f .env ]]; then
    set -a
    source .env
    set +a
  fi
  DB_URL="${SOURCE_DATABASE_URL:-${DATABASE_URL}}"
  if [[ -z "$DB_URL" ]]; then
    echo "Set SOURCE_DATABASE_URL or DATABASE_URL to dump from (e.g. postgresql://postgres:pass@localhost:5432/mobius_rag)"
    exit 1
  fi
  # pg_dump expects postgresql:// (not postgresql+asyncpg)
  PG_URL="${DB_URL/postgresql+asyncpg/postgresql}"
  pg_dump "$PG_URL" --no-owner --no-acl --clean --if-exists > "$DUMP_FILE"
  echo "      Dumped to $DUMP_FILE ($(wc -l < "$DUMP_FILE" 2>/dev/null || echo 0) lines)"
fi

# Step 2: Fix dump for Cloud SQL (replace local user with postgres)
echo "[2/5] Preparing dump for Cloud SQL..."
TEMP_DUMP="${PROJECT_ROOT}/mobius_rag_migration_prepared.sql"
# Replace local owner with postgres so Cloud SQL import succeeds (role "ananth" etc. don't exist)
# Also remove DROP statements - Cloud SQL import with --clean dumps can fail with permission denied.
# We'll import into a fresh DB or one we've cleared separately.
sed -e 's/OWNER TO [a-zA-Z0-9_]*/OWNER TO postgres/g' \
    -e 's/Owner: [a-zA-Z0-9_]*/Owner: postgres/g' \
    -e '/^DROP /d' \
    "$DUMP_FILE" > "$TEMP_DUMP" 2>/dev/null || cp "$DUMP_FILE" "$TEMP_DUMP"
# Prepend pgvector if dump has vector columns
if grep -q "vector\|chunk_embeddings" "$TEMP_DUMP" 2>/dev/null; then
  if ! grep -q "CREATE EXTENSION.*vector" "$TEMP_DUMP" 2>/dev/null; then
    echo "CREATE EXTENSION IF NOT EXISTS vector;" | cat - "$TEMP_DUMP" > "${TEMP_DUMP}.tmp"
    mv "${TEMP_DUMP}.tmp" "$TEMP_DUMP"
  fi
fi
DUMP_FILE="$TEMP_DUMP"
echo "      Prepared: $(basename "$DUMP_FILE")"

# Step 3: Grant Cloud SQL service account read access to bucket (for import)
echo "[3/5] Ensuring Cloud SQL can read from GCS bucket..."
SQL_SA=$(gcloud sql instances describe "$INSTANCE" --format='value(serviceAccountEmailAddress)' 2>/dev/null || true)
if [[ -n "$SQL_SA" ]]; then
  gsutil iam ch "serviceAccount:${SQL_SA}:objectViewer" "gs://${BUCKET}" 2>/dev/null || true
  echo "      Granted objectViewer to $SQL_SA"
else
  echo "      Could not get Cloud SQL SA - import may fail with 403"
fi

# Step 4: Upload to GCS
GCS_PATH="gs://${BUCKET}/migrations/$(basename "$DUMP_FILE")"
echo "[4/5] Uploading dump to GCS..."
gsutil cp "$DUMP_FILE" "$GCS_PATH"
echo "      Uploaded to $GCS_PATH"

# Step 5: Import to Cloud SQL
echo "[5/5] Importing to Cloud SQL (may take several minutes)..."
gcloud sql import sql "$INSTANCE" "$GCS_PATH" \
  --database="$DATABASE" \
  --project="$PROJECT_ID" \
  --quiet

echo ""
echo "=== Migration complete ==="
echo "Connect: gcloud sql connect $INSTANCE --database=$DATABASE --user=postgres --project=$PROJECT_ID"
echo ""
echo "If gcloud sql import fails with 'permission denied', use migrate_to_gcp_psql.sh instead:"
echo "  CLOUD_SQL_PASSWORD=<your-postgres-password> ./scripts/migrate_to_gcp_psql.sh <backup.sql>"
