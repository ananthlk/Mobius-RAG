#!/usr/bin/env bash
# Migrate via Cloud SQL Auth Proxy + psql (bypasses gcloud import permission issues).
# Requires: cloud-sql-proxy, psql, postgres password for Cloud SQL.
#
# Usage:
#   1. Set CLOUD_SQL_PASSWORD (or export it)
#   2. ./scripts/migrate_to_gcp_psql.sh [backup.sql]
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PROJECT_ID="${GCP_PROJECT_ID:-mobiusos-new}"
INSTANCE="${GCP_SQL_INSTANCE:-mobius-platform-db}"
DATABASE="mobius_rag"
CONNECTION_NAME="${PROJECT_ID}:us-central1:${INSTANCE}"
DUMP_FILE="${1:-$HOME/mobius_rag_backup.sql}"

if [[ ! -f "$DUMP_FILE" ]]; then
  echo "Dump file not found: $DUMP_FILE"
  exit 1
fi

# Prepare dump: replace local owner with postgres
PREPARED="${PROJECT_ROOT}/mobius_rag_migration_prepared.sql"
sed -e 's/OWNER TO [a-zA-Z0-9_]*/OWNER TO postgres/g' \
    -e 's/Owner: [a-zA-Z0-9_]*/Owner: postgres/g' \
    -e '/^DROP /d' \
    "$DUMP_FILE" > "$PREPARED"

echo "=== Migrate via Cloud SQL Auth Proxy ==="
echo "Instance: $INSTANCE | Database: $DATABASE"
echo ""

# Download proxy if needed
PROXY_BIN="${PROJECT_ROOT}/cloud-sql-proxy"
if [[ ! -x "$PROXY_BIN" ]]; then
  echo "Downloading Cloud SQL Auth Proxy..."
  curl -o "$PROXY_BIN" "https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.0/cloud-sql-proxy.darwin.arm64"
  chmod +x "$PROXY_BIN"
fi

# Use port 5433 to avoid conflict with local PostgreSQL
PROXY_PORT=5433
echo "Starting Cloud SQL Auth Proxy on port $PROXY_PORT..."
"$PROXY_BIN" "$CONNECTION_NAME" --port "$PROXY_PORT" &
PROXY_PID=$!
trap "kill $PROXY_PID 2>/dev/null" EXIT

sleep 5
echo "Proxy running. Importing..."
PGPASSWORD="${CLOUD_SQL_PASSWORD:?Set CLOUD_SQL_PASSWORD}" psql \
  -h 127.0.0.1 -p "$PROXY_PORT" -U postgres -d "$DATABASE" \
  -f "$PREPARED" \
  -v ON_ERROR_STOP=1

echo "=== Migration complete ==="
