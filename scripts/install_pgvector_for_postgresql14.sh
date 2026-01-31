#!/bin/bash
# Install pgvector for PostgreSQL 14 (Homebrew pgvector only supports PG 17/18)
# Run from project root: ./scripts/install_pgvector_for_postgresql14.sh

set -e
PG14_CONFIG="/opt/homebrew/opt/postgresql@14/bin/pg_config"
PG14_SHAREDIR="/opt/homebrew/share/postgresql@14"

if [ ! -f "$PG14_CONFIG" ]; then
  echo "postgresql@14 not found. Adjust PG14_CONFIG in this script."
  exit 1
fi

echo "Building pgvector for PostgreSQL 14 (ARM64)..."
cd /tmp
rm -rf pgvector
git clone --depth 1 --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector

export PG_CONFIG="$PG14_CONFIG"
export ARCHFLAGS="-arch arm64"  # Force ARM64 on Apple Silicon
make clean 2>/dev/null || true
make
make install

echo ""
echo "pgvector installed for postgresql@14."
echo "Restart PostgreSQL if it's running: brew services restart postgresql@14"
echo "Then run: python -m app.migrations.add_embedding_tables"
