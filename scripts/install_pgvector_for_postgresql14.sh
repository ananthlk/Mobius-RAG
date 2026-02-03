#!/bin/bash
# Install pgvector for PostgreSQL 14 (Homebrew pgvector only supports PG 17/18)
# Run from project root: ./scripts/install_pgvector_for_postgresql14.sh
#
# If you use PostgreSQL 15/16/17 from Homebrew instead, you can skip this and run:
#   brew install pgvector
#   brew services restart postgresql@15  # or your version
# Then in the mobius_rag database: CREATE EXTENSION IF NOT EXISTS vector;

set -e
PG14_CONFIG="/opt/homebrew/opt/postgresql@14/bin/pg_config"

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
# Force native ARM64 so the linker matches Postgres (avoid "required architecture x86_64")
export ARCHFLAGS="-arch arm64"
export LDFLAGS="-arch arm64"
# Run make under native arm64 (avoids Rosetta/x86_64 toolchain)
make clean 2>/dev/null || true
arch -arm64 make
arch -arm64 make install

echo ""
echo "pgvector installed for postgresql@14."
echo "Restart PostgreSQL: brew services restart postgresql@14"
echo "Then in your mobius_rag database run: CREATE EXTENSION IF NOT EXISTS vector;"
echo "  (Or restart the RAG backend; it runs that automatically on startup.)"
