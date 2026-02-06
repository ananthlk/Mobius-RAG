#!/bin/bash
# Start RAG backend server using shared venv

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MOBIUS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$MOBIUS_ROOT/.venv"

cd "$SCRIPT_DIR"

if [ ! -d "$VENV" ]; then
  echo "No shared venv at $VENV. Create it:"
  echo "  cd $MOBIUS_ROOT && python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# Load .env if exists
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

echo "Starting backend server on http://localhost:8001"
echo "Press Ctrl+C to stop"
echo ""

exec "$VENV/bin/python3" -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
