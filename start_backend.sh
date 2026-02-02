#!/bin/bash
# Start backend server

cd "$(cd "$(dirname "$0")" && pwd)"
if [ ! -d .venv ]; then
  echo "No .venv in mobius-rag. Create it and install deps:"
  echo "  cd $(pwd) && python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi
source .venv/bin/activate

echo "Starting backend server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
