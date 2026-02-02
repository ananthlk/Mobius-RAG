#!/bin/bash
# Start backend server

cd "$(cd "$(dirname "$0")" && pwd)"
source .venv/bin/activate

echo "Starting backend server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
