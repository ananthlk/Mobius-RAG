#!/bin/bash
# Start the chunking worker process

cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Start worker
python -m app.worker
