#!/usr/bin/env bash
# Run the category_scores_to_columns migration using your .env DATABASE_URL.
# Usage: ./run_category_migration.sh   (from project root)

set -e
cd "$(dirname "$0")"
source .venv/bin/activate
python -m app.migrations.category_scores_to_columns
