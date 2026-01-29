#!/bin/bash
# Initialize git, initial commit, and push to https://github.com/ananthlk/Mobius-RAG.git
# Run from project root: ./git_init_and_push.sh

set -e
# Go to the directory where this script lives (project root)
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
echo "Using directory: $ROOT"

# Remove .git if it exists as a file (can block init)
if [ -f .git ]; then
  echo "Removing file .git so we can create repo..."
  rm .git
fi

# Create repo (safe to run even if .git already exists)
git init -b main
echo "Git initialized."

git add .
git status

git commit -m "Initial commit: Mobius RAG - document processing, extraction, reader, review facts, navigate to fact" || true

if ! git remote get-url origin 2>/dev/null; then
  git remote add origin https://github.com/ananthlk/Mobius-RAG.git
  echo "Remote 'origin' added."
fi

git push -u origin main
echo "Done."
