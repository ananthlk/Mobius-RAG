#!/bin/bash
# Database setup script

echo "Setting up Mobius RAG database..."

# Check if Postgres is running
if ! pg_isready -q; then
    echo "⚠️  Postgres is not running. Please start it first:"
    echo "   brew services start postgresql"
    exit 1
fi

# Create database (adjust user if needed)
echo "Creating database 'mobius_rag'..."
createdb mobius_rag 2>/dev/null || echo "Database may already exist or you need to run: createdb mobius_rag"

# Initialize tables
echo "Initializing database tables..."
cd "$(dirname "$0")"
source .venv/bin/activate
python app/init_db.py

echo "✅ Database setup complete!"
