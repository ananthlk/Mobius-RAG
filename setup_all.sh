#!/bin/bash
# Complete setup script - run when network is available

set -e  # Exit on error

echo "🚀 Mobius RAG - Complete Setup"
echo "================================"
echo ""

# Check if we're in the right directory
if [ ! -f "app/main.py" ]; then
    echo "❌ Error: Please run this script from the project root"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Install Python dependencies
echo ""
echo "📥 Installing Python dependencies..."
pip install sqlalchemy asyncpg python-dotenv pymupdf greenlet

# Check Postgres
echo ""
echo "🗄️  Checking Postgres..."
if ! pg_isready -h localhost > /dev/null 2>&1; then
    echo "⚠️  Postgres is not running. Attempting to start..."
    brew services start postgresql 2>/dev/null || {
        echo "❌ Could not start Postgres automatically."
        echo "   Please start it manually: brew services start postgresql"
        exit 1
    }
    sleep 2
fi

# Create database
echo ""
echo "🗄️  Creating database..."
createdb mobius_rag 2>/dev/null || echo "   Database may already exist (that's OK)"

# Initialize tables
echo ""
echo "🗄️  Initializing database tables..."
python -m app.init_db

# Verify installation
echo ""
echo "✅ Verifying installation..."
python -c "from app.models import Document, DocumentPage; print('   ✓ Models OK')" || {
    echo "   ❌ Model import failed"
    exit 1
}

python -c "from app.services.extract_text import extract_text_from_gcs; print('   ✓ Extraction service OK')" || {
    echo "   ⚠️  Extraction service import failed (pymupdf may not be installed)"
}

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Start backend:  uvicorn app.main:app --reload"
echo "  2. Start frontend: cd frontend && npm run dev"
echo "  3. Open browser:   http://localhost:5173"
echo ""
