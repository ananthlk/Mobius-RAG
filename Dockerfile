# Mobius RAG - Backend + Frontend for Cloud Run
FROM python:3.11-slim

WORKDIR /app

# Build deps for cffi/cryptography
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python deps
COPY pyproject.toml ./
COPY app ./app/

# Install Python deps (vertex for Vertex AI/GCS)
RUN pip install --no-cache-dir -e ".[vertex]"

# Copy frontend build (built locally before docker build)
COPY frontend/dist ./frontend/dist/

# Cloud Run expects PORT env
ENV PORT=8080
EXPOSE 8080

# Run uvicorn
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
