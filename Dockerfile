# Mobius RAG - Backend + Frontend for Cloud Run
FROM python:3.11-slim

WORKDIR /app

# Build deps:
#   * build-essential, libffi-dev   — cffi / cryptography
#   * pkg-config, libcairo2-dev     — pycairo (pulled in by
#                                     xhtml2pdf → reportlab → rlPyCairo).
#                                     Without libcairo2-dev the PDF
#                                     rendering stack fails to build
#                                     and the image won't produce.
#   * curl                          — debug + Cloud Run liveness probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    pkg-config \
    libcairo2-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python deps
COPY pyproject.toml ./
COPY app ./app/

# Install Python deps. Extras:
#   vertex — Vertex AI/GCS (embeddings + LLM)
#   chroma — chromadb client for publish_sync → published_rag collection
#            (required by embedding worker too; without it, embedding
#            jobs fail with "Chroma required" at import time)
RUN pip install --no-cache-dir -e ".[vertex,chroma]"

# Copy frontend build (built locally before docker build)
COPY frontend/dist ./frontend/dist/

# Cloud Run expects PORT env
ENV PORT=8080
EXPOSE 8080

# Run uvicorn
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
