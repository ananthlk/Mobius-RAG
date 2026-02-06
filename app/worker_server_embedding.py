"""
Minimal HTTP server for Cloud Run: runs the RAG embedding worker in a background thread
and serves /health so Cloud Run keeps the instance alive.
Use: uvicorn app.worker_server_embedding:app --host 0.0.0.0 --port 8080
"""
import logging
import threading

from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [EMBED-WORKER] - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mobius RAG Embedding Worker", version="0.1.0")

_worker_thread: threading.Thread | None = None


@app.on_event("startup")
def start_worker():
    global _worker_thread

    def _run():
        try:
            from app.embedding_worker import main
            main()
        except Exception as e:
            logger.exception("Embedding worker loop exited: %s", e)

    _worker_thread = threading.Thread(target=_run, daemon=True)
    _worker_thread.start()
    logger.info("RAG embedding worker started in background thread")


@app.get("/health")
def health():
    return {"status": "ok", "service": "mobius-rag-embedding-worker"}
