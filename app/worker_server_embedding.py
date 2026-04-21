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


@app.on_event("shutdown")
def stop_worker():
    """Cloud Run SIGTERM drain hook. See worker_server_chunking.py
    for the full rationale; same pattern applies here.
    """
    try:
        from app.worker.shutdown import request_shutdown
        request_shutdown(source="fastapi-shutdown")
        logger.info("Shutdown requested; waiting up to 10s for worker to drain")
        if _worker_thread:
            _worker_thread.join(timeout=10)
            if _worker_thread.is_alive():
                logger.warning("Worker did not drain within 10s — Cloud Run will SIGKILL")
    except Exception as e:
        logger.warning("Shutdown hook failed (non-fatal): %s", e)


@app.get("/health")
def health():
    return {"status": "ok", "service": "mobius-rag-embedding-worker"}
