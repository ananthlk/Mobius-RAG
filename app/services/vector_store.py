"""Vector store abstraction for embedding storage and search.

Interface: add(ids, embeddings, metadata), search(embedding, k, filter?), delete_by_document(document_id).
Implementations: Chroma (optional), pgvector-only (no external store).
"""
from abc import ABC, abstractmethod
from typing import Any
import logging
import os

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract vector store for embeddings."""

    @abstractmethod
    def add(self, ids: list[str], embeddings: list[list[float]], metadata: list[dict]) -> None:
        """Add embeddings with ids and metadata. metadata items: document_id, source_type, source_id."""
        pass

    @abstractmethod
    def search(self, embedding: list[float], k: int = 10, document_id: str | None = None) -> list[dict]:
        """Return top-k results. Each result: {id, document_id, source_type, source_id, distance}."""
        pass

    @abstractmethod
    def delete_by_document(self, document_id: str) -> None:
        """Delete all embeddings for a document."""
        pass


class ChromaVectorStore(VectorStore):
    """Chroma-backed vector store."""

    def __init__(self, collection_name: str = "chunk_embeddings", host: str | None = None, port: int | None = None, persist_directory: str | None = None):
        self._collection_name = collection_name
        self._host = host or os.getenv("CHROMA_HOST", "localhost")
        self._port = port if port is not None else int(os.getenv("CHROMA_PORT", "8000"))
        self._persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIR")
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
        except ImportError as e:
            raise ImportError("Chroma required. Install with: pip install -e '.[chroma]'") from e
        if self._persist_directory:
            self._client = chromadb.PersistentClient(path=self._persist_directory)
        else:
            self._client = chromadb.HttpClient(host=self._host, port=self._port)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def add(self, ids: list[str], embeddings: list[list[float]], metadata: list[dict]) -> None:
        if not ids:
            return
        coll = self._get_collection()
        # Chroma expects metadatas as list of dict; values must be str, int, float, bool
        metadatas = []
        for m in metadata:
            metadatas.append({
                "document_id": str(m.get("document_id", "")),
                "source_type": str(m.get("source_type", "")),
                "source_id": str(m.get("source_id", "")),
            })
        coll.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        logger.debug("Chroma: added %d embeddings", len(ids))

    def search(self, embedding: list[float], k: int = 10, document_id: str | None = None) -> list[dict]:
        coll = self._get_collection()
        where = {"document_id": document_id} if document_id else None
        result = coll.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
            include=["metadatas", "distances"],
        )
        if not result or not result["ids"] or not result["ids"][0]:
            return []
        out = []
        for i, id_ in enumerate(result["ids"][0]):
            meta = (result["metadatas"][0][i] or {}) if result.get("metadatas") else {}
            dist = (result["distances"][0][i]) if result.get("distances") else None
            out.append({
                "id": id_,
                "document_id": meta.get("document_id"),
                "source_type": meta.get("source_type"),
                "source_id": meta.get("source_id"),
                "distance": dist,
            })
        return out

    def delete_by_document(self, document_id: str) -> None:
        coll = self._get_collection()
        coll.delete(where={"document_id": document_id})
        logger.debug("Chroma: deleted embeddings for document %s", document_id)


class NoopVectorStore(VectorStore):
    """No-op store when vector DB is disabled (pgvector-only)."""

    def add(self, ids: list[str], embeddings: list[list[float]], metadata: list[dict]) -> None:
        pass

    def search(self, embedding: list[float], k: int = 10, document_id: str | None = None) -> list[dict]:
        return []

    def delete_by_document(self, document_id: str) -> None:
        pass


def get_vector_store() -> VectorStore:
    """Return configured vector store. Chroma if CHROMA_HOST or CHROMA_PERSIST_DIR set, else NoopVectorStore."""
    if os.getenv("CHROMA_HOST") or os.getenv("CHROMA_PERSIST_DIR"):
        return ChromaVectorStore()
    return NoopVectorStore()
