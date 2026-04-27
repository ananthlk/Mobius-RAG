"""Vector store abstraction for embedding storage and search.

Interface: add(ids, embeddings, metadata), search(embedding, k, filter?), delete_by_document(document_id).
Implementations: Chroma (optional), pgvector-only (no external store).
"""
from abc import ABC, abstractmethod
from typing import Any
import asyncio
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


class PgVectorStore(VectorStore):
    """ANN over ``rag_published_embeddings`` using pgvector + HNSW.

    No external service dependency; queries the same Cloud SQL Postgres
    as the rest of the app. Replaces ``ChromaVectorStore`` as the
    durable retrieval store after the 2026-04-27 mobius-chroma VM
    outage.

    Read-only in Step 3 — ``add()`` / ``delete_by_document()`` raise
    ``NotImplementedError`` here. The embedding worker still writes
    JSONB to ``rag_published_embeddings.embedding`` and Chroma in
    parallel. Step 5 will add a typed-vector write path inside
    ``publish_sync.py`` and wire up these mutators.

    The ``search()`` return shape matches ``ChromaVectorStore.search()``
    so callers don't change. One field differs by name only:
    Chroma returns a cosine ``distance`` (lower=better, 0..2). Per the
    spec we return cosine *similarity* under the same key — the
    chat-side caller in Step 4 needs to handle that. Documented at
    the top of the per-row dict so the contract is explicit.
    """

    # Columns we let the caller filter on. Whitelist (NOT user-supplied
    # column names) so the f-string SQL below is injection-safe.
    _ALLOWED_FILTERS: dict[str, str] = {
        # spec uses friendly names; map to actual columns on
        # rag_published_embeddings (see app/migrations/add_publish_tables.py)
        "payer": "document_payer",
        "state": "document_state",
        "authority_level": "document_authority_level",
        "document_id": "document_id",
        "source_type": "source_type",
    }

    def __init__(self, table_name: str = "rag_published_embeddings"):
        # Whitelist table names; the chunk_embeddings table also has
        # the new column but the published-contract table is what
        # chat consumes.
        if table_name not in {"rag_published_embeddings", "chunk_embeddings"}:
            raise ValueError(f"PgVectorStore: unsupported table {table_name!r}")
        self._table = table_name

    def add(self, ids: list[str], embeddings: list[list[float]], metadata: list[dict]) -> None:
        # Step 5 will implement the write path; for now writes still
        # land via the existing Chroma + JSONB path in publish_sync.
        raise NotImplementedError("PgVectorStore.add() lands in Step 5 (parallel write).")

    def delete_by_document(self, document_id: str) -> None:
        raise NotImplementedError("PgVectorStore.delete_by_document() lands in Step 5.")

    def search(
        self,
        embedding: list[float],
        k: int = 10,
        document_id: str | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[dict]:
        """Return top-k cosine-nearest rows.

        Each result dict::

            {
                "id":            str,    # uuid
                "document_id":   str,
                "source_type":   str,
                "source_id":     str,
                "distance":      float,  # cosine *similarity*, 1.0 = identical
            }

        The ``distance`` key reuses Chroma's name for drop-in
        compatibility with the existing chat-side reader, but the
        value is similarity (``1 - cosine_distance``) per spec —
        callers should treat closer-to-1 as more similar.
        """
        # If we're inside a running event loop (FastAPI handler),
        # callers should ``await store.asearch(...)`` instead. Sync
        # ``search()`` is provided for parity with the ABC and for
        # CLI / worker contexts that aren't async.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.asearch(embedding, k, document_id, filters))
        raise RuntimeError(
            "PgVectorStore.search() called from a running event loop; "
            "use ``await store.asearch(...)`` instead."
        )

    async def asearch(
        self,
        embedding: list[float],
        k: int = 10,
        document_id: str | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[dict]:
        """Async variant of :meth:`search`. Use this from FastAPI handlers."""
        return await self._search_async(embedding, k, document_id, filters)

    async def _search_async(
        self,
        embedding: list[float],
        k: int,
        document_id: str | None,
        filters: dict[str, str] | None,
    ) -> list[dict]:
        # Lazy import: this module is also imported by the chunking
        # worker, which doesn't need a DB session just to construct
        # a Chroma client. Keep DB import deferred so import-time
        # cost stays zero for the legacy path.
        from sqlalchemy import text
        from app.database import AsyncSessionLocal

        # Build WHERE clause from whitelisted filters. Filter values
        # bind via SQLAlchemy params, so they are safe; only column
        # names come from a whitelist.
        clauses: list[str] = []
        params: dict[str, object] = {"k": k}
        if document_id:
            clauses.append("document_id = :document_id")
            params["document_id"] = document_id
        for key, value in (filters or {}).items():
            if value is None or value == "":
                continue
            col = self._ALLOWED_FILTERS.get(key)
            if not col:
                # Unknown filter key — skip silently rather than
                # 500ing. Step 4 callers will pass extra params
                # opportunistically; ignoring is friendlier.
                continue
            param = f"f_{key}"
            clauses.append(f"{col} = :{param}")
            params[param] = value

        # Always filter out rows that haven't been backfilled yet so
        # ANN doesn't have to consider NULL vectors. Combined with the
        # caller's filters into a single AND-joined WHERE.
        clauses.append("embedding_vec IS NOT NULL")
        where_sql = "WHERE " + " AND ".join(clauses)

        # Cast the bound list to vector once at query time. Use the
        # text-form '[f1,f2,...]' which pgvector parses; this avoids
        # depending on the (optional) ``pgvector`` Python adapter.
        params["query_vec"] = "[" + ",".join(repr(float(x)) for x in embedding) + "]"

        sql = text(
            f"""
            SELECT
                id::text             AS id,
                document_id::text    AS document_id,
                source_type          AS source_type,
                source_id::text      AS source_id,
                1 - (embedding_vec <=> CAST(:query_vec AS vector)) AS similarity
            FROM {self._table}
            {where_sql}
            ORDER BY embedding_vec <=> CAST(:query_vec AS vector)
            LIMIT :k
            """
        )

        async with AsyncSessionLocal() as session:
            result = await session.execute(sql, params)
            rows = result.mappings().all()

        out: list[dict] = []
        for row in rows:
            out.append({
                "id": row["id"],
                "document_id": row["document_id"],
                "source_type": row["source_type"],
                "source_id": row["source_id"],
                # Match Chroma's key; value is similarity per spec.
                "distance": float(row["similarity"]) if row["similarity"] is not None else None,
            })
        return out


def get_vector_store() -> VectorStore:
    """Return configured vector store.

    Resolution order:

    * ``VECTOR_STORE=pgvector`` → :class:`PgVectorStore` (Step 3+).
    * ``CHROMA_HOST`` or ``CHROMA_PERSIST_DIR`` set → :class:`ChromaVectorStore`.
    * Otherwise → :class:`NoopVectorStore` (pgvector-only mode without
      the new read path; legacy default).
    """
    explicit = (os.getenv("VECTOR_STORE") or "").strip().lower()
    if explicit == "pgvector":
        return PgVectorStore()
    if os.getenv("CHROMA_HOST") or os.getenv("CHROMA_PERSIST_DIR"):
        return ChromaVectorStore()
    return NoopVectorStore()
