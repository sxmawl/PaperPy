from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import vecs

class ConnectionNotExistException(Exception):
    """Raised when the underlying database connection is not available."""

    def __init__(self, message: str = "Database connection does not exist or is unavailable.", *args: Any) -> None:
        super().__init__(message, *args)
        self.message = message

    def __str__(self) -> str:
        return f"ConnectionNotExistException: {self.message}"


class VectorStore:
    """Wrapper around a Supabase vecs collection."""

    def __init__(
        self,
        collection: str = "knowledge",
        dimension: int = 1536,
        uri: str = "postgresql://postgres:password@localhost:5432/postgres",
        token: Optional[str] = None,  # kept for API compatibility – unused
    ) -> None:
        self.collection_name = collection
        self.dimension = dimension
        self.uri = uri
        
        # In vecs the *client* manages multiple collections.
        self._connect()
        self._setup_collection()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _connect(self) -> None:
        """Create a vecs client (opens a connection pool)."""
        try:
            # create_client returns a connected client ready for usage
            self.client = vecs.create_client(self.uri)
        except Exception as exc:  # pragma: no cover – broad but intentional
            raise ConnectionNotExistException(str(exc)) from exc

    def _setup_collection(self) -> None:
        """Get (or create) the collection and ensure it is indexed."""
        try:
            # get_or_create is idempotent: it returns the existing collection if
            # it already exists, otherwise it creates it with the specified
            # dimension.
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                dimension=self.dimension,
            )
            # For good query performance create an ANN index.  Using defaults
            # (HNSW + cosine distance) gives reliable results for OpenAI
            # embeddings.
            self.collection.create_index()
        except Exception as exc:  # pragma: no cover
            raise ConnectionNotExistException(str(exc)) from exc

    # ------------------------------------------------------------------
    # Public methods – these are used throughout the application
    # ------------------------------------------------------------------
    def store_vectors(self, texts: List[str], embeddings: List[List[float]], sources: Optional[List[str]] = None) -> List[str]:
        """Persist (text, embedding) pairs and return their primary keys."""
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must be the same length")

        records = []
        ids: List[str] = []
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            _id = uuid.uuid4().hex
            ids.append(_id)
            
            # Create metadata with text and optional source
            metadata = {"text": text}
            if sources and i < len(sources):
                metadata["source"] = sources[i]
            
            # vecs upsert signature: (id, vector, metadata)
            records.append((_id, emb, metadata))

        try:
            # No explicit flush is necessary: vecs writes immediately.
            self.collection.upsert(records=records)
            return ids
        except Exception as exc:  # pragma: no cover
            raise ConnectionNotExistException(str(exc)) from exc

    def search_vectors(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Return *top_k* most similar vectors to *query_embedding*."""
        try:
            # We ask vecs to include both the distance (value) and the metadata
            # (where we stored the original text).
            results = self.collection.query(
                data=query_embedding,
                limit=top_k,
                include_value=True,
                include_metadata=True,
            )
            
            formatted: List[Dict[str, Any]] = []
            for row in results:
                # According to vecs API the row format is:
                #   (id, distance, metadata) when include_value & include_metadata
                # Defensive parsing to tolerate potential library changes.
                if len(row) == 3:
                    _id, distance, metadata = row  # type: ignore[misc]
                elif len(row) == 2 and isinstance(row[1], dict):
                    _id, metadata = row  # type: ignore[misc]
                    distance = None
                elif len(row) == 2:
                    _id, distance = row  # type: ignore[misc]
                    metadata = {}
                else:  # pragma: no cover – unexpected format
                    _id, distance, metadata = row[0], None, {}

                # Convert distance to cosine similarity
                # vecs uses cosine distance (1 - cosine_similarity)
                # So cosine_similarity = 1 - distance
                cosine_similarity = 1.0 - distance if distance is not None else 0.0

                formatted.append(
                    {
                        "id": _id,
                        "text": metadata.get("text", ""),
                        "source": metadata.get("source", "unknown"),
                        "distance": distance,
                        "cosine_similarity": cosine_similarity,
                    }
                )
            return formatted
        except Exception as exc:  # pragma: no cover
            raise ConnectionNotExistException(str(exc)) from exc

    # ------------------------------------------------------------------
    # Utility helpers used by the rest of the codebase
    # ------------------------------------------------------------------
    def exists(self) -> bool:
        """Check if the collection exists in the current database."""
        try:
            return self.collection_name in self.client.list_collections()
        except Exception:  # pragma: no cover
            return False

    def drop(self) -> None:
        """Remove **all** vectors from the collection."""
        try:
            if self.exists():
                # Passing an empty filter deletes all records.
                self.collection.delete(filters={})
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def __del__(self) -> None:  # noqa: D401 – simple cleanup
        try:
            self.client.disconnect()
        except Exception:
            # Swallow any errors during interpreter shutdown.
            pass
