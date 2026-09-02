"""Chroma persistent vector store -- stage 9's storage half.

Distance metric: the collection uses Chroma's default squared-L2 space.
Every vector is L2-normalised before insertion, and squared L2 is a strictly
increasing function of cosine distance on normalised vectors, so the ranking
is identical to cosine and only the numeric scale differs. The value is
surfaced as ``vector_distance`` for debugging; nothing thresholds on it.

Lifecycle: the collection is reused via upsert rather than dropped and
recreated on every rebuild, which would orphan HNSW index directories on
disk. A full ``reset`` happens only when the embedding model or its
dimension changes -- the one case where the old vectors are meaningless.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


def _not_found_errors() -> tuple[type[BaseException], ...]:
    """Chroma renamed its "missing collection" error across versions."""

    errors: list[type[BaseException]] = [ValueError]
    try:
        from chromadb.errors import NotFoundError

        errors.append(NotFoundError)
    except ImportError:  # pragma: no cover - very old chromadb
        pass
    return tuple(errors)


class ChromaVectorStore:
    """Implements :class:`engine.ports.VectorStore` on a persistent Chroma DB."""

    def __init__(self, path: Path, collection_name: str) -> None:
        self._path = path
        self._collection_name = collection_name
        self._client = None
        self._collection = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def _ensure_client(self):
        if self._client is None:
            import chromadb

            self._path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._path))
        return self._client

    def _get_collection(self, *, create: bool = False):
        if self._collection is not None:
            return self._collection

        client = self._ensure_client()
        try:
            self._collection = client.get_collection(name=self._collection_name)
        except _not_found_errors():
            if not create:
                return None
            logger.info("Creating collection %s", self._collection_name)
            self._collection = client.create_collection(
                name=self._collection_name
            )
        return self._collection

    def exists(self) -> bool:
        return self._get_collection() is not None

    def count(self) -> int:
        collection = self._get_collection()
        return collection.count() if collection is not None else 0

    def metadata(self) -> Mapping[str, Any]:
        collection = self._get_collection()
        if collection is None:
            return {}
        return dict(collection.metadata or {})

    def set_metadata(self, metadata: Mapping[str, Any]) -> None:
        collection = self._get_collection(create=True)
        collection.modify(metadata=dict(metadata))

    def existing(self) -> dict[str, Mapping[str, Any]]:
        """Indexed ids with their metadata, so a rebuild can diff by hash."""

        collection = self._get_collection()
        if collection is None:
            return {}
        stored = collection.get(include=["metadatas"])
        return {
            record_id: dict(metadata or {})
            for record_id, metadata in zip(
                stored["ids"], stored["metadatas"] or []
            )
        }

    def upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> None:
        if not ids:
            return
        collection = self._get_collection(create=True)
        collection.upsert(
            ids=list(ids),
            documents=list(documents),
            embeddings=[list(vector) for vector in embeddings],
            metadatas=[dict(metadata) for metadata in metadatas],
        )

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        collection = self._get_collection()
        if collection is not None:
            collection.delete(ids=list(ids))

    def reset(self) -> None:
        client = self._ensure_client()
        try:
            client.delete_collection(name=self._collection_name)
            logger.info("Dropped collection %s", self._collection_name)
        except _not_found_errors():
            pass
        self._collection = None

    def query(
        self, embedding: Sequence[float], top_k: int
    ) -> list[tuple[str, str, Mapping[str, Any], float]]:
        collection = self._get_collection()
        if collection is None:
            return []

        result = collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        return [
            (ids[i], documents[i], dict(metadatas[i]), float(distances[i]))
            for i in range(len(ids))
        ]
