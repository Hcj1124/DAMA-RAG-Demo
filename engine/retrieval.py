"""Stage 10 -- vector recall, then cross-encoder precision.

Stage one asks the bi-encoder for ``retrieve_k`` candidates: cheap, and tuned
for recall. Stage two rescores that shortlist with the cross-encoder and
keeps ``rerank_k``: expensive per pair, affordable on twenty pairs, and much
better at telling "mentions the words" from "answers the question".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from engine.config import RetrievalSettings
from engine.errors import IndexNotBuiltError, IndexStaleError
from engine.indexing import METADATA_FINGERPRINT, METADATA_MODEL
from engine.ports import Embedder, Reranker, VectorStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Candidate:
    """A child chunk returned by search, optionally scored by the reranker."""

    record_id: str
    parent_id: str | None
    content_type: str
    title: str
    start_page: int
    end_page: int
    text: str
    vector_distance: float
    rerank_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "record_id": self.record_id,
            "parent_id": self.parent_id,
            "content_type": self.content_type,
            "title": self.title,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "vector_distance": self.vector_distance,
            "text": self.text,
        }
        if self.rerank_score is not None:
            payload["rerank_score"] = self.rerank_score
        return payload


def _rerank_key(candidate: Candidate) -> float:
    """Unscored candidates sort last rather than raising on ``None``."""

    return (
        float("-inf")
        if candidate.rerank_score is None
        else candidate.rerank_score
    )


class Retriever:
    """Turns a question into ranked child chunks."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        reranker: Reranker,
        store: VectorStore,
        settings: RetrievalSettings,
    ) -> None:
        self._embedder = embedder
        self._reranker = reranker
        self._store = store
        self._settings = settings
        self._checked = False

    def _check_index(self) -> None:
        """Refuse to query an index built by a different embedding model."""

        if self._checked:
            return
        if not self._store.exists() or self._store.count() == 0:
            raise IndexNotBuiltError(
                getattr(self._store, "collection_name", "?"),
                getattr(self._store, "path", "?"),
            )
        indexed_with = self._store.metadata().get(METADATA_MODEL)
        indexed_fingerprint = self._store.metadata().get(METADATA_FINGERPRINT)
        expected_fingerprint = self._embedder.index_fingerprint
        if indexed_with and (
            indexed_with != self._embedder.name
            or indexed_fingerprint != expected_fingerprint
        ):
            raise IndexStaleError(
                indexed_with=(
                    f"{indexed_with}@{indexed_fingerprint or 'unknown'}"
                ),
                querying_with=(
                    f"{self._embedder.name}@{expected_fingerprint}"
                ),
            )
        self._checked = True

    def retrieve(self, query: str, top_k: int | None = None) -> list[Candidate]:
        """Stage 1 -- approximate nearest neighbours over child chunks."""

        self._check_index()
        k = top_k or self._settings.retrieve_k
        embedding = self._embedder.embed_query(query)

        return [
            Candidate(
                record_id=record_id,
                parent_id=str(metadata.get("parent_id") or "") or None,
                content_type=str(metadata.get("content_type", "text")),
                title=str(metadata.get("title", "")),
                start_page=int(metadata["start_page"]),
                end_page=int(metadata["end_page"]),
                text=document,
                vector_distance=distance,
            )
            for record_id, document, metadata, distance in self._store.query(
                embedding, k
            )
        ]

    def rerank(
        self,
        query: str,
        candidates: Sequence[Candidate],
        top_k: int | None = None,
    ) -> list[Candidate]:
        """Stage 2 -- cross-encoder rescoring, highest score first."""

        if not candidates:
            return []
        k = top_k or self._settings.rerank_k
        scores = self._reranker.score(query, [c.text for c in candidates])
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = score
        return sorted(candidates, key=_rerank_key, reverse=True)[:k]

    def search(self, query: str) -> list[Candidate]:
        """The full funnel, using the configured stage sizes."""

        return self.rerank(query, self.retrieve(query))
