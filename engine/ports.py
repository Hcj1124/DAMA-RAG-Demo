"""Protocols the core depends on.

Retrieval, context resolution, prompting and the pipeline are written
against these interfaces only. That is what lets the tests exercise the whole
funnel without downloading six gigabytes of weights, and what makes swapping
bge-m3 for Qwen3-Embedding a configuration change instead of a rewrite.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors.

    Documents and queries are embedded through separate methods because
    instruction-aware models (Qwen3-Embedding, E5, GTE) need an asymmetric
    prefix. Models that do not, such as bge-m3, ignore the distinction.
    """

    @property
    def name(self) -> str:
        """Identifier recorded in the index so staleness can be detected."""

    @property
    def index_fingerprint(self) -> str:
        """Stable identity of settings that change stored document vectors."""

    @property
    def dimension(self) -> int: ...

    def embed_documents(
        self, texts: Sequence[str], *, show_progress: bool = False
    ) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class Reranker(Protocol):
    """Scores (query, passage) pairs jointly. Higher is better."""

    @property
    def name(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Persistent nearest-neighbour index over child chunks."""

    def exists(self) -> bool: ...

    def count(self) -> int: ...

    def metadata(self) -> Mapping[str, Any]: ...

    def set_metadata(self, metadata: Mapping[str, Any]) -> None: ...

    def existing(self) -> dict[str, Mapping[str, Any]]:
        """Every indexed id mapped to its stored metadata."""

    def upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> None: ...

    def delete(self, ids: Sequence[str]) -> None: ...

    def reset(self) -> None:
        """Drop the collection and every vector in it."""

    def query(
        self, embedding: Sequence[float], top_k: int
    ) -> list[tuple[str, str, Mapping[str, Any], float]]:
        """Return ``(id, document, metadata, distance)`` nearest first."""


@runtime_checkable
class LanguageModel(Protocol):
    """Generates the final answer from a fully assembled prompt."""

    @property
    def name(self) -> str: ...

    def complete(self, prompt: str) -> str: ...
