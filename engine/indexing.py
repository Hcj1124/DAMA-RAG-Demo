"""Stage 9 -- embed the combined chunks and write them to Chroma.

Re-running this is cheap. Each vector carries the hash of the text it was
built from, so a second run embeds only what actually changed and deletes
records that have disappeared from ``combined-chunks.jsonl``. The expensive
full pass happens on the first run, and whenever the embedding model changes
-- at which point the old vectors are meaningless and the collection is
dropped rather than mixed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from engine.corpus import Corpus, Record
from engine.ports import Embedder, VectorStore

logger = logging.getLogger(__name__)

METADATA_MODEL = "embedding_model"
METADATA_DIMENSION = "embedding_dimension"
METADATA_FINGERPRINT = "embedding_index_fingerprint"


@dataclass(frozen=True, slots=True)
class IndexReport:
    """What one indexing run actually did."""

    embedded: int
    unchanged: int
    deleted: int
    total: int
    model: str
    dimension: int
    rebuilt: bool

    def describe(self) -> str:
        head = "rebuilt" if self.rebuilt else "updated"
        return (
            f"Index {head}: {self.total} records in the collection "
            f"({self.embedded} embedded, {self.unchanged} unchanged, "
            f"{self.deleted} deleted) using {self.model} "
            f"[{self.dimension}-dim]"
        )


class Indexer:
    """Keeps the vector store in step with the corpus on disk."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        batch_size: int = 64,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._batch_size = batch_size

    def build(self, corpus: Corpus, *, rebuild: bool = False) -> IndexReport:
        model = self._embedder.name
        fingerprint = self._embedder.index_fingerprint
        metadata = self._store.metadata()
        indexed_with = metadata.get(METADATA_MODEL)
        indexed_fingerprint = metadata.get(METADATA_FINGERPRINT)

        # Mixing vectors from two models produces plausible-looking nonsense
        # rather than an error, so a model change forces a clean rebuild.
        if indexed_with and (
            indexed_with != model or indexed_fingerprint != fingerprint
        ):
            logger.warning(
                "Index embedding configuration changed (%s/%s -> %s/%s); "
                "dropping the collection",
                indexed_with,
                indexed_fingerprint or "unknown",
                model,
                fingerprint,
            )
            rebuild = True

        if rebuild:
            self._store.reset()

        existing = {} if rebuild else self._store.existing()

        stale = [
            record
            for record in corpus.records
            if existing.get(record.record_id, {}).get("content_hash")
            != record.content_hash
        ]
        removed = sorted(
            set(existing) - {record.record_id for record in corpus.records}
        )
        self._store.delete(removed)

        for start in range(0, len(stale), self._batch_size):
            self._embed_batch(stale[start : start + self._batch_size])

        dimension = self._embedder.dimension
        self._store.set_metadata(
            {
                METADATA_MODEL: model,
                METADATA_DIMENSION: dimension,
                METADATA_FINGERPRINT: fingerprint,
            }
        )

        return IndexReport(
            embedded=len(stale),
            unchanged=len(corpus.records) - len(stale),
            deleted=len(removed),
            total=self._store.count(),
            model=model,
            dimension=dimension,
            rebuilt=rebuild,
        )

    def _embed_batch(self, records: Sequence[Record]) -> None:
        if not records:
            return
        vectors = self._embedder.embed_documents(
            [record.text for record in records]
        )
        self._store.upsert(
            ids=[record.record_id for record in records],
            documents=[record.text for record in records],
            embeddings=vectors,
            metadatas=[record.chroma_metadata() for record in records],
        )
        logger.info("Embedded %d records", len(records))
