"""第 9 階段：將合併後的 chunks 做 embedding 並寫入 Chroma。

每個向量都帶有原文雜湊，因此重跑時只重新處理內容已變更的紀錄，並移除來源檔中
已刪除的項目。首次建立或 embedding 設定改變時才執行完整重建，避免混用不相容向量。
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
    """記錄單次索引作業實際新增、略過、刪除與重建的結果。"""

    embedded: int
    unchanged: int
    deleted: int
    total: int
    model: str
    dimension: int
    rebuilt: bool

    def describe(self) -> str:
        """產生適合 CLI 顯示的索引結果摘要。"""
        head = "rebuilt" if self.rebuilt else "updated"
        return (
            f"Index {head}: {self.total} records in the collection "
            f"({self.embedded} embedded, {self.unchanged} unchanged, "
            f"{self.deleted} deleted) using {self.model} "
            f"[{self.dimension}-dim]"
        )


class Indexer:
    """讓向量資料庫與磁碟上的最新語料保持同步。"""

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
        """依內容雜湊增量更新索引，必要時清除並完整重建。"""
        model = self._embedder.name
        fingerprint = self._embedder.index_fingerprint
        metadata = self._store.metadata()
        indexed_with = metadata.get(METADATA_MODEL)
        indexed_fingerprint = metadata.get(METADATA_FINGERPRINT)

        # 混用不同模型產生的向量不一定報錯，卻會得到看似合理的錯誤結果；因此
        # embedding 模型或影響文件向量的設定一變，就強制乾淨重建。
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
        """批次產生向量，並連同文件與 metadata 寫入向量資料庫。"""
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
