"""第 10 階段：先用向量提高召回率，再以 cross-encoder 提升精確度。

第一階段由 bi-encoder 低成本取回 ``retrieve_k`` 筆候選；第二階段使用
cross-encoder 逐對重新評分並保留 ``rerank_k`` 筆，以區分「提到關鍵字」與
「真正回答問題」的內容。
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
    """檢索回傳的子 chunk，可選擇再附上 reranker 分數。"""

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
        """轉成可序列化並供 CLI 顯示的字典。"""
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
    """讓尚未評分的候選排在最後，避免對 ``None`` 排序時出錯。"""

    return (
        float("-inf")
        if candidate.rerank_score is None
        else candidate.rerank_score
    )


class Retriever:
    """將問題轉成依相關性排序的子 chunks。"""

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
        """拒絕查詢由不同 embedding 設定建立或尚未建立的索引。"""

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
        """第一階段：以子 chunks 的近似最近鄰搜尋取得候選。"""

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
        """第二階段：以 cross-encoder 重評候選，分數最高者優先。"""

        if not candidates:
            return []
        k = top_k or self._settings.rerank_k
        scores = self._reranker.score(query, [c.text for c in candidates])
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = score
        return sorted(candidates, key=_rerank_key, reverse=True)[:k]

    def search(self, query: str) -> list[Candidate]:
        """依設定的候選數量依序執行召回與重排序。"""

        return self.rerank(query, self.retrieve(query))
