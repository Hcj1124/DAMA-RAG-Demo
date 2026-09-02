"""第 9 階段的 Chroma 持久化向量資料庫實作。

Collection 使用 Chroma 預設的平方 L2 距離。因向量寫入前已正規化，其排序與
cosine distance 相同，僅數值尺度不同；``vector_distance`` 只供除錯，不作門檻。

一般更新使用 upsert 重複利用 collection，避免留下孤立的 HNSW 目錄；只有
embedding 模型或其向量設定改變、舊向量失去意義時，才執行完整 ``reset``。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


def _not_found_errors() -> tuple[type[BaseException], ...]:
    """整理不同 Chroma 版本中代表 collection 不存在的例外類型。"""

    errors: list[type[BaseException]] = [ValueError]
    try:
        from chromadb.errors import NotFoundError

        errors.append(NotFoundError)
    except ImportError:  # pragma: no cover - 僅適用非常舊的 chromadb
        pass
    return tuple(errors)


class ChromaVectorStore:
    """以持久化 Chroma DB 實作 :class:`engine.ports.VectorStore`。"""

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
        """首次存取時建立 Chroma client 與資料目錄。"""
        if self._client is None:
            import chromadb

            self._path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._path))
        return self._client

    def _get_collection(self, *, create: bool = False):
        """取得 collection，並在指定時建立不存在的 collection。"""
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
        """取得已索引 ID 與 metadata，供重建時比較內容雜湊。"""

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
        """刪除目前 collection，並清除快取的 collection 物件。"""
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
        """查詢最接近的向量，回傳文件、metadata 與距離。"""
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
