"""定義核心流程依賴的介面協定。

檢索、上下文解析、Prompt 與管線只依賴這些介面，因此測試可用假物件跑完整漏斗，
不需下載大型模型；更換 embedding 或其他 adapter 時也不必重寫核心流程。
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """將文字轉成向量的介面。

    文件與查詢分成不同方法，讓需要非對稱指令前綴的模型可以分別處理；不需要前綴的
    模型則可共用相同編碼邏輯。
    """

    @property
    def name(self) -> str:
        """寫入索引的模型識別名稱，用來判斷索引是否過期。"""

    @property
    def index_fingerprint(self) -> str:
        """會影響文件向量之設定的穩定指紋。"""

    @property
    def dimension(self) -> int: ...

    def embed_documents(
        self, texts: Sequence[str], *, show_progress: bool = False
    ) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class Reranker(Protocol):
    """共同評估查詢與段落配對；分數越高表示越相關。"""

    @property
    def name(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


@runtime_checkable
class VectorStore(Protocol):
    """保存子 chunks 最近鄰索引的持久化向量資料庫介面。"""

    def exists(self) -> bool: ...

    def count(self) -> int: ...

    def metadata(self) -> Mapping[str, Any]: ...

    def set_metadata(self, metadata: Mapping[str, Any]) -> None: ...

    def existing(self) -> dict[str, Mapping[str, Any]]:
        """回傳每個已索引 ID 對應的 metadata。"""

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
        """刪除 collection 與其中所有向量。"""

    def query(
        self, embedding: Sequence[float], top_k: int
    ) -> list[tuple[str, str, Mapping[str, Any], float]]:
        """依距離由近至遠回傳 ``(id, document, metadata, distance)``。"""


@runtime_checkable
class LanguageModel(Protocol):
    """根據已組裝完成的 Prompt 生成最終回答。"""

    @property
    def name(self) -> str: ...

    def complete(self, prompt: str) -> str: ...
