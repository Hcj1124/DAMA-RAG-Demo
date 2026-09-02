"""定義附帶修正方向的引擎錯誤，而不只提供堆疊資訊。"""

from __future__ import annotations


class EngineError(Exception):
    """本套件主動拋出的所有可預期錯誤之基底類別。"""


class ConfigurationError(EngineError):
    """設定缺漏或格式不正確。"""


class CorpusError(EngineError):
    """Chunk 檔案不存在、無法讀取或內部關聯不一致。"""


class IndexNotBuiltError(EngineError):
    """尚未建立可供查詢的向量索引。"""
    def __init__(self, collection: str, path: object) -> None:
        super().__init__(
            f"Collection '{collection}' does not exist under {path}.\n"
            f"Build it first:  dama-rag index"
        )


class IndexStaleError(EngineError):
    """索引由不同的 embedding 設定建立，已不適合目前查詢。"""

    def __init__(self, *, indexed_with: str, querying_with: str) -> None:
        super().__init__(
            f"The index was built with '{indexed_with}' but you are querying "
            f"with '{querying_with}'. Comparing vectors from incompatible "
            f"embedding configurations returns plausible-looking nonsense, "
            f"so this is refused.\n"
            f"Rebuild:  dama-rag index --rebuild"
        )


class LanguageModelError(EngineError):
    """語言模型生成失敗或未回傳可用內容。"""
