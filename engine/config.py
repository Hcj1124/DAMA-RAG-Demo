"""集中管理所有可調整設定，是整個引擎唯一的設定來源。

其他模組不應寫死模型名稱、路徑或 top-k；各欄位都可由 ``DAMA_*``
環境變數覆寫，方便在不修改原始碼的情況下進行參數測試與評估。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from engine.errors import ConfigurationError

# 專案根目錄由此檔位置推導，避免執行時工作目錄影響預設路徑。
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str) -> str | None:
    """讀取 DAMA 前綴的環境變數，空字串視為未設定。"""
    value = os.getenv(f"DAMA_{name}")
    return value if value else None


def _env_int(name: str, default: int) -> int:
    """讀取整數環境變數，格式錯誤時回報可理解的設定錯誤。"""
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ConfigurationError(
            f"DAMA_{name} must be an integer, got {raw!r}"
        ) from error


def _env_float(name: str, default: float) -> float:
    """讀取浮點數環境變數，未設定時使用預設值。"""
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as error:
        raise ConfigurationError(
            f"DAMA_{name} must be a number, got {raw!r}"
        ) from error


def _env_bool(name: str, default: bool) -> bool:
    """將常見真值字串轉成布林設定。"""
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Paths:
    """記錄 chunk 管線的輸出位置；建立後皆使用絕對路徑。"""

    root: Path = PROJECT_ROOT
    output_dir: Path = PROJECT_ROOT / "output"

    @property
    def combined_chunks(self) -> Path:
        """回傳文字與表格子 chunk 的合併檔，也就是實際檢索單位。"""
        return self.output_dir / "chunks" / "combined-chunks.jsonl"

    @property
    def table_parents(self) -> Path:
        """回傳完整表格母紀錄檔，供命中的表格子 chunk 展開上下文。"""
        return self.output_dir / "tables" / "table-parents.jsonl"

    @property
    def chroma_dir(self) -> Path:
        """回傳本機 Chroma 向量資料庫目錄。"""
        return self.output_dir / "chroma_db"

    @classmethod
    def from_env(cls) -> "Paths":
        """依環境變數建立路徑設定，保留相對輸出目錄的既有語意。"""
        root = Path(_env("ROOT") or PROJECT_ROOT).resolve()
        output = _env("OUTPUT_DIR")
        return cls(
            root=root,
            output_dir=Path(output).resolve() if output else root / "output",
        )


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    """索引與查詢共用的 bi-encoder 設定。

    ``BAAI/bge-m3`` 與 chunk 階段記錄的 tokenizer 相同，因此 chunk 大小與
    embedding 視窗一致。它支援多語言且不需指令前綴，所以兩種 prompt
    預設為 ``None``；若改用需指令的模型，只需調整設定而不用改流程。
    """

    model: str = "BAAI/bge-m3"
    batch_size: int = 8
    normalize: bool = True
    max_seq_length: int | None = 1024
    query_prompt: str | None = None
    document_prompt: str | None = None

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        """從環境變數載入 embedding 模型與編碼參數。"""
        max_seq = _env_int("EMBEDDING_MAX_SEQ_LENGTH", 1024)
        return cls(
            model=_env("EMBEDDING_MODEL") or "BAAI/bge-m3",
            batch_size=_env_int("EMBEDDING_BATCH_SIZE", 8),
            normalize=_env_bool("EMBEDDING_NORMALIZE", True),
            max_seq_length=max_seq if max_seq > 0 else None,
            query_prompt=_env("EMBEDDING_QUERY_PROMPT"),
            document_prompt=_env("EMBEDDING_DOCUMENT_PROMPT"),
        )


@dataclass(frozen=True, slots=True)
class RerankSettings:
    """套用於向量檢索候選清單的 cross-encoder 設定。

    分數是未校準的原始 logits，數值越高代表越相關；若要設定門檻，必須先以
    此語料重新校準，不能直接沿用其他模型或資料集的數值。
    """

    model: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 8
    max_length: int = 1024

    @classmethod
    def from_env(cls) -> "RerankSettings":
        """從環境變數載入 reranker 模型與批次參數。"""
        return cls(
            model=_env("RERANK_MODEL") or "BAAI/bge-reranker-v2-m3",
            batch_size=_env_int("RERANK_BATCH_SIZE", 8),
            max_length=_env_int("RERANK_MAX_LENGTH", 1024),
        )


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """控制向量召回、重排序及最終來源數量的漏斗設定。"""

    retrieve_k: int = 20
    rerank_k: int = 8
    max_sources: int = 4
    collection_name: str = "dama_chunks"

    def __post_init__(self) -> None:
        """驗證各階段候選數量彼此相容。"""
        if self.rerank_k > self.retrieve_k:
            raise ConfigurationError(
                f"rerank_k ({self.rerank_k}) cannot exceed "
                f"retrieve_k ({self.retrieve_k})"
            )
        if self.max_sources < 1:
            raise ConfigurationError("max_sources must be at least 1")

    @classmethod
    def from_env(cls) -> "RetrievalSettings":
        """從環境變數載入檢索漏斗與 collection 設定。"""
        return cls(
            retrieve_k=_env_int("RETRIEVE_K", 20),
            rerank_k=_env_int("RERANK_K", 8),
            max_sources=_env_int("MAX_SOURCES", 4),
            collection_name=_env("COLLECTION_NAME") or "dama_chunks",
        )


@dataclass(frozen=True, slots=True)
class PromptSettings:
    """控制 Prompt 組裝長度與回答語言。"""

    max_context_chars: int = 60_000
    answer_language: str = "auto"

    @classmethod
    def from_env(cls) -> "PromptSettings":
        """從環境變數載入 Prompt 設定。"""
        return cls(
            max_context_chars=_env_int("MAX_CONTEXT_CHARS", 60_000),
            answer_language=_env("ANSWER_LANGUAGE") or "auto",
        )


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """透過 Ollama 執行本機生成的設定。

    預設關閉 thinking，因為推理軌跡會增加延遲，且目前流程不會保留它。
    ``num_ctx`` 則明確控制模型可讀取的 Prompt 上下文大小。
    """

    model: str = "qwen3.6:35b-a3b"
    host: str | None = None
    temperature: float = 0.0
    num_ctx: int = 32_768
    think: bool = False

    @classmethod
    def from_env(cls) -> "GenerationSettings":
        """從 DAMA 與 Ollama 環境變數載入生成設定。"""
        return cls(
            model=_env("OLLAMA_MODEL") or "qwen3.6:35b-a3b",
            host=_env("OLLAMA_HOST") or os.getenv("OLLAMA_HOST") or None,
            temperature=_env_float("TEMPERATURE", 0.0),
            num_ctx=_env_int("NUM_CTX", 32_768),
            think=_env_bool("THINK", False),
        )


@dataclass(frozen=True, slots=True)
class Settings:
    """彙整引擎執行所需的全部設定。"""

    paths: Paths = field(default_factory=Paths)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    rerank: RerankSettings = field(default_factory=RerankSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    prompt: PromptSettings = field(default_factory=PromptSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    device: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        """從環境變數一次建立完整設定物件。"""
        return cls(
            paths=Paths.from_env(),
            embedding=EmbeddingSettings.from_env(),
            rerank=RerankSettings.from_env(),
            retrieval=RetrievalSettings.from_env(),
            prompt=PromptSettings.from_env(),
            generation=GenerationSettings.from_env(),
            device=_env("DEVICE"),
        )

    def with_overrides(self, **kwargs: Any) -> "Settings":
        """以不可變資料類別的方式建立指定欄位覆寫版本。"""
        return replace(self, **kwargs)

    def describe(self) -> dict[str, Any]:
        """產生可直接列印的扁平設定摘要，供 ``dama-rag info`` 使用。"""

        return {
            "output_dir": str(self.paths.output_dir),
            "chroma_dir": str(self.paths.chroma_dir),
            "collection": self.retrieval.collection_name,
            "embedding_model": self.embedding.model,
            "rerank_model": self.rerank.model,
            "llm_model": self.generation.model,
            "retrieve_k": self.retrieval.retrieve_k,
            "rerank_k": self.retrieval.rerank_k,
            "max_sources": self.retrieval.max_sources,
            "num_ctx": self.generation.num_ctx,
            "answer_language": self.prompt.answer_language,
            "device": self.device or "auto",
        }
