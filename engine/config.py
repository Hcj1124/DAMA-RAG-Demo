"""Configuration -- the single source of truth for every tunable value.

No other module may hard-code a model name, a path or a top-k. Every field
can be overridden by a ``DAMA_*`` environment variable so that parameter
sweeps and evaluation runs never require editing source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from engine.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str) -> str | None:
    value = os.getenv(f"DAMA_{name}")
    return value if value else None


def _env_int(name: str, default: int) -> int:
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
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Paths:
    """Where the chunk pipeline's outputs live. All paths absolute."""

    root: Path = PROJECT_ROOT
    output_dir: Path = PROJECT_ROOT / "output"

    @property
    def combined_chunks(self) -> Path:
        """Children: text records + table children. The retrieval unit."""
        return self.output_dir / "combined-chunks.jsonl"

    @property
    def table_parents(self) -> Path:
        """Canonical whole tables. The context unit behind a table child."""
        return self.output_dir / "table-parents.jsonl"

    @property
    def chroma_dir(self) -> Path:
        return self.output_dir / "chroma_db"

    @classmethod
    def from_env(cls) -> "Paths":
        root = Path(_env("ROOT") or PROJECT_ROOT).resolve()
        output = _env("OUTPUT_DIR")
        return cls(
            root=root,
            output_dir=Path(output).resolve() if output else root / "output",
        )


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    """Bi-encoder used for both indexing and querying.

    ``BAAI/bge-m3`` is the model the chunker already tokenised against
    (``metadata.tokenizer_name``), so chunk sizes and the embedding window
    agree by construction. It is multilingual with a 8192-token window and
    needs no instruction prefix, which is why both prompts default to
    ``None``; an instruction-aware model such as ``Qwen3-Embedding-0.6B`` is
    a configuration change, not a code change.
    """

    model: str = "BAAI/bge-m3"
    batch_size: int = 8
    normalize: bool = True
    max_seq_length: int | None = 1024
    query_prompt: str | None = None
    document_prompt: str | None = None

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
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
    """Cross-encoder applied to the shortlist from vector search.

    ``BAAI/bge-reranker-v2-m3`` shares the XLM-RoBERTa backbone with the
    embedder, so both stages agree on what "relevant" means across languages.
    Scores are raw logits: higher is better, but never threshold them without
    recalibrating on this corpus.
    """

    model: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 8
    max_length: int = 1024

    @classmethod
    def from_env(cls) -> "RerankSettings":
        return cls(
            model=_env("RERANK_MODEL") or "BAAI/bge-reranker-v2-m3",
            batch_size=_env_int("RERANK_BATCH_SIZE", 8),
            max_length=_env_int("RERANK_MAX_LENGTH", 1024),
        )


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    """How many candidates survive each stage of the funnel."""

    retrieve_k: int = 20
    rerank_k: int = 8
    max_sources: int = 4
    collection_name: str = "dama_chunks"

    def __post_init__(self) -> None:
        if self.rerank_k > self.retrieve_k:
            raise ConfigurationError(
                f"rerank_k ({self.rerank_k}) cannot exceed "
                f"retrieve_k ({self.retrieve_k})"
            )
        if self.max_sources < 1:
            raise ConfigurationError("max_sources must be at least 1")

    @classmethod
    def from_env(cls) -> "RetrievalSettings":
        return cls(
            retrieve_k=_env_int("RETRIEVE_K", 20),
            rerank_k=_env_int("RERANK_K", 8),
            max_sources=_env_int("MAX_SOURCES", 4),
            collection_name=_env("COLLECTION_NAME") or "dama_chunks",
        )


@dataclass(frozen=True, slots=True)
class PromptSettings:
    """Prompt assembly limits."""

    max_context_chars: int = 60_000
    answer_language: str = "auto"

    @classmethod
    def from_env(cls) -> "PromptSettings":
        return cls(
            max_context_chars=_env_int("MAX_CONTEXT_CHARS", 60_000),
            answer_language=_env("ANSWER_LANGUAGE") or "auto",
        )


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Local generation through Ollama.

    ``qwen3.6:35b-a3b`` is a 35B MoE with ~3B active parameters: 35B-class
    quality at small-model speed, with strong Traditional Chinese. Thinking
    is off by default -- the trace costs latency and is discarded anyway.
    """

    model: str = "qwen3.6:35b-a3b"
    host: str | None = None
    temperature: float = 0.0
    num_ctx: int = 32_768
    think: bool = False

    @classmethod
    def from_env(cls) -> "GenerationSettings":
        return cls(
            model=_env("OLLAMA_MODEL") or "qwen3.6:35b-a3b",
            host=_env("OLLAMA_HOST") or os.getenv("OLLAMA_HOST") or None,
            temperature=_env_float("TEMPERATURE", 0.0),
            num_ctx=_env_int("NUM_CTX", 32_768),
            think=_env_bool("THINK", False),
        )


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the engine needs to know, in one object."""

    paths: Paths = field(default_factory=Paths)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    rerank: RerankSettings = field(default_factory=RerankSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    prompt: PromptSettings = field(default_factory=PromptSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    device: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
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
        return replace(self, **kwargs)

    def describe(self) -> dict[str, Any]:
        """A flat, printable summary -- used by ``dama-rag info``."""

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
