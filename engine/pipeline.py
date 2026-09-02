"""組裝整個 RAG 系統的 composition root。

只有此模組決定各介面採用哪個具體 adapter；上層依賴協定，下層實作可替換。
測試也能直接以假物件建立整條管線，而不必呼叫預設工廠。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from engine.adapters.embedding import SentenceTransformerEmbedder
from engine.adapters.llm_ollama import OllamaLanguageModel
from engine.adapters.reranking import CrossEncoderReranker
from engine.adapters.store_chroma import ChromaVectorStore
from engine.config import Settings
from engine.context import ContextBlock, ContextResolver
from engine.corpus import Corpus
from engine.indexing import Indexer
from engine.ports import Embedder, LanguageModel, Reranker, VectorStore
from engine.prompting import Citation, PromptBuilder, PromptBundle
from engine.retrieval import Candidate, Retriever

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Answer:
    """單次問題通過端到端管線後的結果。"""

    question: str
    answer: str
    citations: tuple[Citation, ...]
    model: str
    prompt_chars: int
    latency_s: float

    def to_dict(self) -> dict[str, Any]:
        """將回答、引用與效能資訊轉成可序列化字典。"""
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
            "model": self.model,
            "prompt_chars": self.prompt_chars,
            "latency_s": round(self.latency_s, 3),
        }


@dataclass(slots=True)
class RagPipeline:
    """接收問題並輸出有來源依據回答的主要 RAG 管線。"""

    corpus: Corpus
    retriever: Retriever
    resolver: ContextResolver
    prompt_builder: PromptBuilder
    indexer: Indexer
    settings: Settings
    llm: LanguageModel | None = None

    def search(self, question: str) -> list[Candidate]:
        """只執行檢索與重排序，不呼叫語言模型。"""

        return self.retriever.search(question)

    def context(self, question: str) -> list[ContextBlock]:
        """執行檢索及母紀錄展開，回傳模型實際會讀取的內容。"""

        return self.resolver.resolve(
            self.search(question),
            max_sources=self.settings.retrieval.max_sources,
        )

    def build_prompt(self, question: str) -> PromptBundle:
        """將問題與解析後的來源組成最終 Prompt。"""
        return self.prompt_builder.build(question, self.context(question))

    def answer(self, question: str) -> Answer:
        """執行完整問答管線，並記錄端到端耗時。"""

        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        if self.llm is None:
            raise ValueError(
                "This pipeline was built without a language model; use "
                "search() or build_prompt(), or rebuild with with_llm=True."
            )

        started = time.perf_counter()
        bundle = self.build_prompt(question)
        text = self.llm.complete(bundle.prompt)
        return Answer(
            question=question,
            answer=text,
            citations=bundle.citations,
            model=self.llm.name,
            prompt_chars=bundle.prompt_chars,
            latency_s=time.perf_counter() - started,
        )


def build_embedder(settings: Settings) -> Embedder:
    """依集中設定建立預設 embedding adapter。"""
    return SentenceTransformerEmbedder(settings.embedding, device=settings.device)


def build_reranker(settings: Settings) -> Reranker:
    """依集中設定建立預設 reranker adapter。"""
    return CrossEncoderReranker(settings.rerank, device=settings.device)


def build_store(settings: Settings) -> VectorStore:
    """依集中設定建立 Chroma 向量資料庫 adapter。"""
    return ChromaVectorStore(
        settings.paths.chroma_dir, settings.retrieval.collection_name
    )


def build_pipeline(
    settings: Settings | None = None, *, with_llm: bool = True
) -> RagPipeline:
    """組裝預設管線；模型權重會延後到第一次使用時才載入。"""

    settings = settings or Settings.from_env()
    logger.debug("Building pipeline with %s", settings.describe())

    embedder = build_embedder(settings)
    store = build_store(settings)
    corpus = Corpus.load(settings.paths)

    return RagPipeline(
        corpus=corpus,
        retriever=Retriever(
            embedder=embedder,
            reranker=build_reranker(settings),
            store=store,
            settings=settings.retrieval,
        ),
        resolver=ContextResolver(corpus),
        prompt_builder=PromptBuilder(settings.prompt),
        indexer=Indexer(embedder=embedder, store=store),
        settings=settings,
        llm=OllamaLanguageModel(settings.generation) if with_llm else None,
    )
