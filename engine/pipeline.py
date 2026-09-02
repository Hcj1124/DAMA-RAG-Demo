"""The composition root.

The only module that decides which concrete adapter implements which port.
Everything above it depends on protocols; everything below it is replaceable;
a test builds the whole funnel out of fakes by calling the constructor
directly instead of the factory.
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
    """The end-to-end result of one question."""

    question: str
    answer: str
    citations: tuple[Citation, ...]
    model: str
    prompt_chars: int
    latency_s: float

    def to_dict(self) -> dict[str, Any]:
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
    """Question in, grounded answer out."""

    corpus: Corpus
    retriever: Retriever
    resolver: ContextResolver
    prompt_builder: PromptBuilder
    indexer: Indexer
    settings: Settings
    llm: LanguageModel | None = None

    def search(self, question: str) -> list[Candidate]:
        """Retrieval only -- no generation."""

        return self.retriever.search(question)

    def context(self, question: str) -> list[ContextBlock]:
        """Retrieval plus parent fetch -- what the model would have read."""

        return self.resolver.resolve(
            self.search(question),
            max_sources=self.settings.retrieval.max_sources,
        )

    def build_prompt(self, question: str) -> PromptBundle:
        return self.prompt_builder.build(question, self.context(question))

    def answer(self, question: str) -> Answer:
        """The full pipeline, timed end to end."""

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
    return SentenceTransformerEmbedder(settings.embedding, device=settings.device)


def build_reranker(settings: Settings) -> Reranker:
    return CrossEncoderReranker(settings.rerank, device=settings.device)


def build_store(settings: Settings) -> VectorStore:
    return ChromaVectorStore(
        settings.paths.chroma_dir, settings.retrieval.collection_name
    )


def build_pipeline(
    settings: Settings | None = None, *, with_llm: bool = True
) -> RagPipeline:
    """Build the default pipeline. Model weights load on first use."""

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
