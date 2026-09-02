"""Stages 9-12 of the DAMA RAG demo: embedding, retrieval, reranking,
parent fetch and grounded generation over the chunks in ``output/``.

    from engine.pipeline import build_pipeline

    answer = build_pipeline().answer("What does a Data Steward do?")
    print(answer.answer)

The chunking stages (1-8) stay where they were: the ``build_*.py`` scripts at
the repository root. This package never parses a PDF; it reads
``combined-chunks.jsonl`` and ``table-parents.jsonl``.
"""

from engine.config import Settings

__all__ = ["Settings", "__version__"]
__version__ = "0.1.0"
