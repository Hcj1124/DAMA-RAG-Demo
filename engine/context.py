"""Stage 11 -- runtime parent fetch.

A ranked candidate is a *retrieval* unit. What the model should read is the
smallest unit that is still self-explanatory, and in this corpus those are
not the same thing for both content types:

* ``text`` -- HybridChunker already cut at section boundaries and kept the
  heading, so the chunk is self-explanatory on its own. There is no text
  parent file to fetch, and inventing one would be a different retrieval
  design, not a lookup.
* ``table_child`` -- a row group without its header is close to unreadable,
  and the answer often lives in a neighbouring row. So a matched child is
  expanded to its whole canonical parent from ``table-parents.jsonl``, and
  several children of the same table collapse into one block instead of
  spending three source slots on three slices of one table.

Blocks come back in rerank order and are capped at ``max_sources``, so the
best evidence survives the cap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from engine.corpus import TABLE_CHILD, Corpus
from engine.retrieval import Candidate

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ContextBlock:
    """One piece of evidence, ready to be numbered and put in the prompt."""

    title: str
    start_page: int
    end_page: int
    content_type: str
    matched_record_id: str
    parent_id: str | None
    passage: str
    """The chunk that actually matched the query."""
    section: str
    """The self-explanatory unit the model reads. Equals ``passage`` for text."""
    rerank_score: float | None

    @property
    def expanded(self) -> bool:
        """True when the section is larger than the chunk that matched."""

        return self.section != self.passage


class ContextResolver:
    """Expands ranked candidates into deduplicated context blocks."""

    def __init__(self, corpus: Corpus) -> None:
        self._corpus = corpus

    def resolve(
        self, candidates: Sequence[Candidate], *, max_sources: int
    ) -> list[ContextBlock]:
        blocks: list[ContextBlock] = []
        seen: set[str] = set()

        for candidate in candidates:
            key = (
                candidate.parent_id
                if candidate.content_type == TABLE_CHILD and candidate.parent_id
                else candidate.record_id
            )
            if key in seen:
                continue

            block = self._expand(candidate)
            if block is None:
                continue

            seen.add(key)
            blocks.append(block)
            if len(blocks) >= max_sources:
                break

        return blocks

    def _expand(self, candidate: Candidate) -> ContextBlock | None:
        if candidate.content_type != TABLE_CHILD or not candidate.parent_id:
            return ContextBlock(
                title=candidate.title,
                start_page=candidate.start_page,
                end_page=candidate.end_page,
                content_type=candidate.content_type,
                matched_record_id=candidate.record_id,
                parent_id=None,
                passage=candidate.text,
                section=candidate.text,
                rerank_score=candidate.rerank_score,
            )

        parent = self._corpus.parents.get(candidate.parent_id)
        if parent is None:
            # Corpus.load rejects orphans, so this means the index is older
            # than the chunk files. Degrade to the child rather than drop it.
            logger.warning(
                "Record %s references unknown parent %s; the index is out of "
                "step with table-parents.jsonl. Using the child alone.",
                candidate.record_id,
                candidate.parent_id,
            )
            return ContextBlock(
                title=candidate.title,
                start_page=candidate.start_page,
                end_page=candidate.end_page,
                content_type=candidate.content_type,
                matched_record_id=candidate.record_id,
                parent_id=candidate.parent_id,
                passage=candidate.text,
                section=candidate.text,
                rerank_score=candidate.rerank_score,
            )

        return ContextBlock(
            title=parent.title,
            start_page=parent.start_page,
            end_page=parent.end_page,
            content_type=candidate.content_type,
            matched_record_id=candidate.record_id,
            parent_id=parent.record_id,
            passage=candidate.text,
            section=parent.text,
            rerank_score=candidate.rerank_score,
        )
