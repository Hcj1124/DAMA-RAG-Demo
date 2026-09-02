"""第 11 階段：在執行時將檢索結果解析成可理解的上下文。

文字 chunk 已保留節標題，可直接作為上下文；表格子 chunk 缺少完整表頭與相鄰列，
因此必須展開成標準化母表格。同一母表格的多個命中只占一個來源名額，最後依 rerank
順序保留至 ``max_sources``。
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
    """一段可編號並放入 Prompt 的證據內容。"""

    title: str
    start_page: int
    end_page: int
    content_type: str
    matched_record_id: str
    parent_id: str | None
    passage: str
    """實際命中查詢的 chunk。"""
    section: str
    """模型實際閱讀的完整單位；文字類型與 ``passage`` 相同。"""
    rerank_score: float | None

    @property
    def expanded(self) -> bool:
        """表示模型讀取的內容是否已從命中 chunk 展開。"""

        return self.section != self.passage


class ContextResolver:
    """將排序後的候選項展開並去重，形成上下文區塊。"""

    def __init__(self, corpus: Corpus) -> None:
        self._corpus = corpus

    def resolve(
        self, candidates: Sequence[Candidate], *, max_sources: int
    ) -> list[ContextBlock]:
        """依 rerank 順序解析候選項，並限制最終來源數量。"""
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
        """文字保留原 chunk；表格 child 則盡可能替換成完整母表格。"""
        if candidate.content_type != TABLE_CHILD or not candidate.parent_id:
            record = self._corpus.record_index.get(candidate.record_id)
            return ContextBlock(
                # Chroma 標題只是索引 metadata；此處改讀目前語料標題，更新來源名稱時
                # 就不必為未變動的 embedding 重建索引。
                title=record.title if record is not None else candidate.title,
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
            # Corpus.load 已拒絕孤兒 child，因此這代表索引比 chunk 檔舊；保留 child
            # 作為降級內容，避免直接遺失這筆證據。
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
