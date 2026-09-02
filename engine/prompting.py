"""組裝只依據檢索證據回答的 grounded Prompt。

CLI、未來的評估程式與 UI 都共用這一套 Prompt，確保測試與實際交付行為一致。
``[Source N]`` 是回答句子對應回 DMBOK 頁碼的引用契約，不可任意變更格式。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from engine.config import PromptSettings
from engine.context import ContextBlock

logger = logging.getLogger(__name__)

# 語言規則先由 Python 明確選定，再插入系統指示，避免模型自行猜測。
_LANGUAGE_RULE = {
    "en": (
        "Write the entire answer in English. The context is in English; do "
        "not translate it into any other language."
    ),
    "zh-hant": (
        "Write the entire answer in Traditional Chinese, keeping DMBOK terms "
        "in English on first mention, for example 資料治理 (Data Governance)."
    ),
}

# 涵蓋 CJK 漢字與兩個假名區段，避免把日文問題誤判成英文。
_CJK_RANGES = (
    (0x3040, 0x30FF),  # 平假名與片假名
    (0x3400, 0x4DBF),  # CJK 擴充 A 區
    (0x4E00, 0x9FFF),  # CJK 統一漢字
    (0xF900, 0xFAFF),  # CJK 相容漢字
)


def detect_language(query: str) -> str:
    """依問題使用的字元判定回答語言。

    語言選擇由 Python 完成，再將明確規則送入 Prompt，避免模型因英文語料或自身
    語言偏好而判斷錯誤。目前只要出現 CJK 字元便使用繁體中文，純拉丁字元則使用英文。
    """

    for character in query:
        code = ord(character)
        if any(low <= code <= high for low, high in _CJK_RANGES):
            return "zh-hant"
    return "en"

_INSTRUCTIONS = """You are a DAMA-DMBOK knowledge assistant.

Answer the user's question using ONLY the supplied context.

RULES:

1. Do not use outside knowledge.
2. Do not invent information.
3. Preserve DAMA-DMBOK terminology.
4. Every factual claim must include an inline citation.
5. Use the citation format exactly like this: [Source 1]
6. If a claim is supported by several sources, cite them like: [Source 1][Source 2]
7. Never cite a source number that does not appear in the context.
8. Some sources are Markdown tables. Read them row by row and keep the header
   meaning attached to each cell you quote.
9. If the context does not contain enough information, say:
   "The provided DAMA-DMBOK context does not contain enough information."
10. Be concise but complete.
11. {language_rule}"""

_TEXT_BLOCK = """[{source_id}]

Title: {title}
Pages: {pages}

{section}"""

_TABLE_BLOCK = """[{source_id}]

Title: {title}
Pages: {pages}
Type: table

Rows that matched the question:
{passage}

Full table:
{section}"""

_TEMPLATE = """{instructions}

USER QUESTION:

{query}


CONTEXT:

{context}


ANSWER:"""

_TRUNCATION_NOTE = "\n\n[... section truncated to fit the context budget ...]"


@dataclass(frozen=True, slots=True)
class Citation:
    """一個上下文區塊提供給模型時所附的來源追溯資訊。"""

    source_id: str
    title: str
    start_page: int
    end_page: int
    content_type: str
    record_id: str
    parent_id: str | None

    @property
    def pages(self) -> str:
        """將單頁或頁碼範圍格式化為引用文字。"""
        if self.start_page == self.end_page:
            return f"p. {self.start_page}"
        return f"pp. {self.start_page}-{self.end_page}"

    def to_dict(self) -> dict[str, Any]:
        """將引用資訊轉為可序列化字典。"""
        return {
            "source_id": self.source_id,
            "title": self.title,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "content_type": self.content_type,
            "record_id": self.record_id,
            "parent_id": self.parent_id,
        }


@dataclass(frozen=True, slots=True)
class PromptBundle:
    """完整 Prompt，以及解讀回答引用所需的來源對照。"""

    prompt: str
    citations: tuple[Citation, ...]
    truncated: tuple[str, ...] = ()

    @property
    def prompt_chars(self) -> int:
        """回傳完整 Prompt 的字元數。"""
        return len(self.prompt)


class PromptBuilder:
    """將解析後的上下文區塊組成 Prompt 與引用對照。"""

    def __init__(self, settings: PromptSettings) -> None:
        self._settings = settings

    def build(self, query: str, blocks: Sequence[ContextBlock]) -> PromptBundle:
        """依字元預算加入來源、編號引用，並套用回答語言規則。"""
        rendered: list[str] = []
        citations: list[Citation] = []
        truncated: list[str] = []
        budget = self._settings.max_context_chars

        for block in blocks:
            if budget <= 0:
                break

            number = len(rendered) + 1
            source_id = f"Source {number}"
            pages = (
                str(block.start_page)
                if block.start_page == block.end_page
                else f"{block.start_page} - {block.end_page}"
            )
            template = _TABLE_BLOCK if block.expanded else _TEXT_BLOCK
            overhead = len(
                template.format(
                    source_id=source_id,
                    title=block.title,
                    pages=pages,
                    passage=block.passage,
                    section="",
                )
            )
            if overhead >= budget:
                logger.warning(
                    "Skipping %s because its context header/passage exceeds "
                    "the remaining %d-character budget",
                    block.matched_record_id,
                    budget,
                )
                break

            section, was_truncated = self._fit(
                block.section, budget - overhead
            )
            if was_truncated:
                truncated.append(block.matched_record_id)

            rendered_block = template.format(
                source_id=source_id,
                title=block.title,
                pages=pages,
                passage=block.passage,
                section=section,
            )
            rendered.append(rendered_block)
            budget -= len(rendered_block)
            citations.append(
                Citation(
                    source_id=source_id,
                    title=block.title,
                    start_page=block.start_page,
                    end_page=block.end_page,
                    content_type=block.content_type,
                    record_id=block.matched_record_id,
                    parent_id=block.parent_id,
                )
            )

        if truncated:
            logger.warning(
                "Truncated %d section(s) to stay within %d context characters",
                len(truncated),
                self._settings.max_context_chars,
            )

        language = self._settings.answer_language
        if language not in _LANGUAGE_RULE:
            language = detect_language(query)
        instructions = _INSTRUCTIONS.format(
            language_rule=_LANGUAGE_RULE[language]
        )
        return PromptBundle(
            prompt=_TEMPLATE.format(
                instructions=instructions,
                query=query,
                context="\n\n".join(rendered),
            ),
            citations=tuple(citations),
            truncated=tuple(truncated),
        )

    @staticmethod
    def _fit(text: str, budget: int) -> tuple[str, bool]:
        """將文字限制在字元預算內，並在截斷時加入可見標記。

        讓模型知道來源並不完整；即使剩餘預算很少，也盡量保留開頭而非空區塊。
        """

        if len(text) <= budget:
            return text, False
        if budget <= len(_TRUNCATION_NOTE):
            return _TRUNCATION_NOTE[: max(budget, 0)], True
        keep = max(budget - len(_TRUNCATION_NOTE), 0)
        return text[:keep].rstrip() + _TRUNCATION_NOTE, True
