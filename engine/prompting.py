"""Assembling the grounded prompt.

There is exactly one prompt in this project. Every entry point -- the CLI,
any future evaluation harness, any UI -- goes through it, so what is measured
and what is shipped stay the same system.

The ``[Source N]`` citation contract is load-bearing: it is what maps a
sentence in the answer back to a page range in the DMBOK. Changing the marker
format silently breaks citation checking downstream.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from engine.config import PromptSettings
from engine.context import ContextBlock

logger = logging.getLogger(__name__)

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

# CJK ideographs, plus the two kana blocks so a Japanese question is not
# mistaken for an English one.
_CJK_RANGES = (
    (0x3040, 0x30FF),  # hiragana + katakana
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xF900, 0xFAFF),  # compatibility ideographs
)


def detect_language(query: str) -> str:
    """Pick the answer language from the question's script.

    ``auto`` used to be a sentence in the prompt asking the model to match
    the question's language, and qwen3.6 answered English questions in
    Chinese anyway -- the corpus is English, the model is Chinese-strong, and
    one instruction among eleven lost. This is a decision Python can make
    reliably, so it is made here and the prompt receives a definite rule.

    Any CJK character means Chinese; the DMBOK terms an English question
    contains are all Latin script, so the reverse misfire cannot happen.
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
    """The provenance of one context block offered to the model."""

    source_id: str
    title: str
    start_page: int
    end_page: int
    content_type: str
    record_id: str
    parent_id: str | None

    @property
    def pages(self) -> str:
        if self.start_page == self.end_page:
            return f"p. {self.start_page}"
        return f"pp. {self.start_page}-{self.end_page}"

    def to_dict(self) -> dict[str, Any]:
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
    """A prompt plus the citation map needed to interpret its answer."""

    prompt: str
    citations: tuple[Citation, ...]
    truncated: tuple[str, ...] = ()

    @property
    def prompt_chars(self) -> int:
        return len(self.prompt)


class PromptBuilder:
    """Turns resolved context blocks into a prompt and its citation map."""

    def __init__(self, settings: PromptSettings) -> None:
        self._settings = settings

    def build(self, query: str, blocks: Sequence[ContextBlock]) -> PromptBundle:
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
        """Clip ``text`` to ``budget`` characters, marking it when clipped.

        Truncation is visible to the model so it does not read a cut section
        as a complete one. An exhausted budget still yields a short head
        rather than an empty block.
        """

        if len(text) <= budget:
            return text, False
        if budget <= len(_TRUNCATION_NOTE):
            return _TRUNCATION_NOTE[: max(budget, 0)], True
        keep = max(budget - len(_TRUNCATION_NOTE), 0)
        return text[:keep].rstrip() + _TRUNCATION_NOTE, True
