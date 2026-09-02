"""將 chunk 管線輸出轉成檢索引擎使用的資料型別。

這裡銜接 Docling 切塊與 embedding／檢索／生成流程。下游只操作
:class:`Record` 與 :class:`TableParent`，不直接依賴磁碟上的 JSON 結構。

``text`` 本身同時是檢索與上下文單位；``table_child`` 則是表格列片段，命中後
必須透過 ``parent_id`` 取回完整表格，才能保留表頭語意與相鄰資料列。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from engine.config import Paths
from engine.errors import CorpusError

TEXT = "text"
TABLE_CHILD = "table_child"
TABLE_PARENT = "table_parent"


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """逐行讀取 JSONL，並將缺檔或格式錯誤轉成語料錯誤。"""
    if not path.exists():
        raise CorpusError(
            f"Missing {path.name} at {path}.\n"
            f"Run the chunk pipeline first (see README stages 1-8), or point "
            f"DAMA_OUTPUT_DIR at a directory that has it."
        )
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise CorpusError(
                    f"{path.name} line {line_number} is not valid JSON: {error}"
                ) from error


def _pages(raw: Mapping[str, Any]) -> tuple[int, ...]:
    """從共用 schema 擷取並排序 PDF 頁碼。"""
    return tuple(sorted(int(page) for page in raw["source"]["pages"]))


def _text_title(record: Mapping[str, Any]) -> str:
    """組合文字紀錄的章名與局部節標題。"""

    headings = record["metadata"].get("headings") or []
    section = " > ".join(str(h) for h in headings) or "Untitled section"
    chapter = str(record["metadata"].get("chapter") or "").strip()
    if chapter and chapter != section:
        return f"{chapter} > {section}"
    return section


def _table_title(record: Mapping[str, Any]) -> str:
    """取得表格標題；缺少 caption 時以 Docling 表格索引產生替代名稱。"""

    caption = record["metadata"].get("caption") or []
    if caption:
        return " ".join(str(part) for part in caption)
    index = record["metadata"].get("table_index")
    return f"Table (Docling index {index})" if index is not None else "Table"


@dataclass(frozen=True, slots=True)
class Record:
    """一筆可直接 embedding 的文字 chunk 或表格列群組。"""

    record_id: str
    parent_id: str | None
    content_type: str
    document_id: str
    title: str
    pages: tuple[int, ...]
    text: str
    token_count: int

    @property
    def start_page(self) -> int:
        return self.pages[0]

    @property
    def end_page(self) -> int:
        return self.pages[-1]

    @property
    def content_hash(self) -> str:
        """識別實際送入 embedding 的文字內容。

        此雜湊會與向量一同儲存，重建索引時可跳過內容未變的紀錄。
        """

        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    def chroma_metadata(self) -> dict[str, Any]:
        """建立 Chroma metadata；因其只接受純量，頁碼需展平成字串。"""

        return {
            "content_type": self.content_type,
            "document_id": self.document_id,
            "title": self.title,
            "parent_id": self.parent_id or "",
            "start_page": self.start_page,
            "end_page": self.end_page,
            "pages": ",".join(str(page) for page in self.pages),
            "token_count": self.token_count,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "Record":
        """將共用 JSON schema 紀錄轉成引擎內部的檢索紀錄。"""
        content_type = str(raw["content_type"])
        title = (
            _text_title(raw) if content_type == TEXT else _table_title(raw)
        )
        return cls(
            record_id=str(raw["record_id"]),
            parent_id=raw.get("parent_id") or None,
            content_type=content_type,
            document_id=str(raw["source"]["document_id"]),
            title=title,
            pages=_pages(raw),
            text=str(raw["content"]["text"]),
            token_count=int(raw["metadata"].get("token_count", 0)),
        )


@dataclass(frozen=True, slots=True)
class TableParent:
    """包含 caption、表頭與所有資料列的完整標準化表格。

    ``text`` 優先使用 Markdown，因為它能保留表頭結構，模型也比讀取 tab 對齊
    文字更容易理解欄位關係。
    """

    record_id: str
    document_id: str
    title: str
    pages: tuple[int, ...]
    text: str
    num_rows: int
    num_cols: int

    @property
    def start_page(self) -> int:
        return self.pages[0]

    @property
    def end_page(self) -> int:
        return self.pages[-1]

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "TableParent":
        """將共用 JSON schema 紀錄轉成完整表格母紀錄。"""
        content = raw["content"]
        return cls(
            record_id=str(raw["record_id"]),
            document_id=str(raw["source"]["document_id"]),
            title=_table_title(raw),
            pages=_pages(raw),
            text=str(content.get("markdown") or content["text"]),
            num_rows=int(raw["metadata"].get("num_rows", 0)),
            num_cols=int(raw["metadata"].get("num_cols", 0)),
        )


@dataclass(frozen=True, slots=True)
class Corpus:
    """保存可檢索子紀錄，以及命中表格時可展開的母紀錄。"""

    records: tuple[Record, ...]
    parents: Mapping[str, TableParent]

    @property
    def record_index(self) -> Mapping[str, Record]:
        """建立 record_id 到檢索紀錄的快速查找表。"""
        return {record.record_id: record for record in self.records}

    def counts(self) -> dict[str, int]:
        """統計各內容類型及完整表格的數量。"""
        counts: dict[str, int] = {"total": len(self.records)}
        for record in self.records:
            counts[record.content_type] = counts.get(record.content_type, 0) + 1
        counts["table_parent"] = len(self.parents)
        return counts

    @classmethod
    def load(cls, paths: Paths) -> "Corpus":
        """從磁碟載入語料，並確認每個表格子紀錄都有有效母紀錄。"""
        records = tuple(
            Record.from_raw(raw) for raw in _read_jsonl(paths.combined_chunks)
        )
        if not records:
            raise CorpusError(f"{paths.combined_chunks} contains no records")

        parents = {
            parent.record_id: parent
            for parent in (
                TableParent.from_raw(raw)
                for raw in _read_jsonl(paths.table_parents)
            )
        }

        # 缺少母紀錄的表格 child 會在回答時遺失表頭與相鄰列，因此載入階段即拒絕。
        orphans = sorted(
            record.record_id
            for record in records
            if record.content_type == TABLE_CHILD
            and record.parent_id not in parents
        )
        if orphans:
            raise CorpusError(
                f"{len(orphans)} table child record(s) reference a parent that "
                f"is not in {paths.table_parents.name}, starting with "
                f"{orphans[0]}. combined-chunks.jsonl and table-parents.jsonl "
                f"are out of sync; re-run python -m ingest.combine_chunks."
            )

        return cls(records=records, parents=parents)
