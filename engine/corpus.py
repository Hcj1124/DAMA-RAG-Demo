"""Loading the chunk pipeline's output into retrieval types.

This is the seam between stages 1-8 (Docling chunking, already done and
committed under ``output/``) and stages 9-12 (embedding, retrieval,
generation). Everything downstream sees :class:`Record` and
:class:`TableParent`, never the on-disk JSON envelope.

The corpus has two shapes, and the difference matters:

* ``text`` records have ``parent_id: null``. HybridChunker already cut them
  at section boundaries, so the record *is* both the retrieval unit and the
  context unit -- there is no larger text parent to fetch.
* ``table_child`` records are row slices of a canonical table. Retrieving one
  row group and showing the model only that row group loses the header
  semantics and the neighbouring rows, so a matched child is expanded to its
  whole parent from ``table-parents.jsonl``.
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
    return tuple(sorted(int(page) for page in raw["source"]["pages"]))


def _text_title(record: Mapping[str, Any]) -> str:
    """Section heading for a text record; the chunker guarantees exactly one."""

    headings = record["metadata"].get("headings") or []
    return " > ".join(str(h) for h in headings) or "Untitled section"


def _table_title(record: Mapping[str, Any]) -> str:
    """Caption for a table; two of the 44 tables have none, so fall back."""

    caption = record["metadata"].get("caption") or []
    if caption:
        return " ".join(str(part) for part in caption)
    index = record["metadata"].get("table_index")
    return f"Table (Docling index {index})" if index is not None else "Table"


@dataclass(frozen=True, slots=True)
class Record:
    """One embedding-ready child: a text chunk or a table row group."""

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
        """Identifies the exact bytes that were embedded.

        Stored beside the vector so a rebuild can skip records whose text has
        not changed, which turns a re-index from minutes into seconds.
        """

        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

    def chroma_metadata(self) -> dict[str, Any]:
        """Chroma accepts scalars only, so ``pages`` is flattened to a string."""

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
    """A whole canonical table -- header, every row, caption.

    ``text`` is the Markdown rendering rather than the tab-separated one:
    the header row survives, and every model in this stack reads Markdown
    tables far better than tab alignment.
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
    """Children to search over, plus the table parents to expand into."""

    records: tuple[Record, ...]
    parents: Mapping[str, TableParent]

    @property
    def record_index(self) -> Mapping[str, Record]:
        return {record.record_id: record for record in self.records}

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {"total": len(self.records)}
        for record in self.records:
            counts[record.content_type] = counts.get(record.content_type, 0) + 1
        counts["table_parent"] = len(self.parents)
        return counts

    @classmethod
    def load(cls, paths: Paths) -> "Corpus":
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

        # A table child whose parent is missing would silently lose its
        # header and neighbouring rows at answer time. Fail at load instead.
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
                f"are out of sync; re-run combine_chunks.py."
            )

        return cls(records=records, parents=parents)
