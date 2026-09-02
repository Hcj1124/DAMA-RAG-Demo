"""Phase 5: split canonical table parents into row-aware embedding children.

The final serialized ``content.text`` is counted with the configured embedding
tokenizer. Header rows and captions are repeated, while data rows are assigned to
exactly one child. The resulting JSONL records conform to ``chunk-schema.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from jsonschema import Draft202012Validator
from transformers.utils import logging as transformers_logging


DEFAULT_TOKENIZER_MODEL = "BAAI/bge-m3"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


@dataclass(frozen=True)
class TableRow:
    index: int
    values: list[str]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def escape_markdown(value: str) -> str:
    return value.replace("|", "\\|")


def normalize_value(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def build_rows(structured: dict[str, Any]) -> list[TableRow]:
    num_rows = int(structured["num_rows"])
    num_cols = int(structured["num_cols"])
    matrix = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    for cell in structured["cells"]:
        row = int(cell["start_row_offset_idx"])
        col = int(cell["start_col_offset_idx"])
        if 0 <= row < num_rows and 0 <= col < num_cols:
            # A spanned cell is written once at its top-left location. Span metadata
            # remains authoritative in the parent record.
            value = normalize_value(str(cell.get("text", "")))
            if value:
                matrix[row][col] = value
    return [TableRow(index=index, values=values) for index, values in enumerate(matrix)]


def explicit_header_indices(structured: dict[str, Any]) -> list[int]:
    return sorted(
        {
            int(cell["start_row_offset_idx"])
            for cell in structured["cells"]
            if cell.get("column_header")
        }
    )


def display_header(rows: list[TableRow], header_indices: list[int], num_cols: int) -> tuple[list[str], bool]:
    if not header_indices:
        return [f"Column {index + 1}" for index in range(num_cols)], True
    values = []
    used_fallback = False
    for col in range(num_cols):
        parts = [rows[row].values[col] for row in header_indices if rows[row].values[col]]
        value = " / ".join(parts)
        if not value:
            value = f"Column {col + 1}"
            used_fallback = True
        values.append(value)
    return values, used_fallback


def markdown_table(header: list[str], body: list[TableRow]) -> str:
    lines = ["| " + " | ".join(escape_markdown(value) for value in header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    lines.extend("| " + " | ".join(escape_markdown(value) for value in row.values) + " |" for row in body)
    return "\n".join(lines)


def serialize_markdown(captions: list[str], header: list[str], body: list[TableRow]) -> str:
    sections = []
    if captions:
        sections.append("\n".join(f"Caption: {caption}" for caption in captions))
    sections.append(markdown_table(header, body))
    return "\n".join(sections)


def is_key_value_table(rows: list[TableRow], header_indices: list[int], num_cols: int) -> bool:
    if header_indices or num_cols != 2 or not rows:
        return False
    complete_pairs = [row for row in rows if row.values[0] and row.values[1]]
    if len(complete_pairs) != len(rows):
        return False
    numeric_values = sum(
        1
        for row in complete_pairs
        if sum(char.isdigit() for char in row.values[1]) >= max(1, len(row.values[1]) // 2)
    )
    return numeric_values / len(complete_pairs) >= 0.6


def serialize_row_attribute_value(row: TableRow, header: list[str], key_value_mode: bool) -> str:
    if key_value_mode:
        attribute = row.values[0].rstrip(":").strip()
        return f"Row {row.index}:\n{attribute}: {row.values[1]}"
    fields = [
        f"{header[col]}: {value}"
        for col, value in enumerate(row.values)
        if value
    ]
    if not fields:
        return f"Row {row.index}: (empty)"
    return f"Row {row.index}:\n" + "\n".join(fields)


def serialize_embedding_text(
    captions: list[str],
    header: list[str],
    body: list[TableRow],
    key_value_mode: bool,
) -> str:
    sections = []
    if captions:
        sections.append("\n".join(f"Table: {caption}" for caption in captions))
    sections.extend(serialize_row_attribute_value(row, header, key_value_mode) for row in body)
    if not body:
        sections.append("Columns: " + "; ".join(header))
    return "\n\n".join(sections)


def token_count(tokenizer: HuggingFaceTokenizer, text: str) -> int:
    return int(tokenizer.count_tokens(text=text))


def split_oversize_row(
    row: TableRow,
    captions: list[str],
    header: list[str],
    tokenizer: HuggingFaceTokenizer,
    hard_max_tokens: int,
    key_value_mode: bool,
) -> list[str]:
    """Split an exceptional oversized row without overlapping its text."""
    row_text = serialize_row_attribute_value(row, header, key_value_mode)
    units = re.findall(r"\S+\s*", row_text)
    prefix_parts = ["\n".join(f"Table: {caption}" for caption in captions)] if captions else []
    prefix_parts.append("Columns: " + "; ".join(header))
    prefix = "\n\n".join(prefix_parts) + f"\n\nRow {row.index} fragment: "
    if token_count(tokenizer, prefix) >= hard_max_tokens:
        raise ValueError(f"Table context alone reaches the token limit for row {row.index}")

    fragments: list[str] = []
    current = ""
    for unit in units:
        candidate = current + unit
        if token_count(tokenizer, prefix + candidate) <= hard_max_tokens:
            current = candidate
            continue
        if current:
            fragments.append(current.rstrip())
            current = ""
        if token_count(tokenizer, prefix + unit) <= hard_max_tokens:
            current = unit
            continue
        # Extremely long unbroken strings are split at character boundaries.
        remainder = unit
        while remainder:
            low, high = 1, len(remainder)
            best = 0
            while low <= high:
                middle = (low + high) // 2
                if token_count(tokenizer, prefix + remainder[:middle]) <= hard_max_tokens:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if best == 0:
                raise ValueError(f"Cannot fit any content for oversized row {row.index}")
            fragments.append(remainder[:best].rstrip())
            remainder = remainder[best:]
    if current:
        fragments.append(current.rstrip())
    return fragments


def make_child(
    parent: dict[str, Any],
    child_ordinal: int,
    captions: list[str],
    header: list[str],
    header_indices: list[int],
    synthetic_header: bool,
    rows: list[TableRow],
    embedding_text: str,
    markdown_text: str,
    tokenizer_name: str,
    target_tokens: int,
    hard_max_tokens: int,
    tokenizer: HuggingFaceTokenizer,
    key_value_mode: bool,
    row_fragment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    table_index = int(parent["source"]["locators"]["table_index"])
    document_id = parent["source"]["document_id"]
    record_id = f"{document_id}:table:{table_index:03d}:child:{child_ordinal:03d}"
    row_indices = [row.index for row in rows]
    locators = dict(parent["source"]["locators"])
    locators["row_indices"] = row_indices
    structured: dict[str, Any] = {
        "header_row_indices": header_indices,
        "header": header,
        "row_indices": row_indices,
        "rows": [{"row_index": row.index, "values": row.values} for row in rows],
    }
    if row_fragment:
        structured["row_fragment"] = row_fragment

    return {
        "schema_version": parent["schema_version"],
        "record_id": record_id,
        "parent_id": parent["record_id"],
        "content_type": "table_child",
        "source": {
            "document_id": document_id,
            "source_filename": parent["source"]["source_filename"],
            "pages": parent["source"]["pages"],
            "locators": locators,
        },
        "content": {
            "text": embedding_text,
            "markdown": markdown_text,
            "structured": structured,
        },
        "metadata": {
            "caption": captions,
            "table_index": table_index,
            "header_repeated": True,
            "synthetic_header": synthetic_header,
            "tokenizer_name": tokenizer_name,
            "chunk_target_tokens": target_tokens,
            "chunk_max_tokens": hard_max_tokens,
            "chunk_overlap": 0,
            "token_count": token_count(tokenizer, embedding_text),
            "row_fragmented": row_fragment is not None,
            "embedding_serialization": "key_value_v1" if key_value_mode else "attribute_value_v1",
        },
    }


def chunk_parent(
    parent: dict[str, Any],
    tokenizer: HuggingFaceTokenizer,
    tokenizer_name: str,
    target_tokens: int,
    hard_max_tokens: int,
) -> list[dict[str, Any]]:
    structured = parent["content"]["structured"]
    all_rows = build_rows(structured)
    header_indices = explicit_header_indices(structured)
    header, synthetic_header = display_header(all_rows, header_indices, int(structured["num_cols"]))
    data_rows = [row for row in all_rows if row.index not in set(header_indices)]
    captions = list(parent["metadata"].get("caption", []))
    key_value_mode = is_key_value_table(
        rows=data_rows,
        header_indices=header_indices,
        num_cols=int(structured["num_cols"]),
    )

    children: list[dict[str, Any]] = []
    current: list[TableRow] = []

    def emit(rows: list[TableRow]) -> None:
        embedding_text = serialize_embedding_text(captions, header, rows, key_value_mode)
        markdown_text = serialize_markdown(captions, header, rows)
        children.append(
            make_child(
                parent=parent,
                child_ordinal=len(children) + 1,
                captions=captions,
                header=header,
                header_indices=header_indices,
                synthetic_header=synthetic_header,
                rows=rows,
                embedding_text=embedding_text,
                markdown_text=markdown_text,
                tokenizer_name=tokenizer_name,
                target_tokens=target_tokens,
                hard_max_tokens=hard_max_tokens,
                tokenizer=tokenizer,
                key_value_mode=key_value_mode,
            )
        )

    for row in data_rows:
        candidate = current + [row]
        candidate_text = serialize_embedding_text(captions, header, candidate, key_value_mode)
        if token_count(tokenizer, candidate_text) <= target_tokens:
            current = candidate
            continue
        if current:
            emit(current)
            current = []
        single_text = serialize_embedding_text(captions, header, [row], key_value_mode)
        if token_count(tokenizer, single_text) <= hard_max_tokens:
            current = [row]
            continue

        fragments = split_oversize_row(
            row=row,
            captions=captions,
            header=header,
            tokenizer=tokenizer,
            hard_max_tokens=hard_max_tokens,
            key_value_mode=key_value_mode,
        )
        for fragment_index, fragment in enumerate(fragments, start=1):
            context = []
            if captions:
                context.append("\n".join(f"Table: {caption}" for caption in captions))
            context.append("Columns: " + "; ".join(header))
            context.append(f"Row {row.index} fragment {fragment_index}/{len(fragments)}: {fragment}")
            fragment_text = "\n\n".join(context)
            fragment_markdown = serialize_markdown(captions, header, []) + f"\n\nRow {row.index} fragment: {fragment}"
            children.append(
                make_child(
                    parent=parent,
                    child_ordinal=len(children) + 1,
                    captions=captions,
                    header=header,
                    header_indices=header_indices,
                    synthetic_header=synthetic_header,
                    rows=[],
                    embedding_text=fragment_text,
                    markdown_text=fragment_markdown,
                    tokenizer_name=tokenizer_name,
                    target_tokens=target_tokens,
                    hard_max_tokens=hard_max_tokens,
                    tokenizer=tokenizer,
                    key_value_mode=key_value_mode,
                    row_fragment={
                        "row_index": row.index,
                        "fragment_index": fragment_index,
                        "fragment_count": len(fragments),
                        "text": fragment,
                    },
                )
            )
    if current:
        emit(current)

    if not children:
        # Header-only tables are retained rather than silently dropped.
        emit([])
    return children


def validate_children(
    parents: list[dict[str, Any]],
    children: list[dict[str, Any]],
    hard_max_tokens: int,
    schema: dict[str, Any],
) -> dict[str, Any]:
    parent_by_id = {parent["record_id"]: parent for parent in parents}
    child_ids = [child["record_id"] for child in children]
    if len(child_ids) != len(set(child_ids)):
        raise AssertionError("Duplicate child record_id detected")
    if any(child["parent_id"] not in parent_by_id for child in children):
        raise AssertionError("Child references an unknown parent")
    if any(child["metadata"]["token_count"] > hard_max_tokens for child in children):
        raise AssertionError("A child exceeds the hard token limit")
    if any(not child["content"]["text"].strip() for child in children):
        raise AssertionError("An empty child was generated")
    attribute_value_errors = []
    for child in children:
        child_structured = child["content"]["structured"]
        if child_structured.get("row_fragment"):
            continue
        header = child_structured["header"]
        embedding_text = child["content"]["text"]
        key_value_mode = child["metadata"]["embedding_serialization"] == "key_value_v1"
        for row in child_structured["rows"]:
            if key_value_mode:
                attribute = row["values"][0].rstrip(":").strip()
                expected_pair = f"{attribute}: {row['values'][1]}"
                if expected_pair not in embedding_text:
                    attribute_value_errors.append(
                        (child["record_id"], row["row_index"], 0, expected_pair)
                    )
                continue
            for col, value in enumerate(row["values"]):
                if value and f"{header[col]}: {value}" not in embedding_text:
                    attribute_value_errors.append(
                        (child["record_id"], row["row_index"], col, value)
                    )
    if attribute_value_errors:
        raise AssertionError(
            f"Attribute-value serialization failed: {attribute_value_errors[:3]}"
        )
    schema_validator = Draft202012Validator(schema)
    schema_errors = [
        (child["record_id"], error.message)
        for child in children
        for error in schema_validator.iter_errors(child)
    ]
    if schema_errors:
        raise AssertionError(f"Chunk schema validation failed: {schema_errors[:3]}")

    fragmented_rows = 0
    for parent in parents:
        structured = parent["content"]["structured"]
        headers = set(explicit_header_indices(structured))
        expected = set(range(int(structured["num_rows"]))) - headers
        actual: list[int] = []
        fragment_groups: dict[int, list[int]] = {}
        for child in children:
            if child["parent_id"] != parent["record_id"]:
                continue
            actual.extend(child["content"]["structured"]["row_indices"])
            fragment = child["content"]["structured"].get("row_fragment")
            if fragment:
                fragment_groups.setdefault(fragment["row_index"], []).append(fragment["fragment_index"])
        fragmented_rows += len(fragment_groups)
        if len(actual) != len(set(actual)):
            raise AssertionError(f"Duplicate data row in {parent['record_id']}")
        actual_set = set(actual) | set(fragment_groups)
        if actual_set != expected:
            raise AssertionError(
                f"Row coverage mismatch in {parent['record_id']}: expected={sorted(expected)} actual={sorted(actual_set)}"
            )
        for row_index, indices in fragment_groups.items():
            if sorted(indices) != list(range(1, len(indices) + 1)):
                raise AssertionError(f"Fragment sequence mismatch for row {row_index} in {parent['record_id']}")

    token_counts = sorted(child["metadata"]["token_count"] for child in children)
    return {
        "parents": len(parents),
        "children": len(children),
        "fragmented_rows": fragmented_rows,
        "min_observed_tokens": token_counts[0],
        "max_observed_tokens": token_counts[-1],
        "over_limit": sum(count > hard_max_tokens for count in token_counts),
        "schema_errors": len(schema_errors),
        "attribute_value_errors": len(attribute_value_errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/table-parents.jsonl")
    parser.add_argument("--output", default="output/table-chunks.jsonl")
    parser.add_argument("--qa-output", default="output/table-chunks-qa.json")
    parser.add_argument("--schema", default="output/chunk-schema.json")
    parser.add_argument("--tokenizer-model", default=DEFAULT_TOKENIZER_MODEL)
    parser.add_argument("--target-tokens", type=int, default=480)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.target_tokens <= 0 or args.max_tokens <= 0 or args.target_tokens > args.max_tokens:
        raise ValueError("Require 0 < target-tokens <= max-tokens")

    input_path = Path(args.input)
    output_path = Path(args.output)
    qa_output_path = Path(args.qa_output)
    schema_path = Path(args.schema)
    parents = read_jsonl(input_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    # Candidate groups may temporarily exceed the model limit while the packer
    # decides where to split; final children are checked strictly below.
    transformers_logging.set_verbosity_error()
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        model_name=args.tokenizer_model,
        max_tokens=args.max_tokens,
    )
    children = [
        child
        for parent in parents
        for child in chunk_parent(
            parent=parent,
            tokenizer=tokenizer,
            tokenizer_name=args.tokenizer_model,
            target_tokens=args.target_tokens,
            hard_max_tokens=args.max_tokens,
        )
    ]
    qa = validate_children(
        parents=parents,
        children=children,
        hard_max_tokens=args.max_tokens,
        schema=schema,
    )
    qa.update(
        {
            "tokenizer_name": args.tokenizer_model,
            "target_tokens": args.target_tokens,
            "max_tokens": args.max_tokens,
            "row_overlap": 0,
            "input_sha256": file_sha256(input_path),
            "schema_sha256": file_sha256(schema_path),
            "docling_core_version": importlib.metadata.version("docling-core"),
            "transformers_version": importlib.metadata.version("transformers"),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qa_output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, children)
    qa_output_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False))


if __name__ == "__main__":
    main()
