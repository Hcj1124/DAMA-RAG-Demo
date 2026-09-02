"""Combine validated text and table chunks for embedding.

Inputs:
    output/text-chunks.jsonl
    output/table-chunks.jsonl

Outputs:
    output/combined-chunks.jsonl
    output/combined-chunks-qa.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def validate_combined(
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    table_parent_ids: set[str],
    hard_max_tokens: int,
) -> dict[str, Any]:

    if not records:
        raise AssertionError(
            "No combined records were produced"
        )

    # ---------------------------------------------------------
    # Record IDs must remain globally unique
    # ---------------------------------------------------------

    record_ids = [
        record["record_id"]
        for record in records
    ]

    if len(record_ids) != len(set(record_ids)):
        raise AssertionError(
            "Duplicate record_id detected across combined chunks"
        )

    # ---------------------------------------------------------
    # Only embedding-ready records belong here
    # ---------------------------------------------------------

    invalid_types = [
        (
            record["record_id"],
            record["content_type"],
        )
        for record in records
        if record["content_type"]
        not in {"text", "table_child"}
    ]

    if invalid_types:
        raise AssertionError(
            "combined-chunks.jsonl may contain only "
            f"text and table_child records: {invalid_types[:5]}"
        )

    # ---------------------------------------------------------
    # Embedding input must exist
    # ---------------------------------------------------------

    empty_text = [
        record["record_id"]
        for record in records
        if not record["content"]["text"].strip()
    ]

    if empty_text:
        raise AssertionError(
            f"Empty embedding text detected: {empty_text[:5]}"
        )

    over_limit = [
        (
            record["record_id"],
            record["metadata"].get("token_count"),
        )
        for record in records
        if record["metadata"].get("token_count", hard_max_tokens + 1)
        > hard_max_tokens
    ]

    if over_limit:
        raise AssertionError(
            f"Combined chunks exceed token limit: {over_limit[:5]}"
        )

    tokenizer_names = {
        record["metadata"].get("tokenizer_name")
        for record in records
    }

    if len(tokenizer_names) != 1 or None in tokenizer_names:
        raise AssertionError(
            "Combined inputs use incompatible tokenizers: "
            f"{sorted(str(value) for value in tokenizer_names)}"
        )

    document_ids = {
        record["source"].get("document_id")
        for record in records
    }

    if len(document_ids) != 1 or None in document_ids:
        raise AssertionError(
            "Combined inputs contain incompatible documents: "
            f"{sorted(str(value) for value in document_ids)}"
        )

    invalid_parent_links = [
        (record["record_id"], record.get("parent_id"))
        for record in records
        if (
            record["content_type"] == "table_child"
            and record.get("parent_id") not in table_parent_ids
        )
    ]

    if invalid_parent_links:
        raise AssertionError(
            "Table children reference missing parents: "
            f"{invalid_parent_links[:5]}"
        )

    # ---------------------------------------------------------
    # Shared schema validation
    # ---------------------------------------------------------

    validator = Draft202012Validator(schema)

    schema_errors = [
        (
            record["record_id"],
            error.message,
        )
        for record in records
        for error in validator.iter_errors(record)
    ]

    if schema_errors:
        raise AssertionError(
            f"Combined schema validation failed: "
            f"{schema_errors[:5]}"
        )

    content_type_counts = Counter(
        record["content_type"]
        for record in records
    )

    return {
        "records": len(records),
        "text_records": content_type_counts["text"],
        "table_child_records": content_type_counts["table_child"],
        "duplicate_ids": (
            len(record_ids)
            - len(set(record_ids))
        ),
        "invalid_content_types": len(invalid_types),
        "empty_embedding_text": len(empty_text),
        "over_limit": len(over_limit),
        "tokenizer_name": next(iter(tokenizer_names)),
        "document_id": next(iter(document_ids)),
        "invalid_parent_links": len(invalid_parent_links),
        "schema_errors": len(schema_errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text-input",
        default="output/text-chunks.jsonl",
    )

    parser.add_argument(
        "--table-input",
        default="output/table-chunks.jsonl",
    )

    parser.add_argument(
        "--output",
        default="output/combined-chunks.jsonl",
    )

    parser.add_argument(
        "--qa-output",
        default="output/combined-chunks-qa.json",
    )

    parser.add_argument(
        "--schema",
        default="output/chunk-schema.json",
    )

    parser.add_argument(
        "--table-parents",
        default="output/table-parents.jsonl",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
    )

    args = parser.parse_args()

    text_path = Path(args.text_input)
    table_path = Path(args.table_input)
    output_path = Path(args.output)
    qa_path = Path(args.qa_output)
    schema_path = Path(args.schema)
    table_parents_path = Path(args.table_parents)

    for path in (
        text_path,
        table_path,
        schema_path,
        table_parents_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}"
            )

    text_records = read_jsonl(text_path)
    table_records = read_jsonl(table_path)
    table_parent_records = read_jsonl(table_parents_path)
    table_parent_ids = {
        record["record_id"]
        for record in table_parent_records
        if record.get("content_type") == "table_parent"
    }

    # Keep source order within each pipeline.
    combined_records = (
        text_records
        + table_records
    )

    schema = json.loads(
        schema_path.read_text(
            encoding="utf-8"
        )
    )

    qa = validate_combined(
        records=combined_records,
        schema=schema,
        table_parent_ids=table_parent_ids,
        hard_max_tokens=args.max_tokens,
    )

    qa.update(
        {
            "text_input": str(text_path),
            "table_input": str(table_path),
            "table_parents_input": str(table_parents_path),
            "text_input_sha256": file_sha256(text_path),
            "table_input_sha256": file_sha256(table_path),
            "table_parents_input_sha256": file_sha256(table_parents_path),
            "schema_sha256": file_sha256(schema_path),
            "max_tokens": args.max_tokens,
        }
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    qa_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        output_path,
        combined_records,
    )

    qa_path.write_text(
        json.dumps(
            qa,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            qa,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
