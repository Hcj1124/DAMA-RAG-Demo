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
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator


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

    args = parser.parse_args()

    text_path = Path(args.text_input)
    table_path = Path(args.table_input)
    output_path = Path(args.output)
    qa_path = Path(args.qa_output)
    schema_path = Path(args.schema)

    for path in (
        text_path,
        table_path,
        schema_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}"
            )

    text_records = read_jsonl(text_path)
    table_records = read_jsonl(table_path)

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
    )

    qa.update(
        {
            "text_input": str(text_path),
            "table_input": str(table_path),
        }
    )

    output_path.parent.mkdir(
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