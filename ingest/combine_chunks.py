"""合併已驗證的文字 chunks 與表格 child，形成統一 embedding 語料。

輸入：
    output/chunks/text-chunks.jsonl
    output/chunks/table-chunks.jsonl

輸出：
    output/chunks/combined-chunks.jsonl
    output/qa/combined-chunks-qa.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ingest.paths import project_path
from typing import Any, Iterable

from jsonschema import Draft202012Validator


def file_sha256(path: Path) -> str:
    """計算各輸入檔雜湊，供合併 QA 追溯資料版本。"""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """讀取 JSONL 紀錄並忽略空白行。"""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    """將合併後的 embedding 紀錄逐行寫成 JSONL。"""
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
    """驗證合併語料的相容性、完整性、parent 關聯與共用 schema。"""
    if not records:
        raise AssertionError(
            "No combined records were produced"
        )

    # 文字與表格兩條管線合併後，record_id 仍須全域唯一。
    record_ids = [
        record["record_id"]
        for record in records
    ]

    if len(record_ids) != len(set(record_ids)):
        raise AssertionError(
            "Duplicate record_id detected across combined chunks"
        )

    # 最終語料只接受可直接 embedding 的 text 與 table_child。
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

    # 所有紀錄必須有 embedding 文字，且使用相同 tokenizer、文件與 token 上限。
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

    # 最後以共用 schema 驗證兩條來源管線的輸出契約一致。
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
    """串接輸入載入、跨管線驗證、合併輸出與 QA 報告。"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text-input",
        default=project_path("output/chunks/text-chunks.jsonl"),
    )

    parser.add_argument(
        "--table-input",
        default=project_path("output/chunks/table-chunks.jsonl"),
    )

    parser.add_argument(
        "--output",
        default=project_path("output/chunks/combined-chunks.jsonl"),
    )

    parser.add_argument(
        "--qa-output",
        default=project_path("output/qa/combined-chunks-qa.json"),
    )

    parser.add_argument(
        "--schema",
        default=project_path("schemas/chunk-schema.json"),
    )

    parser.add_argument(
        "--table-parents",
        default=project_path("output/tables/table-parents.jsonl"),
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

    # table parent 不進入 embedding，但用來驗證每個 table child 的父紀錄存在。
    text_records = read_jsonl(text_path)
    table_records = read_jsonl(table_path)
    table_parent_records = read_jsonl(table_parents_path)
    table_parent_ids = {
        record["record_id"]
        for record in table_parent_records
        if record.get("content_type") == "table_parent"
    }

    # 保留文字及表格各自的原始順序，文字紀錄排列在表格紀錄之前。
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

    # QA 報告記錄各輸入路徑與雜湊，方便確認合併時使用的確切版本。
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
