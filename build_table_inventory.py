"""Build the phase-1 schema contract and phase-2 table inventory from Docling JSON.

The script is intentionally dependency-free so that the inventory is reproducible from
the saved DoclingDocument, without rerunning PDF conversion.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0.0"

# Phase-3 decisions supplied by the user. Keeping these here makes every later
# output reproducible when this script is rerun.
USER_REVIEW_OVERRIDES = {
    21: ("figure", "Figure 22 Simplified Zachman Framework，不是表格"),
    25: ("er_diagram", "ER 圖中的實體，不是表格"),
    30: ("attribute_template", "只有屬性的示意表格，包含空值列"),
    44: ("flowchart", "流程圖，不是表格"),
    59: ("acknowledgements", "Primary Contributors 名單，不是知識表格"),
    60: ("acknowledgements", "Reviewers and Commenters 名單，不是知識表格"),
    61: ("index", "書末索引，不是資料表格"),
    62: ("index", "書末索引，不是資料表格"),
}


def norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def cell_text(cell: Any) -> str:
    return norm_text(str((cell or {}).get("text", "")))


def table_markdown(table: dict[str, Any]) -> str:
    """Render Docling's resolved grid to a compact Markdown preview."""
    grid = table["data"].get("grid", [])
    rows = []
    for row in grid:
        rows.append([cell_text(cell).replace("|", "\\|") for cell in row])
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(rows[0]) + " |"]
    lines.append("| " + " | ".join(["---"] * width) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def captions(table: dict[str, Any], texts_by_ref: dict[str, dict[str, Any]]) -> list[str]:
    """Resolve both inline captions and Docling ``$ref`` caption links."""
    output = []
    for item in table.get("captions", []):
        caption_item = texts_by_ref.get(item.get("$ref"), item)
        value = norm_text(str(caption_item.get("text", "")))
        if value:
            output.append(value)
    return output


def is_toc_layout(table: dict[str, Any]) -> tuple[bool, str | None]:
    """Identify navigational tables, including Contents and lists of figures."""
    data = table["data"]
    text = " ".join(cell_text(cell) for cell in data.get("table_cells", []))
    normalized = norm_text(text).lower()
    rows = data.get("num_rows", 0)
    page_no = min((p.get("page_no", 10**9) for p in table.get("prov", [])), default=10**9)
    leader_chars = text.count("_") + text.count(".")
    numeric_tail_rows = sum(
        1
        for row in data.get("grid", [])
        if len(row) >= 2 and re.fullmatch(r"\d+", cell_text(row[-1])) is not None
    )

    if "contents" in normalized and rows >= 10:
        return True, "contents"
    if page_no <= 20 and rows >= 20 and leader_chars >= 40 and numeric_tail_rows >= 5:
        return True, "navigational_list"
    if page_no <= 20 and rows >= 20 and normalized.startswith("figure") and numeric_tail_rows >= 5:
        return True, "list_of_figures"
    return False, None


def inventory_record(
    doc_id: str,
    index: int,
    table: dict[str, Any],
    texts_by_ref: dict[str, dict[str, Any]],
    caption_overrides: dict[int, list[str]],
) -> dict[str, Any]:
    data = table["data"]
    provenance = table.get("prov", [])
    pages = sorted({item["page_no"] for item in provenance if "page_no" in item})
    bbox = [item["bbox"] for item in provenance if "bbox" in item]
    toc, toc_reason = is_toc_layout(table)
    user_override = USER_REVIEW_OVERRIDES.get(index)
    if toc:
        review_status = "excluded"
        is_true_table = False
        exclusion_reason = toc_reason
        review_source = "automatic_toc_rule"
    elif user_override:
        review_status = "excluded"
        is_true_table = False
        exclusion_reason = user_override[0]
        review_source = "user_review"
    else:
        # The user identified the false positives and requested phase 4, so the
        # remaining candidates are the accepted canonical-table set.
        review_status = "accepted"
        is_true_table = True
        exclusion_reason = None
        review_source = "accepted_after_user_review"
    record_id = f"{doc_id}:table:{index:03d}"
    preview = table_markdown(table)
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": record_id,
        "parent_id": None,
        "content_type": "table_inventory",
        "source": {
            "document_id": doc_id,
            "source_filename": None,
            "docling_ref": table.get("self_ref"),
            "table_index": index,
            "pages": pages,
            "bounding_boxes": bbox,
        },
        "table": {
            "caption": caption_overrides.get(index, captions(table, texts_by_ref)),
            "num_rows": data.get("num_rows"),
            "num_cols": data.get("num_cols"),
            "orientation": data.get("orientation"),
            "markdown_preview": preview,
        },
        "classification": {
            "is_toc": toc,
            "toc_reason": toc_reason,
            "review_status": review_status,
            "is_true_table": is_true_table,
            "exclusion_reason": exclusion_reason,
            "review_source": review_source,
            "review_note": user_override[1] if user_override else None,
        },
    }


def plain_table_text(table: dict[str, Any]) -> str:
    rows = []
    for row in table["data"].get("grid", []):
        values = [cell_text(cell) for cell in row]
        if any(values):
            rows.append("\t".join(values))
    return "\n".join(rows)


def structured_table(table: dict[str, Any]) -> dict[str, Any]:
    data = table["data"]
    cell_keys = (
        "text",
        "row_span",
        "col_span",
        "start_row_offset_idx",
        "end_row_offset_idx",
        "start_col_offset_idx",
        "end_col_offset_idx",
        "column_header",
        "row_header",
        "row_section",
    )
    cells = []
    for cell in data.get("table_cells", []):
        item = {key: cell.get(key) for key in cell_keys}
        item["text"] = cell_text(cell)
        if cell.get("bbox") is not None:
            item["bbox"] = cell["bbox"]
        cells.append(item)
    return {
        "num_rows": data.get("num_rows"),
        "num_cols": data.get("num_cols"),
        "orientation": data.get("orientation"),
        "cells": cells,
    }


def canonical_parent(record: dict[str, Any], table: dict[str, Any]) -> dict[str, Any]:
    source = record["source"]
    table_index = source["table_index"]
    parent_id = f"{source['document_id']}:table:{table_index:03d}:parent:000"
    markdown = record["table"]["markdown_preview"]
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": parent_id,
        "parent_id": None,
        "content_type": "table_parent",
        "source": {
            "document_id": source["document_id"],
            "source_filename": source["source_filename"],
            "pages": source["pages"],
            "locators": {
                "docling_ref": source["docling_ref"],
                "table_index": table_index,
                "bounding_boxes": source["bounding_boxes"],
            },
        },
        "content": {
            "text": plain_table_text(table),
            "markdown": markdown,
            "structured": structured_table(table),
        },
        "metadata": {
            "caption": record["table"]["caption"],
            "num_rows": record["table"]["num_rows"],
            "num_cols": record["table"]["num_cols"],
            "review_status": "accepted",
            "review_source": record["classification"]["review_source"],
        },
    }


def schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.invalid/schemas/chunk-schema-1.0.0.json",
        "title": "Unified text and table RAG record",
        "description": "Common envelope for text chunks, table parents, and table embedding children.",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "record_id", "parent_id", "content_type", "source", "content", "metadata"],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "record_id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9._-]*:(text|table):[0-9]{3,}(?::(parent|child):[0-9]{3,})?$"},
            "parent_id": {"type": ["string", "null"]},
            "content_type": {"enum": ["text", "table_parent", "table_child"]},
            "source": {
                "type": "object",
                "additionalProperties": False,
                "required": ["document_id", "source_filename", "pages", "locators"],
                "properties": {
                    "document_id": {"type": "string"},
                    "source_filename": {"type": "string"},
                    "pages": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 1, "uniqueItems": True},
                    "locators": {"type": "object", "additionalProperties": True},
                },
            },
            "content": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "markdown", "structured"],
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "markdown": {"type": ["string", "null"]},
                    "structured": {"type": ["object", "array", "null"]},
                },
            },
            "metadata": {"type": "object", "additionalProperties": True},
        },
        "allOf": [
            {
                "if": {"properties": {"content_type": {"const": "table_child"}}},
                "then": {"properties": {"parent_id": {"type": "string", "minLength": 1}}},
            },
            {
                "if": {"properties": {"content_type": {"enum": ["text", "table_parent"]}}},
                "then": {"properties": {"parent_id": {"const": None}}},
            },
        ],
        "x_id_rules": {
            "document_id": "lowercase source stem; e.g. dama-dmbok-2nd-edition",
            "text": "{document_id}:text:{zero-padded ordinal}",
            "table_parent": "{document_id}:table:{Docling tables[] index zero-padded to 3 digits}:parent:000",
            "table_child": "{document_id}:table:{Docling tables[] index zero-padded to 3 digits}:child:{zero-padded ordinal}",
            "parent_id": "null for text/table_parent; exact table_parent record_id for table_child",
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_review_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "docling_ref",
        "table_index",
        "pages",
        "decision",
        "classification",
        "review_source",
        "note",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            classification = record["classification"]
            writer.writerow(
                {
                    "docling_ref": record["source"]["docling_ref"],
                    "table_index": record["source"]["table_index"],
                    "pages": ";".join(map(str, record["source"]["pages"])),
                    "decision": "include" if classification["is_true_table"] else "exclude",
                    "classification": classification["exclusion_reason"] or "true_table",
                    "review_source": classification["review_source"],
                    "note": classification["review_note"] or "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/sample.json")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--caption-overrides", default="table-caption-overrides.json")
    args = parser.parse_args()
    input_path, output_dir = Path(args.input), Path(args.output_dir)
    doc = json.loads(input_path.read_text(encoding="utf-8"))
    doc_id = re.sub(r"[^a-z0-9._-]+", "-", doc["name"].lower()).strip("-")
    overrides_path = Path(args.caption_overrides)
    raw_caption_overrides = (
        json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else {}
    )
    caption_overrides = {int(index): values for index, values in raw_caption_overrides.items()}
    texts_by_ref = {item["self_ref"]: item for item in doc.get("texts", []) if item.get("self_ref")}
    records = [
        inventory_record(doc_id, i, table, texts_by_ref, caption_overrides)
        for i, table in enumerate(doc.get("tables", []))
    ]
    for record in records:
        record["source"]["source_filename"] = doc.get("origin", {}).get("filename")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chunk-schema.json").write_text(json.dumps(schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(output_dir / "table-inventory.all.jsonl", records)
    included = [record for record in records if record["classification"]["is_true_table"]]
    excluded_toc = [record for record in records if record["classification"]["is_toc"]]
    excluded_review = [
        record
        for record in records
        if not record["classification"]["is_toc"] and not record["classification"]["is_true_table"]
    ]
    write_jsonl(output_dir / "table-inventory.jsonl", included)
    write_jsonl(output_dir / "table-inventory-excluded-toc.jsonl", excluded_toc)
    write_jsonl(output_dir / "table-inventory-excluded-review.jsonl", excluded_review)
    write_review_csv(output_dir / "table-review.csv", records)

    accepted_indices = {record["source"]["table_index"] for record in included}
    parents = [
        canonical_parent(records[index], table)
        for index, table in enumerate(doc.get("tables", []))
        if index in accepted_indices
    ]
    write_jsonl(output_dir / "table-parents.jsonl", parents)
    canonical_dir = output_dir / "canonical-tables"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    for parent in parents:
        file_stem = parent["record_id"].replace(":", "_")
        (canonical_dir / f"{file_stem}.json").write_text(
            json.dumps(parent, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (canonical_dir / f"{file_stem}.md").write_text(parent["content"]["markdown"] + "\n", encoding="utf-8")
    print(
        f"all={len(records)} included={len(included)} "
        f"excluded_toc={len(excluded_toc)} excluded_review={len(excluded_review)} "
        f"parents={len(parents)}"
    )


if __name__ == "__main__":
    main()
