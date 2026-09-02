"""由 Docling JSON 建立表格盤點與 canonical table parent。

此程式直接使用已儲存的 DoclingDocument，不必重新轉換 PDF；處理結果會依
自動規則及人工覆核決策，分成納入與排除清單，再產生後續切塊所需的表格母紀錄。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from ingest.paths import project_path
from typing import Any


# 表格母紀錄與後續文字／表格子 chunk 共用的 schema 版本。
SCHEMA_VERSION = "1.0.0"

# 人工覆核後確認的非資料表格；固定保存於程式中以維持重跑結果一致。
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
    """統一空白字元，讓分類、比較與輸出文字保持穩定。"""
    return re.sub(r"\s+", " ", value).strip()


def cell_text(cell: Any) -> str:
    """安全取得並正規化 Docling 儲存格文字。"""
    return norm_text(str((cell or {}).get("text", "")))


def table_markdown(table: dict[str, Any]) -> str:
    """將 Docling 已解析的表格網格轉為供閱讀的 Markdown 預覽。"""
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
    """解析表格內嵌 caption 及透過 Docling $ref 連結的 caption。"""
    output = []
    for item in table.get("captions", []):
        caption_item = texts_by_ref.get(item.get("$ref"), item)
        value = norm_text(str(caption_item.get("text", "")))
        if value:
            output.append(value)
    return output


def is_toc_layout(table: dict[str, Any]) -> tuple[bool, str | None]:
    """依頁碼、列數及文字特徵辨識目錄、圖表清單等導覽型表格。"""
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
    """建立單一表格的盤點紀錄，整合自動分類與人工覆核結果。"""
    data = table["data"]
    provenance = table.get("prov", [])
    pages = sorted({item["page_no"] for item in provenance if "page_no" in item})
    bbox = [item["bbox"] for item in provenance if "bbox" in item]
    toc, toc_reason = is_toc_layout(table)
    user_override = USER_REVIEW_OVERRIDES.get(index)
    # 分類優先序：自動目錄規則、人工排除清單，其餘視為已接受表格。
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
    """建立表格的純文字表示，保留列與欄的基本界線。"""
    rows = []
    for row in table["data"].get("grid", []):
        values = [cell_text(cell) for cell in row]
        if any(values):
            rows.append("\t".join(values))
    return "\n".join(rows)


def structured_table(table: dict[str, Any]) -> dict[str, Any]:
    """保留儲存格位置、跨列欄及標題屬性，形成結構化表格內容。"""
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
    """將已接受的盤點項目轉成後續 table child 共用的母紀錄。"""
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


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """將盤點或 canonical 紀錄逐行寫成 JSONL。"""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_review_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """輸出方便人工檢查的表格納入／排除決策清單。"""
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
    """串接表格盤點、分類輸出與 canonical parent 產製流程。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=project_path("output/docling/document.json"))
    parser.add_argument("--output-dir", default=project_path("output/tables"))
    parser.add_argument("--caption-overrides", default=project_path("config/table-caption-overrides.json"))
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
    # 每個 Docling table 先建立完整盤點紀錄，再依分類結果分流輸出。
    records = [
        inventory_record(doc_id, i, table, texts_by_ref, caption_overrides)
        for i, table in enumerate(doc.get("tables", []))
    ]
    for record in records:
        record["source"]["source_filename"] = doc.get("origin", {}).get("filename")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "table-inventory-all.jsonl", records)
    included = [record for record in records if record["classification"]["is_true_table"]]
    excluded_toc = [record for record in records if record["classification"]["is_toc"]]
    excluded_review = [
        record
        for record in records
        if not record["classification"]["is_toc"] and not record["classification"]["is_true_table"]
    ]
    write_jsonl(output_dir / "table-inventory-accepted.jsonl", included)
    write_jsonl(output_dir / "table-inventory-excluded-toc.jsonl", excluded_toc)
    write_jsonl(output_dir / "table-inventory-excluded-review.jsonl", excluded_review)
    write_review_csv(output_dir / "table-review.csv", records)

    # 僅將通過覆核的真實資料表格轉成後續 row-aware 切塊所需的母紀錄。
    accepted_indices = {record["source"]["table_index"] for record in included}
    parents = [
        canonical_parent(records[index], table)
        for index, table in enumerate(doc.get("tables", []))
        if index in accepted_indices
    ]
    write_jsonl(output_dir / "table-parents.jsonl", parents)
    canonical_dir = output_dir / "canonical-parents"
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
