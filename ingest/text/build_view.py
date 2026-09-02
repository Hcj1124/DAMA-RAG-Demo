"""建立僅保留正文的 DoclingDocument，供後續 HybridChunker 切塊。

此程式會排除表格、圖片及其附屬內容，也會移除指定頁面範圍外與
不適合檢索的文字；原始的 output/docling/document.json 不會被修改。

執行方式：
    python -m ingest.text.build_view

或指定輸入、輸出與 QA 報告：
    python -m ingest.text.build_view \
        --input output/docling/document.json \
        --output output/docling/text-only.json \
        --qa-output output/qa/text-only-qa.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ingest.paths import project_path

from docling_core.types.doc import DoclingDocument

# 正文起始頁，以及不納入文字檢索語料的 Docling 類型與父節點。
START_PAGE = 21

EXCLUDED_LABELS = {
    "caption",
    "page_footer",
    "page_header",
}

EXCLUDED_PARENT_PREFIXES = (
    "#/tables/",
    "#/pictures/",
)


def item_pages(item) -> list[int]:
    """取得 Docling 項目在來源 PDF 中出現的所有頁碼。"""
    return [
        int(prov.page_no)
        for prov in getattr(item, "prov", [])
        if getattr(prov, "page_no", None) is not None
    ]


def file_sha256(path: Path) -> str:
    """計算輸入檔案雜湊，讓 QA 報告可追溯到確切的來源版本。"""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def item_ref(item) -> str:
    """取得項目本身的 Docling 參照路徑。"""
    return str(getattr(item, "self_ref", ""))


def referenced_item_ref(ref) -> str:
    """將 Docling 參照物件統一轉成可比較的字串。"""
    return str(getattr(ref, "cref", ref))


def linked_caption_and_footnote_refs(doc: DoclingDocument) -> set[str]:
    """收集表格與圖片所連結的標題、註腳，避免殘留於純文字語料。"""
    refs: set[str] = set()

    for owner in [*doc.tables, *doc.pictures]:
        for attribute in ("captions", "footnotes"):
            for ref in getattr(owner, attribute, []) or []:
                refs.add(referenced_item_ref(ref))

    return refs


def find_index_start_page(doc: DoclingDocument) -> int | None:
    """找出書末 Index 的起始頁；找不到時回傳 None。"""
    candidates = []

    for item in doc.texts:
        if str(getattr(item, "label", "")) != "section_header":
            continue

        if str(getattr(item, "text", "")).strip().casefold() != "index":
            continue

        candidates.extend(item_pages(item))

    return min(candidates) if candidates else None


def validate_output(
    path: Path,
    *,
    start_page: int,
    index_start_page: int | None,
) -> dict[str, int]:
    """驗證輸出只保留目標頁面內的正文，並回傳各項 QA 統計。"""

    data = json.loads(path.read_text(encoding="utf-8"))

    tables = data.get("tables", [])
    pictures = data.get("pictures", [])
    texts = data.get("texts", [])

    dangling_text_refs = []
    forbidden_labels = []
    outside_page_range = []

    for item in texts:
        parent = item.get("parent")

        parent_ref = (
            str(parent.get("$ref", ""))
            if isinstance(parent, dict)
            else ""
        )

        if parent_ref.startswith(EXCLUDED_PARENT_PREFIXES):
            dangling_text_refs.append(
                {
                    "self_ref": item.get("self_ref"),
                    "parent_ref": parent_ref,
                    "text": item.get("text", ""),
                }
            )

        label = str(item.get("label", ""))
        if label in EXCLUDED_LABELS:
            forbidden_labels.append(item.get("self_ref"))

        pages = [
            int(prov["page_no"])
            for prov in item.get("prov", [])
            if prov.get("page_no") is not None
        ]

        if pages and min(pages) < start_page:
            outside_page_range.append(item.get("self_ref"))

        if (
            pages
            and index_start_page is not None
            and max(pages) >= index_start_page
        ):
            outside_page_range.append(item.get("self_ref"))

    if tables:
        raise AssertionError(
            f"Filtered document still contains {len(tables)} table items."
        )

    if pictures:
        raise AssertionError(
            f"Filtered document still contains {len(pictures)} picture items."
        )

    if dangling_text_refs:
        raise AssertionError(
            "Filtered document still contains text owned by tables/pictures: "
            f"{dangling_text_refs[:5]}"
        )

    if forbidden_labels:
        raise AssertionError(
            "Filtered document still contains excluded labels: "
            f"{forbidden_labels[:5]}"
        )

    if outside_page_range:
        raise AssertionError(
            "Filtered document still contains excluded pages: "
            f"{outside_page_range[:5]}"
        )

    return {
        "text_items": len(texts),
        "tables": len(tables),
        "pictures": len(pictures),
        "dangling_table_picture_text_refs": len(dangling_text_refs),
        "forbidden_labels": len(forbidden_labels),
        "outside_page_range": len(outside_page_range),
    }

def is_before_start_page(item, start_page: int) -> bool:
    """判斷項目是否完全位於正文起始頁之前。"""
    pages = item_pages(item)
    return bool(pages) and max(pages) < start_page


def is_index_or_later(item, index_start_page: int | None) -> bool:
    """判斷項目是否位於書末 Index 起始頁或其後。"""
    if index_start_page is None:
        return False

    pages = item_pages(item)
    return bool(pages) and max(pages) >= index_start_page

def main() -> None:
    """串接載入、內容過濾、輸出驗證及 QA 報告產生流程。"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=project_path("output/docling/document.json"),
    )

    parser.add_argument(
        "--output",
        default=project_path("output/docling/text-only.json"),
    )

    parser.add_argument(
        "--qa-output",
        default=project_path("output/qa/text-only-qa.json"),
    )

    parser.add_argument(
        "--start-page",
        type=int,
        default=START_PAGE,
        help="First PDF page retained in the retrieval corpus.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    qa_output_path = Path(args.qa_output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input DoclingDocument not found: {input_path}"
        )

    # 載入原始 DoclingDocument；後續結果另存新檔，不覆寫來源。
    doc = DoclingDocument.load_from_json(input_path)

    # 保留過濾前統計，並先找出頁面邊界與表格／圖片附屬文字。
    original_texts = len(doc.texts)
    original_tables = len(doc.tables)
    original_pictures = len(doc.pictures)
    index_start_page = find_index_start_page(doc)
    linked_refs = linked_caption_and_footnote_refs(doc)

    # 第一階段：刪除表格與圖片節點；delete_items() 會連同其子項目刪除。
    items_to_remove = [
        *doc.tables,
        *doc.pictures,
    ]

    if items_to_remove:
        doc.delete_items(node_items=items_to_remove)

    # 第二階段：排除封面／目錄、書末索引、頁首頁尾及媒體附屬文字。
    filtered_text_items = [
        item
        for item in doc.texts
        if (
            is_before_start_page(item, args.start_page)
            or is_index_or_later(item, index_start_page)
            or str(getattr(item, "label", "")) in EXCLUDED_LABELS
            or item_ref(item) in linked_refs
        )
    ]

    if filtered_text_items:
        doc.delete_items(node_items=filtered_text_items)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 將純文字文件另存新檔，再以序列化後的實際內容進行 QA。
    doc.save_as_json(output_path)

    qa = validate_output(
        output_path,
        start_page=args.start_page,
        index_start_page=index_start_page,
    )

    # QA 摘要同時記錄來源雜湊、過濾邊界及前後數量，便於重現與稽核。
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "input_sha256": file_sha256(input_path),
        "start_page": args.start_page,
        "index_start_page": index_start_page,
        "removed_text_items": original_texts - qa["text_items"],
        "before": {
            "text_items": original_texts,
            "tables": original_tables,
            "pictures": original_pictures,
        },
        "after": qa,
    }


    qa_output_path.parent.mkdir(parents=True, exist_ok=True)
    qa_output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
