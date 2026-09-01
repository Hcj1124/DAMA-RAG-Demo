"""Build a text-only DoclingDocument for later HybridChunker processing.

Removes all TableItem and PictureItem nodes, including their child items,
while leaving the original output/sample.json unchanged.

Run:
    python build_text_only_document.py

Or:
    python build_text_only_document.py \
        --input output/sample.json \
        --output output/sample-text-only.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docling_core.types.doc import DoclingDocument

START_PAGE = 21

EXCLUDED_PARENT_PREFIXES = (
    "#/tables/",
    "#/pictures/",
)


def item_pages(item) -> list[int]:
    return [
        int(prov.page_no)
        for prov in getattr(item, "prov", [])
        if getattr(prov, "page_no", None) is not None
    ]


def validate_output(path: Path) -> dict[str, int]:
    """Check the serialized JSON for tables, pictures, or dangling child refs."""

    data = json.loads(path.read_text(encoding="utf-8"))

    tables = data.get("tables", [])
    pictures = data.get("pictures", [])
    texts = data.get("texts", [])

    dangling_text_refs = []

    for item in texts:
        parent = item.get("parent")

        if not isinstance(parent, dict):
            continue

        parent_ref = str(parent.get("$ref", ""))

        if parent_ref.startswith(EXCLUDED_PARENT_PREFIXES):
            dangling_text_refs.append(
                {
                    "self_ref": item.get("self_ref"),
                    "parent_ref": parent_ref,
                    "text": item.get("text", ""),
                }
            )

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

    return {
        "text_items": len(texts),
        "tables": len(tables),
        "pictures": len(pictures),
        "dangling_table_picture_text_refs": len(dangling_text_refs),
    }

def is_before_start_page(item) -> bool:
    pages = item_pages(item)
    return bool(pages) and max(pages) < START_PAGE

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="output/sample.json",
    )

    parser.add_argument(
        "--output",
        default="output/sample-text-only.json",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input DoclingDocument not found: {input_path}"
        )

    # Load the original serialized DoclingDocument.
    # The original file itself is not modified.
    doc = DoclingDocument.load_from_json(input_path)

    original_texts = len(doc.texts)
    original_tables = len(doc.tables)
    original_pictures = len(doc.pictures)

    # delete_items() also removes child items belonging to these nodes.
    items_to_remove = [
        *doc.tables,
        *doc.pictures,
    ]

    if items_to_remove:
        doc.delete_items(node_items=items_to_remove)

    items_before_page_21 = [
        item
        for item in doc.texts
        if is_before_start_page(item)
    ]

    if items_before_page_21:
        doc.delete_items(node_items=items_before_page_21)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save to a NEW file.
    doc.save_as_json(output_path)

    qa = validate_output(output_path)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "before": {
            "text_items": original_texts,
            "tables": original_tables,
            "pictures": original_pictures,
        },
        "after": qa,
    }

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()