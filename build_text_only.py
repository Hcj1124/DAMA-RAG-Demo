"""Build a text-only DoclingDocument for later HybridChunker processing.

Removes all TableItem and PictureItem nodes, including their child items,
while leaving the original output/sample.json unchanged.

Run:
    python build_text_only.py

Or:
    python build_text_only.py \
        --input output/sample.json \
        --output output/sample-text-only.json \
        --qa-output output/sample-text-only-qa.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from docling_core.types.doc import DoclingDocument

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
    return [
        int(prov.page_no)
        for prov in getattr(item, "prov", [])
        if getattr(prov, "page_no", None) is not None
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def item_ref(item) -> str:
    return str(getattr(item, "self_ref", ""))


def referenced_item_ref(ref) -> str:
    return str(getattr(ref, "cref", ref))


def linked_caption_and_footnote_refs(doc: DoclingDocument) -> set[str]:
    refs: set[str] = set()

    for owner in [*doc.tables, *doc.pictures]:
        for attribute in ("captions", "footnotes"):
            for ref in getattr(owner, attribute, []) or []:
                refs.add(referenced_item_ref(ref))

    return refs


def find_index_start_page(doc: DoclingDocument) -> int | None:
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
    """Check the serialized JSON for tables, pictures, or dangling child refs."""

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
    pages = item_pages(item)
    return bool(pages) and max(pages) < start_page


def is_index_or_later(item, index_start_page: int | None) -> bool:
    if index_start_page is None:
        return False

    pages = item_pages(item)
    return bool(pages) and max(pages) >= index_start_page

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

    parser.add_argument(
        "--qa-output",
        default="output/sample-text-only-qa.json",
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

    # Load the original serialized DoclingDocument.
    # The original file itself is not modified.
    doc = DoclingDocument.load_from_json(input_path)

    original_texts = len(doc.texts)
    original_tables = len(doc.tables)
    original_pictures = len(doc.pictures)
    index_start_page = find_index_start_page(doc)
    linked_refs = linked_caption_and_footnote_refs(doc)

    # delete_items() also removes child items belonging to these nodes.
    items_to_remove = [
        *doc.tables,
        *doc.pictures,
    ]

    if items_to_remove:
        doc.delete_items(node_items=items_to_remove)

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

    # Save to a NEW file.
    doc.save_as_json(output_path)

    qa = validate_output(
        output_path,
        start_page=args.start_page,
        index_start_page=index_start_page,
    )

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
