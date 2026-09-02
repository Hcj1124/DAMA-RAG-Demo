"""Build embedding-ready text chunks from the text-only DoclingDocument.

Uses Docling HybridChunker, converts each DocChunk to the shared
chunk-schema.json format, validates the results, and writes text-chunks.jsonl.

Run:
    python build_text_chunks.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any, Iterable

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument
from jsonschema import Draft202012Validator
from transformers.utils import logging as transformers_logging


SCHEMA_VERSION = "1.0.0"
DEFAULT_TOKENIZER_MODEL = "BAAI/bge-m3"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


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


def document_id_from_name(name: str) -> str:
    return re.sub(
        r"[^a-z0-9._-]+",
        "-",
        name.lower(),
    ).strip("-")


def chunk_pages(chunk: Any) -> list[int]:
    return sorted(
        {
            int(prov.page_no)
            for item in chunk.meta.doc_items
            for prov in getattr(item, "prov", [])
            if getattr(prov, "page_no", None) is not None
        }
    )


def chunk_docling_refs(chunk: Any) -> list[str]:
    refs = []

    for item in chunk.meta.doc_items:
        ref = str(item.self_ref)

        if ref not in refs:
            refs.append(ref)

    return refs


def chunk_headings(chunk: Any) -> list[str]:
    headings = getattr(
        chunk.meta,
        "headings",
        None,
    )

    if not headings:
        return []

    return [
        str(value)
        for value in headings
        if str(value).strip()
    ]


def chunk_captions(chunk: Any) -> list[str]:
    meta = chunk.meta.export_json_dict()
    captions = meta.get("captions", [])

    return [
        str(value)
        for value in captions
        if str(value).strip()
    ]

def token_count(
    tokenizer: HuggingFaceTokenizer,
    text: str,
) -> int:
    return int(
        tokenizer.count_tokens(text=text)
    )


def make_text_record(
    *,
    chunk: Any,
    ordinal: int,
    embedding_text: str,
    tokenizer: HuggingFaceTokenizer,
    tokenizer_name: str,
    document_id: str,
    source_filename: str,
    target_tokens: int,
    hard_max_tokens: int,
) -> dict[str, Any]:

    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": (
            f"{document_id}:text:{ordinal:06d}"
        ),
        "parent_id": None,
        "content_type": "text",
        "source": {
            "document_id": document_id,
            "source_filename": source_filename,
            "pages": chunk_pages(chunk),
            "locators": {
                "docling_refs": chunk_docling_refs(
                    chunk
                ),
            },
        },
        "content": {
            "text": embedding_text,
            "markdown": None,
            "structured": None,
        },
        "metadata": {
            "headings": chunk_headings(chunk),
            "captions": chunk_captions(chunk),
            "tokenizer_name": tokenizer_name,
            "chunker": "docling_hybrid",
            "chunk_target_tokens": target_tokens,
            "chunk_max_tokens": hard_max_tokens,
            "chunk_overlap": 0,
            "token_count": token_count(
                tokenizer,
                embedding_text,
            ),
        },
    }


def build_records(
    *,
    doc: DoclingDocument,
    chunker: HybridChunker,
    tokenizer: HuggingFaceTokenizer,
    tokenizer_name: str,
    document_id: str,
    source_filename: str,
    target_tokens: int,
    hard_max_tokens: int,
) -> list[dict[str, Any]]:

    records = []

    for ordinal, chunk in enumerate(
        chunker.chunk(dl_doc=doc),
        start=1,
    ):
        embedding_text = chunker.contextualize(
            chunk=chunk
        )

        records.append(
            make_text_record(
                chunk=chunk,
                ordinal=ordinal,
                embedding_text=embedding_text,
                tokenizer=tokenizer,
                tokenizer_name=tokenizer_name,
                document_id=document_id,
                source_filename=source_filename,
                target_tokens=target_tokens,
                hard_max_tokens=hard_max_tokens,
            )
        )

    return records


def validate_records(
    records: list[dict[str, Any]],
    hard_max_tokens: int,
    schema: dict[str, Any],
) -> dict[str, Any]:

    if not records:
        raise AssertionError(
            "No text chunks were generated"
        )

    record_ids = [
        record["record_id"]
        for record in records
    ]

    if len(record_ids) != len(set(record_ids)):
        raise AssertionError(
            "Duplicate text record_id detected"
        )

    empty_chunks = [
        record["record_id"]
        for record in records
        if not record["content"]["text"].strip()
    ]

    if empty_chunks:
        raise AssertionError(
            f"Empty text chunks detected: "
            f"{empty_chunks[:5]}"
        )

    over_limit = [
        (
            record["record_id"],
            record["metadata"]["token_count"],
        )
        for record in records
        if record["metadata"]["token_count"]
        > hard_max_tokens
    ]

    if over_limit:
        raise AssertionError(
            f"Text chunks exceed token limit: "
            f"{over_limit[:5]}"
        )

    missing_pages = [
        record["record_id"]
        for record in records
        if not record["source"]["pages"]
    ]

    if missing_pages:
        raise AssertionError(
            f"Text chunks missing pages: "
            f"{missing_pages[:5]}"
        )

    missing_refs = [
        record["record_id"]
        for record in records
        if not record["source"]["locators"][
            "docling_refs"
        ]
    ]

    if missing_refs:
        raise AssertionError(
            f"Text chunks missing Docling refs: "
            f"{missing_refs[:5]}"
        )

    forbidden_refs = []

    for record in records:
        refs = record["source"]["locators"][
            "docling_refs"
        ]

        for ref in refs:
            if (
                ref.startswith("#/tables/")
                or ref.startswith("#/pictures/")
            ):
                forbidden_refs.append(
                    (
                        record["record_id"],
                        ref,
                    )
                )

    if forbidden_refs:
        raise AssertionError(
            "Table/picture refs found in text chunks: "
            f"{forbidden_refs[:5]}"
        )

    navigation_chunks = [
        record["record_id"]
        for record in records
        if any(
            str(heading).strip().casefold() == "index"
            for heading in record["metadata"].get("headings", [])
        )
    ]

    if navigation_chunks:
        raise AssertionError(
            "Index/navigation chunks found in text output: "
            f"{navigation_chunks[:5]}"
        )

    schema_validator = Draft202012Validator(
        schema
    )

    schema_errors = [
        (
            record["record_id"],
            error.message,
        )
        for record in records
        for error in schema_validator.iter_errors(
            record
        )
    ]

    if schema_errors:
        raise AssertionError(
            f"Chunk schema validation failed: "
            f"{schema_errors[:5]}"
        )

    token_counts = sorted(
        record["metadata"]["token_count"]
        for record in records
    )

    return {
        "chunks": len(records),
        "min_observed_tokens": token_counts[0],
        "max_observed_tokens": token_counts[-1],
        "over_limit": len(over_limit),
        "empty_chunks": len(empty_chunks),
        "duplicate_ids": (
            len(record_ids)
            - len(set(record_ids))
        ),
        "missing_pages": len(missing_pages),
        "missing_docling_refs": len(
            missing_refs
        ),
        "forbidden_table_picture_refs": len(
            forbidden_refs
        ),
        "navigation_chunks": len(navigation_chunks),
        "schema_errors": len(schema_errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="output/sample-text-only.json",
    )

    parser.add_argument(
        "--output",
        default="output/text-chunks.jsonl",
    )

    parser.add_argument(
        "--qa-output",
        default="output/text-chunks-qa.json",
    )

    parser.add_argument(
        "--schema",
        default="output/chunk-schema.json",
    )

    parser.add_argument(
        "--tokenizer-model",
        default=(
            DEFAULT_TOKENIZER_MODEL
        ),
    )

    parser.add_argument(
        "--target-tokens",
        type=int,
        default=480,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
    )

    args = parser.parse_args()

    if (
        args.target_tokens <= 0
        or args.max_tokens <= 0
        or args.target_tokens
        > args.max_tokens
    ):
        raise ValueError(
            "Require "
            "0 < target-tokens <= max-tokens"
        )

    input_path = Path(args.input)
    output_path = Path(args.output)
    qa_output_path = Path(args.qa_output)
    schema_path = Path(args.schema)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input document not found: "
            f"{input_path}"
        )

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Chunk schema not found: "
            f"{schema_path}"
        )

    doc = DoclingDocument.load_from_json(
        input_path
    )

    if doc.tables:
        raise AssertionError(
            "Text-only document still contains "
            f"{len(doc.tables)} tables"
        )

    if doc.pictures:
        raise AssertionError(
            "Text-only document still contains "
            f"{len(doc.pictures)} pictures"
        )

    if (
        doc.origin is None
        or not doc.origin.filename
    ):
        raise ValueError(
            "DoclingDocument source filename "
            "is missing"
        )

    document_id = document_id_from_name(
        doc.name
    )

    source_filename = doc.origin.filename

    schema = json.loads(
        schema_path.read_text(
            encoding="utf-8"
        )
    )

    transformers_logging.set_verbosity_error()

    tokenizer = (
        HuggingFaceTokenizer.from_pretrained(
            model_name=args.tokenizer_model,
            max_tokens=args.target_tokens,
        )
    )

    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
    )

    records = build_records(
        doc=doc,
        chunker=chunker,
        tokenizer=tokenizer,
        tokenizer_name=args.tokenizer_model,
        document_id=document_id,
        source_filename=source_filename,
        target_tokens=args.target_tokens,
        hard_max_tokens=args.max_tokens,
    )

    qa = validate_records(
        records=records,
        hard_max_tokens=args.max_tokens,
        schema=schema,
    )

    qa.update(
        {
            "tokenizer_name": (
                args.tokenizer_model
            ),
            "target_tokens": (
                args.target_tokens
            ),
            "max_tokens": args.max_tokens,
            "text_overlap": 0,
            "chunker": "docling_hybrid",
            "merge_peers": True,
            "input_sha256": file_sha256(input_path),
            "schema_sha256": file_sha256(schema_path),
            "docling_version": importlib.metadata.version("docling"),
            "docling_core_version": importlib.metadata.version("docling-core"),
            "transformers_version": importlib.metadata.version("transformers"),
        }
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    qa_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        output_path,
        records,
    )

    qa_output_path.write_text(
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
