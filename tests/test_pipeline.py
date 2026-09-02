from __future__ import annotations

import unittest
from types import SimpleNamespace

from ingest.text.build_view import is_before_start_page, is_index_or_later
from ingest.combine_chunks import validate_combined


def item_on_page(page: int) -> SimpleNamespace:
    return SimpleNamespace(
        prov=[SimpleNamespace(page_no=page)],
    )


def record(
    record_id: str,
    content_type: str,
    *,
    parent_id: str | None = None,
    tokenizer_name: str = "BAAI/bge-m3",
) -> dict:
    return {
        "schema_version": "1.0.0",
        "record_id": record_id,
        "parent_id": parent_id,
        "content_type": content_type,
        "source": {
            "document_id": "dama-dmbok-2nd-edition",
            "source_filename": "dama-dmbok-2nd-edition.pdf",
            "pages": [21],
            "locators": {},
        },
        "content": {
            "text": "Data governance",
            "markdown": None,
            "structured": None,
        },
        "metadata": {
            "token_count": 10,
            "tokenizer_name": tokenizer_name,
        },
    }


class TextFilterTests(unittest.TestCase):
    def test_page_boundaries(self) -> None:
        self.assertTrue(is_before_start_page(item_on_page(20), 21))
        self.assertFalse(is_before_start_page(item_on_page(21), 21))
        self.assertFalse(is_index_or_later(item_on_page(618), 619))
        self.assertTrue(is_index_or_later(item_on_page(619), 619))


class CombinedValidationTests(unittest.TestCase):
    def test_valid_parent_child_and_text_records(self) -> None:
        result = validate_combined(
            records=[
                record("doc:text:000001", "text"),
                record(
                    "doc:table:001:child:001",
                    "table_child",
                    parent_id="doc:table:001:parent:000",
                ),
            ],
            schema={"type": "object"},
            table_parent_ids={"doc:table:001:parent:000"},
            hard_max_tokens=512,
        )

        self.assertEqual(result["invalid_parent_links"], 0)
        self.assertEqual(result["tokenizer_name"], "BAAI/bge-m3")

    def test_missing_table_parent_is_rejected(self) -> None:
        with self.assertRaisesRegex(AssertionError, "missing parents"):
            validate_combined(
                records=[
                    record(
                        "doc:table:001:child:001",
                        "table_child",
                        parent_id="doc:table:001:parent:000",
                    )
                ],
                schema={"type": "object"},
                table_parent_ids=set(),
                hard_max_tokens=512,
            )

    def test_mixed_tokenizers_are_rejected(self) -> None:
        with self.assertRaisesRegex(AssertionError, "incompatible tokenizers"):
            validate_combined(
                records=[
                    record("doc:text:000001", "text"),
                    record(
                        "doc:text:000002",
                        "text",
                        tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
                    ),
                ],
                schema={"type": "object"},
                table_parent_ids=set(),
                hard_max_tokens=512,
            )


if __name__ == "__main__":
    unittest.main()
