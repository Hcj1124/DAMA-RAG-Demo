from pathlib import Path

from docling.document_converter import DocumentConverter


source = "input/dama-dmbok-2nd-edition.pdf"

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)


converter = DocumentConverter()
result = converter.convert(source)

doc = result.document


# 1. 儲存完整 DoclingDocument
doc.save_as_json(
    output_dir / "sample.json"
)


# 2. 額外存 Markdown
doc.save_as_markdown(
    output_dir / "sample.md"
)

print("完成")