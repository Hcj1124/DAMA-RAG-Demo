"""將來源 PDF 轉換成完整的 Docling JSON 與方便閱讀的 Markdown。"""

from __future__ import annotations

import argparse
from pathlib import Path

from ingest.paths import project_path


def main() -> None:
    """解析命令列參數並執行一次 PDF 轉換。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=project_path("input/dama-dmbok-2nd-edition.pdf"))
    parser.add_argument("--output-dir", default=project_path("output/docling"))
    args = parser.parse_args()
    source, output_dir = Path(args.input), Path(args.output_dir)
    if not source.is_file():
        raise FileNotFoundError(f"Source PDF not found: {source}")

    # 延後匯入轉換器，避免單純匯入入口或查看說明時就啟動 PDF 轉換。
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(source)
    doc = result.document
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 將完整 DoclingDocument 結構儲存為 JSON，保留版面與內容資訊。
    doc.save_as_json(output_dir / "document.json")
    # 2. 將文件內容另存為較方便閱讀與後續處理的 Markdown。
    doc.save_as_markdown(output_dir / "document.md")
    print("完成")


if __name__ == "__main__":
    main()
