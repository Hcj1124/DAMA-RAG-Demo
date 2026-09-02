"""讓引擎測試在未安裝切塊依賴時仍可執行。

``tests/test_pipeline.py`` 會匯入 ingestion 套件，因此需要 ``ingest`` extra
提供的 ``docling-core`` 與 ``jsonschema``。它們刻意不放入預設安裝，因為一般
問答流程不會解析 PDF。

若沒有此 hook，直接執行 ``uv run pytest`` 會在收集測試時失敗。現在會略過需要
額外依賴的檔案，並在標頭說明略過原因與完整測試指令。
"""

from __future__ import annotations

import importlib.util

_INGEST_ONLY = ("test_pipeline.py",)

collect_ignore: list[str] = []
if importlib.util.find_spec("docling_core") is None:
    collect_ignore.extend(_INGEST_ONLY)


def pytest_report_header(config) -> str | None:
    """在 pytest 標頭顯示因缺少 ingestion extra 而略過的測試。"""
    if not collect_ignore:
        return None
    return (
        f"skipped {', '.join(collect_ignore)} (needs the `ingest` extra) -- "
        f"run the full suite with: uv run --extra ingest pytest"
    )
