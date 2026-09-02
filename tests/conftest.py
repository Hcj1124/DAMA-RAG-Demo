"""Let the engine tests run without the chunking dependencies.

``tests/test_pipeline.py`` imports ``build_text_only`` and ``combine_chunks``
from the repository root, which need ``docling-core`` and ``jsonschema`` --
the ``ingest`` extra. Those are deliberately not part of the default install,
because answering a question never parses a PDF.

Without this hook a bare ``uv run pytest`` dies during collection and the
engine tests never run. With it they run, and the header says plainly what
was skipped and how to include it.
"""

from __future__ import annotations

import importlib.util

_INGEST_ONLY = ("test_pipeline.py",)

collect_ignore: list[str] = []
if importlib.util.find_spec("docling_core") is None:
    collect_ignore.extend(_INGEST_ONLY)


def pytest_report_header(config) -> str | None:
    if not collect_ignore:
        return None
    return (
        f"skipped {', '.join(collect_ignore)} (needs the `ingest` extra) -- "
        f"run the full suite with: uv run --extra ingest pytest"
    )
