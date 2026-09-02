"""驗證管線與資料移動後，路徑解析仍維持正確的回歸測試。"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from engine.config import Paths
from engine.corpus import Corpus
from ingest.paths import PROJECT_ROOT, project_path


def test_default_corpus_and_schema_survive_working_directory_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DAMA_ROOT", raising=False)
    monkeypatch.delenv("DAMA_OUTPUT_DIR", raising=False)
    paths = Paths.from_env()
    corpus = Corpus.load(paths)
    assert paths.root == PROJECT_ROOT
    assert paths.chroma_dir == PROJECT_ROOT / "output/chroma_db"
    assert corpus.records and corpus.parents
    schema = json.loads(project_path("schemas/chunk-schema.json").read_text(encoding="utf-8"))
    assert schema["properties"]["content_type"]["enum"] == ["text", "table_parent", "table_child"]


def test_environment_output_root_keeps_existing_relative_path_semantics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DAMA_ROOT", "checkout")
    monkeypatch.setenv("DAMA_OUTPUT_DIR", "custom-output")
    paths = Paths.from_env()
    assert paths.root == tmp_path / "checkout"
    assert paths.combined_chunks == tmp_path / "custom-output/chunks/combined-chunks.jsonl"
    assert paths.table_parents == tmp_path / "custom-output/tables/table-parents.jsonl"


def test_pdf_entry_point_writes_only_to_requested_directory(tmp_path, monkeypatch):
    # 不匯入真正 Docling、也不執行模型推論，直接測試命令列入口的檔案行為。
    from ingest import convert_pdf

    source = tmp_path / "input.pdf"
    source.write_bytes(b"conversion is mocked")
    destination = tmp_path / "nested/docling"
    seen = []

    class Converter:
        def convert(self, path):
            seen.append(path)
            return SimpleNamespace(document=SimpleNamespace(
                save_as_json=lambda target: target.write_text("{}", encoding="utf-8"),
                save_as_markdown=lambda target: target.write_text("markdown", encoding="utf-8"),
            ))

    module = ModuleType("docling.document_converter")
    module.DocumentConverter = Converter
    monkeypatch.setitem(sys.modules, "docling.document_converter", module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["convert_pdf", "--input", str(source), "--output-dir", str(destination)])
    convert_pdf.main()
    assert seen == [source]
    assert {p.name for p in destination.iterdir()} == {"document.json", "document.md"}
    assert not (tmp_path / "output").exists()


@pytest.mark.skipif(importlib.util.find_spec("jsonschema") is None, reason="needs ingest extra (jsonschema)")
def test_combination_cli_from_another_directory_preserves_records(tmp_path):
    # 刻意使用文件所述的 editable install，而不依賴 pytest 注入的專案路徑。
    output = tmp_path / "chunks/combined.jsonl"
    qa_path = tmp_path / "qa/combined.json"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-m", "ingest.combine_chunks", "--output", str(output), "--qa-output", str(qa_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    read = lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert read(output) == read(Paths().combined_chunks)
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    assert qa["schema_errors"] == qa["invalid_parent_links"] == 0
    assert Path(qa["text_input"]).is_file()
    assert not (tmp_path / "output").exists()
