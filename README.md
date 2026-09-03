# DAMA RAG Demo

這是一個在本機執行的 DAMA-DMBOK 雙語 RAG 示範專案。專案以 Docling 處理 PDF 的文字與表格，再使用 BGE-M3、Chroma、BGE reranker 與 Ollama 進行檢索、重排、上下文組裝及具來源引用的問答。

專案已附上可直接建立索引的 chunks，一般使用者不需要原始 PDF，也不需要重跑 Docling 轉檔。

## 系統流程

```text
問題 → BGE-M3 向量檢索 → BGE reranker 重排
     → 表格 child 回取完整 parent → 組裝引用上下文
     → Ollama / Qwen 產生答案
```

預設模型與資料庫：

| 用途 | 預設值 |
|---|---|
| Tokenizer / embedding | `BAAI/bge-m3` |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| LLM | `qwen3.6:35b-a3b` |
| Vector store | Chroma（`output/chroma_db/`） |

## 環境需求

- Windows、macOS 或 Linux
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://docs.ollama.com/windows)
- 足以存放並執行 `qwen3.6:35b-a3b` 的磁碟空間與記憶體

專案以 `.python-version` 指定 Python 3.12，並以 `pyproject.toml` 與 `uv.lock` 管理環境。uv 會自動建立 `.venv/`，不需要手動啟用虛擬環境。

## 快速開始

1. 下載專案並安裝鎖定的 Python 依賴：

   ```powershell
   git clone https://github.com/Hcj1124/DAMA-RAG-Demo.git
   cd DAMA-RAG-Demo
   uv sync --locked
   ```

2. 下載預設 LLM：

   ```powershell
   ollama pull qwen3.6:35b-a3b
   ```

3. 建立本機 Chroma 索引：

   ```powershell
   uv run dama-rag index
   ```

   首次建立索引會下載 BGE embedding 模型；第一次搜尋或問答時才會載入 reranker。索引建立後，可用下列指令檢查整體狀態：

   ```powershell
   uv run dama-rag doctor
   ```

   `doctor` 會確認索引筆數與語料一致，並檢查 embedding 模型、向量維度及完整設定指紋。若索引來自舊設定或 metadata 不完整，請執行 `uv run dama-rag index --rebuild`。

4. 進行問答：

   ```powershell
   uv run dama-rag ask "資料治理的核心目標是什麼？"
   ```

   省略問題即進入互動模式；輸入 `exit`、`quit` 或 `:q` 結束：

   ```powershell
   uv run dama-rag ask
   ```

## 常用指令

```powershell
uv run dama-rag info                              # 顯示實際生效的設定
uv run dama-rag doctor                            # 檢查語料、索引、裝置與 Ollama
uv run dama-rag search "Data Governance" -k 3    # 只檢索與重排
uv run dama-rag context "GDPR 原則"             # 查看會送給 LLM 的來源
uv run dama-rag context "GDPR 原則" --prompt    # 顯示完整 prompt
uv run dama-rag index --rebuild                   # 強制重建向量索引
```

所有可調整設定集中在 `engine/config.py`，並可以 `DAMA_*` 環境變數覆寫。例如：

```powershell
$env:DAMA_DEVICE = "cpu"
$env:DAMA_ANSWER_LANGUAGE = "zh-hant"
uv run dama-rag ask "What is Data Governance?"
```

## 重建 Docling 語料

只有在更換或重新處理原始 PDF 時才需要此流程。將檔案放在：

```text
input/dama-dmbok-2nd-edition.pdf
```

安裝 `ingest` extra 後依序執行：

```powershell
uv sync --extra ingest --locked
uv run --extra ingest python -m ingest.convert_pdf
uv run --extra ingest python -m ingest.tables.build_inventory
uv run --extra ingest python -m ingest.tables.build_chunks
uv run --extra ingest python -m ingest.text.build_view
uv run --extra ingest python -m ingest.text.build_chunks
uv run --extra ingest python -m ingest.combine_chunks
```

## 專案結構

```text
engine/          RAG 執行引擎、CLI 與 adapters
ingest/          PDF、文字、表格與 chunk 處理流程
config/          人工補充的表格 caption 設定
schemas/         文字與表格共用的 chunk schema
output/chunks/   已產生的 embedding-ready JSONL
output/tables/   表格盤點、審查與 parent records
output/qa/       資料處理的 QA 結果
tests/           引擎、資料與路徑回歸測試
```

`input/` 內的 PDF、`output/docling/` 的大型中間檔、`output/chroma_db/`、模型 cache 與 `.venv/` 都是本機資料，不會納入 Git。

## 測試

```powershell
uv run --frozen pytest
```

包含 Docling ingestion 測試的完整測試：

```powershell
uv run --extra ingest --frozen pytest
```

若尚未準備預設的大型模型，可暫時使用較小的 `qwen3:8b` 做端到端 smoke test；這不會改變專案預設值：

```powershell
ollama pull qwen3:8b
$env:DAMA_OLLAMA_MODEL = "qwen3:8b"
uv run dama-rag doctor
uv run dama-rag ask "資料治理的核心目標是什麼？" --language zh-hant
Remove-Item Env:DAMA_OLLAMA_MODEL
```

## 相關資源

- [完整操作指南](docs/guide.md)：環境準備、逐步操作、語料重建、設定與故障排除
- [uv 專案環境](https://docs.astral.sh/uv/guides/projects/)
- [Ollama for Windows](https://docs.ollama.com/windows)
- [Qwen3.6 35B-A3B for Ollama](https://ollama.com/library/qwen3.6:35b-a3b)
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Docling](https://docling-project.github.io/docling/)
