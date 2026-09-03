# DAMA RAG Demo 完整操作指南

本指南說明如何在本機使用 DAMA-DMBOK 語料完成檢索、重排序與問答，以及需要時如何從來源 PDF 重建語料。若只想快速認識專案，請先閱讀根目錄的 [README](../README.md)。

以下指令均從含有 `pyproject.toml` 的專案根目錄執行。

## 1. 用途與整體流程

專案分為兩個部分：

1. `ingest/` 將 PDF 轉換成 Docling 文件、文字區塊與表格區塊，再合併成統一 JSONL 語料。
2. `engine/` 將語料寫入 ChromaDB，以 BGE-M3 檢索、BGE reranker 重排序，最後把證據交給本機 Ollama 模型回答。

```text
來源 PDF
  └─ Docling 轉換
      ├─ 文字專用文件 → 文字 chunks ─┐
      └─ 表格盤點 → 表格 parents → 表格 child chunks
                                      └─ 合併語料
                                          └─ ChromaDB 索引
                                              └─ 檢索 → 重排序 → 證據 → Ollama 回答
```

一般使用者不必重新處理 PDF。專案目前已提供：

- `output/chunks/combined-chunks.jsonl`：1,250 筆可檢索區塊。
- `output/tables/table-parents.jsonl`：44 筆完整表格父資料。
- `schemas/chunk-schema.json`：合併語料的資料格式。

文字命中會直接成為證據；表格 child chunk 命中後會展開成完整 parent，並合併相同 parent，避免重複放入同一張表。

## 2. 操作前準備

### 必要工具

- Git。
- [uv](https://docs.astral.sh/uv/)。
- [Ollama](https://ollama.com/)。
- 可下載 Python、PyPI 套件、Hugging Face 模型與 Ollama 模型的網路環境。

專案要求 Python `>=3.12,<3.14`，`.python-version` 指定 3.12。uv 會建立並管理環境，不需要手動建立傳統虛擬環境。

```powershell
git --version
uv --version
ollama --version
```

取得專案：

```powershell
git clone <REPOSITORY_URL>
cd <PROJECT_DIRECTORY>
```

請將占位值換成實際儲存庫網址與目錄名稱。

### 輸入與路徑

直接使用已提交語料時不需要來源 PDF。只有執行第 4 節的重建流程時，才需要自行準備：

```text
input/dama-dmbok-2nd-edition.pdf
```

`input/` 與 PDF 已被 `.gitignore` 排除，不應提交含版權或敏感內容的來源檔案。

未覆蓋設定時，程式會以專案位置推導預設路徑。明確傳入的相對路徑，以及相對的 `DAMA_ROOT`、`DAMA_OUTPUT_DIR`，則以目前工作目錄為基準；因此建議始終從專案根目錄執行。

## 3. 使用既有語料完成問答

### 步驟 1：安裝相依套件

```powershell
uv lock --check
uv sync --locked
```

第一個命令確認 `uv.lock` 與專案設定一致；第二個命令依鎖定檔同步 Python 與套件。後續均以 `uv run` 執行，不必手動啟用環境。

### 步驟 2：準備 Ollama 模型

```powershell
ollama pull qwen3.6:35b-a3b
ollama list
```

`ollama list` 應顯示相同模型名稱。若服務未自動啟動，請先依作業系統的 Ollama 安裝方式啟動服務。

### 步驟 3：確認有效設定

```powershell
uv run dama-rag info
```

重點預設值：

- Embedding：`BAAI/bge-m3`
- Reranker：`BAAI/bge-reranker-v2-m3`
- LLM：`qwen3.6:35b-a3b`
- Chroma collection：`dama_chunks`
- 語料：`output/chunks/combined-chunks.jsonl`

`info` 顯示程式將採用的設定，不代表模型、語料與索引都已就緒。

### 步驟 4：建立向量索引

```powershell
uv run dama-rag index
```

- 輸入：`output/chunks/combined-chunks.jsonl`
- 表格父資料：`output/tables/table-parents.jsonl`
- 輸出：`output/chroma_db/`
- Embedding：`BAAI/bge-m3`

首次執行會下載 embedding 模型並建立索引。依目前語料建立完成後，索引筆數應等於 1,250。

一般執行採增量同步，並保存 embedding 設定指紋。遇到下列情況時應完整重建：

```powershell
uv run dama-rag index --rebuild
```

- 更換 embedding 模型、文件提示前綴、正規化方式或最大序列長度。
- 索引筆數與語料不符。
- `doctor` 顯示索引 metadata 缺漏。
- 程式回報 embedding index fingerprint 不相容。

### 步驟 5：執行就緒檢查

```powershell
uv run dama-rag doctor
```

此命令檢查語料、表格父資料、索引、運算裝置與 Ollama 模型。索引檢查會自動確認：

- index 筆數與 corpus 相同。
- embedding 模型與目前設定相同。
- embedding dimension metadata 是有效正整數。
- embedding 設定 fingerprint 完整且與目前設定相同。

以目前提交的語料，corpus 應為 1,250 筆、table parents 應為 44 筆；Ollama 清單也應包含目前設定的 LLM。

全部通過時，末尾會提示可執行 `dama-rag ask`。

### 步驟 6：先驗證檢索與證據

```powershell
uv run dama-rag search "What is data governance?"
uv run dama-rag search "What is data governance?" -k 5 --json
```

成功時會列出分數、來源頁碼、章節與 chunk 識別碼。第一次搜尋會載入 reranking 模型，可能需要下載模型檔案。

查看實際送入 LLM 的證據，或連同完整提示詞顯示：

```powershell
uv run dama-rag context "What is data governance?"
uv run dama-rag context "What is data governance?" --prompt
```

建議先確認來源與問題相關，再生成答案，藉此區分「檢索不準」與「模型回答不佳」。

### 步驟 7：執行問答

```powershell
uv run dama-rag ask "What is data governance?"
uv run dama-rag ask "什麼是資料治理？" --language zh-hant
uv run dama-rag ask "What is data governance?" "What is metadata?"
uv run dama-rag ask "什麼是資料治理？" --language zh-hant --json
```

不帶問題會進入互動模式：

```powershell
uv run dama-rag ask
```

輸出包含回答及引用來源。`--language` 可用 `auto`、`en`、`zh-hant`，預設為 `auto`。

### 簡化入口

```powershell
uv run python -m engine.chat
```

此入口會在沒有索引時自動建索引，並為較低記憶體環境套用 `DAMA_DEVICE=cpu`、`DAMA_NUM_CTX=8192`、`DAMA_MAX_CONTEXT_CHARS=18000`。外部已設定的同名變數不會被覆蓋。若要明確掌握建索引與檢索結果，仍建議使用前述分步流程。

## 4. 從 PDF 重建語料（選用）

只有更換來源 PDF、調整切分策略或重建中間資料時才執行本節。這些命令會改寫 `output/` 下的對應產物。

```powershell
uv sync --extra ingest --locked
```

### 步驟 1：轉換 PDF

```powershell
uv run python -m ingest.convert_pdf
```

- 輸入：`input/dama-dmbok-2nd-edition.pdf`
- 輸出目錄：`output/docling/`
- 主要產物：`document.json`、`document.md`

成功時會顯示「完成」及輸出位置。

### 步驟 2：盤點表格

```powershell
uv run python -m ingest.tables.build_inventory
```

預設讀取 `output/docling/document.json`，輸出盤點、待檢視 CSV、排除清單與 `output/tables/table-parents.jsonl`。目前來源的 QA 基準為：全部 63 張、納入 44 張、目錄型排除 11 張、人工檢視排除 8 張、parents 44 筆。

更換 PDF 或排除設定後，數量可能合理改變；應檢查盤點與待檢視清單，而非只追求相同數字。

### 步驟 3：建立表格 child chunks

```powershell
uv run python -m ingest.tables.build_chunks
```

- 輸入：`output/tables/table-parents.jsonl`
- 輸出：`output/chunks/table-chunks.jsonl`
- QA：`output/qa/table-chunks-qa.json`
- Tokenizer：`BAAI/bge-m3`
- 目標／上限：480／512 tokens
- 列重疊：0

目前資料應產生 51 個 child chunks，QA 不應有超限、結構或屬性錯誤。

### 步驟 4：建立文字專用文件

```powershell
uv run python -m ingest.text.build_view
```

- 輸入：`output/docling/document.json`
- 輸出：`output/docling/text-only.json`
- QA：`output/qa/text-only-qa.json`
- 起始頁：21

此步驟依 Docling reading order 保留文字，排除表格與圖片來源內容。

### 步驟 5：建立文字 chunks

```powershell
uv run python -m ingest.text.build_chunks
```

- 輸入：`output/docling/text-only.json`
- 輸出：`output/chunks/text-chunks.jsonl`
- QA：`output/qa/text-chunks-qa.json`
- Chunker：Docling `HybridChunker`
- Tokenizer：`BAAI/bge-m3`
- 目標／上限：480／512 tokens
- 文字重疊：0；合併相鄰區塊：啟用

目前資料應產生 1,199 個文字 chunks，且 QA 不應有 schema、空內容、重複 ID 或 token 超限錯誤。

### 步驟 6：合併並驗證語料

```powershell
uv run python -m ingest.combine_chunks
```

輸出 `output/chunks/combined-chunks.jsonl` 與 `output/qa/combined-chunks-qa.json`。目前預期為 1,250 筆，包含 1,199 個文字 chunks 與 51 個表格 child chunks。QA 中重複 ID、無效紀錄、空文字、token 超限、無效 parent reference 與 schema 錯誤都應為 0。

### 步驟 7：重建索引

```powershell
uv run dama-rag index --rebuild
uv run dama-rag doctor
```

確認 index 等於新語料筆數，再依序執行 `search`、`context` 與 `ask`。

### 查詢與調整 ingestion 參數

```powershell
uv run python -m ingest.convert_pdf --help
uv run python -m ingest.tables.build_inventory --help
uv run python -m ingest.tables.build_chunks --help
uv run python -m ingest.text.build_view --help
uv run python -m ingest.text.build_chunks --help
uv run python -m ingest.combine_chunks --help
```

| 參數 | 預設值 | 調整時機與影響 |
| --- | ---: | --- |
| `--start-page` | 21 | 正文起始頁改變時調整；設錯會漏掉正文或混入前置頁面。 |
| `--target-tokens` | 480 | 控制理想 chunk 大小；降低通常增加筆數與局部精度。 |
| `--max-tokens` | 512 | chunk 硬上限。 |
| `--tokenizer-model` | `BAAI/bge-m3` | 應與 embedding tokenizer 一致，否則 token QA 無法代表實際輸入。 |

目前文字重疊與表格列重疊都固定為 0，文字 `HybridChunker` 固定啟用 `merge_peers=True`，並不是 CLI 參數。若輸入、輸出或 schema 不在預設位置，請依各命令 `--help` 顯示的選項明確指定。

## 5. 常用執行設定

環境變數會覆蓋 `engine/config.py` 的預設值。

| 環境變數 | 預設值 | 用途與影響 |
| --- | --- | --- |
| `DAMA_EMBEDDING_MODEL` | `BAAI/bge-m3` | 建立與查詢向量；更換後須重建索引。 |
| `DAMA_EMBEDDING_BATCH_SIZE` | `8` | Embedding 批次；記憶體不足時降低。 |
| `DAMA_EMBEDDING_NORMALIZE` | `true` | 向量正規化；改動後須重建索引。 |
| `DAMA_EMBEDDING_MAX_SEQ_LENGTH` | `1024` | Embedding 序列上限；改動後須重建索引。 |
| `DAMA_EMBEDDING_QUERY_PROMPT` | 空值 | 查詢前綴；只影響之後產生的查詢向量，不須重建文件索引。 |
| `DAMA_EMBEDDING_DOCUMENT_PROMPT` | 空值 | 文件前綴；改動後須重建索引。 |
| `DAMA_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | 對初步命中重新排序。 |
| `DAMA_RERANK_BATCH_SIZE` | `8` | Reranking 批次；記憶體不足時降低。 |
| `DAMA_RERANK_MAX_LENGTH` | `1024` | Reranker 輸入上限。 |
| `DAMA_RETRIEVE_K` | `20` | 向量候選數；提高會增加召回與重排成本。 |
| `DAMA_RERANK_K` | `8` | 重排後保留的候選數。 |
| `DAMA_MAX_SOURCES` | `4` | 最後送入回答的來源上限。 |
| `DAMA_MAX_CONTEXT_CHARS` | `60000` | 證據總字元上限。 |
| `DAMA_ANSWER_LANGUAGE` | `auto` | 回答語言，也可由 `--language` 覆蓋。 |
| `DAMA_OLLAMA_MODEL` | `qwen3.6:35b-a3b` | Ollama 生成模型。 |
| `DAMA_TEMPERATURE` | `0` | 生成溫度。 |
| `DAMA_NUM_CTX` | `32768` | Context window；記憶體不足時降低。 |
| `DAMA_THINK` | `false` | 是否使用 thinking 模式。 |
| `DAMA_OLLAMA_HOST` | Ollama 預設位址 | 指定服務；未設時也接受 `OLLAMA_HOST`。 |
| `DAMA_DEVICE` | `auto` | 可設 `auto`、`cuda`、`mps` 或 `cpu`。 |
| `DAMA_COLLECTION_NAME` | `dama_chunks` | Chroma collection 名稱。 |
| `DAMA_ROOT` | 專案根目錄 | 改變整組預設路徑的根。 |
| `DAMA_OUTPUT_DIR` | `<root>/output` | 改變語料與索引輸出根目錄。 |

PowerShell 單一工作階段範例：

```powershell
$env:DAMA_DEVICE = "cpu"
$env:DAMA_NUM_CTX = "8192"
uv run dama-rag doctor
uv run dama-rag ask "什麼是資料治理？" --language zh-hant
```

取消覆蓋：

```powershell
Remove-Item Env:DAMA_DEVICE
Remove-Item Env:DAMA_NUM_CTX
```

不要把金鑰、內部服務網址或其他敏感設定寫入版控。

## 6. 結果檢查與測試

### 語料 QA

檢查下列檔案：

- `output/qa/text-only-qa.json`
- `output/qa/text-chunks-qa.json`
- `output/qa/table-chunks-qa.json`
- `output/qa/combined-chunks-qa.json`

確認總數符合預期，且 schema、空內容、重複 ID、token 超限和 parent reference 等錯誤都是 0。更換來源後數量可以改變，但不應忽略錯誤欄位。

### 自動化測試

```powershell
uv run --frozen pytest
uv run --extra ingest --frozen pytest
```

目前測試套件共有 36 項，涵蓋設定、語料、索引同步、`doctor` 索引診斷、檢索／重排流程、提示詞、Ollama 介面與 ingestion 邏輯。自動化測試仍不等同於大型模型的實際端到端回答。

### 人工驗證順序

1. `dama-rag doctor`：確認檔案、索引與模型存在。
2. `dama-rag search`：確認候選來源與問題相關。
3. `dama-rag context --prompt`：確認表格展開、來源標示與提示詞。
4. `dama-rag ask`：確認模型依證據回答並附來源。

專案目前沒有固定問答 golden set，因此檢索品質仍需以實際問題人工評估。

## 7. 常見問題

### 找不到 `uv`

重新開啟終端機並執行 `uv --version`。若仍找不到，依 [uv 官方安裝說明](https://docs.astral.sh/uv/getting-started/installation/) 安裝並確認安裝目錄已加入 `PATH`。

### 鎖定檔檢查或同步失敗

`pyproject.toml` 與 `uv.lock` 可能不同步。先確認版本是否正確；只有確定要更新相依結果時才執行 `uv lock`，並審查 `uv.lock` 的變更。

### 找不到來源 PDF

只有重建 Docling 產物時需要 PDF。確認預設檔名，或查看 `ingest.convert_pdf --help` 後指定其他輸入。

### Hugging Face 模型下載失敗

確認網路、代理、磁碟空間與 Hugging Face 存取狀態。Embedding 與 reranker 是兩個模型，可能在不同步驟首次下載。

### 索引不存在、筆數不符或 metadata 為 `?`

```powershell
uv run dama-rag index --rebuild
uv run dama-rag doctor
```

目前新索引應與 1,250 筆合併語料一致。

### Embedding fingerprint 不相容

索引的 embedding 設定與目前環境不同。確認 `dama-rag info` 後完整重建，不要沿用舊向量。

### 表格 parent reference 錯誤

合併語料與 `table-parents.jsonl` 可能不是同次建置產物。依第 4 節重建表格 chunks、合併語料與索引。

### Ollama 無法連線或找不到模型

確認服務已啟動，執行 `ollama list`，並比較模型名稱與 `dama-rag info`。遠端服務另檢查 `DAMA_OLLAMA_HOST` 或 `OLLAMA_HOST`。

### 記憶體或 GPU 不足

可改用 CPU，降低 embedding／reranking batch、`DAMA_NUM_CTX` 或 `DAMA_MAX_CONTEXT_CHARS`。簡化入口的預設較保守，但大型 LLM 仍須符合硬體容量。

### Ingestion 因 QA assertion 中止

不要略過檢查。閱讀對應 `*-qa.json`、待檢視 CSV 與錯誤訊息，確認是來源結構改變、token 超限、schema 不符或 parent reference 遺失，再調整輸入或參數。

## 8. 本指南的驗證範圍

本指南已依目前程式碼、設定與產物核對；同一工作區最近一次實際執行的結果為：

- uv 鎖定檔可解析。
- CLI 的 `--help`、`info` 與六個 ingestion 模組的 `--help` 可執行。
- 包含 ingestion 相依的 36 項測試通過。
- 合併語料為 1,250 筆，表格 parents 為 44 筆，現有 QA 未記錄結構或 token 錯誤。

本次已用 `BAAI/bge-m3` 重建 1,250 筆本機索引，並以 `BAAI/bge-reranker-v2-m3` 和較小的 `qwen3:8b` 完成一題實際端到端問答。預設 `qwen3.6:35b-a3b` 尚未下載及實測，完整 PDF 轉換也未重新執行；實際耗時、記憶體需求與回答品質仍取決於執行環境。

## 9. 延伸閱讀

- [README](../README.md)：專案概述、最短安裝與快速開始。
- [Chunk schema](../schemas/chunk-schema.json)：合併 JSONL 的欄位與規則。
- [uv 官方文件](https://docs.astral.sh/uv/)：安裝、環境與鎖定檔。
- [Ollama 官方文件](https://docs.ollama.com/)：模型下載、服務與設定。
- [Sentence Transformers 文件](https://www.sbert.net/)：Embedding 與 CrossEncoder 概念。
- [Docling 官方文件](https://docling-project.github.io/docling/)：PDF 解析與文件格式。
