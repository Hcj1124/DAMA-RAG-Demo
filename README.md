# DAMA RAG Demo

本專案以 Docling 解析 DAMA-DMBOK PDF，分別處理一般文字與表格，再將兩者轉成共同 record schema，供後續 embedding、Chroma hybrid retrieval 與 parent-child table retrieval 使用。

目前已完成表格 pipeline、text-only 過濾、Hybrid text chunking、共同 schema 驗證、文字與表格 children 合併，以及 `engine/` 底下的檢索與生成引擎：bge-m3 embedding、Chroma、bge-reranker-v2-m3 重排、runtime table parent fetch 與 Qwen 生成。

## 這個 fork 做了什麼

上游（[Hcj1124/DAMA-RAG-Demo](https://github.com/Hcj1124/DAMA-RAG-Demo)）完成到階段 8：
Docling 解析、表格與文字各自 chunking、共同 schema 驗證、合併成 `combined-chunks.jsonl`。
階段 9–12 是空的 —— 有 embedding-ready 的 chunk，但沒有東西讀它們。

這個 fork 補上那條路徑，接上地端三件套：

| 階段 | 新增檔案 | 內容 |
|---:|---|---|
| 9 | [`engine/indexing.py`](engine/indexing.py) | `BAAI/bge-m3` embedding → Chroma，1250 筆 / 1024 維 |
| 10 | [`engine/retrieval.py`](engine/retrieval.py) | 向量召回 top-20 → `BAAI/bge-reranker-v2-m3` 重排 → top-8 |
| 11 | [`engine/context.py`](engine/context.py) | runtime table parent fetch |
| 12 | [`engine/pipeline.py`](engine/pipeline.py) | `qwen3.6:35b-a3b` via Ollama，帶 `[Source N]` 引用 |

另外新增 `pyproject.toml` / `uv.lock`（uv 環境，docling 移到 `ingest` extra）、
`engine/cli.py`（`doctor` / `index` / `search` / `context` / `ask` 五個指令）、
以及 `tests/test_engine.py`（21 個測試，用假 adapter 跑完整條漏斗，不需下載模型）。

從乾淨的 clone 到問出第一個問題：

```bash
uv sync
ollama pull qwen3.6:35b-a3b
uv run dama-rag index                            # 約 68 秒
uv run dama-rag ask "資料治理的核心目標是什麼？"
```

### 實測

跨語言檢索成立 —— 語料是純英文，中文問句仍能命中正確段落：

```text
$ uv run dama-rag search "資料治理的核心目標是什麼？" -k 1
 1. [+0.988] 1.2 Goals and Principles  (p. 75, text)
     The goal of Data Governance is to enable an organization to manage data as an asset...
```

表格路徑成立 —— 問 GDPR 原則時，命中的是一組 row，送進 prompt 的是整張表，
所以七條原則答得完整，且 `[Source 1]` 指回 p.58 的 `Table 1 GDPR Principles`。

單題端到端約 26 秒（M5 Pro / 48GB / MPS）。

### 兩個設計判斷

**階段 11 只對表格做 parent fetch。** text records 的 `parent_id` 全是 `null`，
HybridChunker 已經在章節邊界切好並保留標題，chunk 本身就看得懂，也沒有 text parent 檔案可取；
硬造一個是另一種檢索設計，不是查表。所以 text 用自己，table child 展開成完整 parent，
同一張表的多個 child 收斂成一個 source，不會用掉三個引用槽。

**回答語言由程式判定，不交給模型。** 原本是 prompt 裡一句「用問句的語言回答」，
實測 qwen3.6 會把英文問句用中文回答 —— 語料英文、模型中文強，那條規則在十一條裡輸掉。
現在 `detect_language()` 在 Python 看問句字集（CJK → 繁中，否則英文），
prompt 收到的是確定的規則而不是選擇題。

### 尚未完成

Retrieval evaluation 還沒做：沒有 golden set，所以「換成 bge-m3 到底買到多少分」目前是沒有數字的。
上游的 chunking 階段（1–8）未做任何修改。

## 整體架構

```text
原始 PDF
  │
  ▼
DoclingDocument（output/sample.json）
  │
  ├─ 文字 items
  │    └─ text-only view
  │         └─ HybridChunker
  │              └─ schema adapter
  │                   └─ text-chunks.jsonl          [已完成]
  │
  └─ tables[]
       └─ inventory + review
            └─ canonical table parents
                 └─ row-aware table chunker
                      └─ table-chunks.jsonl          [已完成]

text-chunks.jsonl + table-chunks.jsonl
  └─ chunk-schema validation
       └─ combined-chunks.jsonl                      [已完成]
            └─ BAAI/bge-m3 embedding
                 └─ Chroma dama_chunks collection    [已完成]
```

查詢時（`engine/`）：

```text
問句（中文或英文）
  │
  ▼
BAAI/bge-m3 query embedding
  └─ Chroma top-20 recall
       └─ BAAI/bge-reranker-v2-m3 重排 → top-8
            └─ context resolution（階段 11）
                 ├─ text child      → 直接使用該 chunk
                 └─ table child     → 取回 table-parents.jsonl 的完整表格
                      └─ [Source N] 帶頁碼的 prompt
                           └─ qwen3.6:35b-a3b via Ollama
                                └─ 帶引用的答案
```

三個模型的選擇理由：

| 階段 | 模型 | 為什麼 |
|---|---|---|
| Embedding | `BAAI/bge-m3` | 1024 維、8192 token 視窗、100+ 語言，不需要 instruction prefix。也是 chunker 已經用來算 token 的同一個 tokenizer，切塊大小與 embedding 視窗天生一致 |
| Reranking | `BAAI/bge-reranker-v2-m3` | 與 embedding 同一個 XLM-RoBERTa 底座，兩階段對「什麼叫相關」跨語言的判斷一致 |
| Generation | `qwen3.6:35b-a3b` via Ollama | MoE 只活化 3B 參數：35B 級品質、小模型速度，繁體中文輸出穩定 |

文字與表格不使用相同的 chunking 演算法：

- 文字使用 HybridChunker，保留章節標題、段落邊界與語意切點。
- 表格使用 row-aware chunker，保證資料列完整、表頭重複、無漏列及無重複列。
- 兩者最後使用相同的 `chunk-schema.json` 外層格式、相同 tokenizer、相同 embedding model 與相同 collection。

共同 schema 不是 embedding model 的要求。Embedding model 實際只接收每筆 record 的 `content.text`；共同 schema 的目的，是統一 ID、來源追蹤、QA、合併、Chroma metadata 與 parent retrieval。

## 專案結構

```text
.
├─ input/
│  └─ dama-dmbok-2nd-edition.pdf     # 本機來源，不提交 Git
├─ output/
│  ├─ sample.json                    # 完整 DoclingDocument，不提交 Git
│  ├─ sample.md                      # Docling Markdown，不提交 Git
│  ├─ chunk-schema.json              # 文字與表格共同 record schema
│  ├─ table-inventory.all.jsonl      # 原始 63 個 table items
│  ├─ table-inventory.jsonl          # 44 張核准表格
│  ├─ table-inventory-excluded-toc.jsonl
│  ├─ table-inventory-excluded-review.jsonl
│  ├─ table-review.csv               # 自動與人工審查結果
│  ├─ table-parents.jsonl            # 44 個完整 canonical parents
│  ├─ canonical-tables/              # 每張 parent 的 JSON 與 Markdown
│  ├─ table-chunks.jsonl             # embedding-ready table children
│  ├─ table-chunks-qa.json           # 表格 chunk QA 摘要
│  ├─ text-chunks.jsonl              # embedding-ready text records
│  ├─ text-chunks-qa.json            # 文字 chunk QA 摘要
│  ├─ combined-chunks.jsonl          # text + table_child records
│  └─ combined-chunks-qa.json        # 合併與交叉 QA
├─ table-caption-overrides.json      # 人工補充的跨頁表格 captions
├─ build_table_inventory.py          # 提取目標表格與前處理
├─ build_table_chunks.py             # 表格 chunking
├─ build_text_only.py                # 排除 table/picture/navigation 內容
├─ build_text_chunks.py              # HybridChunker 與 text schema adapter
├─ combine_chunks.py                 # 合併與交叉驗證
├─ engine/                           # 階段 9–12：檢索與生成
│  ├─ config.py                      # 所有可調參數的唯一來源（DAMA_* 環境變數）
│  ├─ corpus.py                      # 讀 combined-chunks.jsonl / table-parents.jsonl
│  ├─ ports.py                       # Embedder / Reranker / VectorStore / LLM 協定
│  ├─ adapters/                      # bge-m3、bge-reranker-v2-m3、Chroma、Ollama
│  ├─ indexing.py                    # 階段 9：embedding 與寫入 Chroma
│  ├─ retrieval.py                   # 階段 10：向量召回 + cross-encoder 重排
│  ├─ context.py                     # 階段 11：runtime table parent fetch
│  ├─ prompting.py                   # [Source N] 引用契約與 context 預算
│  ├─ pipeline.py                    # composition root
│  └─ cli.py                         # dama-rag doctor / index / search / context / ask
├─ pyproject.toml                    # uv 環境；docling 在 `ingest` extra
├─ tests/                             # pipeline regression tests
└─ test_docling.py                   # PDF → Docling JSON/Markdown
```

## 目前進度

| 階段 | 工作 | 狀態 | 產出 |
|---:|---|---|---|
| 1 | 定義共同 record schema | 完成 | `chunk-schema.json` |
| 2 | 匯出 Docling table inventory | 完成 | `table-inventory.all.jsonl` |
| 3 | 排除目錄與錯誤 table items | 完成 | `table-review.csv` |
| 4 | 建立 canonical table parents | 完成 | `table-parents.jsonl` |
| 5 | 建立 row-aware table children | 完成 | `table-chunks.jsonl` |
| 6 | 表格 chunk QA | 完成 | `table-chunks-qa.json` |
| 7 | 建立及驗證 text chunks | 完成 | `text-chunks.jsonl` |
| 8 | 合併文字與表格 chunks | 完成 | `combined-chunks.jsonl` |
| 9 | BGE-M3 embedding 與 Chroma 寫入 | 完成 | `output/chroma_db/`（`dama_chunks`，1250 筆、1024 維） |
| 10 | BGE reranking | 完成 | `engine/retrieval.py` |
| 11 | Runtime parent store 與 parent fetch | 完成 | `engine/context.py` |
| 12 | Qwen generation via Ollama | 完成 | `engine/pipeline.py` |
| – | Retrieval evaluation | 未開始 | 尚無 golden set |

### 表格審查結果

```text
Docling 原始 table items：63
排除的目錄：11
人工排除／非檢索用 items：8
核准 canonical tables：44
```

人工排除項目包含 Figure、ER 圖、只有屬性的示意版面、流程圖、書末 index，以及不具知識檢索價值的 contributors/reviewers 名單。人工補充的 caption 保存在 `table-caption-overrides.json`，重跑產生器時不會被原始 Docling JSON 覆寫。

## 環境設定

`output/` 底下的 chunk 檔案已經進版控，所以**問問題不需要 Docling，也不需要原始 PDF**。
安裝因此分成兩層：預設只裝檢索引擎，重跑 chunking 才裝 `ingest` extra。

### 檢索引擎（uv，預設）

```bash
uv sync
uv run dama-rag doctor
```

`uv sync` 依 `pyproject.toml` 與 `uv.lock` 建立 `.venv`。`doctor` 會逐項檢查 chunk 檔案、
torch device、Chroma 索引與 Ollama，任何一項不過都會直接告訴你補救指令。

### 檢索引擎（Windows PowerShell，手動 `.venv`）

不使用 uv 時，可直接建立並啟用專案內的 `.venv`：

```powershell
cd D:\docling_testing

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

若 PowerShell 阻擋 `Activate.ps1`，只調整目前這個終端工作階段：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

啟用後，命令列前方會出現 `(.venv)`，接著執行：

```powershell
dama-rag doctor
dama-rag index
dama-rag search "資料治理的核心目標是什麼？" -k 3
dama-rag ask "資料治理的核心目標是什麼？"
```

不啟用環境也可以使用完整路徑：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\dama-rag.exe doctor
.\.venv\Scripts\dama-rag.exe index
```

執行測試：

```powershell
python -m pip install pytest
python -m pytest
```

離開虛擬環境：

```powershell
deactivate
```

已驗證環境：

```text
Python 3.13.3          Windows / NVIDIA CUDA
torch 2.12.1+cu132    sentence-transformers 6.0.1
chromadb 1.5.9        ollama 0.6.2 (client)

Python 3.12.11          macOS 26.6 / Apple Silicon / MPS
torch 2.13.0            sentence-transformers 6.0.1
chromadb 1.5.9          ollama 0.6.2 (client) / 0.33.2 (server)
```

生成需要本機 Ollama 已經拉好模型：

```bash
ollama pull qwen3.6:35b-a3b
```

Embedding 與 reranker 會在第一次使用時從 Hugging Face 下載（合計約 6.4 GB）並快取在
`~/.cache/huggingface`。之後要完全離線跑，設 `HF_HUB_OFFLINE=1` 即可。

### 重跑 chunking（`ingest` extra）

只有在要從 PDF 重新產生 `output/` 時才需要：

```bash
uv sync --extra ingest
```

手動 `.venv` 對應命令：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[ingest]"
```

原始 chunk pipeline 的驗證環境是 Python 3.13.3 / docling 2.123.1 / docling-core 2.92.0；
`requirements.txt` 保留那組已驗證的版本，`docling-core` 由相容版本的 `docling` 帶入，
不另外釘選以避免和 Docling 的相依約束衝突。

來源 PDF 必須放在：

```text
input/dama-dmbok-2nd-edition.pdf
```

若只想安裝原始 chunk pipeline 的固定版本：

```powershell
cd D:\docling_testing

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

原始 PDF、`.venv`、`sample.json`、`sample.md` 與 `output/chroma_db/` 已由 `.gitignore` 排除。

## 建立 DoclingDocument

```powershell
.\.venv\Scripts\python.exe test_docling.py
```

產出：

```text
output/sample.json
output/sample.md
```

後續 inventory 與文字 chunking 都以 `sample.json` 為解析來源，不需要重複解析 PDF。

## 表格處理流程

### 階段 1–4：Inventory、review 與 canonical parents

```powershell
.\.venv\Scripts\python.exe build_table_inventory.py
```

預期摘要：

```text
all=63 included=44 excluded_toc=11 excluded_review=8 parents=44
```

此步驟會：

1. 產生共同 `chunk-schema.json`。
2. 從 Docling `tables[]` 建立 inventory。
3. 解析 caption `$ref`，並套用 `table-caption-overrides.json`。
4. 排除 Contents、Figure、ER 圖、流程圖及 index。
5. 將每張核准表格保存為完整 Markdown 與結構化 JSON parent。

Canonical JSON 是權威版本，保留原始 cell、row/column span 與 bbox；Markdown 無法完整表達合併儲存格，只用於閱讀和顯示。

### 階段 5：Row-aware table children

```powershell
.\.venv\Scripts\python.exe build_table_chunks.py
```

預設參數：

```text
tokenizer：BAAI/bge-m3
target tokens：480
hard limit：512
row overlap：0
```

完整命令：

```powershell
.\.venv\Scripts\python.exe build_table_chunks.py `
  --input output/table-parents.jsonl `
  --output output/table-chunks.jsonl `
  --qa-output output/table-chunks-qa.json `
  --schema output/chunk-schema.json `
  --tokenizer-model "BAAI/bge-m3" `
  --target-tokens 480 `
  --max-tokens 512
```

Chunker 會：

1. 從 parent 的 `structured.cells` 重建列與欄。
2. 使用 Docling `column_header=true` 判定正式表頭。
3. 對缺少表頭的表格使用 synthetic column names；可可靠判定的兩欄 key-value 表則使用左欄作動態屬性。
4. 將每一個非空 cell 轉成 `屬性: 值`。
5. 以最終 `content.text` 計算 token，而不是用 Markdown、JSON 或字元數。
6. 依原始列順序打包，目標 480 tokens、hard limit 512 tokens。
7. 每個 child 重複 caption 與欄位屬性，不重疊資料列。
8. 建立 child ID 與 `parent_id`。
9. 逐筆執行共同 schema、token 與 row coverage QA。

### Attribute-value embedding text

一般表格：

```text
Table: Table 1 GDPR Principles

Row 2:
GDPR Principle: Purpose Limitation
Description of Principle: Personal data must be collected...
```

Key-value 表格：

```text
Row 0:
ISBN, Print ed.: 9781634622349
```

`metadata.embedding_serialization` 會記錄轉換方式：一般表格為 `attribute_value_v1`，可可靠判定的兩欄 key-value 表為 `key_value_v1`。未來若調整序列化規則，可依此欄位區分並重建 embeddings。

每個 child 同時保存三種內容：

| 欄位 | 用途 |
|---|---|
| `content.text` | Attribute-value 格式，實際送 embedding model |
| `content.markdown` | 子表 Markdown，供顯示與回答引用 |
| `content.structured` | Row/cell 結構，供 QA、重建與除錯 |

空 cell 不寫入 `content.text`，避免增加無語意的 `(empty)` token，但仍保留在 canonical parent 的 structured cells 中。

### 目前 table chunk QA

```text
Parents：44
Children：51
Fragmented rows：0
Minimum observed tokens：34
Maximum observed tokens：479
Over 512：0
Schema errors：0
Attribute-value errors：0
Row overlap：0
```

QA 會確認：

- child ID 唯一。
- 每個 `parent_id` 都能找到 canonical parent。
- 每個非表頭資料列恰好出現一次。
- 沒有資料列 overlap、漏列或重複列。
- 每個非空 cell 都能在 `content.text` 找到對應的 `屬性: 值`。
- 沒有空 chunk。
- 最終 embedding text 不超過 512 tokens。
- 所有 records 均符合 `chunk-schema.json`。

## 共同 chunk schema

文字與表格 children 使用相同頂層欄位：

```json
{
  "schema_version": "1.0.0",
  "record_id": "...",
  "parent_id": null,
  "content_type": "text | table_parent | table_child",
  "source": {
    "document_id": "...",
    "source_filename": "...",
    "pages": [1],
    "locators": {}
  },
  "content": {
    "text": "...",
    "markdown": null,
    "structured": null
  },
  "metadata": {}
}
```

差異如下：

| 項目 | Text child | Table child | Table parent |
|---|---|---|---|
| `content_type` | `text` | `table_child` | `table_parent` |
| `parent_id` | `null` | 指向 table parent | `null` |
| `content.text` | Heading context + paragraph | Attribute-value rows | 完整表格純文字 |
| `content.markdown` | 通常 `null` | 子表 Markdown | 完整表格 Markdown |
| `content.structured` | 通常 `null` | 子表 rows | 完整 cells 與 spans |

### 具體轉換作法：先建契約，再由 adapter 寫入 records

共同 schema 是 JSONL 的「資料契約」，不是把 Docling 原始物件直接複製出去。實作時固定依下列順序處理：

1. 執行 `build_table_inventory.py`，由 `schema()` 產生 `output/chunk-schema.json`。這個檔案規定頂層必填欄位、允許的 `content_type`、`source` 與 `content` 的結構，以及 ID 格式。
2. 每個來源單位建立一筆新的 Python `dict`，明確填入共同欄位；不可直接將 `DocChunk` 或 `TableItem` 寫成 JSONL。
3. 將適合檢索的可讀文字寫到 `content.text`；將顯示用 Markdown 寫到 `content.markdown`；將不可遺失的列、儲存格、span 等結構寫到 `content.structured`。
4. 將頁碼、Docling reference、table index 等追溯資訊放在 `source`，將 chunker、token 數、caption、序列化規則等處理資訊放在 `metadata`。
5. 以 `Draft202012Validator(output/chunk-schema.json)` 驗證每筆新 record，完成後才寫入 JSONL；合併 text/table JSONL 前再次驗證 `record_id` 唯一性。

欄位來源對照如下：

| 共同 schema 欄位 | Text adapter（規劃） | Table adapter（已實作） |
|---|---|---|
| `record_id` | `{document_id}:text:{ordinal:06d}` | `{document_id}:table:{table_index:03d}:parent:000` 或 `...:child:{ordinal:03d}` |
| `parent_id` | `null` | parent 為 `null`；child 指向 canonical parent 的 `record_id` |
| `content_type` | `text` | `table_parent` 或 `table_child` |
| `source.document_id/source_filename` | 固定由輸入檔識別 | 由 table parent 繼承 |
| `source.pages` | `doc_items[].prov[].page_no` 去重排序 | `TableItem.prov[].page_no` |
| `source.locators` | `doc_items[].self_ref` → `docling_refs` | `TableItem.self_ref`、`tables[]` index、child 的 `row_indices` |
| `content.text` | `chunker.contextualize(chunk)` | caption + `欄名: 值` 的 row-aware 序列化 |
| `content.markdown` | `null`（除非需要顯示用版本） | 完整或子集表格的 Markdown |
| `content.structured` | `null` 或文字區塊結構 | parent 保留 cells/spans；child 保留 header、rows、row fragments |
| `metadata` | headings、captions、tokenizer、token count、chunker 設定 | caption、header 狀態、tokenizer、token count、`embedding_serialization` |

表格轉換的實際路徑是 `DoclingDocument.tables[]` → inventory/review → `table_parent` → `table_child`。`build_table_inventory.py` 先將核准表格轉為一筆可追溯的 canonical parent；`build_table_chunks.py` 從 parent 的 `structured.cells` 還原 grid，辨識 `column_header`，依列打包，並建立具有同一 `parent_id` 的 children。每個 child 的 `content.text` 才是 embedding 的輸入，parent 則保留供命中 child 後回取完整表格。

文字轉換採用相同 adapter 觀念：先由 text-only view 排除 table/picture/navigation items，交給 `HybridChunker`，再將 contextualized text、來源 refs、頁碼與 token count 映射到上述固定欄位。因此文字與表格的內容生成方式不同，但能用同一個 schema 驗證、合併、embedding 與 Chroma 寫入流程。

## Provenance 欄位是否需要

`docling_ref`、`table_index` 與 `bounding_boxes` 不參與 embedding，也不一定需要寫入 Chroma；它們目前保留於離線 JSONL，主要用於追溯與除錯。

| 欄位 | 用途 | 建議是否寫入 Chroma |
|---|---|---|
| `docling_ref` | 回到原始 Docling item，例如 `#/tables/48` | 通常不需要 |
| `table_index` | 對照 `tables[]`、review CSV 與產生器 log | 可選；和 ID 有部分重複 |
| `bounding_boxes` | PDF 頁面框選、截圖、citation highlight | 沒有頁面高亮 UI 時不需要 |
| `pages` | 顯示引用頁碼、metadata filter | 建議保留 |
| `row_indices` | Row coverage QA、回取 parent 中的原始列 | 建議離線保留；Chroma 可序列化成字串 |
| `parent_id` | Child 命中後取得完整表格 | 必須保留 |

建議分成兩層：

- 離線 JSONL：保留完整 provenance，方便重建、QA 與除錯。
- Chroma metadata：只保留 retrieval 必需的 scalar 欄位，例如 `parent_id`、`content_type`、`document_id`、`page`、`table_index` 與 `token_count`。

`docling_ref` 與 `table_index` 有資訊重複。若後續確定不需要回到 DoclingDocument，可從 Chroma metadata 省略 `docling_ref`。

## Tokenizer 與 embedding model

目標模型組合：

| 階段 | 模型 | 目前狀態 |
|---|---|---|
| Embedding | `BAAI/bge-m3` | tokenizer、embedding adapter 與 Chroma indexing 已實作 |
| Reranking | `BAAI/bge-reranker-v2-m3` | cross-encoder adapter 與 retrieval funnel 已實作 |
| Generation | `qwen3.6:35b-a3b` via Ollama | Ollama adapter、prompt 與 CLI 已實作 |

Tokenizer 是 embedding model 的前置處理：

```text
content.text
→ tokenizer
→ token IDs
→ embedding model
→ vector
```

文字與表格 chunker 預設只載入 `BAAI/bge-m3` 的 tokenizer 來計數，沒有載入 embedding model 權重，也沒有產生 embeddings。BGE-M3 可接受較長輸入，但本專案仍採 target 480／hard limit 512，因為模型上限不等於適合檢索的 chunk 粒度。

Embedding 模型已選定為 `BAAI/bge-m3`，因此文字與表格 children 必須都使用該 tokenizer 重新產生。不同 tokenizer 對同一段文字可能產生不同 token 數，不能沿用舊 MiniLM 的 token metadata。

更換 HuggingFace embedding model 時：

```powershell
.\.venv\Scripts\python.exe build_table_chunks.py `
  --tokenizer-model "<new-embedding-model>" `
  --target-tokens 480 `
  --max-tokens 512
```

Canonical parents 不需要重建；只需以新 tokenizer 重新建立 children 與 embeddings。若新模型不是 HuggingFace tokenizer，需要實作相對應的 tokenizer adapter。

## 文字處理方法

### 1. 建立 text-only DoclingDocument view

輸入仍是：

```text
output/sample.json (加入 sample.json 至 /output)
```

但在 HybridChunker 前先排除：

- `TableItem`，避免和 table children 重複。
- `PictureItem` 與已知圖形版面。
- Contents／index 等導航文字。
- 重複頁首頁尾與純頁碼。
- 已由表格管線使用的 table caption，避免同一 caption 同時成為獨立 text chunk。

不要直接對完整文件 chunk 後，只靠 `content_type` 丟棄表格 chunks；HybridChunker 可能在結構合併時產生混合 doc items。較安全的方法是先建立 text-only view，再執行 HybridChunker。

以本機 `output/sample.json` 建立 text-only view：

```powershell
.\.venv\Scripts\python.exe build_text_only.py
```

此步驟會移除 table、picture、caption、page header/footer、第 21 頁以前內容，以及偵測到 `Index` 章節後的內容，並將來源 `sample.json` SHA-256 寫入 QA。

### 2. 使用和 embedding model 相同的 tokenizer

規劃值：

```text
HybridChunker packing limit：480 tokens
最終 QA hard limit：512 tokens
text overlap：先採 0
```

HybridChunker 沒有一般 splitter 的 `chunk_overlap` 參數。它會先依 Docling 文件階層與 metadata 分組，再以 tokenizer 做超限切分，並可合併相同 heading metadata 下的短 chunks。

```powershell
.\.venv\Scripts\python.exe build_text_chunks.py `
  --tokenizer-model "BAAI/bge-m3" `
  --target-tokens 480 `
  --max-tokens 512
```

### 3. 使用 HybridChunker 保留文件結構

文字 chunk 應保留：

- Chapter／section headings。
- Paragraph 與 list 邊界。
- 原始頁碼。
- Docling text refs。
- Chunker 與 tokenizer 設定。

Embedding text 建議使用 HybridChunker 的 contextualized output：

```text
Chapter 3: Data Management Frameworks
The DAMA-DMBOK Framework

The framework identifies...
```

這樣即使原始段落很短，embedding 仍知道其所屬章節。

### 4. 將 DocChunk 轉成共同 schema

HybridChunker 原始輸出不會自動符合 `chunk-schema.json`，需要 schema adapter：

```text
chunker.contextualize(chunk)     → content.text
chunk.meta.headings              → metadata.headings
chunk.meta.captions              → metadata.captions
doc_items[].self_ref             → source.locators.docling_refs
doc_items[].prov[].page_no       → source.pages
tokenizer count                  → metadata.token_count
```

預期文字 record：

```json
{
  "schema_version": "1.0.0",
  "record_id": "dama-dmbok-2nd-edition:text:000001",
  "parent_id": null,
  "content_type": "text",
  "source": {
    "document_id": "dama-dmbok-2nd-edition",
    "source_filename": "dama-dmbok-2nd-edition.pdf",
    "pages": [21],
    "locators": {
      "docling_refs": ["#/texts/120", "#/texts/121"]
    }
  },
  "content": {
    "text": "Chapter heading\n\nContextualized paragraph...",
    "markdown": null,
    "structured": null
  },
  "metadata": {
    "headings": ["Chapter 1", "Data Management"],
    "token_count": 420,
    "tokenizer_name": "<embedding tokenizer>",
    "chunker": "docling_hybrid",
    "chunk_target_tokens": 480,
    "chunk_max_tokens": 512,
    "chunk_overlap": 0
  }
}
```

### 5. Text chunk QA

建立 `text-chunks.jsonl` 前至少檢查：

- 每筆符合 `chunk-schema.json`。
- `content.text` 不為空。
- 最終 token count 不超過 512。
- `record_id` 唯一。
- 頁碼與 source refs 不為空。
- 無大量重複頁首頁尾。
- 無表格內容重複進入 text chunks。
- 抽樣確認 heading context 與段落相符。

## 合併與 embedding

階段 8 僅合併可被 embedding 的 records：

```text
text-chunks.jsonl
+ table-chunks.jsonl
= combined-chunks.jsonl
```

```powershell
.\.venv\Scripts\python.exe combine_chunks.py
```

合併 QA 會檢查 schema、全域 ID、token 上限、tokenizer/document 相容性，以及每個 table child 的 `parent_id` 是否存在於 `table-parents.jsonl`。

目前以本機 `sample.json` 與 `BAAI/bge-m3` tokenizer 產生：

```text
Text records：1199
Table children：51
Combined records：1250
Over 512：0
Invalid parent links：0
Schema errors：0
Index/navigation chunks：0
```

`table-parents.jsonl` 不直接放入 `combined-chunks.jsonl`：parent 不參與檢索，只在命中 child 之後被取回。這條 runtime 路徑實作在 `engine/context.py`（見下一節）。Text records `parent_id=null`，沒有 text parent-child。

Embedding model 實際輸入：

```python
embedding_inputs = [record["content"]["text"] for record in combined_records]
```

Chroma metadata 通常只接受 scalar value。`caption`、`pages`、`row_indices`、bbox 等 list/object 在寫入前要拆成簡單欄位或序列化成 JSON 字串，不應將完整 schema record 原封不動當成 Chroma metadata。

建議的 Chroma mapping：

```text
id        = record_id
document  = content.text
metadata  = content_type, parent_id, document_id, page,
            table_index, token_count, schema_version
```


## 檢索與生成引擎（階段 9–12）

引擎住在 `engine/`，完全不碰 PDF：它只讀 `combined-chunks.jsonl` 與 `table-parents.jsonl`。
根目錄的 `build_*.py` 負責產生那些檔案，兩邊透過 `chunk-schema.json` 這個契約溝通。

### 四個指令

```bash
uv run dama-rag doctor                    # chunk 檔、device、索引、Ollama 是否就緒
uv run dama-rag index                     # 階段 9：把 1250 筆 chunk 寫進 Chroma
uv run dama-rag ask "資料治理的核心目標是什麼？"   # 階段 10–12：完整流程
uv run dama-rag ask                       # 不帶問題則進入互動模式
```

診斷用，不花生成成本：

```bash
uv run dama-rag search "GDPR 的資料保護原則"     # 只看重排後的候選與分數
uv run dama-rag context "GDPR 的資料保護原則"    # 看模型實際會讀到哪些 source
uv run dama-rag context "..." --prompt          # 印出完整 prompt
```

### 建索引

```text
$ uv run dama-rag index
Index updated: 1250 records in the collection
(1250 embedded, 0 unchanged, 0 deleted) using BAAI/bge-m3 [1024-dim]
```

M5 Pro 上約 68 秒。每個向量會存下原文的 hash，所以重跑只會 embed 真正改過的 record、
刪掉已經從 `combined-chunks.jsonl` 消失的 record。換 embedding model 時會偵測到並自動
整個重建 —— 混用兩個模型的向量不會報錯，只會安靜地回傳看起來很合理的垃圾。

### 階段 11：為什麼只有表格需要 parent fetch

檢索單位與閱讀單位在這個語料裡不是同一件事，而且兩種 content type 的答案不同：

- **text**：HybridChunker 已經在章節邊界切開並保留標題，chunk 本身就看得懂。
  沒有 text parent 檔案可以取，硬造一個是另一種檢索設計，不是查表。
- **table_child**：一組 row 沒有表頭幾乎無法閱讀，答案又常在隔壁列。
  所以命中的 child 會被展開成 `table-parents.jsonl` 裡的完整表格，
  而且同一張表的多個 child 會收斂成一個 source，不會用掉三個引用槽。

實測（`uv run dama-rag context "GDPR 的資料保護原則有哪些？"`）：

```text
[Source 1] Table 1 GDPR Principles  (p. 58, table)
[Source 2] 3.2 Principles Behind Data Privacy Law  (pp. 58-60, text)
```

Source 1 命中的是一組 row，送進 prompt 的是整張表，因此模型答得出全部七條原則。

### 跨語言檢索

語料是純英文，問句常是中文。這正是選 bge-m3 的理由：

```text
$ uv run dama-rag search "資料治理的核心目標是什麼？" -k 3
 1. [+0.988] 1.2 Goals and Principles  (p. 75, text)
     The goal of Data Governance is to enable an organization to manage data as an asset...
 2. [+0.961] 1.2 Goals  (p. 22, text)
 3. [+0.926] 1.2 Goals and Principles  (p. 538, text)
```

### 設定

所有可調參數集中在 `engine/config.py`，每一項都能用 `DAMA_*` 環境變數覆寫，
不需要改程式。常用的幾個：

| 變數 | 預設 | 說明 |
|---|---|---|
| `DAMA_EMBEDDING_MODEL` | `BAAI/bge-m3` | 換掉會強制重建索引 |
| `DAMA_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | |
| `DAMA_OLLAMA_MODEL` | `qwen3.6:35b-a3b` | |
| `DAMA_RETRIEVE_K` | `20` | 向量召回數量 |
| `DAMA_RERANK_K` | `8` | 重排後保留數量 |
| `DAMA_MAX_SOURCES` | `4` | 進 prompt 的 source 上限 |
| `DAMA_NUM_CTX` | `32768` | Ollama context window |
| `DAMA_ANSWER_LANGUAGE` | `auto` | `auto` 依問句字集自動判定；也可固定成 `en` 或 `zh-hant` |
| `DAMA_DEVICE` | 自動 | `mps` / `cuda` / `cpu` |
| `DAMA_THINK` | `0` | qwen3.6 預設開 thinking，RAG 不需要 |

`uv run dama-rag info` 會印出實際生效的設定。

### 程式介面

```python
from engine.pipeline import build_pipeline

pipeline = build_pipeline()
answer = pipeline.answer("What does a Data Steward do?")

print(answer.answer)
for citation in answer.citations:
    print(citation.source_id, citation.title, citation.pages)
```

`engine/ports.py` 定義了 Embedder、Reranker、VectorStore、LanguageModel 四個協定，
`engine/pipeline.py` 是唯一決定「哪個實作接哪個協定」的地方。換模型是改設定，
不是改程式；測試則可以直接用假的 adapter 組出整條流程。

### 測試

```bash
uv run pytest tests/test_engine.py      # 引擎：26 個測試，不需要下載模型
uv run --extra ingest pytest            # 全部，含 chunk pipeline 的既有測試
```

引擎測試用假的 adapter 跑完整條漏斗，因此不到一秒就跑完；
`RealCorpusTest` 則會真的去讀 `output/` 裡已進版控的 chunk 檔案，
確保 chunk pipeline 的改動若破壞了載入契約，會在這裡失敗而不是在回答問題時才爆。

### 回答語言由程式決定，不由模型決定

`auto` 原本是 prompt 裡的一句「用問句的語言回答」，實測 qwen3.6 會把英文問句用中文回答 ——
語料是英文、模型中文強，那一句規則在十一條裡輸掉了。
現在 `engine/prompting.py` 的 `detect_language()` 在 Python 裡看問句字集（CJK → 繁中，否則英文），
prompt 收到的是確定的規則而不是選擇題。英文問句只含拉丁字母，所以不會誤判成中文。
