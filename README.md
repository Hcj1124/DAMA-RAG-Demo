# DAMA RAG Demo

本專案以 Docling 解析 DAMA-DMBOK PDF，分別處理一般文字與表格，再將兩者轉成共同 record schema，供後續 embedding、Chroma hybrid retrieval 與 parent-child table retrieval 使用。

目前已完成表格 pipeline 的盤點、人工審查、canonical parents、row-aware children 與自動 QA；文字 pipeline、合併、embedding 與 Chroma 寫入仍是後續工作。

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
  │                   └─ text-chunks.jsonl          [尚未實作]
  │
  └─ tables[]
       └─ inventory + review
            └─ canonical table parents
                 └─ row-aware table chunker
                      └─ table-chunks.jsonl          [已完成]

text-chunks.jsonl + table-chunks.jsonl
  └─ chunk-schema validation
       └─ combined-chunks.jsonl                      [尚未實作]
            └─ embedding model
                 └─ Chroma dama_chunks collection   [尚未實作]
```

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
│  ├─ table-inventory.jsonl          # 46 張核准表格
│  ├─ table-inventory-excluded-toc.jsonl
│  ├─ table-inventory-excluded-review.jsonl
│  ├─ table-review.csv               # 自動與人工審查結果
│  ├─ table-parents.jsonl            # 46 個完整 canonical parents
│  ├─ canonical-tables/              # 每張 parent 的 JSON 與 Markdown
│  ├─ table-chunks.jsonl             # 52 個 embedding children
│  └─ table-chunks-qa.json           # 表格 chunk QA 摘要
├─ table-caption-overrides.json      # 人工補充的跨頁表格 captions
├─ build_table_inventory.py          # 提取目標表格與前處理
├─ build_table_chunks.py             # 表格 chunking
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
| 7 | 合併文字與表格 chunks | 未開始 | `combined-chunks.jsonl` |
| 8 | Embedding 與 Chroma 寫入 | 未開始 | `dama_chunks` |
| 9 | Retrieval 測試問題 | 未開始 | Retrieval evaluation |
| 10 | 決定 parent-child retrieval 策略 | 未開始 | v0.2 design |

### 表格審查結果

```text
Docling 原始 table items：63
排除的目錄：11
人工排除的非表格 items：6
核准 canonical tables：46
```

人工排除項目包含 Figure、ER 圖、只有屬性的示意版面、流程圖與書末 index。人工補充的 caption 保存在 `table-caption-overrides.json`，重跑產生器時不會被原始 Docling JSON 覆寫。

## 環境設定

目前驗證環境：

```text
Python 3.13.3
docling 2.123.1
docling-core 2.92.0
```

Windows PowerShell：

```powershell
cd D:\docling_testing

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` 列出本專案程式直接使用、且已在目前環境驗證過的版本。`docling-core` 由相容版本的 `docling` 安裝；不另外釘選它，避免和 Docling 發布的相依約束衝突。

來源 PDF 必須放在：

```text
input/dama-dmbok-2nd-edition.pdf
```

原始 PDF、`.venv`、`sample.json` 與 `sample.md` 已由 `.gitignore` 排除。

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
all=63 included=46 excluded_toc=11 excluded_review=6 parents=46
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
tokenizer：sentence-transformers/all-MiniLM-L6-v2
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
  --tokenizer-model "sentence-transformers/all-MiniLM-L6-v2" `
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
Parents：46
Children：52
Fragmented rows：0
Minimum observed tokens：35
Maximum observed tokens：439
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

Tokenizer 是 embedding model 的前置處理：

```text
content.text
→ tokenizer
→ token IDs
→ embedding model
→ vector
```

目前 table chunker 只載入 `all-MiniLM-L6-v2` 的 tokenizer 來計數，沒有載入 embedding model 權重，也沒有產生 embeddings。

最終決定 embedding model 後，必須使用該模型對應的 tokenizer 重新產生文字與表格 children。不同 tokenizer 對同一段文字可能產生不同 token 數，不能用目前的 MiniLM token count 保證另一個模型仍低於 512。

更換 HuggingFace embedding model 時：

```powershell
.\.venv\Scripts\python.exe build_table_chunks.py `
  --tokenizer-model "<new-embedding-model>" `
  --target-tokens 480 `
  --max-tokens 512
```

Canonical parents 不需要重建；只需以新 tokenizer 重新建立 children 與 embeddings。若新模型不是 HuggingFace tokenizer，需要實作相對應的 tokenizer adapter。

## 後續文字處理方法

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

### 2. 使用和 embedding model 相同的 tokenizer

規劃值：

```text
HybridChunker packing limit：480 tokens
最終 QA hard limit：512 tokens
text overlap：先採 0
```

HybridChunker 沒有一般 splitter 的 `chunk_overlap` 參數。它會先依 Docling 文件階層與 metadata 分組，再以 tokenizer 做超限切分，並可合併相同 heading metadata 下的短 chunks。

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

階段 7 僅合併可被 embedding 的 records：

```text
text-chunks.jsonl
+ table-chunks.jsonl
= combined-chunks.jsonl
```

`table-parents.jsonl` 通常不直接送 embedding，而是放入 parent store，或以另一個 collection 保存。Table child 被檢索命中後，再由 `parent_id` 取得完整表格。

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
