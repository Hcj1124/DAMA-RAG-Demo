"""DAMA RAG 示範的第 9 至 12 階段。

此套件針對 ``output/`` 中的 chunks 執行 embedding、檢索、重排序、母紀錄展開，
以及有來源依據的本機回答生成。

    from engine.pipeline import build_pipeline

    answer = build_pipeline().answer("What does a Data Steward do?")
    print(answer.answer)

第 1 至 8 階段的切塊流程位於獨立的 ``ingest`` 套件。本套件不解析 PDF，只讀取
``combined-chunks.jsonl`` 與 ``table-parents.jsonl``。
"""

from engine.config import Settings

__all__ = ["Settings", "__version__"]
__version__ = "0.1.0"
