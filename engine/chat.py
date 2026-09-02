"""單一指令即可啟動的 DAMA-DMBOK 互動問答程式。

請從專案根目錄執行：
    ./.venv/Scripts/python.exe -m engine.chat

第一次執行時會視需要下載檢索模型並建立本機 Chroma 索引，之後可直接進入問答。
預設讓檢索留在 CPU，避免與 Ollama 中的 Qwen 爭用 GPU 記憶體。
"""

from __future__ import annotations

import os

# 只設定預設值，進階使用者仍可透過 PowerShell 環境變數覆寫而不需修改程式。
os.environ.setdefault("DAMA_OLLAMA_MODEL", "qwen3.6:35b-a3b")
os.environ.setdefault("DAMA_DEVICE", "cpu")
os.environ.setdefault("DAMA_NUM_CTX", "8192")
os.environ.setdefault("DAMA_MAX_CONTEXT_CHARS", "18000")

from engine.config import Settings
from engine.errors import EngineError
from engine.pipeline import build_pipeline, build_store


def _print_sources(citations) -> None:
    """列出回答引用的標題與頁碼。"""
    print("\n來源：")
    for citation in citations:
        page = (
            f"p. {citation.start_page}"
            if citation.start_page == citation.end_page
            else f"pp. {citation.start_page}-{citation.end_page}"
        )
        print(f"  [{citation.source_id}] {citation.title} ({page})")


def main() -> int:
    """確保索引存在後啟動持續問答迴圈。"""
    settings = Settings.from_env()
    pipeline = build_pipeline(settings)
    store = build_store(settings)

    # 新 clone 會包含 chunks，但不提交可重新產生的 Chroma 資料庫；若索引不存在，
    # 在此自動建立一次，省去額外的初始化指令。
    if not store.exists() or not store.count():
        print("首次啟動：正在建立本機檢索索引，完成後會直接進入問答。")
        try:
            print(pipeline.indexer.build(pipeline.corpus).describe())
        except EngineError as error:
            print(f"\n無法建立索引：\n{error}")
            return 1

    print(
        "\nDAMA-DMBOK 問答已就緒（Qwen3.6 35B-A3B）。"
        "輸入問題後按 Enter；輸入 exit 結束。\n"
    )
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            return 0

        try:
            answer = pipeline.answer(question)
        except EngineError as error:
            print(f"\n{error}\n")
            continue

        print(f"\n{answer.answer}")
        _print_sources(answer.citations)
        print(f"\n（{answer.latency_s:.1f} 秒）\n")


if __name__ == "__main__":
    raise SystemExit(main())
