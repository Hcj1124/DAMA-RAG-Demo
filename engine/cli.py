"""DAMA RAG 命令列介面。

    dama-rag doctor            # 檢查執行條件是否完整
    dama-rag index             # 第 9 階段：將 chunks 寫入 Chroma
    dama-rag ask "question"    # 第 10 至 12 階段：完整問答管線
    dama-rag search "query"    # 只做檢索，不生成回答
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Mapping, Sequence

from engine.config import Settings
from engine.corpus import Corpus
from engine.errors import EngineError
from engine.indexing import (
    METADATA_DIMENSION,
    METADATA_FINGERPRINT,
    METADATA_MODEL,
)
from engine.pipeline import build_embedder, build_pipeline, build_store

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """依 verbose 選項設定命令列日誌層級。"""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _print_citations(citations: Sequence) -> None:
    """以人類可讀格式列出回答所引用的來源。"""
    if not citations:
        print("  (none)")
        return
    for citation in citations:
        kind = "table" if citation.content_type != "text" else "text"
        print(
            f"  [{citation.source_id}] {citation.title}  "
            f"({citation.pages}, {kind})"
        )


def _cmd_info(args: argparse.Namespace) -> int:
    """顯示目前實際生效的設定。"""
    settings = Settings.from_env()
    for key, value in settings.describe().items():
        print(f"{key:>18}: {value}")
    return 0


def _index_problems(
    *,
    vector_count: int,
    corpus_count: int | None,
    metadata: Mapping[str, object],
    expected_model: str,
    expected_fingerprint: str,
) -> list[str]:
    """比對索引數量與 embedding metadata，回傳需要修復的項目。"""

    if vector_count <= 0:
        return ["no vectors yet -- run: dama-rag index"]

    issues: list[str] = []
    if corpus_count is not None and vector_count != corpus_count:
        issues.append(
            f"vector count {vector_count} does not match corpus count "
            f"{corpus_count}"
        )

    indexed_model = metadata.get(METADATA_MODEL)
    if not indexed_model:
        issues.append(f"missing index metadata: {METADATA_MODEL}")
    elif indexed_model != expected_model:
        issues.append(
            f"index model {indexed_model!r} does not match configured model "
            f"{expected_model!r}"
        )

    indexed_fingerprint = metadata.get(METADATA_FINGERPRINT)
    if not indexed_fingerprint:
        issues.append(f"missing index metadata: {METADATA_FINGERPRINT}")
    elif indexed_fingerprint != expected_fingerprint:
        issues.append(
            "index fingerprint does not match the current embedding settings"
        )

    dimension = metadata.get(METADATA_DIMENSION)
    if not isinstance(dimension, int) or dimension <= 0:
        issues.append(f"missing or invalid index metadata: {METADATA_DIMENSION}")

    return issues


def _cmd_doctor(args: argparse.Namespace) -> int:
    """逐項檢查問答所需條件，並指出失敗項目。"""

    settings = Settings.from_env()
    problems = 0
    corpus_count: int | None = None

    print("chunk files")
    try:
        corpus = Corpus.load(settings.paths)
        counts = corpus.counts()
        corpus_count = counts["total"]
        print(
            f"  ok    {counts['total']} children "
            f"({counts.get('text', 0)} text, "
            f"{counts.get('table_child', 0)} table children), "
            f"{counts['table_parent']} table parents"
        )
    except EngineError as error:
        problems += 1
        print(f"  FAIL  {error}")

    print("torch device")
    try:
        from engine.device import resolve_device

        print(f"  ok    {resolve_device(settings.device)}")
    except Exception as error:  # pragma: no cover - 僅在 torch 匯入失敗時發生
        problems += 1
        print(f"  FAIL  {error}")

    print("vector index")
    try:
        store = build_store(settings)
        vector_count = store.count()
        meta = store.metadata()
        expected_fingerprint = build_embedder(settings).index_fingerprint
        index_problems = _index_problems(
            vector_count=vector_count,
            corpus_count=corpus_count,
            metadata=meta,
            expected_model=settings.embedding.model,
            expected_fingerprint=expected_fingerprint,
        )

        if index_problems:
            problems += 1
            for issue in index_problems:
                print(f"  FAIL  {issue}")
            if vector_count:
                print("        run: dama-rag index --rebuild")
        else:
            print(
                f"  ok    {vector_count} vectors in "
                f"'{settings.retrieval.collection_name}' "
                f"(model: {meta[METADATA_MODEL]}, "
                f"dim: {meta[METADATA_DIMENSION]}, "
                f"fingerprint: {meta[METADATA_FINGERPRINT]})"
            )
    except Exception as error:
        problems += 1
        print(f"  FAIL  cannot inspect vector index ({error})")

    print("ollama")
    try:
        import ollama

        client = (
            ollama.Client(host=settings.generation.host)
            if settings.generation.host
            else ollama.Client()
        )
        available = {
            model.get("model") or model.get("name")
            for model in client.list().get("models", [])
        }
        wanted = settings.generation.model
        if wanted in available:
            print(f"  ok    {wanted} is available")
        else:
            problems += 1
            print(
                f"  FAIL  {wanted} is not pulled. Run: ollama pull {wanted}\n"
                f"        available: {', '.join(sorted(m for m in available if m))}"
            )
    except Exception as error:
        problems += 1
        print(f"  FAIL  cannot reach Ollama ({error}). Is `ollama serve` up?")

    print()
    if problems:
        print(f"{problems} problem(s) found.")
        return 1
    print("Ready. Ask a question:  dama-rag ask \"...\"")
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    """建立或重建本機向量索引。"""
    settings = Settings.from_env()
    pipeline = build_pipeline(settings, with_llm=False)
    report = pipeline.indexer.build(pipeline.corpus, rebuild=args.rebuild)
    print(report.describe())
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    """執行檢索與重排序，並以文字或 JSON 顯示結果。"""
    pipeline = build_pipeline(Settings.from_env(), with_llm=False)
    candidates = pipeline.retriever.rerank(
        args.query, pipeline.retriever.retrieve(args.query), top_k=args.k
    )
    if args.json:
        print(
            json.dumps(
                [candidate.to_dict() for candidate in candidates],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    for position, candidate in enumerate(candidates, start=1):
        preview = " ".join(candidate.text.split())[:160]
        print(
            f"{position:>2}. [{candidate.rerank_score:+.3f}] "
            f"{candidate.title}  (p. {candidate.start_page}, "
            f"{candidate.content_type})"
        )
        print(f"    {preview}...")
    return 0


def _cmd_context(args: argparse.Namespace) -> int:
    """顯示模型將讀取的來源或完整 Prompt，但不執行生成。"""

    pipeline = build_pipeline(Settings.from_env(), with_llm=False)
    bundle = pipeline.build_prompt(args.query)
    if args.prompt:
        print(bundle.prompt)
    else:
        print(f"{bundle.prompt_chars} prompt characters")
        _print_citations(bundle.citations)
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    """回答單次或多次問題；未提供問題時進入互動模式。"""
    settings = Settings.from_env()
    if args.language:
        settings = settings.with_overrides(
            prompt=settings.prompt.__class__(
                max_context_chars=settings.prompt.max_context_chars,
                answer_language=args.language,
            )
        )
    pipeline = build_pipeline(settings)

    questions = args.question or []
    if not questions:
        return _interactive(pipeline)

    for question in questions:
        answer = pipeline.answer(question)
        if args.json:
            print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
            continue
        print(f"\nQUESTION:\n{answer.question}")
        print(f"\nANSWER:\n{answer.answer}")
        print("\nSOURCES:")
        _print_citations(answer.citations)
        print(
            f"\n({answer.model}, {answer.prompt_chars} prompt chars, "
            f"{answer.latency_s:.1f}s)"
        )
    return 0


def _interactive(pipeline) -> int:
    """持續接受彼此獨立的問題，直到使用者退出。"""
    print("DAMA-DMBOK RAG. Ask a question, or press Ctrl-D to quit.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question in {"exit", "quit", ":q"}:
            return 0
        try:
            answer = pipeline.answer(question)
        except EngineError as error:
            print(f"\n{error}\n")
            continue
        print(f"\n{answer.answer}\n")
        print("SOURCES:")
        _print_citations(answer.citations)
        print(f"\n({answer.latency_s:.1f}s)\n")


def build_parser() -> argparse.ArgumentParser:
    """建立所有子命令與選項的 argparse parser。"""
    parser = argparse.ArgumentParser(
        prog="dama-rag",
        description="Local bilingual RAG over the DAMA-DMBOK.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="log progress to stderr"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="print the effective settings")
    info.set_defaults(func=_cmd_info)

    doctor = subparsers.add_parser(
        "doctor", help="check chunks, index, device and Ollama"
    )
    doctor.set_defaults(func=_cmd_doctor)

    index = subparsers.add_parser(
        "index", help="embed the combined chunks into Chroma (stage 9)"
    )
    index.add_argument(
        "--rebuild",
        action="store_true",
        help="drop the collection and re-embed everything",
    )
    index.set_defaults(func=_cmd_index)

    search = subparsers.add_parser(
        "search", help="retrieve and rerank only, no generation"
    )
    search.add_argument("query")
    search.add_argument("-k", type=int, default=None, help="results to show")
    search.add_argument("--json", action="store_true")
    search.set_defaults(func=_cmd_search)

    context = subparsers.add_parser(
        "context", help="show the assembled context or the whole prompt"
    )
    context.add_argument("query")
    context.add_argument(
        "--prompt", action="store_true", help="print the full prompt text"
    )
    context.set_defaults(func=_cmd_context)

    ask = subparsers.add_parser("ask", help="the full pipeline")
    ask.add_argument("question", nargs="*", help="omit for an interactive loop")
    ask.add_argument("--json", action="store_true")
    ask.add_argument(
        "--language",
        choices=["auto", "en", "zh-hant"],
        default=None,
        help="override DAMA_ANSWER_LANGUAGE for this run",
    )
    ask.set_defaults(func=_cmd_ask)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """解析參數、執行子命令，並將已知錯誤轉成結束碼。"""
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return args.func(args)
    except EngineError as error:
        print(f"\n{error}\n", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - 使用者主動中斷
        print(file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover - 模組直接執行入口
    raise SystemExit(main())
