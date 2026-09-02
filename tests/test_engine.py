"""Tests for stages 9-12.

The retrieval funnel is exercised with fake adapters, so the whole suite runs
in under a second and without downloading a model. The corpus tests do read
the real ``output/`` files -- those are committed, and a change to the chunk
pipeline that breaks the loader should fail here rather than at answer time.
"""

from __future__ import annotations

import unittest

from engine.config import Paths, PromptSettings, RetrievalSettings
from engine.context import ContextResolver
from engine.corpus import Corpus, Record, TableParent
from engine.errors import IndexNotBuiltError, IndexStaleError
from engine.indexing import METADATA_MODEL, Indexer
from engine.prompting import PromptBuilder
from engine.retrieval import Retriever

DOCUMENT = "dama-dmbok-2nd-edition"


def text_record(ordinal: int, text: str = "Data Governance body of work.") -> Record:
    return Record(
        record_id=f"{DOCUMENT}:text:{ordinal:06d}",
        parent_id=None,
        content_type="text",
        document_id=DOCUMENT,
        title="1.2 Goals and Principles",
        pages=(75,),
        text=text,
        token_count=42,
    )


def table_child(ordinal: int, parent_id: str, text: str = "Row 0: lawfulness") -> Record:
    return Record(
        record_id=f"{DOCUMENT}:table:013:child:{ordinal:03d}",
        parent_id=parent_id,
        content_type="table_child",
        document_id=DOCUMENT,
        title="Table 1 GDPR Principles",
        pages=(58,),
        text=text,
        token_count=20,
    )


def table_parent(record_id: str) -> TableParent:
    return TableParent(
        record_id=record_id,
        document_id=DOCUMENT,
        title="Table 1 GDPR Principles",
        pages=(58,),
        text="| Principle | Meaning |\n| --- | --- |\n| Lawfulness | ... |",
        num_rows=8,
        num_cols=2,
    )


class FakeEmbedder:
    """Embeds by counting characters -- deterministic and dependency-free."""

    name = "fake-embedder"
    dimension = 2

    def embed_documents(self, texts, *, show_progress=False):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text):
        return [float(len(text)), 1.0]


class FakeReranker:
    """Scores by term overlap, so the expected ordering is obvious."""

    name = "fake-reranker"

    def score(self, query, passages):
        terms = set(query.lower().split())
        return [
            len(terms & set(passage.lower().split())) / (len(terms) or 1)
            for passage in passages
        ]


class FakeStore:
    """An in-memory stand-in for Chroma with the same protocol."""

    collection_name = "fake"
    path = "(memory)"

    def __init__(self):
        self._records = {}
        self._metadata = {}
        self._created = False

    def exists(self):
        return self._created

    def count(self):
        return len(self._records)

    def metadata(self):
        return dict(self._metadata)

    def set_metadata(self, metadata):
        self._created = True
        self._metadata.update(metadata)

    def existing(self):
        return {key: value[1] for key, value in self._records.items()}

    def upsert(self, *, ids, documents, embeddings, metadatas):
        self._created = True
        for record_id, document, metadata in zip(ids, documents, metadatas):
            self._records[record_id] = (document, dict(metadata))

    def delete(self, ids):
        for record_id in ids:
            self._records.pop(record_id, None)

    def reset(self):
        self._records.clear()
        self._metadata.clear()
        self._created = False

    def query(self, embedding, top_k):
        # Distance is irrelevant to these tests; insertion order is stable.
        return [
            (record_id, document, metadata, 0.1 * position)
            for position, (record_id, (document, metadata)) in enumerate(
                self._records.items()
            )
        ][:top_k]


class RealCorpusTest(unittest.TestCase):
    """The committed chunk files must load and stay internally consistent."""

    @classmethod
    def setUpClass(cls):
        cls.corpus = Corpus.load(Paths())

    def test_loads_children_and_parents(self):
        counts = self.corpus.counts()
        self.assertEqual(counts["total"], counts["text"] + counts["table_child"])
        self.assertGreater(counts["text"], 0)
        self.assertGreater(counts["table_parent"], 0)

    def test_every_table_child_resolves_to_a_parent(self):
        # Corpus.load raises on orphans; assert it explicitly so the reason
        # this matters stays documented next to the check.
        orphans = [
            record.record_id
            for record in self.corpus.records
            if record.content_type == "table_child"
            and record.parent_id not in self.corpus.parents
        ]
        self.assertEqual(orphans, [])

    def test_every_record_has_text_and_pages(self):
        for record in self.corpus.records:
            self.assertTrue(record.text.strip(), record.record_id)
            self.assertTrue(record.pages, record.record_id)
            self.assertLessEqual(record.start_page, record.end_page)


class IndexingTest(unittest.TestCase):
    def setUp(self):
        self.store = FakeStore()
        self.indexer = Indexer(embedder=FakeEmbedder(), store=self.store)

    def test_second_run_embeds_nothing_when_nothing_changed(self):
        corpus = Corpus(records=(text_record(1), text_record(2)), parents={})
        first = self.indexer.build(corpus)
        second = self.indexer.build(corpus)
        self.assertEqual(first.embedded, 2)
        self.assertEqual(second.embedded, 0)
        self.assertEqual(second.unchanged, 2)

    def test_changed_text_is_re_embedded(self):
        self.indexer.build(Corpus(records=(text_record(1),), parents={}))
        report = self.indexer.build(
            Corpus(records=(text_record(1, "different text"),), parents={})
        )
        self.assertEqual(report.embedded, 1)

    def test_removed_record_is_deleted(self):
        self.indexer.build(
            Corpus(records=(text_record(1), text_record(2)), parents={})
        )
        report = self.indexer.build(Corpus(records=(text_record(1),), parents={}))
        self.assertEqual(report.deleted, 1)
        self.assertEqual(report.total, 1)

    def test_changing_the_embedding_model_forces_a_rebuild(self):
        corpus = Corpus(records=(text_record(1),), parents={})
        self.indexer.build(corpus)

        class OtherEmbedder(FakeEmbedder):
            name = "other-embedder"

        report = Indexer(embedder=OtherEmbedder(), store=self.store).build(corpus)
        self.assertTrue(report.rebuilt)
        self.assertEqual(report.embedded, 1)
        self.assertEqual(self.store.metadata()[METADATA_MODEL], "other-embedder")


class RetrievalGuardTest(unittest.TestCase):
    def _retriever(self, store):
        return Retriever(
            embedder=FakeEmbedder(),
            reranker=FakeReranker(),
            store=store,
            settings=RetrievalSettings(),
        )

    def test_empty_index_is_refused_with_a_fix(self):
        with self.assertRaises(IndexNotBuiltError):
            self._retriever(FakeStore()).retrieve("anything")

    def test_index_from_another_model_is_refused(self):
        store = FakeStore()
        Indexer(embedder=FakeEmbedder(), store=store).build(
            Corpus(records=(text_record(1),), parents={})
        )
        store.set_metadata({METADATA_MODEL: "some-other-model"})
        with self.assertRaises(IndexStaleError):
            self._retriever(store).retrieve("anything")


class ContextResolutionTest(unittest.TestCase):
    """Stage 11: text stays as-is, table children expand to their parent."""

    def setUp(self):
        parent_id = f"{DOCUMENT}:table:013:parent:000"
        self.parent = table_parent(parent_id)
        self.corpus = Corpus(
            records=(
                text_record(1, "Data Governance enables managing data as an asset"),
                table_child(1, parent_id, "Row 0: lawfulness fairness transparency"),
                table_child(2, parent_id, "Row 1: purpose limitation"),
            ),
            parents={parent_id: self.parent},
        )
        self.store = FakeStore()
        Indexer(embedder=FakeEmbedder(), store=self.store).build(self.corpus)
        self.retriever = Retriever(
            embedder=FakeEmbedder(),
            reranker=FakeReranker(),
            store=self.store,
            settings=RetrievalSettings(),
        )
        self.resolver = ContextResolver(self.corpus)

    def test_text_block_is_not_expanded(self):
        candidates = self.retriever.search("data governance asset")
        blocks = self.resolver.resolve(candidates, max_sources=4)
        text_blocks = [b for b in blocks if b.content_type == "text"]
        self.assertTrue(text_blocks)
        for block in text_blocks:
            self.assertEqual(block.section, block.passage)
            self.assertFalse(block.expanded)

    def test_table_child_expands_to_the_whole_table(self):
        candidates = self.retriever.search("lawfulness fairness")
        blocks = self.resolver.resolve(candidates, max_sources=4)
        table_blocks = [b for b in blocks if b.content_type == "table_child"]
        self.assertTrue(table_blocks)
        block = table_blocks[0]
        self.assertEqual(block.section, self.parent.text)
        self.assertIn("Row 0", block.passage)
        self.assertTrue(block.expanded)
        self.assertEqual(block.title, self.parent.title)

    def test_two_children_of_one_table_collapse_into_one_source(self):
        candidates = self.retriever.search("row purpose limitation lawfulness")
        blocks = self.resolver.resolve(candidates, max_sources=4)
        parent_ids = [b.parent_id for b in blocks if b.parent_id]
        self.assertEqual(len(parent_ids), len(set(parent_ids)))

    def test_max_sources_is_respected(self):
        candidates = self.retriever.search("data")
        self.assertLessEqual(len(self.resolver.resolve(candidates, max_sources=1)), 1)


class PromptingTest(unittest.TestCase):
    def setUp(self):
        parent_id = f"{DOCUMENT}:table:013:parent:000"
        parent = table_parent(parent_id)
        corpus = Corpus(
            records=(text_record(1), table_child(1, parent_id)),
            parents={parent_id: parent},
        )
        self.resolver = ContextResolver(corpus)
        self.builder = PromptBuilder(PromptSettings())

    def _blocks(self):
        from engine.retrieval import Candidate

        return self.resolver.resolve(
            [
                Candidate(
                    record_id=f"{DOCUMENT}:text:000001",
                    parent_id=None,
                    content_type="text",
                    title="1.2 Goals and Principles",
                    start_page=75,
                    end_page=75,
                    text="Data Governance body of work.",
                    vector_distance=0.1,
                    rerank_score=0.9,
                ),
                Candidate(
                    record_id=f"{DOCUMENT}:table:013:child:001",
                    parent_id=f"{DOCUMENT}:table:013:parent:000",
                    content_type="table_child",
                    title="Table 1 GDPR Principles",
                    start_page=58,
                    end_page=58,
                    text="Row 0: lawfulness",
                    vector_distance=0.2,
                    rerank_score=0.8,
                ),
            ],
            max_sources=4,
        )

    def test_sources_are_numbered_from_one_and_match_the_citations(self):
        bundle = self.builder.build("What is DG?", self._blocks())
        self.assertEqual(
            [citation.source_id for citation in bundle.citations],
            ["Source 1", "Source 2"],
        )
        for citation in bundle.citations:
            self.assertIn(f"[{citation.source_id}]", bundle.prompt)

    def test_table_source_carries_both_the_matched_rows_and_the_full_table(self):
        bundle = self.builder.build("GDPR principles", self._blocks())
        self.assertIn("Rows that matched the question:", bundle.prompt)
        self.assertIn("Full table:", bundle.prompt)
        self.assertIn("| Principle | Meaning |", bundle.prompt)

    def test_context_budget_truncates_visibly(self):
        from engine.context import ContextBlock

        long_block = ContextBlock(
            title="Long section",
            start_page=1,
            end_page=2,
            content_type="text",
            matched_record_id="x",
            parent_id=None,
            passage="word " * 500,
            section="word " * 500,
            rerank_score=1.0,
        )
        builder = PromptBuilder(PromptSettings(max_context_chars=600))
        bundle = builder.build("q", [long_block])
        self.assertIn("section truncated", bundle.prompt)
        self.assertEqual(bundle.truncated, ("x",))


class LanguageSelectionTest(unittest.TestCase):
    """The answer language is decided in Python, not asked of the model."""

    def test_detects_chinese_and_english(self):
        from engine.prompting import detect_language

        self.assertEqual(detect_language("資料治理的核心目標是什麼？"), "zh-hant")
        self.assertEqual(detect_language("What is Data Governance?"), "en")

    def test_english_question_with_dmbok_terms_stays_english(self):
        from engine.prompting import detect_language

        self.assertEqual(
            detect_language("Explain CDO, DAMA-DMBOK and metadata."), "en"
        )

    def test_chinese_question_quoting_english_terms_is_chinese(self):
        from engine.prompting import detect_language

        self.assertEqual(
            detect_language("Data Governance 的目標是什麼?"), "zh-hant"
        )

    def test_prompt_carries_the_detected_rule(self):
        from engine.context import ContextBlock

        block = ContextBlock(
            title="1.2 Goals",
            start_page=75,
            end_page=75,
            content_type="text",
            matched_record_id="x",
            parent_id=None,
            passage="Data Governance body of work.",
            section="Data Governance body of work.",
            rerank_score=1.0,
        )
        builder = PromptBuilder(PromptSettings())
        self.assertIn(
            "entire answer in English",
            builder.build("What is Data Governance?", [block]).prompt,
        )
        self.assertIn(
            "Traditional Chinese",
            builder.build("資料治理是什麼？", [block]).prompt,
        )

    def test_explicit_setting_overrides_detection(self):
        from engine.context import ContextBlock

        block = ContextBlock(
            title="1.2 Goals",
            start_page=75,
            end_page=75,
            content_type="text",
            matched_record_id="x",
            parent_id=None,
            passage="text",
            section="text",
            rerank_score=1.0,
        )
        builder = PromptBuilder(PromptSettings(answer_language="zh-hant"))
        self.assertIn(
            "Traditional Chinese",
            builder.build("What is Data Governance?", [block]).prompt,
        )


if __name__ == "__main__":
    unittest.main()
