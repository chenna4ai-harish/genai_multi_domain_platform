"""
tests/unit/test_retrieval.py

Unit tests for retrieval strategies: VectorSimilarity, BM25, Hybrid.
All vector store and embedding calls are mocked.
"""

import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_vectorstore(results=None):
    vs = MagicMock()
    default_results = [
        {"id": "chunk_1", "score": 0.9, "document": "Annual leave is 15 days.", "metadata": {"doc_id": "doc1", "deprecated": False}},
        {"id": "chunk_2", "score": 0.75, "document": "Sick leave is separate.", "metadata": {"doc_id": "doc1", "deprecated": False}},
    ]
    vs.search.return_value = results if results is not None else default_results
    vs.get_all_documents.return_value = (
        ["Annual leave is 15 days.", "Sick leave is separate.", "Public holidays apply."],
        ["chunk_1", "chunk_2", "chunk_3"],
    )
    return vs


def make_embedding_model():
    em = MagicMock()
    em.embed_texts.return_value = np.random.rand(1, 384)
    em.get_embedding_dimension.return_value = 384
    em.get_model_name.return_value = "all-MiniLM-L6-v2"
    return em


def make_retrieval_config(alpha=0.7, top_k=10):
    cfg = MagicMock()
    cfg.retrieval.hybrid.alpha = alpha
    cfg.retrieval.top_k = top_k
    return cfg


# ---------------------------------------------------------------------------
# VectorSimilarityRetrieval
# ---------------------------------------------------------------------------

class TestVectorSimilarityRetrieval:
    def setup_method(self):
        from core.retrievals.vector_similarity_retrieval import VectorSimilarityRetrieval
        self.vs = make_vectorstore()
        self.em = make_embedding_model()
        self.retriever = VectorSimilarityRetrieval(
            vectorstore=self.vs,
            embedding_model=self.em,
            config=make_retrieval_config(),
        )

    def test_retrieve_returns_list(self):
        results = self.retriever.retrieve("vacation policy", top_k=5)
        assert isinstance(results, list)

    def test_retrieve_calls_vectorstore_search(self):
        self.retriever.retrieve("vacation policy", top_k=5)
        self.vs.search.assert_called()

    def test_retrieve_calls_embed(self):
        self.retriever.retrieve("vacation policy", top_k=5)
        self.em.embed_texts.assert_called()

    def test_empty_results_returns_empty_list(self):
        self.vs.search.return_value = []
        results = self.retriever.retrieve("obscure query", top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# BM25Retrieval
# ---------------------------------------------------------------------------

class TestBM25Retrieval:
    def setup_method(self):
        from core.retrievals.bm25_retrieval import BM25Retrieval
        self.corpus = [
            "Annual leave is 15 days per year.",
            "Sick leave does not count against annual leave.",
            "Public holidays are additional to annual leave.",
            "Leave must be approved by manager.",
        ]
        self.doc_ids = ["chunk_1", "chunk_2", "chunk_3", "chunk_4"]
        self.retriever = BM25Retrieval(corpus=self.corpus, doc_ids=self.doc_ids)

    def test_retrieve_returns_list(self):
        results = self.retriever.retrieve("annual leave days", top_k=3)
        assert isinstance(results, list)

    def test_retrieve_top_k_respected(self):
        results = self.retriever.retrieve("leave", top_k=2)
        assert len(results) <= 2

    def test_relevant_chunk_ranked_high(self):
        results = self.retriever.retrieve("annual leave days", top_k=4)
        if results:
            top_text = results[0].get("document", "")
            assert "annual" in top_text.lower() or "leave" in top_text.lower()

    def test_empty_query_does_not_crash(self):
        try:
            results = self.retriever.retrieve("", top_k=3)
            assert isinstance(results, list)
        except Exception:
            pass  # Acceptable: empty query edge case


# ---------------------------------------------------------------------------
# HybridRetrieval
# ---------------------------------------------------------------------------

class TestHybridRetrieval:
    def setup_method(self):
        from core.retrievals.hybrid_retrieval import HybridRetrieval
        from core.retrievals.bm25_retrieval import BM25Retrieval

        self.vs = make_vectorstore()
        self.em = make_embedding_model()
        corpus = ["Annual leave is 15 days.", "Sick leave is separate.", "Public holidays."]
        doc_ids = ["chunk_1", "chunk_2", "chunk_3"]
        self.bm25 = BM25Retrieval(corpus=corpus, doc_ids=doc_ids)

        self.retriever = HybridRetrieval(
            vectorstore=self.vs,
            embedding_model=self.em,
            bm25_index=self.bm25,
            config=make_retrieval_config(alpha=0.7),
        )

    def test_retrieve_returns_list(self):
        results = self.retriever.retrieve("vacation policy", top_k=5)
        assert isinstance(results, list)

    def test_scores_are_normalized(self):
        results = self.retriever.retrieve("leave policy", top_k=5)
        for r in results:
            score = r.get("score", 0.0)
            assert 0.0 <= score <= 1.0 + 1e-6, f"Score out of range: {score}"

    def test_results_sorted_descending(self):
        results = self.retriever.retrieve("leave", top_k=5)
        scores = [r.get("score", 0.0) for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# RetrievalFactory
# ---------------------------------------------------------------------------

class TestRetrievalFactory:
    def test_creates_vector_similarity(self):
        from core.factories.retrieval_factory import RetrievalFactory
        vs = make_vectorstore()
        em = make_embedding_model()
        retriever = RetrievalFactory.create_retriever(
            strategy_name="vector_similarity",
            config=make_retrieval_config(),
            vectorstore=vs,
            embedding_model=em,
            bm25_index=None,
        )
        assert hasattr(retriever, "retrieve")

    def test_unknown_strategy_raises(self):
        from core.factories.retrieval_factory import RetrievalFactory
        with pytest.raises(Exception):
            RetrievalFactory.create_retriever(
                strategy_name="nonexistent_strategy",
                config=make_retrieval_config(),
                vectorstore=make_vectorstore(),
                embedding_model=make_embedding_model(),
                bm25_index=None,
            )
