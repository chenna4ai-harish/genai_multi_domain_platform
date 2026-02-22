"""
tests/unit/test_chunking.py

Unit tests for chunking strategies (RecursiveChunker, SemanticChunker).
"""

import sys
from pathlib import Path
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunking_config(strategy="recursive", chunk_size=200, overlap=20):
    cfg = MagicMock()
    cfg.strategy = strategy
    cfg.chunk_size = chunk_size
    cfg.overlap = overlap
    return cfg


SAMPLE_TEXT = (
    "Employees are entitled to 15 days of annual leave per calendar year. "
    "Leave must be approved by the line manager at least two weeks in advance. "
    "Unused leave can be carried forward to the next year up to a maximum of 5 days. "
    "Sick leave is separate and does not count against annual leave entitlement. "
    "Public holidays are in addition to annual leave and are governed by local law. "
    "Employees must submit a leave request via the HR portal before taking any leave. "
    "Leave during notice period requires special approval from HR. "
    "Part-time employees receive leave on a pro-rata basis proportional to their hours."
)


# ---------------------------------------------------------------------------
# RecursiveChunker tests
# ---------------------------------------------------------------------------

class TestRecursiveChunker:
    """Tests for the fixed-size recursive chunker."""

    def setup_method(self):
        from core.chunking.recursive_chunker import RecursiveChunker
        config = make_chunking_config(strategy="recursive", chunk_size=200, overlap=20)
        self.chunker = RecursiveChunker(config=config, embedding_model_name="test-model")

    def test_produces_chunks(self):
        chunks = self.chunker.chunk_text(
            text=SAMPLE_TEXT,
            doc_id="test_doc",
            domain="hr",
            source_file_path="test.txt",
            file_hash="abc123",
        )
        assert len(chunks) >= 1

    def test_chunk_text_not_empty(self):
        chunks = self.chunker.chunk_text(
            text=SAMPLE_TEXT,
            doc_id="test_doc",
            domain="hr",
            source_file_path="test.txt",
            file_hash="abc123",
        )
        for chunk in chunks:
            assert chunk.chunk_text.strip() != ""

    def test_chunk_has_required_metadata(self):
        chunks = self.chunker.chunk_text(
            text=SAMPLE_TEXT,
            doc_id="doc_001",
            domain="hr",
            source_file_path="policy.pdf",
            file_hash="deadbeef",
            uploader_id="alice",
        )
        for chunk in chunks:
            assert chunk.doc_id == "doc_001"
            assert chunk.domain == "hr"
            assert chunk.chunk_id is not None

    def test_empty_text_returns_no_chunks(self):
        chunks = self.chunker.chunk_text(
            text="",
            doc_id="empty_doc",
            domain="hr",
            source_file_path="empty.txt",
            file_hash="000",
        )
        assert chunks == [] or len(chunks) == 0

    def test_short_text_produces_one_chunk(self):
        short = "This is a very short document."
        chunks = self.chunker.chunk_text(
            text=short,
            doc_id="short_doc",
            domain="hr",
            source_file_path="short.txt",
            file_hash="111",
        )
        assert len(chunks) == 1
        assert short in chunks[0].chunk_text or chunks[0].chunk_text in short

    def test_chunk_size_respected(self):
        """No chunk should exceed chunk_size by a large margin."""
        chunks = self.chunker.chunk_text(
            text=SAMPLE_TEXT * 5,
            doc_id="long_doc",
            domain="hr",
            source_file_path="long.txt",
            file_hash="222",
        )
        for chunk in chunks:
            # Allow some slack for word-boundary splitting
            assert len(chunk.chunk_text) <= 200 + 50


# ---------------------------------------------------------------------------
# ChunkingFactory tests
# ---------------------------------------------------------------------------

class TestChunkingFactory:
    """Tests for ChunkingFactory routing."""

    def test_creates_recursive_chunker(self):
        from core.factories.chunking_factory import ChunkingFactory
        config = make_chunking_config("recursive")
        chunker = ChunkingFactory.create_chunker(config=config, embedding_model_name="model")
        assert hasattr(chunker, "chunk_text")

    def test_unknown_strategy_raises(self):
        from core.factories.chunking_factory import ChunkingFactory
        config = make_chunking_config("unknown_strategy_xyz")
        with pytest.raises(Exception):
            ChunkingFactory.create_chunker(config=config, embedding_model_name="model")
