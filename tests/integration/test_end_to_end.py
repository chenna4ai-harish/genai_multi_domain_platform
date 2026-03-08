"""
tests/integration/test_end_to_end.py

End-to-end integration tests for the full RAG pipeline.

These tests require:
  - A configured domain (hr) with a valid YAML config
  - ChromaDB available at the configured persist_directory
  - SentenceTransformers installed (or Gemini API key for Gemini embeddings)

Run with:
    pytest tests/integration/ -v

Skip these in CI if dependencies are not available by using:
    pytest tests/unit/ -v   (unit tests only)
"""

import sys
import os
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Mark entire module as integration so it can be skipped with: pytest -m "not integration"
pytestmark = pytest.mark.integration

DOMAIN = "hr"
TEST_TEXT = (
    "Employees are entitled to 15 days of annual leave per calendar year. "
    "Leave must be approved by the line manager at least two weeks in advance. "
    "Unused leave can be carried forward up to 5 days. "
    "Sick leave is tracked separately and does not affect annual leave balance."
)
TEST_DOC_ID = "integration_test_doc_001"


@pytest.fixture(scope="module")
def service():
    """
    Create a DocumentService for integration tests.
    Skips if domain config is missing or dependencies are not installed.
    """
    try:
        from core.services.document_service import DocumentService
        svc = DocumentService(DOMAIN)
        return svc
    except Exception as e:
        pytest.skip(f"Cannot initialize DocumentService for domain '{DOMAIN}': {e}")


@pytest.fixture(scope="module", autouse=True)
def cleanup(service):
    """Remove test document after the test module runs."""
    yield
    try:
        service.delete_document(TEST_DOC_ID)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIngestionAndQuery:
    def test_ingest_text_document(self, service):
        """Ingest a plain-text document via the service layer."""
        import io
        content = TEST_TEXT.encode("utf-8")
        fake_file = io.BytesIO(content)
        fake_file.name = f"{TEST_DOC_ID}.txt"

        result = service.upload_document(
            file_obj=fake_file,
            metadata={
                "doc_id": TEST_DOC_ID,
                "title": "Integration Test Document",
                "doc_type": "policy",
                "uploader_id": "test_runner",
            },
            replace_existing=True,
        )

        assert result["status"] == "success"
        assert result["chunks_ingested"] >= 1
        assert result["doc_id"] == TEST_DOC_ID

    def test_list_documents_contains_ingested(self, service):
        docs = service.list_documents()
        doc_ids = [d["doc_id"] for d in docs]
        assert TEST_DOC_ID in doc_ids

    def test_query_returns_results(self, service):
        results = service.query(
            query_text="annual leave days",
            strategy="vector_similarity",
            top_k=3,
        )
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_query_result_contains_ingested_doc(self, service):
        results = service.query(
            query_text="annual leave days",
            strategy="vector_similarity",
            top_k=5,
        )
        doc_ids_in_results = {
            r.get("metadata", {}).get("doc_id") for r in results if r.get("metadata")
        }
        assert TEST_DOC_ID in doc_ids_in_results

    def test_get_document_info(self, service):
        info = service.get_document_info(TEST_DOC_ID)
        assert info["doc_id"] == TEST_DOC_ID
        assert info["chunk_count"] >= 1
        assert info["deprecated"] is False

    def test_list_chunks(self, service):
        chunks = service.list_chunks(TEST_DOC_ID)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.get("doc_id") == TEST_DOC_ID or chunk.get("metadata", {}).get("doc_id") == TEST_DOC_ID

    def test_deprecate_document(self, service):
        result = service.deprecate_document(
            doc_id=TEST_DOC_ID,
            reason="Integration test deprecation",
        )
        assert result["chunks_deprecated"] >= 1
        assert result["doc_id"] == TEST_DOC_ID

    def test_deprecated_doc_excluded_from_query(self, service):
        """After deprecation, the doc should NOT appear in default queries."""
        results = service.query(
            query_text="annual leave days",
            top_k=10,
            include_deprecated=False,
        )
        doc_ids = {r.get("metadata", {}).get("doc_id") for r in results if r.get("metadata")}
        assert TEST_DOC_ID not in doc_ids

    def test_deprecated_doc_included_when_requested(self, service):
        """With include_deprecated=True, the doc should appear."""
        results = service.query(
            query_text="annual leave days",
            top_k=10,
            include_deprecated=True,
        )
        doc_ids = {r.get("metadata", {}).get("doc_id") for r in results if r.get("metadata")}
        assert TEST_DOC_ID in doc_ids
