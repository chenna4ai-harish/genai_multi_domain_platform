"""
tests/unit/test_document_service.py

Unit tests for DocumentService — all external dependencies are mocked
so these tests run without ChromaDB, real embedding models, or file system setup.
"""

import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch, mock_open
from io import BytesIO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.services.document_service import (
    DocumentService,
    ValidationError,
    ProcessingError,
    DocumentNotFoundError,
)


# ---------------------------------------------------------------------------
# Fixtures — patch everything that touches disk / network
# ---------------------------------------------------------------------------

def make_mock_pipeline():
    pipeline = MagicMock()
    pipeline.query.return_value = [
        {
            "id": "chunk_001",
            "score": 0.92,
            "document": "Employees receive 15 days annual leave.",
            "metadata": {"doc_id": "handbook_2025", "title": "HR Handbook", "deprecated_flag": False},
            "strategy": "hybrid",
        }
    ]
    pipeline.process_document.return_value = {
        "doc_id": "handbook_2025",
        "chunks_ingested": 10,
        "status": "success",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunking_strategy": "recursive",
    }
    pipeline.list_documents.return_value = [
        {"doc_id": "handbook_2025", "title": "HR Handbook", "chunk_count": 10, "deprecated_flag": False}
    ]
    pipeline.list_chunks.return_value = [
        {"id": "chunk_001", "doc_id": "handbook_2025", "text": "Annual leave policy..."}
    ]
    pipeline.get_document_info.return_value = {
        "doc_id": "handbook_2025",
        "title": "HR Handbook",
        "chunk_count": 10,
        "deprecated": False,
    }
    pipeline.deprecate_document.return_value = {
        "doc_id": "handbook_2025",
        "chunks_deprecated": 10,
        "deprecated_date": "2026-02-22T00:00:00",
        "reason": "Superseded",
        "superseded_by": None,
    }
    return pipeline


def make_mock_domain_config():
    config = MagicMock()
    config.domain_id = "hr"
    security = MagicMock()
    security.allowed_file_types = ["pdf", "docx", "txt"]
    security.max_file_size_mb = 20
    config.security = security
    return config


@pytest.fixture
def service():
    """Return a DocumentService with all heavy dependencies mocked."""
    mock_pipeline = make_mock_pipeline()
    mock_config = make_mock_domain_config()

    with patch("core.services.document_service.ConfigManager") as MockCM, \
         patch("core.services.document_service.DocumentPipeline") as MockPipeline:

        MockCM.return_value.load_domain_config.return_value = mock_config
        MockPipeline.return_value = mock_pipeline

        svc = DocumentService("hr")
        svc._mock_pipeline = mock_pipeline  # expose for assertions
        yield svc


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_file_type_validation_rejects_exe(self, service):
        fake_file = BytesIO(b"fake content")
        fake_file.name = "malware.exe"
        with pytest.raises((ValidationError, Exception)):
            service.upload_document(fake_file, {"doc_id": "x", "title": "x", "doc_type": "policy", "uploader_id": "u"})

    def test_metadata_missing_doc_id_raises(self, service):
        with pytest.raises((ValidationError, Exception)):
            service._validate_metadata({"title": "HR Doc", "doc_type": "policy", "uploader_id": "u"})

    def test_metadata_missing_title_raises(self, service):
        with pytest.raises((ValidationError, Exception)):
            service._validate_metadata({"doc_id": "x", "doc_type": "policy", "uploader_id": "u"})

    def test_metadata_all_required_passes(self, service):
        # Should not raise
        service._validate_metadata({
            "doc_id": "handbook_2025",
            "title": "HR Handbook",
            "doc_type": "policy",
            "uploader_id": "admin",
        })

    def test_file_type_allowed_extensions(self, service):
        # These must not raise
        for ext in ["policy.pdf", "manual.docx", "notes.txt"]:
            service._validate_file_type(ext)

    def test_file_type_disallowed_raises(self, service):
        with pytest.raises(ValidationError):
            service._validate_file_type("script.sh")


# ---------------------------------------------------------------------------
# Query tests
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_returns_results(self, service):
        results = service.query("vacation policy", strategy="hybrid", top_k=5)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_query_adds_deprecated_filter(self, service):
        service.query("leave policy", include_deprecated=False)
        call_kwargs = service._mock_pipeline.query.call_args
        filters = call_kwargs[1].get("metadata_filters") or call_kwargs[0][2]
        assert filters.get("deprecated_flag") is False

    def test_query_empty_text_still_delegates(self, service):
        # Service does not block empty queries — pipeline handles it
        results = service.query("")
        assert results is not None

    def test_query_pipeline_exception_raises_processing_error(self, service):
        service._mock_pipeline.query.side_effect = RuntimeError("vectorstore down")
        with pytest.raises(ProcessingError):
            service.query("test query")


# ---------------------------------------------------------------------------
# Deprecation tests
# ---------------------------------------------------------------------------

class TestDeprecateDocument:
    def test_deprecate_calls_pipeline(self, service):
        result = service.deprecate_document("handbook_2025", reason="Superseded")
        service._mock_pipeline.deprecate_document.assert_called_once_with(
            doc_id="handbook_2025",
            reason="Superseded",
            superseded_by=None,
        )
        assert result["chunks_deprecated"] == 10

    def test_deprecate_missing_doc_id_raises_validation_error(self, service):
        with pytest.raises(ValidationError):
            service.deprecate_document("", reason="test")

    def test_deprecate_missing_reason_raises_validation_error(self, service):
        with pytest.raises(ValidationError):
            service.deprecate_document("handbook_2025", reason="")

    def test_deprecate_not_found_raises_document_not_found(self, service):
        service._mock_pipeline.deprecate_document.side_effect = ValueError("no document found")
        with pytest.raises(DocumentNotFoundError):
            service.deprecate_document("nonexistent", reason="test")


# ---------------------------------------------------------------------------
# List documents / chunks
# ---------------------------------------------------------------------------

class TestListMethods:
    def test_list_documents_returns_list(self, service):
        docs = service.list_documents()
        assert isinstance(docs, list)

    def test_list_documents_injects_domain_filter(self, service):
        service.list_documents()
        call_kwargs = service._mock_pipeline.list_documents.call_args
        filters = call_kwargs[1].get("filters") or call_kwargs[0][0]
        assert filters.get("domain") == "hr"

    def test_list_chunks_returns_list(self, service):
        chunks = service.list_chunks("handbook_2025")
        assert isinstance(chunks, list)

    def test_list_chunks_empty_doc_id_raises(self, service):
        with pytest.raises(ValidationError):
            service.list_chunks("")

    def test_list_chunks_not_found_raises(self, service):
        service._mock_pipeline.list_chunks.return_value = []
        with pytest.raises(DocumentNotFoundError):
            service.list_chunks("nonexistent")


# ---------------------------------------------------------------------------
# delete_document
# ---------------------------------------------------------------------------

class TestDeleteDocument:
    def test_delete_calls_pipeline(self, service):
        service.delete_document("handbook_2025")
        service._mock_pipeline.delete_document.assert_called_once_with("handbook_2025")

    def test_delete_pipeline_failure_raises_processing_error(self, service):
        service._mock_pipeline.delete_document.side_effect = RuntimeError("db error")
        with pytest.raises(ProcessingError):
            service.delete_document("handbook_2025")
