"""

core/services/document_service.py

This module implements the Service Layer for Phase 2 architecture.

What is the Service Layer?
---------------------------
The Service Layer is the MANDATORY interface between the UI and core business logic.
ALL business logic, validation, orchestration, and error handling resides here.

Phase 2 Critical Architectural Constraint:
-------------------------------------------
**ZERO BUSINESS LOGIC IN UI LAYER**

The UI (app.py, Gradio, any web framework) SHALL:
- ✅ Accept user input (file uploads, query text, dropdown selections)
- ✅ Call service layer methods
- ✅ Display results returned by service
- ✅ Format error messages for presentation

The UI SHALL NOT:
- ❌ Validate file types or metadata
- ❌ Directly instantiate factories
- ❌ Call pipeline methods directly
- ❌ Process or parse documents
- ❌ Execute chunking or embedding
- ❌ Compute file hashes
- ❌ Manage metadata
- ❌ Import core.pipeline, core.factories, or core.vectorstores

Why Service Layer?
------------------
1. **Separation of Concerns**: Business logic separate from UI presentation
2. **Testability**: Core logic fully testable independent of UI framework
3. **Reusability**: Same service used by web UI, CLI, API, batch jobs
4. **Flexibility**: Easy to swap UI frameworks (Gradio → FastAPI → React)
5. **Maintainability**: Business rules in one place
6. **Security**: Centralized validation and authorization

Architecture Position:
----------------------
UI Layer (app.py, Gradio)
    ↓ calls ONLY
**Service Layer (DocumentService)** ← YOU ARE HERE
    ↓ delegates to
Pipeline Layer (DocumentPipeline)
    ↓ uses
Factory Layer (ChunkingFactory, EmbeddingFactory, etc.)

Key Responsibilities:
---------------------
- Input validation (file types, metadata, file size)
- File text extraction (PDF, DOCX, TXT)
- File hash computation (provenance)
- Metadata enrichment (timestamps, user IDs)
- Business rule enforcement (security policies)
- Pipeline orchestration (call pipeline methods)
- Error translation (technical → user-friendly)
- Structured logging (audit trails)

Example Usage:
--------------
# Initialize service for domain
service = DocumentService("hr")

# Upload document (UI calls this)
with open("handbook.pdf", "rb") as f:
    result = service.upload_document(
        file_obj=f,
        metadata={
            "doc_id": "handbook_2025",
            "title": "Employee Handbook 2025",
            "author": "HR Department",
            "doc_type": "policy",
            "uploader_id": "admin@company.com"
        },
        replace_existing=True
    )

# Query (UI calls this)
results = service.query(
    query_text="vacation policy",
    strategy="hybrid",
    metadata_filters={"doc_type": "policy", "deprecated": False},
    top_k=5
)

References:
-----------
- Phase 2 Architecture: Section 2.1 (Zero Business Logic in UI)
- Service Layer: Section 6 (Service Layer Specification)
- Validation: Section 10 (File Processing Validation)

"""

import logging
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

# Service layer imports (interfaces to core)
from core.pipeline.document_pipeline import DocumentPipeline
from core.config_manager import ConfigManager

# File processing utilities
from core.utils.file_parsers import extract_text_from_file

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class ProcessingError(Exception):
    """Raised when document processing fails."""
    pass


class DocumentNotFoundError(Exception):
    """Raised when document not found."""
    pass


# =============================================================================
# SERVICE LAYER CLASS
# =============================================================================

class DocumentService:
    """
    Service layer providing high-level document management APIs.

    This is the SOLE interface between UI and core business logic.
    UI MUST ONLY call methods from this class.

    Responsibilities:
    - Input validation (files, metadata, size)
    - File text extraction
    - File hash computation
    - Metadata enrichment
    - Business logic enforcement
    - Pipeline orchestration
    - Error handling and logging

    All methods are designed to be called directly from UI handlers.
    """

    def __init__(self, domain_id: str):
        """
        Initialize service for a specific domain.

        This method:
        1. Loads domain configuration
        2. Initializes pipeline with domain config
        3. Extracts security settings (allowed file types, max file size)

        Parameters:
        -----------
        domain_id : str
            Domain identifier (hr, finance, legal, engineering, etc.)

        Raises:
        -------
        ValueError:
            If domain_id not found in config

        Example:
        --------
        hr_service = DocumentService("hr")
        finance_service = DocumentService("finance")
        """
        self.domain_id = domain_id
        self.config_manager = ConfigManager()

        logger.info(f"Initializing DocumentService for domain: {domain_id}")

        # Load domain-specific configuration
        try:
            self.domain_config = self.config_manager.load_domain_config(domain_id)
        except Exception as e:
            logger.error(f"Failed to load config for domain '{domain_id}': {e}")
            raise ValueError(f"Domain '{domain_id}' not found or invalid config")

        # Initialize pipeline (handles factories, chunking, embedding, storage)
        self.pipeline = DocumentPipeline(self.domain_config)

        # Extract security settings from config
        security_config = getattr(self.domain_config, 'security', None)
        if security_config:
            self.allowed_file_types = set(
                ext.lower() for ext in getattr(security_config, 'allowed_file_types', ['pdf', 'docx', 'txt'])
            )
            self.max_file_size_mb = getattr(security_config, 'max_file_size_mb', 20)
        else:
            # Fallback defaults
            self.allowed_file_types = {'pdf', 'docx', 'txt'}
            self.max_file_size_mb = 20

        logger.info(
            f"✅ DocumentService initialized: domain={domain_id}, "
            f"allowed_types={self.allowed_file_types}, "
            f"max_size={self.max_file_size_mb}MB"
        )

    # =========================================================================
    # VALIDATION METHODS (Private - used internally)
    # =========================================================================

    def _validate_file_type(self, filename: str) -> None:
        """
        Validate file extension against allowed types.

        Parameters:
        -----------
        filename : str
            Filename with extension

        Raises:
        -------
        ValidationError:
            If file type not allowed
        """
        if '.' not in filename:
            raise ValidationError(f"File '{filename}' has no extension")

        ext = filename.rsplit('.', 1)[-1].lower()

        if ext not in self.allowed_file_types:
            logger.error(
                f"File extension '.{ext}' not allowed for domain '{self.domain_id}'"
            )
            raise ValidationError(
                f"File type '.{ext}' not allowed. "
                f"Allowed types: {', '.join(sorted(self.allowed_file_types))}"
            )

        logger.debug(f"✅ File type validation passed: .{ext}")

    def _validate_file_size(self, file_obj: Any) -> None:
        """
        Validate file size against maximum allowed.

        Parameters:
        -----------
        file_obj : Any
            File object with .seek() and .tell() methods

        Raises:
        -------
        ValidationError:
            If file size exceeds limit
        """
        # Get file size
        file_obj.seek(0, 2)  # Seek to end
        file_size_bytes = file_obj.tell()
        file_obj.seek(0)  # Reset to beginning

        file_size_mb = file_size_bytes / (1024 * 1024)

        if file_size_mb > self.max_file_size_mb:
            logger.error(
                f"File size {file_size_mb:.2f}MB exceeds limit "
                f"{self.max_file_size_mb}MB for domain '{self.domain_id}'"
            )
            raise ValidationError(
                f"File size {file_size_mb:.2f}MB exceeds maximum allowed "
                f"size of {self.max_file_size_mb}MB"
            )

        logger.debug(f"✅ File size validation passed: {file_size_mb:.2f}MB")

    def _validate_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Validate required metadata fields are present and non-empty.

        Parameters:
        -----------
        metadata : Dict[str, Any]
            Metadata dictionary

        Raises:
        -------
        ValidationError:
            If required fields missing or empty
        """
        # Phase 2 required fields (from spec Section 4.2)
        required_fields = [
            "doc_id",
            "title",
            "doc_type",
            "uploader_id"
        ]

        missing = []
        empty = []

        for field in required_fields:
            if field not in metadata:
                missing.append(field)
            elif not metadata[field]:  # None, empty string, empty list
                empty.append(field)

        if missing:
            logger.error(f"Missing required metadata fields: {missing}")
            raise ValidationError(f"Missing required metadata fields: {', '.join(missing)}")

        if empty:
            logger.error(f"Empty required metadata fields: {empty}")
            raise ValidationError(f"Empty required metadata fields: {', '.join(empty)}")

        logger.debug("✅ Metadata validation passed")

    def _compute_file_hash(self, file_obj: Any) -> str:
        """
        Compute SHA-256 hash of file for provenance tracking.

        Parameters:
        -----------
        file_obj : Any
            File object with .read() method

        Returns:
        --------
        str:
            Hexadecimal hash string (64 characters)
        """
        sha256_hash = hashlib.sha256()

        # Read file in chunks (handles large files efficiently)
        file_obj.seek(0)  # Reset to beginning
        for chunk in iter(lambda: file_obj.read(4096), b""):
            sha256_hash.update(chunk)

        file_obj.seek(0)  # Reset to beginning for subsequent reads

        file_hash = sha256_hash.hexdigest()
        logger.debug(f"✅ Computed file hash: {file_hash[:16]}...")

        return file_hash

    # =========================================================================
    # PUBLIC API METHODS (Called by UI)
    # =========================================================================

    def upload_document(
            self,
            file_obj: Any,
            metadata: Dict[str, Any],
            replace_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Upload and process a document end-to-end.

        This is the PRIMARY INGESTION METHOD called by the UI.

        Workflow:
        ---------
        1. Validate file type (extension check)
        2. Validate file size (max MB check)
        3. Validate required metadata fields
        4. Extract text from file (PDF, DOCX, TXT)
        5. Compute file hash (SHA-256 for provenance)
        6. Enrich metadata with system fields
        7. Delegate to pipeline for processing
        8. Return ingestion summary

        Parameters:
        -----------
        file_obj : Any
            File object with .name, .read(), .seek() attributes
            Typically from Gradio: UploadedFile or similar

        metadata : Dict[str, Any]
            Metadata dictionary with required fields:
            - doc_id: str (unique identifier)
            - title: str (document title)
            - doc_type: str (policy, faq, manual, etc.)
            - uploader_id: str (user who uploaded)

            Optional fields:
            - author: str (original document author)
            - version: str (document version, default: "1.0")

        replace_existing : bool
            If True, deletes existing document before ingestion
            Default: False

        Returns:
        --------
        Dict[str, Any]:
            Ingestion summary with:
            - doc_id: Document identifier
            - chunks_ingested: Number of chunks created
            - status: "success"
            - embedding_model: Model name used
            - chunking_strategy: Strategy used
            - file_hash: SHA-256 hash

        Raises:
        -------
        ValidationError:
            If validation fails (file type, size, metadata)
        ProcessingError:
            If text extraction or pipeline processing fails

        Example:
        --------
        # In UI handler:
        def upload_handler(file, title, author, doc_type):
            service = DocumentService("hr")

            try:
                result = service.upload_document(
                    file_obj=file,
                    metadata={
                        "doc_id": f"hr_doc_{datetime.now().timestamp()}",
                        "title": title,
                        "author": author,
                        "doc_type": doc_type,
                        "uploader_id": "current_user@company.com"
                    },
                    replace_existing=True
                )
                return f"✅ Success! Ingested {result['chunks_ingested']} chunks"
            except ValidationError as e:
                return f"❌ Validation Error: {e}"
            except ProcessingError as e:
                return f"❌ Processing Error: {e}"
        """
        logger.info(
            f"Upload request: doc_id={metadata.get('doc_id')}, "
            f"domain={self.domain_id}"
        )

        # Step 1: Validate file has name attribute
        filename = getattr(file_obj, 'name', None)
        if not filename:
            raise ValidationError("Uploaded file missing 'name' attribute")

        # Step 2: Validate file type
        self._validate_file_type(filename)

        # Step 3: Validate file size
        self._validate_file_size(file_obj)

        # Step 4: Validate required metadata
        self._validate_metadata(metadata)

        # Step 5: Extract text from file
        logger.info(f"Extracting text from file: {filename}")
        try:
            extracted_data = extract_text_from_file(file_obj, filename)
            text = extracted_data.get('text', '')

            if not text or not text.strip():
                raise ProcessingError(f"No text could be extracted from file: {filename}")

            logger.info(f"✅ Extracted {len(text)} characters from {filename}")

        except Exception as e:
            logger.error(f"Text extraction failed for {filename}: {e}")
            raise ProcessingError(f"Failed to extract text from file: {e}")

        # Step 6: Compute file hash for provenance
        file_hash = self._compute_file_hash(file_obj)

        # Step 7: Enrich metadata with system fields
        enriched_metadata = {
            **metadata,  # User-provided fields
            "domain": self.domain_id,  # Enforce domain
            "source_file_path": filename,
            "file_hash": file_hash,
            "upload_timestamp": datetime.utcnow().isoformat(),
            "version": metadata.get("version", "1.0"),
            "author": metadata.get("author", "Unknown")
        }

        # Step 8: Delegate to pipeline for processing
        logger.info(f"Delegating to pipeline: doc_id={metadata['doc_id']}")
        try:
            result = self.pipeline.process_document(
                text=text,
                doc_id=enriched_metadata["doc_id"],
                domain=self.domain_id,
                source_file_path=filename,
                file_hash=file_hash,
                uploader_id=enriched_metadata["uploader_id"],
                title=enriched_metadata.get("title"),
                doc_type=enriched_metadata.get("doc_type"),
                author=enriched_metadata.get("author"),
                version=enriched_metadata.get("version"),
                replace_existing=replace_existing
            )

            # Add file_hash to result
            result['file_hash'] = file_hash

            logger.info(f"✅ Upload successful: {result}")
            return result

        except Exception as e:
            logger.exception(
                f"Pipeline processing failed for doc_id={metadata.get('doc_id')}"
            )
            raise ProcessingError(f"Document processing failed: {e}")

    def query(
            self,
            query_text: str,
            strategy: Optional[str] = None,
            metadata_filters: Optional[Dict[str, Any]] = None,
            top_k: int = 10,
            include_deprecated: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Execute semantic search over domain documents.

        This is the PRIMARY QUERY METHOD called by the UI.

        Parameters:
        -----------
        query_text : str
            Natural language query
            Example: "How many vacation days do employees get?"

        strategy : str, optional
            Retrieval strategy to use ("hybrid", "vector_similarity", etc.)
            If None, uses all configured strategies and aggregates results

        metadata_filters : Dict[str, Any], optional
            Metadata filters to apply
            Example: {"doc_type": "policy", "author": "HR Department"}

        top_k : int
            Number of results to return
            Default: 10

        include_deprecated : bool
            If False (default), filters out deprecated documents
            Default: False

        Returns:
        --------
        List[Dict[str, Any]]:
            List of results, each with:
            - id: chunk_id
            - score: similarity score
            - metadata: chunk metadata
            - document: chunk text
            - strategy: which strategy returned this (if multi-strategy)

        Example:
        --------
        # In UI handler:
        def query_handler(query, doc_type_filter):
            service = DocumentService("hr")

            results = service.query(
                query_text=query,
                strategy="hybrid",
                metadata_filters={"doc_type": doc_type_filter},
                top_k=5,
                include_deprecated=False
            )

            return results
        """
        logger.info(
            f"Query request: '{query_text}', domain={self.domain_id}, "
            f"strategy={strategy}, top_k={top_k}"
        )

        # Add default filter for deprecated documents
        if not include_deprecated:
            if metadata_filters is None:
                metadata_filters = {}
            metadata_filters['deprecated'] = False

        # Delegate to pipeline
        try:
            results = self.pipeline.query(
                query_text=query_text,
                strategy_name=strategy,
                metadata_filters=metadata_filters,
                top_k=top_k
            )

            logger.info(f"✅ Query returned {len(results)} results")
            return results

        except Exception as e:
            logger.exception(f"Query failed: {query_text}")
            raise ProcessingError(f"Query execution failed: {e}")

    def delete_document(self, doc_id: str) -> None:
        """
        Delete all chunks for a document.

        Parameters:
        -----------
        doc_id : str
            Document identifier

        Raises:
        -------
        ProcessingError:
            If deletion fails

        Example:
        --------
        service.delete_document("old_handbook_2023")
        """
        logger.info(f"Delete request: doc_id={doc_id}, domain={self.domain_id}")

        try:
            self.pipeline.delete_document(doc_id)
            logger.info(f"✅ Document deleted: {doc_id}")

        except Exception as e:
            logger.exception(f"Deletion failed for doc_id={doc_id}")
            raise ProcessingError(f"Document deletion failed: {e}")

    def get_document_info(self, doc_id: str) -> Dict[str, Any]:
        """
        Get information about a document.

        Parameters:
        -----------
        doc_id : str
            Document identifier

        Returns:
        --------
        Dict[str, Any]:
            Document metadata and statistics

        Raises:
        -------
        DocumentNotFoundError:
            If document not found
        ProcessingError:
            If retrieval fails
        """
        logger.info(f"Document info request: doc_id={doc_id}")

        try:
            info = self.pipeline.get_document_info(doc_id)
            return info

        except NotImplementedError:
            raise ProcessingError("Document info retrieval not yet implemented")

        except Exception as e:
            logger.exception(f"Failed to get document info: {doc_id}")
            raise DocumentNotFoundError(f"Document not found: {doc_id}")


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of DocumentService usage.
    Run: python core/services/document_service.py
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("DocumentService Usage Examples")
    print("=" * 70)

    print("""
Example 1: Upload Document (UI Handler Pattern)
------------------------------------------------
def upload_handler(file, title, author, doc_type):
    '''Gradio handler for file upload.'''
    service = DocumentService("hr")

    try:
        result = service.upload_document(
            file_obj=file,
            metadata={
                "doc_id": f"hr_{datetime.now().timestamp()}",
                "title": title,
                "author": author,
                "doc_type": doc_type,
                "uploader_id": "current_user@company.com"
            },
            replace_existing=True
        )

        return f"✅ Success! Ingested {result['chunks_ingested']} chunks"

    except ValidationError as e:
        return f"❌ Validation Error: {e}"

    except ProcessingError as e:
        return f"❌ Processing Error: {e}"


Example 2: Query Documents (UI Handler Pattern)
------------------------------------------------
def query_handler(query, doc_type_filter, strategy):
    '''Gradio handler for queries.'''
    service = DocumentService("hr")

    results = service.query(
        query_text=query,
        strategy=strategy,
        metadata_filters={"doc_type": doc_type_filter} if doc_type_filter else None,
        top_k=10,
        include_deprecated=False
    )

    # Format for UI display
    formatted_results = []
    for result in results:
        formatted_results.append({
            "score": f"{result['score']:.3f}",
            "text": result['document'][:200] + "...",
            "source": result['metadata']['title'],
            "page": result['metadata'].get('page_num', 'N/A')
        })

    return formatted_results


Example 3: Complete UI Integration (Gradio)
--------------------------------------------
import gradio as gr
from core.services.document_service import DocumentService

# Initialize service ONCE (not in handlers!)
service = DocumentService("hr")

def upload(file, title, doc_type):
    '''Upload handler - delegates to service.'''
    result = service.upload_document(
        file_obj=file,
        metadata={
            "doc_id": f"doc_{datetime.now().timestamp()}",
            "title": title,
            "doc_type": doc_type,
            "uploader_id": "user123"
        }
    )
    return f"Uploaded: {result['chunks_ingested']} chunks"

def query(query_text, doc_type_filter):
    '''Query handler - delegates to service.'''
    results = service.query(
        query_text=query_text,
        strategy="hybrid",
        metadata_filters={"doc_type": doc_type_filter} if doc_type_filter else None,
        top_k=5
    )
    return results

# Build UI
with gr.Blocks() as demo:
    with gr.Tab("Upload"):
        file_input = gr.File()
        title_input = gr.Textbox(label="Title")
        type_input = gr.Dropdown(["policy", "faq", "manual"], label="Type")
        upload_btn = gr.Button("Upload")
        upload_output = gr.Textbox(label="Result")

        upload_btn.click(upload, [file_input, title_input, type_input], upload_output)

    with gr.Tab("Query"):
        query_input = gr.Textbox(label="Query")
        filter_input = gr.Dropdown(["policy", "faq", "manual"], label="Filter")
        query_btn = gr.Button("Search")
        query_output = gr.JSON(label="Results")

        query_btn.click(query, [query_input, filter_input], query_output)

demo.launch()

# Notice: UI has ZERO business logic - only calls service methods!
    """)

    print("\n" + "=" * 70)
    print("DocumentService examples completed!")
    print("=" * 70)
