"""
core/pipeline/document_pipeline.py

This module implements the main document processing pipeline that orchestrates
the complete workflow from raw documents to searchable vector embeddings.

What is the Document Pipeline?
-------------------------------
The document pipeline is the ORCHESTRATOR that coordinates all components
(chunking, embedding, vector storage) to process documents end-to-end.

Think of it as an assembly line:
1. Raw Document (PDF, DOCX, TXT) →
2. Extract Text →
3. Split into Chunks →
4. Generate Embeddings →
5. Store in Vector Database →
6. Done! Document is now searchable

Why Do We Need This?
---------------------
Without a pipeline, you'd need to manually:
- Load config for each domain
- Create chunker, embedder, vector store
- Extract text from files
- Chunk text
- Generate embeddings
- Upsert to vector store
- Handle errors at each step

The pipeline encapsulates ALL of this into one simple call:
    pipeline.process_document(file_path, doc_id, domain)

Design Pattern:
---------------
This implements the Pipeline Pattern (a.k.a. Chain of Responsibility):
- Each stage processes data and passes to next stage
- Stages are independent and can be swapped
- Error handling and logging at each stage
- Easy to add new stages (e.g., metadata extraction, OCR)

Config-Driven Benefits:
-----------------------
The pipeline uses factories to create components based on config:
- Different domains can use different chunking strategies
- Different domains can use different embedding models
- Different domains can use different vector stores
- All controlled by YAML config, no code changes!

Example:
--------
# Initialize pipeline for HR domain
pipeline = DocumentPipeline(
    domain="hr",
    config_dir="./configs"
)

# Process a document (one line!)
result = pipeline.process_document(
    file_path="./docs/employee_handbook.pdf",
    doc_id="handbook_2025",
    uploader_id="hr_admin@company.com"
)

# That's it! Document is now chunked, embedded, and searchable
print(f"Processed {result['chunks_created']} chunks")
print(f"Processing time: {result['processing_time']:.2f}s")

Architecture:
-------------
DocumentPipeline
    ├── ConfigManager (loads domain config)
    ├── ChunkingFactory (creates chunker)
    ├── EmbeddingFactory (creates embedder)
    ├── VectorStoreFactory (creates vector store)
    └── FileParser (extracts text from files)

The pipeline coordinates these components to execute the workflow.
"""



from typing import Dict, Any, Optional, List
from pathlib import Path
import hashlib
import time
import logging

# Core components
from core.config_manager import ConfigManager
from core.factories.chunking_factory import ChunkingFactory
from core.factories.embedding_factory import EmbeddingFactory
from core.factories.vector_store_factory import VectorStoreFactory

# Interfaces
from core.interfaces.chunking_interface import ChunkerInterface
from core.interfaces.embedding_interface import EmbeddingInterface
from core.interfaces.vector_store_interface import VectorStoreInterface

# Models
from models.metadata_models import ChunkMetadata

# Configure logging
logger = logging.getLogger(__name__)


class DocumentPipeline:
    """
    Main document processing pipeline that orchestrates the complete workflow.

    This class is the CORE of your document ingestion system. It coordinates
    all components to transform raw documents into searchable embeddings.

    Responsibilities:
    -----------------
    1. Load domain configuration
    2. Initialize components (chunker, embedder, vector store)
    3. Extract text from files
    4. Chunk documents
    5. Generate embeddings
    6. Store in vector database
    7. Handle errors and logging
    8. Track metrics (time, chunks, etc.)

    Key Design Decisions:
    ---------------------
    - **One pipeline per domain**: Each domain has its own config and components
    - **Lazy initialization**: Components created on first use (faster startup)
    - **Error isolation**: Errors in one document don't affect others
    - **Comprehensive logging**: Every step is logged for debugging
    - **Metrics tracking**: Returns detailed results for monitoring

    Usage Patterns:
    ---------------
    1. Single document processing:
       pipeline = DocumentPipeline(domain="hr")
       result = pipeline.process_document("file.pdf", "doc123")

    2. Batch processing:
       pipeline = DocumentPipeline(domain="hr")
       for file in files:
           result = pipeline.process_document(file, f"doc_{i}")

    3. Different domains:
       hr_pipeline = DocumentPipeline(domain="hr")
       finance_pipeline = DocumentPipeline(domain="finance")

    Example:
    --------
    # Initialize pipeline
    pipeline = DocumentPipeline(
        domain="hr",
        config_dir="./configs"
    )

    # Process document
    result = pipeline.process_document(
        file_path="./docs/handbook.pdf",
        doc_id="handbook_2025",
        uploader_id="hr_admin@company.com"
    )

    # Check results
    print(f"✅ Success: {result['success']}")
    print(f"📄 Chunks: {result['chunks_created']}")
    print(f"⏱️  Time: {result['processing_time']:.2f}s")
    """

    def __init__(self, domain: str, config_dir: str = "configs"):
        """
        Initialize the document pipeline for a specific domain.

        Parameters:
        -----------
        domain : str
            Domain name (must match a config file)
            Examples: "hr", "finance", "engineering"

            This will load: configs/domains/{domain}_domain.yaml

        config_dir : str, optional
            Root directory containing config files
            Default: "configs"

        Raises:
        -------
        FileNotFoundError:
            If domain config file doesn't exist
        ValueError:
            If domain config is invalid

        Notes:
        ------
        - Components (chunker, embedder, vector store) are initialized lazily
        - Config is loaded immediately to fail fast if invalid
        - Domain-specific settings are merged with global defaults

        Example:
        --------
        # Initialize for HR domain
        hr_pipeline = DocumentPipeline(domain="hr")

        # Initialize for Finance domain with custom config dir
        finance_pipeline = DocumentPipeline(
            domain="finance",
            config_dir="./custom_configs"
        )
        """
        self.domain = domain
        self.config_dir = config_dir

        logger.info(
            f"Initializing DocumentPipeline for domain: {domain}\n"
            f"  Config directory: {config_dir}"
        )

        # Step 1: Load domain configuration
        try:
            self.config_manager = ConfigManager(config_dir=config_dir)
            self.config = self.config_manager.load_domain_config(domain)

            logger.info(
                f"✅ Loaded config for domain: {domain}\n"
                f"   Display name: {self.config.display_name}\n"
                f"   Description: {self.config.description}\n"
                f"   Chunking: {self.config.chunking.strategy}\n"
                f"   Embeddings: {self.config.embeddings.provider}\n"
                f"   Vector store: {self.config.vector_store.provider}"
            )

        except FileNotFoundError as e:
            logger.error(f"Domain config not found: {domain}")
            raise FileNotFoundError(
                f"Configuration file not found for domain: {domain}\n"
                f"Expected file: {config_dir}/domains/{domain}_domain.yaml\n"
                f"Create this file with domain-specific settings."
            )
        except Exception as e:
            logger.error(f"Failed to load config for domain {domain}: {e}")
            raise

        # Step 2: Initialize component placeholders (lazy initialization)
        self._chunker: Optional[ChunkerInterface] = None
        self._embedder: Optional[EmbeddingInterface] = None
        self._vector_store: Optional[VectorStoreInterface] = None

        logger.info(f"DocumentPipeline initialized for domain: {domain}")

    @property
    def chunker(self) -> ChunkerInterface:
        """
        Get or create the chunker instance (lazy initialization).

        Lazy initialization means we only create the chunker when it's first needed.
        Benefits:
        - Faster pipeline startup
        - Avoid loading resources for unused components
        - Better memory efficiency

        Returns:
        --------
        ChunkerInterface:
            Chunker instance for this domain's chunking strategy
        """
        if self._chunker is None:
            logger.debug(f"Initializing chunker for domain: {self.domain}")
            self._chunker = ChunkingFactory.create_chunker(
                self.config.chunking,
                embedding_model_name=self.config.embeddings.model_name
            )
        return self._chunker

    @property
    def embedder(self) -> EmbeddingInterface:
        """
        Get or create the embedder instance (lazy initialization).

        Returns:
        --------
        EmbeddingInterface:
            Embedder instance for this domain's embedding provider
        """
        if self._embedder is None:
            logger.debug(f"Initializing embedder for domain: {self.domain}")
            self._embedder = EmbeddingFactory.create_embedder(
                self.config.embeddings
            )
        return self._embedder

    @property
    def vector_store(self) -> VectorStoreInterface:
        """
        Get or create the vector store instance (lazy initialization).

        Returns:
        --------
        VectorStoreInterface:
            Vector store instance for this domain's vector store provider
        """
        if self._vector_store is None:
            logger.debug(f"Initializing vector store for domain: {self.domain}")

            # Get embedding dimension from embedder
            # This ensures vector store dimension matches embedding dimension
            embedding_dim = 384  # Default
            if hasattr(self.embedder, 'get_embedding_dimension'):
                embedding_dim = self.embedder.get_embedding_dimension()
            elif hasattr(self.embedder, 'embedding_dim'):
                embedding_dim = self.embedder.embedding_dim

            self._vector_store = VectorStoreFactory.create_store(
                self.config.vector_store,
                embedding_dimension=embedding_dim
            )
        return self._vector_store

    def process_document(
            self,
            file_path: str,
            doc_id: str,
            uploader_id: Optional[str] = None,
            page_num: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process a single document through the complete pipeline."""
        """
        Process a single document through the complete pipeline.

        This is the MAIN METHOD that executes the end-to-end workflow:
        File → Text → Chunks → Embeddings → Vector Store

        Parameters:
        -----------
        file_path : str
            Path to the document file
            Supported formats: PDF, DOCX, TXT, CSV (based on file parsers)
            Example: "./data/raw_documents/hr/employee_handbook.pdf"

        doc_id : str
            Unique identifier for this document
            Used to group chunks and enable updates/deletions
            Example: "employee_handbook_2025"

        uploader_id : str, optional
            Identifier of user/system who uploaded this document
            Used for audit trails and access control
            Example: "hr_admin@company.com"

        page_num : int, optional
            Page number (for paginated documents like PDFs)
            Used for citations in search results
            Example: 12 (for page 12)

        Returns:
        --------
        Dict[str, Any]:
            Result dictionary containing:
            - success (bool): Whether processing succeeded
            - doc_id (str): Document identifier
            - domain (str): Domain name
            - chunks_created (int): Number of chunks created
            - processing_time (float): Time taken in seconds
            - file_hash (str): SHA256 hash of source file
            - error (str, optional): Error message if failed

        Raises:
        -------
        FileNotFoundError:
            If file_path doesn't exist
        ValueError:
            If file format is not supported
        RuntimeError:
            If any pipeline stage fails

        Workflow:
        ---------
        1. Validate file exists and is readable
        2. Extract text from file (PDF, DOCX, TXT, etc.)
        3. Calculate file hash (for idempotency and change detection)
        4. Chunk text using domain's chunking strategy
        5. Generate embeddings for all chunks
        6. Upsert chunks + embeddings to vector store
        7. Log success and return metrics

        Example:
        --------
        pipeline = DocumentPipeline(domain="hr")

        result = pipeline.process_document(
            file_path="./docs/handbook.pdf",
            doc_id="handbook_2025",
            uploader_id="hr_admin@company.com"
        )

        if result['success']:
            print(f"✅ Processed {result['chunks_created']} chunks")
            print(f"⏱️  Time: {result['processing_time']:.2f}s")
        else:
            print(f"❌ Error: {result['error']}")

        Idempotency:
        ------------
        Processing the same file multiple times with same doc_id:
        - First time: Creates new chunks
        - Second time: Updates existing chunks (upsert)
        - File hash is stored to detect changes

        If file content changes:
        - Old chunks are updated with new content
        - File hash is updated
        - Metadata reflects new upload timestamp
        """

        start_time = time.time()
        logger.info(
            f"Processing document:\n"
            f"  File: {file_path}\n"
            f"  Doc ID: {doc_id}\n"
            f"  Domain: {self.domain}\n"
            f"  Uploader: {uploader_id or 'N/A'}"
        )

        try:
            # ================================================================
            # STAGE 1: Validate and read file
            # ================================================================
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            if not file_path_obj.is_file():
                raise ValueError(f"Path is not a file: {file_path}")

            # Read file content for hash
            logger.debug("Reading file content...")
            with open(file_path, 'rb') as f:
                file_content = f.read()

            # Calculate file hash (for change detection)
            file_hash = hashlib.sha256(file_content).hexdigest()
            logger.debug(f"File hash: {file_hash}")

            # ================================================================
            # STAGE 2: Extract text from file using appropriate processor
            # ================================================================
            logger.debug("Extracting text from file...")

            # Import file processors
            from utils.file_parsers.pdf_processor import PDFProcessor
            from utils.file_parsers.docx_processor import DOCXProcessor
            from utils.file_parsers.txt_processor import TXTProcessor

            # Determine file type and extract text
            file_ext = file_path_obj.suffix.lower()

            if file_ext == '.pdf':
                processor = PDFProcessor()
                text = processor.extract_text(str(file_path))
            elif file_ext == '.docx':
                processor = DOCXProcessor()
                text = processor.extract_text(str(file_path))
            elif file_ext == '.txt':
                processor = TXTProcessor()
                text = processor.extract_text(str(file_path))
            else:
                raise ValueError(
                    f"Unsupported file format: {file_ext}\n"
                    f"Supported formats: .pdf, .docx, .txt"
                )

            if not text or not text.strip():
                raise ValueError(f"File is empty or contains no extractable text: {file_path}")

            logger.info(f"Extracted {len(text)} characters from {file_ext} file")

            # ================================================================
            # STAGE 3: Chunk text
            # ================================================================
            logger.debug(
                f"Chunking text using {self.config.chunking.strategy} strategy..."
            )

            chunks = self.chunker.chunk_text(
                text=text,
                doc_id=doc_id,
                domain=self.domain,
                source_file_path=str(file_path),
                file_hash=file_hash,
                uploader_id=uploader_id,
                page_num=page_num
            )

            if not chunks:
                raise ValueError(
                    f"No chunks created from document: {doc_id}\n"
                    f"Text length: {len(text)} characters\n"
                    f"Check chunking configuration"
                )

            logger.info(f"Created {len(chunks)} chunks")

            # ================================================================
            # STAGE 4: Generate embeddings
            # ================================================================
            logger.debug("Generating embeddings...")

            # Extract text from chunks for embedding
            chunk_texts = [chunk.chunk_text for chunk in chunks]

            # Generate embeddings (batched internally by embedder)
            embeddings = self.embedder.embed_texts(chunk_texts)

            logger.info(
                f"Generated embeddings: shape={embeddings.shape}, "
                f"model={self.embedder.get_model_name()}"
            )

            # ================================================================
            # STAGE 5: Upsert to vector store
            # ================================================================
            logger.debug("Upserting to vector store...")
            self.vector_store.upsert(chunks, embeddings)
            logger.info(f"Upserted {len(chunks)} chunks to vector store")

            # ================================================================
            # STAGE 6: Return success result
            # ================================================================
            processing_time = time.time() - start_time

            result = {
                'success': True,
                'doc_id': doc_id,
                'domain': self.domain,
                'chunks_created': len(chunks),
                'processing_time': processing_time,
                'file_path': str(file_path),
                'file_size': len(file_content),
                'file_hash': file_hash,
                'embedding_model': self.embedder.get_model_name(),
                'chunking_strategy': self.config.chunking.strategy,
                'vector_store': self.config.vector_store.provider
            }

            logger.info(
                f"✅ Document processed successfully!\n"
                f"  Doc ID: {doc_id}\n"
                f"  Chunks: {len(chunks)}\n"
                f"  Time: {processing_time:.2f}s\n"
                f"  Embedding model: {self.embedder.get_model_name()}\n"
                f"  Vector store: {self.config.vector_store.provider}"
            )

            return result

        except Exception as e:
            # Error handling: Log and return error result
            processing_time = time.time() - start_time
            logger.error(
                f"❌ Document processing failed:\n"
                f"  Doc ID: {doc_id}\n"
                f"  Error: {str(e)}\n"
                f"  Time: {processing_time:.2f}s",
                exc_info=True
            )

            return {
                'success': False,
                'doc_id': doc_id,
                'domain': self.domain,
                'chunks_created': 0,
                'processing_time': processing_time,
                'file_path': str(file_path) if 'file_path' in locals() else None,
                'error': str(e),
                'error_type': type(e).__name__
            }


    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Delete all chunks for a document from the vector store.

        Parameters:
        -----------
        doc_id : str
            Document ID to delete

        Returns:
        --------
        Dict[str, Any]:
            Result dictionary with success status

        Example:
        --------
        result = pipeline.delete_document("old_handbook_2023")
        if result['success']:
            print("✅ Document deleted")
        """
        logger.info(f"Deleting document: {doc_id} from domain: {self.domain}")

        try:
            self.vector_store.delete_by_doc_id(doc_id)

            logger.info(f"✅ Document deleted: {doc_id}")

            return {
                'success': True,
                'doc_id': doc_id,
                'domain': self.domain
            }

        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")

            return {
                'success': False,
                'doc_id': doc_id,
                'domain': self.domain,
                'error': str(e)
            }


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of DocumentPipeline usage.
    Run: python core/pipeline/document_pipeline.py

    Note: Requires configs/ directory with domain config files
    """

    import logging
    import tempfile

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("DocumentPipeline Usage Examples")
    print("=" * 70)

    # Example 1: Create a test document
    print("\n1. Creating Test Document")
    print("-" * 70)

    # Create temporary test file
    test_content = """
    Employee Benefits Overview

    All full-time employees receive comprehensive benefits including:
    - 15 vacation days per year
    - Health insurance (medical, dental, vision)
    - 401k retirement plan with 6% company match
    - Professional development budget of $2000 annually

    For more information, contact HR at hr@company.com
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        test_file_path = f.name

    print(f"Created test file: {test_file_path}")
    print(f"Content: {len(test_content)} characters")

    # Example 2: Initialize pipeline
    print("\n2. Initializing Pipeline")
    print("-" * 70)

    from pathlib import Path

    print(f"Current directory: {Path.cwd()}")
    print(f"HR config exists: {Path('configs/domains/hr_domain.yaml').exists()}")
    # Use relative path that works from core/pipeline/

    # Note: This requires configs/domains/hr_domain.yaml to exist
    try:
        # Use relative path that works from core/pipeline/
        pipeline = DocumentPipeline(domain="hr", config_dir="../../configs")
        print(f"✅ Pipeline initialized for domain: hr")
        print(f"   Display name: {pipeline.config.display_name}")
        print(f"   Chunking: {pipeline.config.chunking.strategy}")
        print(f"   Embeddings: {pipeline.config.embeddings.provider}")
    except FileNotFoundError as e:
        print(f"⚠️  Config not found: {e}")
        print("   Create configs/domains/hr_domain.yaml to run this example")
        import sys

        sys.exit(0)

    # Example 3: Process document
    print("\n3. Processing Document")
    print("-" * 70)

    result = pipeline.process_document(
        file_path=test_file_path,
        doc_id="test_benefits_doc",
        uploader_id="admin@company.com"
    )

    if result['success']:
        print(f"✅ Processing successful!")
        print(f"   Chunks created: {result['chunks_created']}")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        print(f"   Embedding model: {result['embedding_model']}")
        print(f"   Vector store: {result['vector_store']}")
    else:
        print(f"❌ Processing failed: {result['error']}")

    # Example 4: Delete document
    print("\n4. Deleting Document")
    print("-" * 70)

    delete_result = pipeline.delete_document("test_benefits_doc")

    if delete_result['success']:
        print(f"✅ Document deleted: {delete_result['doc_id']}")
    else:
        print(f"❌ Deletion failed: {delete_result['error']}")

    # Cleanup
    import os

    os.unlink(test_file_path)

    print("\n" + "=" * 70)
    print("DocumentPipeline examples completed!")
    print("=" * 70)
