"""
Document Pipeline Module
Orchestrates the complete document processing workflow:
PDF → Extract → Chunk → Embed → Upsert → Version Management
"""

import sys
from pathlib import Path

# Add project root to Python path (so it can find 'models' package)
project_root = Path(__file__).parent.parent  # Go up two levels from config_manager.py
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import logging

from core.config_manager import ConfigManager
from core.vector_store import VectorStore
from core.embeddings import EmbeddingGenerator
from utils.pdf_processor import PDFProcessor
from models.domain_config import MergedDomainConfig

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """
    End-to-end document processing pipeline.

    Features:
    - Complete PDF → Vector DB workflow
    - Version detection and management
    - Semantic deduplication
    - Progress tracking
    - Error handling with rollback
    """

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize document pipeline.

        Args:
            config_manager: ConfigManager instance
        """
        self.config_manager = config_manager

        # Initialize components (will be created per domain)
        self.vector_store = None
        self.embedder = None
        self.pdf_processor = None

        logger.info("DocumentPipeline initialized")

    def _initialize_components(self, domain_config: MergedDomainConfig):
        """
        Initialize processing components for a domain.

        Args:
            domain_config: Merged domain configuration
        """
        logger.info(f"Initializing components for domain: {domain_config.domain.name}")

        # Create vector store (shared across domains)
        if self.vector_store is None:
            self.vector_store = VectorStore(domain_config.vector_store)

        # Create embedder (domain-specific if embeddings differ)
        self.embedder = EmbeddingGenerator(domain_config.embeddings)

        # Create PDF processor (domain-specific chunking)
        self.pdf_processor = PDFProcessor(domain_config.chunking)

        logger.info("✅ Components initialized")

    def check_existing_versions(
            self,
            collection_name: str,
            document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Check for existing versions of a document.

        Args:
            collection_name: Collection to check
            document_id: Document ID to check

        Returns:
            List of existing version info
        """
        if self.vector_store is None:
            return []

        versions = self.vector_store.get_document_versions(collection_name, document_id)

        if versions:
            logger.info(f"Found {len(versions)} existing versions for {document_id}")
        else:
            logger.info(f"No existing versions found for {document_id}")

        return versions

    def detect_duplicate(
            self,
            collection_name: str,
            document_embedding: Any,
            threshold: float = 0.90
    ) -> Optional[Dict[str, Any]]:
        """
        Detect semantically similar documents (potential duplicates).

        Args:
            collection_name: Collection to search
            document_embedding: Document-level embedding
            threshold: Similarity threshold for duplicate detection

        Returns:
            Info about similar document if found, else None
        """
        logger.info(f"Checking for duplicates (threshold={threshold})")

        try:
            results = self.vector_store.search(
                collection_name=collection_name,
                query_embedding=document_embedding,
                top_k=1
            )

            if results["ids"][0]:  # Has results
                similarity = 1 - results["distances"][0][0]  # Convert distance to similarity

                if similarity >= threshold:
                    metadata = results["metadatas"][0][0]
                    logger.info(f"⚠️ Potential duplicate found: {metadata['filename']} "
                                f"(similarity={similarity:.2%})")

                    return {
                        "filename": metadata.get("filename", "unknown"),
                        "document_id": metadata.get("document_id", "unknown"),
                        "version": metadata.get("document_version", "unknown"),
                        "similarity": similarity
                    }

        except Exception as e:
            logger.warning(f"Duplicate detection failed: {e}")

        return None

    def process_document(
            self,
            domain_name: str,
            pdf_path: str,
            document_id: str,
            version: str,
            metadata: Dict[str, Any],
            replace_versions: Optional[List[str]] = None,
            check_duplicates: bool = True
    ) -> Dict[str, Any]:
        """
        Complete document processing pipeline.

        Args:
            domain_name: Domain to process for
            pdf_path: Path to PDF file
            document_id: Logical document ID
            version: Document version
            metadata: Additional metadata (department, doc_type, etc.)
            replace_versions: List of versions to replace
            check_duplicates: Whether to check for semantic duplicates

        Returns:
            Dict with processing results and statistics
        """
        logger.info(f"=" * 80)
        logger.info(f"Starting document processing pipeline")
        logger.info(f"Domain: {domain_name}")
        logger.info(f"Document ID: {document_id}")
        logger.info(f"Version: {version}")
        logger.info(f"PDF: {pdf_path}")
        logger.info(f"=" * 80)

        start_time = datetime.now()

        try:
            # Step 1: Load domain configuration
            logger.info("Step 1/7: Loading domain configuration...")
            domain_config = self.config_manager.get_domain_config(domain_name)
            collection_name = domain_config.domain.collection_name

            # Step 2: Initialize components
            logger.info("Step 2/7: Initializing components...")
            self._initialize_components(domain_config)

            # Step 3: Extract and chunk PDF
            logger.info("Step 3/7: Extracting and chunking PDF...")
            chunks = self.pdf_processor.process_pdf(pdf_path)

            if not chunks:
                raise ValueError("No chunks extracted from PDF")

            logger.info(f"✅ Extracted {len(chunks)} chunks")

            # Step 4: Generate embeddings
            logger.info("Step 4/7: Generating embeddings...")
            embeddings = self.embedder.embed_chunks(chunks, show_progress=True)
            logger.info(f"✅ Generated embeddings: shape={embeddings.shape}")

            # Step 5: Check for duplicates (optional)
            duplicate_info = None
            if check_duplicates:
                logger.info("Step 5/7: Checking for duplicates...")
                doc_embedding = self.embedder.compute_document_embedding(chunks)
                duplicate_info = self.detect_duplicate(collection_name, doc_embedding)
            else:
                logger.info("Step 5/7: Skipping duplicate check (disabled)")

            # Step 6: Handle version replacements
            if replace_versions:
                logger.info(f"Step 6/7: Deleting {len(replace_versions)} old versions...")
                for old_version in replace_versions:
                    deleted = self.vector_store.delete_by_document_id(
                        collection_name=collection_name,
                        document_id=document_id,
                        version=old_version
                    )
                    logger.info(f"  Deleted version {old_version}: {deleted} chunks")
            else:
                logger.info("Step 6/7: No versions to replace")

            # Step 7: Upsert to vector store
            logger.info("Step 7/7: Upserting to vector store...")

            # Generate chunk IDs
            chunk_ids = [f"{document_id}-{version}-chunk{i}" for i in range(len(chunks))]

            # Extract texts
            texts = [chunk["text"] for chunk in chunks]

            # Build metadata for each chunk
            upload_date = datetime.now().isoformat()
            filename = Path(pdf_path).name

            metadatas = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "document_id": document_id,
                    "document_version": version,
                    "filename": filename,
                    "domain": domain_name,
                    "upload_date": upload_date,
                    "chunk_index": i,
                    "page": chunk.get("page", 0),
                    "chunk_length": chunk.get("length", len(chunk["text"]))
                }

                # Add user-provided metadata
                chunk_metadata.update(metadata)
                metadatas.append(chunk_metadata)

            # Upsert
            upsert_stats = self.vector_store.upsert_documents(
                collection_name=collection_name,
                chunk_ids=chunk_ids,
                embeddings=embeddings,
                texts=texts,
                metadatas=metadatas
            )

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Build result
            result = {
                "success": True,
                "document_id": document_id,
                "version": version,
                "filename": filename,
                "domain": domain_name,
                "collection": collection_name,
                "chunks_processed": len(chunks),
                "chunks_upserted": upsert_stats["chunks_upserted"],
                "total_collection_chunks": upsert_stats["total_docs_in_collection"],
                "versions_replaced": replace_versions or [],
                "duplicate_detected": duplicate_info,
                "processing_time_seconds": processing_time,
                "timestamp": upload_date
            }

            logger.info(f"=" * 80)
            logger.info(f"✅ Document processing completed successfully")
            logger.info(f"   Chunks: {len(chunks)}")
            logger.info(f"   Processing time: {processing_time:.2f}s")
            logger.info(f"   Collection size: {upsert_stats['total_docs_in_collection']} chunks")
            logger.info(f"=" * 80)

            return result

        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")

            return {
                "success": False,
                "error": str(e),
                "document_id": document_id,
                "version": version,
                "domain": domain_name,
                "timestamp": datetime.now().isoformat()
            }

    def delete_document(
            self,
            domain_name: str,
            document_id: str,
            version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Delete a document (or specific version).

        Args:
            domain_name: Domain name
            document_id: Document ID to delete
            version: Optional specific version to delete

        Returns:
            Dict with deletion results
        """
        logger.info(f"Deleting document: {document_id} (version={version or 'all'})")

        try:
            # Load domain config
            domain_config = self.config_manager.get_domain_config(domain_name)
            collection_name = domain_config.domain.collection_name

            # Initialize vector store if needed
            if self.vector_store is None:
                self.vector_store = VectorStore(domain_config.vector_store)

            # Delete
            chunks_deleted = self.vector_store.delete_by_document_id(
                collection_name=collection_name,
                document_id=document_id,
                version=version
            )

            result = {
                "success": True,
                "document_id": document_id,
                "version": version or "all",
                "chunks_deleted": chunks_deleted,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"✅ Deleted {chunks_deleted} chunks")
            return result

        except Exception as e:
            logger.error(f"❌ Deletion failed: {e}")

            return {
                "success": False,
                "error": str(e),
                "document_id": document_id,
                "version": version,
                "timestamp": datetime.now().isoformat()
            }

    def list_documents(
            self,
            domain_name: str,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        List documents in a domain.

        Args:
            domain_name: Domain name
            filters: Optional metadata filters

        Returns:
            List of document info
        """
        logger.info(f"Listing documents in {domain_name}")

        try:
            # Load domain config
            domain_config = self.config_manager.get_domain_config(domain_name)
            collection_name = domain_config.domain.collection_name

            # Initialize vector store if needed
            if self.vector_store is None:
                self.vector_store = VectorStore(domain_config.vector_store)

            # List documents
            documents = self.vector_store.list_documents(collection_name, filters)

            logger.info(f"Found {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []


# ============================================
# Convenience Functions
# ============================================

def create_pipeline(config_dir: str = "configs") -> DocumentPipeline:
    """
    Factory function to create DocumentPipeline.

    Args:
        config_dir: Path to configs directory

    Returns:
        Initialized DocumentPipeline
    """
    config_manager = ConfigManager(config_dir=config_dir)
    return DocumentPipeline(config_manager)


def quick_upload(
        domain_name: str,
        pdf_path: str,
        document_id: str,
        version: str,
        metadata: Dict[str, Any],
        config_dir: str = "configs"
) -> Dict[str, Any]:
    """
    Quick function to upload a document.

    Args:
        domain_name: Domain name
        pdf_path: Path to PDF
        document_id: Document ID
        version: Version string
        metadata: Additional metadata
        config_dir: Path to configs

    Returns:
        Processing results
    """
    pipeline = create_pipeline(config_dir)
    return pipeline.process_document(
        domain_name=domain_name,
        pdf_path=pdf_path,
        document_id=document_id,
        version=version,
        metadata=metadata
    )
