"""

core/pipeline/document_pipeline.py

This module implements the Document Processing Pipeline for Phase 2 architecture.

What is the Document Pipeline?
-------------------------------
The DocumentPipeline orchestrates the complete document processing workflow:
1. Chunking: Split documents into semantic chunks
2. Embedding: Convert chunks into vector representations
3. Storage: Store vectors with metadata in vector database
4. Retrieval: Query documents using multiple strategies

This is the CORE business logic layer that sits between the service layer
and the factory layer in Phase 2 architecture.

Phase 2 Architecture Position:
-------------------------------
UI Layer (app.py)
    ↓ calls only
Service Layer (DocumentService)
    ↓ delegates to
**Pipeline Layer (DocumentPipeline)** ← YOU ARE HERE
    ↓ uses
Factory Layer (ChunkingFactory, EmbeddingFactory, etc.)

Why Use a Pipeline?
--------------------
1. **Orchestration**: Coordinates multiple components (chunker, embedder, vector store)
2. **Workflow Management**: Enforces processing order and dependencies
3. **Configuration-Driven**: All components created via factories from config
4. **Separation of Concerns**: Business logic separate from UI and data access
5. **Testability**: Can test complete workflows without UI
6. **Reusability**: Same pipeline used by web UI, CLI, API, batch jobs

Key Responsibilities:
---------------------
- Initialize all components via factories (NOT direct instantiation!)
- Execute document ingestion workflow
- Execute multi-strategy retrieval workflow
- Manage document lifecycle (deprecation, updates)
- Attach comprehensive metadata to all chunks
- Provide document information and listing

Key Principles:
---------------
1. **Factory-Based**: All components created via factories, never direct instantiation
2. **Config-Driven**: All behavior controlled by domain config
3. **Metadata-Rich**: Every chunk gets complete Phase 2 metadata
4. **Multi-Strategy**: Supports multiple retrieval strategies simultaneously
5. **Error-Transparent**: Let errors bubble up (service layer handles them)
6. **Stateless**: No mutable state beyond config (thread-safe)

Example Usage:
--------------
# Initialize pipeline with domain config
config = ConfigManager().load_domain_config("hr")
pipeline = DocumentPipeline(config)

# Process document
result = pipeline.process_document(
    text="Employee handbook content...",
    doc_id="handbook_2025",
    domain="hr",
    source_file_path="./docs/handbook.pdf",
    file_hash="abc123...",
    uploader_id="admin@company.com"
)

# Query with multiple strategies
results = pipeline.query(
    query_text="vacation policy",
    strategy_name="hybrid",  # or None for all strategies
    metadata_filters={"domain": "hr", "deprecated": False},
    top_k=10
)

References:
-----------
- Phase 2 Architecture: Section 5 (Architecture Overview)
- Pipeline Layer: Section 7 (Pipeline Layer Enhancements)
- Multi-Strategy Retrieval: Section 8 (Hybrid Retrieval Implementation)

"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

# Import factories (ONLY factories, never concrete implementations!)
from core.factories.chunking_factory import ChunkingFactory
from core.factories.embedding_factory import EmbeddingFactory
from core.factories.vectorstore_factory import VectorStoreFactory
from core.factories.retrieval_factory import RetrievalFactory

# Import interfaces and models
from core.interfaces.chunking_interface import ChunkerInterface
from core.interfaces.embedding_interface import EmbeddingInterface
from core.interfaces.vectorstore_interface import VectorStoreInterface
from core.interfaces.retrieval_interface import RetrievalInterface
from models.metadata_models import ChunkMetadata

# Configure logging
logger = logging.getLogger(__name__)


class DocumentPipeline:
    """
    Orchestrates document processing workflows for Phase 2 architecture.

    This pipeline:
    - Initializes all components via factories (config-driven)
    - Executes ingestion workflow (chunk → embed → store)
    - Executes multi-strategy retrieval
    - Manages document lifecycle (deprecation, listing)

    All components are created from domain configuration using factories.
    No direct instantiation of concrete classes!
    """

    def __init__(self, domain_config: Any):
        """
        Initialize pipeline with domain configuration.

        This method:
        1. Stores domain config
        2. Creates embedding model via factory
        3. Creates chunker via factory (needs embedding model name)
        4. Creates vector store via factory (needs embedding dimension)
        5. Creates retrieval strategies via factory

        Parameters:
        -----------
        domain_config : Any
            Domain configuration object (usually Pydantic model)
            Must contain:
            - embeddings: Embedding provider config
            - chunking: Chunking strategy config
            - vectorstore: Vector store config
            - retrieval: Retrieval strategies config (optional)

        Raises:
        -------
        ValueError:
            If config is missing required sections
        RuntimeError:
            If factory initialization fails
        """
        self.config = domain_config
        self.domain_id = getattr(domain_config, 'domain_id', 'default')

        logger.info(f"Initializing DocumentPipeline for domain: {self.domain_id}")

        # Step 1: Create embedding model (needed first for dimension)
        logger.debug("Creating embedding model via EmbeddingFactory...")
        self.embedding_model: EmbeddingInterface = EmbeddingFactory.create_embedder(
            config=self.config.embeddings  # Note: 'embeddings' not 'embedding'
        )

        embedding_model_name = self.embedding_model.get_model_name()
        embedding_dimension = self.embedding_model.get_embedding_dimension()

        logger.info(
            f"✅ Embedding model created: {embedding_model_name} "
            f"({embedding_dimension}-dim)"
        )

        # Step 2: Create chunker (needs embedding model name for metadata)
        logger.debug("Creating chunker via ChunkingFactory...")
        self.chunker: ChunkerInterface = ChunkingFactory.create_chunker(
            config=self.config.chunking,
            embedding_model_name=embedding_model_name
        )

        logger.info(f"✅ Chunker created: {self.config.chunking.strategy}")

        # Step 3: Define metadata fields schema
        self.metadata_fields = [
            # Identity
            "doc_id", "chunk_id", "title", "domain", "doc_type",
            # Provenance
            "author", "uploader_id", "upload_timestamp",
            "source_file", "source_file_hash",
            # Versioning
            "version", "document_version",
            # Processing
            "embedding_model_name", "embedding_dimension",
            "chunking_strategy", "chunking_params", "processing_timestamp",
            # Lifecycle
            "deprecated", "deprecated_date", "deprecation_reason",
            # Quality
            "authority_level", "review_status", "confidence_score",
            # Content
            "page_num", "char_range", "chunk_text"
        ]

        # Step 4: Create vector store (needs dimension and metadata schema)
        logger.debug("Creating vector store via VectorStoreFactory...")
        self.vectorstore: VectorStoreInterface = VectorStoreFactory.create_store(
            config=self.config.vectorstore,
            embedding_dimension=embedding_dimension,
            metadata_fields=self.metadata_fields
        )

        logger.info(f"✅ Vector store created: {self.config.vectorstore.provider}")

        # Step 5: Create retrieval strategies
        logger.debug("Creating retrieval strategies via RetrievalFactory...")
        self.retrieval_strategies = self._init_retrieval_strategies()

        logger.info(
            f"✅ DocumentPipeline initialized for domain '{self.domain_id}' "
            f"with {len(self.retrieval_strategies)} retrieval strategies"
        )

    def _init_retrieval_strategies(self) -> Dict[str, RetrievalInterface]:
        """
        Initialize all configured retrieval strategies.

        Creates retriever instances for each strategy in config.
        Supports: vector_similarity, hybrid, bm25, etc.

        Returns:
        --------
        Dict[str, RetrievalInterface]:
            Dictionary mapping strategy name to retriever instance
            Example: {"hybrid": HybridRetrieval(...), "vector_similarity": VectorSimilarityRetrieval(...)}
        """
        retrieval_config = getattr(self.config, 'retrieval', None)
        if not retrieval_config:
            logger.warning("No retrieval config found, using default vector_similarity")
            retrieval_config = {"strategies": ["vector_similarity"]}

        # Get list of strategies to initialize
        strategies = getattr(retrieval_config, 'strategies', ["vector_similarity"])
        if isinstance(strategies, str):
            strategies = [strategies]

        retrievers = {}
        bm25_index = None  # Shared BM25 index for hybrid retrieval

        for strategy_name in strategies:
            try:
                # Get strategy-specific config
                strategy_config = getattr(retrieval_config, strategy_name, {})
                if isinstance(strategy_config, dict):
                    strategy_config['strategy'] = strategy_name

                # Build BM25 index if needed (for hybrid/bm25 strategies)
                if strategy_name in ["hybrid", "bm25"] and bm25_index is None:
                    logger.info("Building BM25 index for sparse retrieval...")
                    # BM25 index built by factory from vector store corpus
                    bm25_index = None  # Factory will build it

                # Create retriever via factory
                retriever = RetrievalFactory.create_retriever(
                    config=strategy_config,
                    vectorstore=self.vectorstore,
                    embedding_model=self.embedding_model,
                    bm25_index=bm25_index
                )

                retrievers[strategy_name] = retriever
                logger.info(f"✅ Loaded retrieval strategy: {strategy_name}")

            except Exception as e:
                logger.error(f"Failed to initialize strategy '{strategy_name}': {e}")
                # Continue with other strategies instead of failing completely
                continue

        if not retrievers:
            raise RuntimeError("Failed to initialize any retrieval strategies")

        return retrievers

    def process_document(
            self,
            text: str,
            doc_id: str,
            domain: str,
            source_file_path: str,
            file_hash: str,
            uploader_id: str = None,
            title: str = None,
            doc_type: str = "document",
            author: str = None,
            version: str = "1.0",
            replace_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Process document end-to-end: chunk → embed → store.

        This is the CORE ingestion workflow for Phase 2.

        Workflow:
        ---------
        1. Delete existing chunks if replace_existing=True
        2. Chunk document using configured strategy
        3. Embed chunks using configured embedding model
        4. Attach comprehensive metadata to each chunk
        5. Upsert chunks + embeddings + metadata to vector store
        6. Return ingestion summary

        Parameters:
        -----------
        text : str
            Full document text to process
        doc_id : str
            Unique document identifier
        domain : str
            Domain/department (hr, finance, legal, etc.)
        source_file_path : str
            Original file path for provenance
        file_hash : str
            SHA-256 hash of source file
        uploader_id : str, optional
            User who uploaded the document
        title : str, optional
            Document title
        doc_type : str
            Document type (policy, faq, manual, etc.)
        author : str, optional
            Original document author
        version : str
            Document version (default: "1.0")
        replace_existing : bool
            If True, delete existing doc before ingestion

        Returns:
        --------
        Dict[str, Any]:
            Ingestion summary with:
            - doc_id: Document identifier
            - chunks_ingested: Number of chunks created
            - status: "success"
            - embedding_model: Model name used
            - chunking_strategy: Strategy used

        Raises:
        -------
        ValueError:
            If required parameters are missing or invalid
        RuntimeError:
            If chunking, embedding, or storage fails

        Example:
        --------
        result = pipeline.process_document(
            text="Employee benefits include...",
            doc_id="handbook_2025",
            domain="hr",
            source_file_path="./docs/handbook.pdf",
            file_hash="abc123...",
            uploader_id="admin@company.com",
            title="Employee Handbook 2025",
            doc_type="policy",
            replace_existing=True
        )

        print(f"Ingested {result['chunks_ingested']} chunks")
        """
        logger.info(f"Processing document: doc_id={doc_id}, domain={domain}")

        # Step 1: Delete existing if requested
        if replace_existing:
            logger.info(f"Deleting existing chunks for doc_id: {doc_id}")
            try:
                self.vectorstore.delete_by_doc_id(doc_id)
                logger.info(f"✅ Deleted existing chunks for doc_id: {doc_id}")
            except Exception as e:
                logger.warning(f"No existing chunks to delete: {e}")

        # Step 2: Chunk document
        logger.info(f"Chunking document with strategy: {self.config.chunking.strategy}")
        chunks: List[ChunkMetadata] = self.chunker.chunk_text(
            text=text,
            doc_id=doc_id,
            domain=domain,
            source_file_path=source_file_path,
            file_hash=file_hash,
            uploader_id=uploader_id
        )

        if not chunks:
            logger.warning(f"No chunks created for doc_id: {doc_id}")
            return {
                "doc_id": doc_id,
                "chunks_ingested": 0,
                "status": "no_chunks_created"
            }

        logger.info(f"✅ Created {len(chunks)} chunks")

        # Step 3: Extract chunk texts for embedding
        chunk_texts = [chunk.chunk_text for chunk in chunks]

        # Step 4: Embed chunks
        logger.info(f"Embedding {len(chunk_texts)} chunks...")
        import numpy as np
        embeddings: np.ndarray = self.embedding_model.embed_texts(chunk_texts)

        logger.info(f"✅ Generated embeddings: shape={embeddings.shape}")

        # Step 5: Enrich metadata with additional fields
        # ChunkMetadata from chunker has basic fields, we can add more here if needed
        # For now, chunker provides complete metadata

        # Step 6: Upsert to vector store
        logger.info(f"Upserting {len(chunks)} chunks to vector store...")
        self.vectorstore.upsert(chunks=chunks, embeddings=embeddings)

        logger.info(f"✅ Successfully upserted {len(chunks)} chunks for doc_id: {doc_id}")

        # Step 7: Return summary
        return {
            "doc_id": doc_id,
            "chunks_ingested": len(chunks),
            "status": "success",
            "embedding_model": self.embedding_model.get_model_name(),
            "chunking_strategy": self.config.chunking.strategy,
            "embedding_dimension": self.embedding_model.get_embedding_dimension()
        }

    def query(
            self,
            query_text: str,
            strategy_name: Optional[str] = None,
            metadata_filters: Optional[Dict[str, Any]] = None,
            top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Execute multi-strategy retrieval query.

        If strategy_name is None, queries ALL configured strategies and
        aggregates results. If specified, uses only that strategy.

        Parameters:
        -----------
        query_text : str
            Natural language query
            Example: "How many vacation days do employees get?"

        strategy_name : str, optional
            Specific strategy to use ("hybrid", "vector_similarity", etc.)
            If None, uses all configured strategies

        metadata_filters : Dict[str, Any], optional
            Metadata filters to apply
            Example: {"domain": "hr", "deprecated": False, "doc_type": "policy"}

        top_k : int
            Number of results to return per strategy
            Default: 10

        Returns:
        --------
        List[Dict[str, Any]]:
            List of results, each with:
            - id: chunk_id
            - score: similarity score
            - metadata: chunk metadata dictionary
            - document: chunk text
            - strategy: which strategy returned this (if multi-strategy)

        Example:
        --------
        # Single strategy
        results = pipeline.query(
            query_text="vacation policy",
            strategy_name="hybrid",
            metadata_filters={"domain": "hr", "deprecated": False},
            top_k=5
        )

        # All strategies (aggregated)
        results = pipeline.query(
            query_text="vacation policy",
            metadata_filters={"domain": "hr"},
            top_k=10
        )

        for result in results:
            print(f"{result['score']:.3f}: {result['document'][:100]}...")
        """
        logger.info(
            f"Query: '{query_text}' "
            f"(strategy={strategy_name or 'ALL'}, filters={metadata_filters}, top_k={top_k})"
        )

        # Single strategy query
        if strategy_name:
            retriever = self.retrieval_strategies.get(strategy_name)
            if not retriever:
                available = list(self.retrieval_strategies.keys())
                raise ValueError(
                    f"Retrieval strategy '{strategy_name}' not found. "
                    f"Available: {available}"
                )

            results = retriever.retrieve(
                query_text=query_text,
                metadata_filters=metadata_filters,
                top_k=top_k
            )

            logger.info(f"✅ Strategy '{strategy_name}' returned {len(results)} results")
            return results

        # Multi-strategy query (aggregate all strategies)
        else:
            aggregated_results = []

            for name, retriever in self.retrieval_strategies.items():
                try:
                    results = retriever.retrieve(
                        query_text=query_text,
                        metadata_filters=metadata_filters,
                        top_k=top_k
                    )

                    # Tag results with strategy name
                    for result in results:
                        result['strategy'] = name

                    aggregated_results.extend(results)
                    logger.debug(f"Strategy '{name}' returned {len(results)} results")

                except Exception as e:
                    logger.error(f"Strategy '{name}' failed: {e}")
                    continue

            # Optional: Deduplicate by chunk_id (keep highest score)
            deduped = self._deduplicate_results(aggregated_results)

            logger.info(
                f"✅ Queried {len(self.retrieval_strategies)} strategies, "
                f"returned {len(deduped)} deduplicated results"
            )

            return deduped

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """
        Remove duplicate chunks, keeping highest score.

        When multiple strategies return the same chunk, keep the one
        with the highest score.
        """
        seen = {}
        for result in results:
            chunk_id = result.get('id')
            score = result.get('score', 0)

            if chunk_id not in seen or score > seen[chunk_id]['score']:
                seen[chunk_id] = result

        # Sort by score descending
        return sorted(seen.values(), key=lambda x: x.get('score', 0), reverse=True)

    def delete_document(self, doc_id: str) -> None:
        """
        Delete all chunks for a document.

        Parameters:
        -----------
        doc_id : str
            Document identifier

        Example:
        --------
        pipeline.delete_document("old_handbook_2023")
        """
        logger.info(f"Deleting document: {doc_id}")
        self.vectorstore.delete_by_doc_id(doc_id)
        logger.info(f"✅ Deleted all chunks for doc_id: {doc_id}")

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

        Note:
        -----
        Implementation depends on vector store capabilities.
        May need to search and aggregate chunk metadata.
        """
        logger.info(f"Fetching document info: {doc_id}")
        # This would need vector store support or search + aggregate
        raise NotImplementedError("get_document_info not yet implemented")


    def list_documents(self, filters: dict | None = None) -> list[dict]:
        # Implement via your vector store / metadata backend
        raise NotImplementedError

    def list_chunks(
        self,
        doc_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict]:
        # Implement via your vector store / metadata backend
        raise NotImplementedError


    def list_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Return document-level metadata aggregated from the vector store.

        This is used by the Playground "Corpus Explorer" to show:
        - doc_id
        - title
        - doc_type
        - domain
        - uploader_id
        - version
        - file_hash
        - chunk_count
        - first_seen / last_seen timestamps
        - deprecated flag

        Parameters
        ----------
        filters : dict, optional
            Simple equality filters on document-level fields
            Example: {"deprecated": False, "doc_type": "policy"}

        Returns
        -------
        List[Dict[str, Any]]:
            One dict per document.
        """
        logger.info(
            f"Listing documents for domain={self.domain_id}, filters={filters}"
        )

        # 1) If vector store has a native list_documents(), just use it
        if hasattr(self.vectorstore, "list_documents"):
            logger.debug("Using vectorstore.list_documents()")
            return self.vectorstore.list_documents(filters=filters)

        # 2) Generic fallback: scan all metadata from the underlying collection
        collection = getattr(self.vectorstore, "collection", None)
        if collection is None:
            raise NotImplementedError(
                "Vector store does not support list_documents and has no 'collection' handle."
            )

        raw = collection.get(include=["metadatas"])
        metadatas = raw.get("metadatas") or []

        docs: Dict[str, Dict[str, Any]] = {}

        for md in metadatas:
            if not md:
                continue
            doc_id = md.get("doc_id")
            if not doc_id:
                continue

            # Initialize record if first time
            record = docs.setdefault(
                doc_id,
                {
                    "doc_id": doc_id,
                    "title": md.get("title"),
                    "doc_type": md.get("doc_type"),
                    "domain": md.get("domain"),
                    "uploader_id": md.get("uploader_id"),
                    "version": md.get("version"),
                    "file_hash": md.get("source_file_hash")
                    or md.get("file_hash"),
                    "chunk_count": 0,
                    "first_seen": None,
                    "last_seen": None,
                    "deprecated": md.get("deprecated", False),
                },
            )

            # Increment chunk count
            record["chunk_count"] += 1

            # Track timestamps (upload or processing)
            ts = md.get("upload_timestamp") or md.get("processing_timestamp")
            if ts:
                if record["first_seen"] is None or ts < record["first_seen"]:
                    record["first_seen"] = ts
                if record["last_seen"] is None or ts > record["last_seen"]:
                    record["last_seen"] = ts

        # Apply simple equality filters in Python if provided
        if filters:
            def _match(rec: Dict[str, Any]) -> bool:
                for k, v in filters.items():
                    if rec.get(k) != v:
                        return False
                return True

            doc_list = [d for d in docs.values() if _match(d)]
        else:
            doc_list = list(docs.values())

        # Sort newest first
        doc_list.sort(key=lambda d: d.get("last_seen") or "", reverse=True)

        logger.info(f"list_documents: returning {len(doc_list)} docs.")
        return doc_list


    def list_chunks(
        self,
        doc_id: str,
        *,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Return chunk-level metadata for a given document.

        Used by the Playground 'Chunks' view to show:
        - id (chunk_id)
        - doc_id
        - text
        - page_num
        - char_start / char_end (or char_range)
        - full metadata

        Parameters
        ----------
        doc_id : str
            Document identifier
        limit : int, optional
            Max number of chunks to return
        offset : int
            Offset into the result set (for pagination)

        Returns
        -------
        List[Dict[str, Any]]:
            One dict per chunk.

        Raises
        ------
        ValueError:
            If doc_id is missing
        """
        if not doc_id:
            raise ValueError("doc_id is required for list_chunks")

        logger.info(
            f"Listing chunks for doc_id={doc_id}, limit={limit}, offset={offset}"
        )

        # 1) If vector store has a native list_chunks(), use that
        if hasattr(self.vectorstore, "list_chunks"):
            logger.debug("Using vectorstore.list_chunks()")
            return self.vectorstore.list_chunks(
                doc_id=doc_id, limit=limit, offset=offset
            )

        # 2) Generic fallback: query underlying collection by metadata
        collection = getattr(self.vectorstore, "collection", None)
        if collection is None:
            raise NotImplementedError(
                "Vector store does not support list_chunks and has no 'collection' handle."
            )

        results = collection.get(
            where={"doc_id": doc_id},
            include=["ids", "documents", "metadatas"],
        )

        ids = results.get("ids") or []
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []

        n = len(ids)
        if n == 0:
            logger.warning(f"No chunks found for doc_id={doc_id}")
            return []

        start = max(0, offset)
        end = n if limit is None else min(n, offset + limit)

        out: List[Dict[str, Any]] = []

        for i in range(start, end):
            md = metas[i] or {}
            text = docs[i] if i < len(docs) else ""

            # Try to derive char_start / char_end
            char_start = md.get("char_start")
            char_end = md.get("char_end")

            # If only char_range is stored (like [start, end]), use that
            char_range = md.get("char_range")
            if char_range and len(char_range) == 2:
                if char_start is None:
                    char_start = char_range[0]
                if char_end is None:
                    char_end = char_range[1]

            out.append(
                {
                    "id": ids[i],
                    "doc_id": md.get("doc_id", doc_id),
                    "chunk_index": md.get("chunk_index", i),
                    "text": text,
                    "page_num": md.get("page_num"),
                    "char_start": char_start,
                    "char_end": char_end,
                    "metadata": md,
                }
            )

        logger.info(f"list_chunks: returning {len(out)} chunks for doc_id={doc_id}")
        return out


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of DocumentPipeline usage.
    Run: python core/pipeline/document_pipeline.py
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("DocumentPipeline Usage Examples")
    print("=" * 70)

    # Note: This requires actual config files and dependencies
    # This is a conceptual example showing the API

    print("""
Example 1: Initialize Pipeline
-------------------------------
from core.config_manager import ConfigManager

config_manager = ConfigManager()
hr_config = config_manager.load_domain_config("hr")
pipeline = DocumentPipeline(hr_config)

print(f"Pipeline initialized for domain: {hr_config.domain_id}")
print(f"Embedding model: {pipeline.embedding_model.get_model_name()}")
print(f"Chunking strategy: {pipeline.config.chunking.strategy}")
print(f"Retrieval strategies: {list(pipeline.retrieval_strategies.keys())}")


Example 2: Process Document
----------------------------
result = pipeline.process_document(
    text="Employee benefits include 15 vacation days per year...",
    doc_id="handbook_2025",
    domain="hr",
    source_file_path="./docs/employee_handbook_2025.pdf",
    file_hash="abc123def456...",
    uploader_id="admin@company.com",
    title="Employee Handbook 2025",
    doc_type="policy",
    author="HR Department",
    version="2.0",
    replace_existing=True
)

print(f"Ingestion result: {result}")
# Output: {'doc_id': 'handbook_2025', 'chunks_ingested': 42, 'status': 'success', ...}


Example 3: Query with Hybrid Retrieval
---------------------------------------
results = pipeline.query(
    query_text="How many vacation days do employees get?",
    strategy_name="hybrid",  # Use hybrid (dense + sparse)
    metadata_filters={
        "domain": "hr",
        "doc_type": "policy",
        "deprecated": False,
        "authority_level": "official"
    },
    top_k=5
)

for i, result in enumerate(results, 1):
    print(f"{i}. Score: {result['score']:.3f}")
    print(f"   Text: {result['document'][:100]}...")
    print(f"   Metadata: domain={result['metadata']['domain']}, "
          f"doc_type={result['metadata']['doc_type']}")


Example 4: Query All Strategies
--------------------------------
# Query with all configured strategies and aggregate results
results = pipeline.query(
    query_text="vacation policy",
    metadata_filters={"domain": "hr"},
    top_k=10
)

# Results are automatically deduplicated
print(f"Total results from all strategies: {len(results)}")

# See which strategy contributed each result
for result in results[:5]:
    print(f"{result['score']:.3f} [{result['strategy']}]: {result['document'][:80]}...")


Example 5: Delete Document
---------------------------
# Delete old version before uploading new
pipeline.delete_document("handbook_2023")
print("Old handbook deleted")

# Upload new version
result = pipeline.process_document(...)
print(f"New handbook uploaded: {result['chunks_ingested']} chunks")
    """)

    print("\n" + "=" * 70)
    print("DocumentPipeline examples completed!")
    print("=" * 70)
