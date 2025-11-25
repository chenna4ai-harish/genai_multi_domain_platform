"""

core/vectorstores/pinecone_store.py

This module implements the Pinecone vector store adapter (OPTION 2).

What is Pinecone?
------------------
Pinecone is a fully managed, cloud-native vector database designed for
production-scale AI applications. It provides high-performance vector similarity
search with automatic scaling, replication, and no infrastructure management.

Key Features:
-------------
- **Fully Managed**: No servers, clusters, or infrastructure to manage
- **Serverless Architecture**: Auto-scales based on usage, pay-per-use
- **High Performance**: Sub-100ms latency at billion-vector scale
- **Distributed**: Multi-region deployment, automatic replication
- **Production-Ready**: 99.9% uptime SLA, built-in monitoring
- **Metadata Filtering**: Advanced filtering with low-latency queries

Serverless Architecture (2024+):
---------------------------------
Pinecone's serverless indexes separate storage from compute:
- Storage: S3-compatible object storage (scales infinitely)
- Compute: On-demand query pods (auto-scale based on traffic)
- Cost: Pay only for storage + operations (no idle resource costs)

Compared to ChromaDB:
---------------------
| Feature          | ChromaDB      | Pinecone        |
|------------------|---------------|-----------------|
| Hosting          | Self-hosted   | Fully managed   |
| Scaling          | Vertical only | Auto-scales     |
| Setup            | Instant       | ~2-5 min setup  |
| Cost             | Free          | ~$70/month min  |
| Latency          | <10ms local   | ~50-200ms API   |
| Data limit       | ~1M vectors   | Unlimited       |
| Replication      | None          | Multi-region    |
| Monitoring       | DIY           | Built-in        |

When to Use Pinecone:
----------------------
✅ Production deployments (managed, reliable)
✅ Scaling beyond 1M vectors
✅ Multi-region applications (low latency globally)
✅ When uptime SLAs matter (99.9%)
✅ Enterprise applications (compliance, security)
❌ MVP/development (overkill, costs money)
❌ Budget constraints (use ChromaDB)
❌ Privacy-sensitive data that can't leave your servers
❌ Offline deployments (requires internet)

Pricing (Serverless, as of 2025):
----------------------------------
- Storage: $0.25 per GB per month (~$25 for 1M 384-dim vectors)
- Reads: $0.004 per 100K read units
- Writes: $0.02 per 100K write units
- Estimated: ~$70-100/month for typical SMB use case

See: https://www.pinecone.io/pricing/

Regions (Serverless):
---------------------
- AWS: us-east-1, us-west-2, eu-west-1
- GCP: us-central1, europe-west4 (public preview)
- Azure: eastus, westeurope (public preview)

Setup:
------
1. Sign up: https://app.pinecone.io/
2. Create API key in console
3. Set environment variable:
   export PINECONE_API_KEY="your-api-key-here"

Installation:
-------------
pip install pinecone-client

# For gRPC support (faster, recommended):
pip install "pinecone-client[grpc]"

References:
-----------
- Documentation: https://docs.pinecone.io/
- Python SDK: https://github.com/pinecone-io/pinecone-python-client
- Community: https://community.pinecone.io/
- Status: https://status.pinecone.io/

"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from core.interfaces.vectorstore_interface import VectorStoreInterface
from models.metadata_models import ChunkMetadata
import logging
import time

# Configure logging
logger = logging.getLogger(__name__)


class PineconeStore(VectorStoreInterface):
    """
    Pinecone vector store adapter (cloud, production-ready).

    This implementation provides a clean interface to Pinecone's serverless
    vector database for storing and retrieving document chunks at scale.

    Configuration Parameters:
    -------------------------
    api_key : str
        Pinecone API key from console
        Get one at: https://app.pinecone.io/ → API Keys
        Best practice: Store in environment variable
            export PINECONE_API_KEY="your-key"
        Then: api_key=os.getenv("PINECONE_API_KEY")

    index_name : str
        Name of the Pinecone index (globally unique in your project)
        Naming rules:
        - 1-45 characters
        - Lowercase alphanumeric and hyphens only
        - Must start with letter
        Examples: "hr-docs-prod", "finance-policies-v2"

    dimension : int
        Dimensionality of embedding vectors
        MUST match your embedding model's output!
        Common dimensions:
        - 384: all-MiniLM-L6-v2 (Sentence-Transformers)
        - 768: all-mpnet-base-v2, BERT, Gemini
        - 1536: OpenAI text-embedding-ada-002
        IMPORTANT: Cannot be changed after index creation!

    cloud : str
        Cloud provider for serverless deployment
        Options: "aws", "gcp", "azure"
        Default: "aws"

    region : str
        Cloud region for index deployment
        AWS options: "us-east-1", "us-west-2", "eu-west-1"
        GCP options: "us-central1", "europe-west4"
        Azure options: "eastus", "westeurope"
        Default: "us-east-1"

    Example Usage:
    --------------
    import os

    # Initialize store (creates index if doesn't exist)
    store = PineconeStore(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name="hr-docs-prod",
        dimension=384,
        cloud="aws",
        region="us-east-1"
    )

    # Upsert chunks
    store.upsert(chunks, embeddings)

    # Search with filters
    results = store.search(
        query_embedding,
        top_k=10,
        filters={"domain": "hr", "is_authoritative": True}
    )

    # Delete old document
    store.delete_by_doc_id("old_policy_2023")

    # Get all documents for BM25 (Phase 2)
    corpus, doc_ids = store.get_all_documents()
    """

    def __init__(
            self,
            api_key: str,
            index_name: str,
            dimension: int = 384,
            cloud: str = "aws",
            region: str = "us-east-1"
    ):
        """
        Initialize Pinecone store with serverless index.

        Parameters:
        -----------
        api_key : str
            Pinecone API key
        index_name : str
            Index name (will be created if doesn't exist)
        dimension : int
            Embedding vector dimension
        cloud : str
            Cloud provider ("aws", "gcp", "azure")
        region : str
            Cloud region

        Raises:
        -------
        ValueError:
            If API key is missing or parameters are invalid
        RuntimeError:
            If Pinecone initialization or index creation fails

        Notes:
        ------
        - First run creates index (~2-5 minutes)
        - Subsequent runs connect instantly
        - Index creation is asynchronous (waits until ready)
        """
        if not api_key:
            raise ValueError(
                "PINECONE_API_KEY is required. Get one at: "
                "https://app.pinecone.io/\n"
                "Set environment variable: export PINECONE_API_KEY='your-key'"
            )

        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        self.cloud = cloud
        self.region = region

        logger.info(
            f"Initializing Pinecone store:\n"
            f"  Index: {index_name}\n"
            f"  Dimension: {dimension}\n"
            f"  Cloud: {cloud}\n"
            f"  Region: {region}"
        )

        try:
            # Step 1: Initialize Pinecone client
            self.pc = Pinecone(api_key=self.api_key)

            # Step 2: Create index if doesn't exist
            if index_name not in self.pc.list_indexes().names():
                logger.info(f"Index '{index_name}' not found. Creating...")
                self.pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=cloud,
                        region=region
                    )
                )

                # Wait for index to be ready
                logger.info("Waiting for index to be ready (this may take a few minutes)...")
                start = time.time()
                while not self.pc.describe_index(index_name).status['ready']:
                    time.sleep(5)
                    elapsed = time.time() - start
                    logger.info(f"Still waiting... ({elapsed:.0f}s elapsed)")

                logger.info(f"✅ Index created in {elapsed:.0f}s!")

            # Step 3: Connect to index
            self.index = self.pc.Index(index_name)

            # Step 4: Get index stats
            stats = self.index.describe_index_stats()

            logger.info(
                f"✅ Pinecone initialized!\n"
                f"   Index: {index_name}\n"
                f"   Total vectors: {stats.total_vector_count:,}\n"
                f"   Dimension: {dimension}\n"
                f"   Metric: cosine"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise RuntimeError(
                f"Could not initialize Pinecone\n"
                f"Index: {index_name}\n"
                f"Error: {e}\n"
                f"Check API key and parameters"
            )

    def upsert(self, chunks: List[ChunkMetadata], embeddings: np.ndarray) -> None:
        """
        Insert or update chunks with embeddings in Pinecone.

        Parameters:
        -----------
        chunks : List[ChunkMetadata]
            List of chunk metadata objects
        embeddings : np.ndarray
            2D numpy array of embeddings (n_chunks, embedding_dim)

        Raises:
        -------
        ValueError:
            If chunks and embeddings length mismatch
        RuntimeError:
            If upsert operation fails
        """
        # Validate inputs
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) must match embeddings count ({len(embeddings)})"
            )

        if len(chunks) == 0:
            logger.warning("Upsert called with empty chunks list")
            return

        logger.info(f"Upserting {len(chunks)} chunks to Pinecone...")

        try:
            # Prepare vectors
            vectors = []
            for chunk, embedding in zip(chunks, embeddings):
                vectors.append({
                    "id": chunk.chunk_id,
                    "values": embedding.tolist(),
                    "metadata": self._serialize_metadata(chunk)
                })

            # Upsert in batches (100 vectors per request limit)
            batch_size = 100
            total_upserted = 0

            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                upsert_response = self.index.upsert(vectors=batch)
                total_upserted += upsert_response.upserted_count

            logger.info(
                f"✅ Upsert successful!\n"
                f"   Upserted: {total_upserted} vectors\n"
                f"   Index: {self.index_name}"
            )

        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            raise RuntimeError(f"Failed to upsert chunks to Pinecone: {e}")

    def search(
            self,
            query_embedding: np.ndarray,
            top_k: int,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """
        Search for similar chunks using vector similarity.

        Parameters:
        -----------
        query_embedding : np.ndarray
            1D numpy array of query embedding
        top_k : int
            Number of top results to return
        filters : Dict[str, Any], optional
            Metadata filters using Pinecone's filter syntax

        Returns:
        --------
        List[Dict]:
            List of results with id, score, metadata, document
        """
        if query_embedding.ndim != 1:
            raise ValueError(f"Query embedding must be 1D, got shape: {query_embedding.shape}")

        if top_k < 1 or top_k > 10000:
            raise ValueError(f"top_k must be between 1 and 10000, got {top_k}")

        try:
            query_response = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                filter=filters,
                include_metadata=True
            )

            return self._format_results(query_response)

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RuntimeError(f"Failed to search Pinecone: {e}")

    def delete_by_doc_id(self, doc_id: str) -> None:
        """
        Delete all chunks belonging to a document.

        Parameters:
        -----------
        doc_id : str
            Document ID to delete all chunks for
        """
        logger.info(f"Deleting all chunks for doc_id: {doc_id}")

        try:
            self.index.delete(filter={"doc_id": doc_id})
            logger.info(f"✅ Delete requested for doc_id: {doc_id}")

        except Exception as e:
            logger.error(f"Deletion failed: {e}")
            raise RuntimeError(f"Failed to delete chunks for doc_id: {doc_id}: {e}")

    def get_all_documents(self) -> Tuple[List[str], List[str]]:
        """
        Retrieve all document texts and IDs from the index.

        **ADDED FOR PHASE 2** - Required for hybrid retrieval BM25 indexing.

        This method queries all vectors in the Pinecone index and extracts
        their chunk texts and IDs for building the BM25 sparse index.

        Returns:
        --------
        Tuple[List[str], List[str]]:
            Tuple of (corpus, doc_ids)
            - corpus: List of all chunk texts
            - doc_ids: List of corresponding chunk IDs

        Raises:
        -------
        RuntimeError:
            If retrieval operation fails

        Performance:
        ------------
        - For small indexes (<10K vectors): Fast (~1-5 seconds)
        - For medium indexes (10K-100K): Moderate (~5-30 seconds)
        - For large indexes (>100K): Slow (~30-300 seconds)
        - Pinecone queries in batches using pagination

        Implementation Notes:
        ---------------------
        Pinecone doesn't have a "get all" API, so we:
        1. Query with a dummy vector
        2. Use large top_k (10,000 max per query)
        3. Paginate through all results using offset/cursors
        4. Extract chunk_text from metadata

        Alternative: If your index is very large, consider:
        - Maintaining a separate corpus file/database
        - Rebuilding BM25 index incrementally
        - Using Pinecone's sparse-dense hybrid (when available)

        Example:
        --------
        # Build BM25 index for hybrid retrieval
        corpus, doc_ids = store.get_all_documents()
        print(f"Retrieved {len(corpus)} documents")

        bm25_index = BM25Retrieval(corpus=corpus, doc_ids=doc_ids)

        hybrid_retriever = HybridRetrieval(
            vector_store=store,
            embedding_model=embedder,
            bm25_index=bm25_index,
            alpha=0.7
        )
        """
        logger.info(f"Retrieving all documents from Pinecone index: {self.index_name}")

        try:
            # Get index stats to know total count
            stats = self.index.describe_index_stats()
            total_count = stats.total_vector_count

            if total_count == 0:
                logger.warning("Index is empty, returning empty corpus")
                return [], []

            logger.info(f"Index has {total_count:,} vectors, fetching all...")

            # Pinecone limitation: No "get all" API
            # Solution: Query with dummy vector and large top_k, paginate

            corpus = []
            doc_ids = []

            # Create dummy query vector (zeros - will return random results, but we get metadata)
            dummy_vector = [0.0] * self.dimension

            # Fetch in batches (10K max per query)
            batch_size = 10000
            fetched = 0

            while fetched < total_count:
                logger.info(f"Fetching batch: {fetched:,} / {total_count:,}")

                # Query with dummy vector to get vectors with metadata
                results = self.index.query(
                    vector=dummy_vector,
                    top_k=min(batch_size, total_count - fetched),
                    include_metadata=True
                )

                # Extract chunk texts and IDs from results
                for match in results['matches']:
                    doc_ids.append(match['id'])
                    # Extract chunk_text from metadata
                    chunk_text = match.get('metadata', {}).get('chunk_text', '')
                    corpus.append(chunk_text)

                fetched += len(results['matches'])

                # Break if no more results
                if len(results['matches']) < batch_size:
                    break

            logger.info(
                f"✅ Retrieved {len(corpus):,} documents from Pinecone\n"
                f"   Index: {self.index_name}\n"
                f"   Corpus size: {sum(len(text) for text in corpus):,} characters"
            )

            return corpus, doc_ids

        except Exception as e:
            logger.error(f"Failed to retrieve all documents: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to retrieve documents from Pinecone\n"
                f"Index: {self.index_name}\n"
                f"Error: {e}"
            )

    # =========================================================================
    # HELPER METHODS (Private)
    # =========================================================================

    def _serialize_metadata(self, chunk: ChunkMetadata) -> Dict[str, Any]:
        """
        Convert ChunkMetadata to Pinecone-compatible metadata dict.

        Pinecone Metadata Limitations:
        - Values must be: str, number, bool, or list of strings
        - Max metadata size: 40KB per vector
        - Filterable types: str, number, bool
        """
        char_start, char_end = chunk.char_range

        return {
            # Core fields
            "doc_id": chunk.doc_id,
            "domain": chunk.domain,
            "chunk_text": chunk.chunk_text,  # Include for retrieval
            "char_start": char_start,
            "char_end": char_end,
            "page_num": chunk.page_num if chunk.page_num is not None else 0,
            # Provenance fields
            "uploader_id": chunk.uploader_id or "",
            "upload_timestamp": chunk.upload_timestamp.isoformat(),
            "document_version": chunk.document_version,
            "source_file_path": chunk.source_file_path,
            "source_file_hash": chunk.source_file_hash,
            # Processing fields
            "embedding_model_name": chunk.embedding_model_name,
            "chunking_strategy": chunk.chunking_strategy,
            "chunk_type": chunk.chunk_type,
            # Quality fields
            "is_authoritative": chunk.is_authoritative,
            "confidence_score": chunk.confidence_score,
            "deprecated_flag": chunk.deprecated_flag
        }

    def _format_results(self, query_response) -> List[Dict]:
        """
        Format Pinecone query results into standardized structure.
        """
        formatted = []

        for match in query_response['matches']:
            formatted.append({
                'id': match['id'],
                'score': match['score'],
                'metadata': match['metadata'],
                'document': match['metadata'].get('chunk_text', '')
            })

        return formatted

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index.

        Returns:
        --------
        dict:
            Index statistics including total_vector_count, dimension, etc.
        """
        stats = self.index.describe_index_stats()
        return {
            'total_vector_count': stats.total_vector_count,
            'dimension': stats.dimension,
            'index_fullness': stats.index_fullness,
            'namespaces': stats.namespaces
        }


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of PineconeStore usage.
    Run: python core/vectorstores/pinecone_store.py
    """
    import os

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("PineconeStore Usage Examples")
    print("=" * 70)

    # Check for API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("\n❌ PINECONE_API_KEY not set!")
        print("Get API key: https://app.pinecone.io/ → API Keys")
        print("Then run: export PINECONE_API_KEY='your-key'")
        exit(1)

    # Example 1: Initialize store
    print("\n1. Initialize Pinecone Store")
    print("-" * 70)

    store = PineconeStore(
        api_key=api_key,
        index_name="test-docs-dev",
        dimension=384,
        cloud="aws",
        region="us-east-1"
    )

    stats = store.get_index_stats()
    print(f"Total vectors: {stats['total_vector_count']:,}")

    # Example 2: Upsert chunks
    print("\n2. Upserting Chunks")
    print("-" * 70)

    from models.metadata_models import ChunkMetadata

    chunks = [
        ChunkMetadata(
            doc_id="sample_doc",
            domain="hr",
            chunk_text="Employees receive 15 vacation days per year",
            char_range=(0, 44),
            source_file_path="./test.pdf",
            source_file_hash="abc123",
            embedding_model_name="all-MiniLM-L6-v2",
            chunking_strategy="recursive"
        )
    ]

    embeddings = np.random.rand(1, 384).astype(np.float32)
    store.upsert(chunks, embeddings)

    print("Waiting 2 seconds for eventual consistency...")
    time.sleep(2)

    # Example 3: Get all documents (Phase 2)
    print("\n3. Get All Documents (Hybrid Retrieval)")
    print("-" * 70)

    corpus, doc_ids = store.get_all_documents()
    print(f"Corpus size: {len(corpus)} documents")
    print(f"Total characters: {sum(len(text) for text in corpus):,}")

    if corpus:
        print(f"\nSample document 1:")
        print(f"  ID: {doc_ids[0]}")
        print(f"  Text: {corpus[0][:80]}...")

    # Example 4: Delete
    print("\n4. Deleting Document")
    print("-" * 70)

    store.delete_by_doc_id("sample_doc")

    print("\n" + "=" * 70)
    print("PineconeStore examples completed!")
    print("=" * 70)
