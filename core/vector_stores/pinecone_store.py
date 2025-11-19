"""
core/vector_stores/pinecone_store.py

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
pip install pinecone

# For gRPC support (faster, recommended):
pip install "pinecone[grpc]"

References:
-----------
- Documentation: https://docs.pinecone.io/
- Python SDK: https://github.com/pinecone-io/pinecone-python-client
- Community: https://community.pinecone.io/
- Status: https://status.pinecone.io/
"""


from typing import List, Dict, Any, Optional
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from core.interfaces.vector_store_interface import VectorStoreInterface
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

        Choose based on:
        - Your application's region (minimize latency)
        - Data residency requirements
        - Availability in your tier (GCP/Azure in preview)

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
        dimension=384,  # Match your embedding model!
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
                    metric="cosine",  # Similarity metric (cosine, euclidean, dotproduct)
                    spec=ServerlessSpec(
                        cloud=cloud,
                        region=region
                    )
                )

                # Wait for index to be ready (can take 2-5 minutes)
                logger.info("Waiting for index to be ready (this may take a few minutes)...")
                start = time.time()

                while not self.pc.describe_index(index_name).status['ready']:
                    time.sleep(5)  # Check every 5 seconds
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

        Pinecone's upsert operation:
        - If vector ID exists → update
        - If vector ID is new → insert

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

        Performance:
        ------------
        - Batch size: 100 vectors per API call (Pinecone limit)
        - Speed: ~500-2000 vectors/sec
        - Async: Upserts are eventually consistent (~1-2 seconds)

        Example:
        --------
        chunks = chunker.chunk_text(...)
        embeddings = embedder.embed_texts([c.chunk_text for c in chunks])
        store.upsert(chunks, embeddings)

        Notes:
        ------
        - Vectors become searchable within 1-2 seconds (eventual consistency)
        - Metadata is stored with vectors (no separate database)
        - Duplicate chunk_ids will be updated, not create duplicates
        """
        # Step 1: Validate inputs
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) must match embeddings count ({len(embeddings)})"
            )

        if len(chunks) == 0:
            logger.warning("Upsert called with empty chunks list")
            return

        logger.info(f"Upserting {len(chunks)} chunks to Pinecone...")

        try:
            # Step 2: Prepare vectors for upsert
            # Pinecone format: [(id, values, metadata), ...]
            vectors = []

            for chunk, embedding in zip(chunks, embeddings):
                vectors.append({
                    "id": chunk.chunk_id,
                    "values": embedding.tolist(),  # Convert numpy to list
                    "metadata": self._serialize_metadata(chunk)
                })

            # Step 3: Upsert in batches (Pinecone limit: 100 vectors per request)
            batch_size = 100
            total_upserted = 0

            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(vectors) + batch_size - 1) // batch_size

                logger.debug(
                    f"Upserting batch {batch_num}/{total_batches} "
                    f"({len(batch)} vectors)..."
                )

                # Upsert batch
                upsert_response = self.index.upsert(vectors=batch)
                total_upserted += upsert_response.upserted_count

                logger.debug(f"Batch {batch_num} completed")

            # Step 4: Log success
            logger.info(
                f"✅ Upsert successful!\n"
                f"   Upserted: {total_upserted} vectors\n"
                f"   Index: {self.index_name}\n"
                f"   Note: Vectors searchable in ~1-2 seconds (eventual consistency)"
            )

        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            raise RuntimeError(
                f"Failed to upsert chunks to Pinecone\n"
                f"Index: {self.index_name}\n"
                f"Chunks: {len(chunks)}\n"
                f"Error: {e}"
            )

    def search(
            self,
            query_embedding: np.ndarray,
            top_k: int,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """
        Search for similar chunks using vector similarity and optional metadata filters.

        Pinecone uses approximate nearest neighbor (ANN) search optimized for
        high-dimensional vectors at scale.

        Parameters:
        -----------
        query_embedding : np.ndarray
            1D numpy array of query embedding
            Shape: (embedding_dim,)

        top_k : int
            Number of top results to return (1-10000)
            Recommended: 10-100 for most use cases

        filters : Dict[str, Any], optional
            Metadata filters using Pinecone's filter syntax

            Examples:
            ---------
            # Simple equality
            filters = {"domain": "hr"}

            # Multiple conditions (AND)
            filters = {
                "$and": [
                    {"domain": "hr"},
                    {"is_authoritative": True}
                ]
            }

            # Multiple conditions (OR)
            filters = {
                "$or": [
                    {"domain": "hr"},
                    {"domain": "finance"}
                ]
            }

            # Comparison operators
            filters = {"confidence_score": {"$gte": 0.8}}  # >= 0.8
            filters = {"page_num": {"$lt": 10}}  # < 10

            # String matching
            filters = {"domain": {"$in": ["hr", "finance"]}}
            filters = {"domain": {"$ne": "legal"}}  # not equal

            # Field existence
            filters = {"page_num": {"$exists": True}}

            See: https://docs.pinecone.io/guides/data/filter-with-metadata

        Returns:
        --------
        List[Dict]:
            List of result dictionaries, each containing:
            - 'id': chunk_id
            - 'score': similarity score (0-1, higher = more similar)
            - 'metadata': metadata dict (includes chunk_text)
            - 'document': chunk_text (extracted for convenience)

            Sorted by similarity (most similar first)

        Performance:
        ------------
        - Latency: 50-200ms (depends on index size and filters)
        - Scales: Handles billions of vectors
        - Filters: Low overhead (~10-20% latency increase)

        Example:
        --------
        # Simple search
        query_emb = embedder.embed_texts(["vacation policy"])[0]
        results = store.search(query_emb, top_k=10)

        # Search with filters
        results = store.search(
            query_emb,
            top_k=10,
            filters={
                "$and": [
                    {"domain": "hr"},
                    {"is_authoritative": True},
                    {"deprecated_flag": False}
                ]
            }
        )

        for result in results:
            print(f"{result['score']:.3f} - {result['document'][:100]}...")
        """
        # Step 1: Validate inputs
        if query_embedding.ndim != 1:
            raise ValueError(
                f"Query embedding must be 1D array, got shape: {query_embedding.shape}"
            )

        if top_k < 1 or top_k > 10000:
            raise ValueError(f"top_k must be between 1 and 10000, got {top_k}")

        logger.debug(
            f"Searching Pinecone: top_k={top_k}, "
            f"filters={'Yes' if filters else 'No'}"
        )

        try:
            # Step 2: Query Pinecone
            # API: index.query(vector, top_k, filter, include_metadata)
            query_response = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                filter=filters,  # Optional metadata filters
                include_metadata=True  # Include metadata in response
            )

            # Step 3: Format results
            formatted_results = self._format_results(query_response)

            logger.debug(f"Found {len(formatted_results)} results")

            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RuntimeError(
                f"Failed to search Pinecone\n"
                f"Index: {self.index_name}\n"
                f"top_k: {top_k}\n"
                f"Filters: {filters}\n"
                f"Error: {e}"
            )

    def delete_by_doc_id(self, doc_id: str) -> None:
        """
        Delete all chunks belonging to a document.

        Uses metadata filtering to delete all chunks with matching doc_id.

        Parameters:
        -----------
        doc_id : str
            Document ID to delete all chunks for

        Example:
        --------
        # Delete all chunks from old policy
        store.delete_by_doc_id("employee_handbook_2023")

        # Replace document
        store.delete_by_doc_id("policy_v1")
        store.upsert(new_chunks, new_embeddings)

        Notes:
        ------
        - Deletion is asynchronous (~1-2 seconds to complete)
        - Metadata filter must match exactly
        - Operation is logged but count not immediately available
        """
        logger.info(f"Deleting all chunks for doc_id: {doc_id}")

        try:
            # Delete using metadata filter
            self.index.delete(
                filter={"doc_id": doc_id}
            )

            logger.info(
                f"✅ Delete requested for doc_id: {doc_id}\n"
                f"   Note: Deletion completes in ~1-2 seconds (async)"
            )

        except Exception as e:
            logger.error(f"Deletion failed: {e}")
            raise RuntimeError(
                f"Failed to delete chunks for doc_id: {doc_id}\n"
                f"Error: {e}"
            )

    def _serialize_metadata(self, chunk: ChunkMetadata) -> Dict[str, Any]:
        """
        Convert ChunkMetadata to Pinecone-compatible metadata dict.

        Pinecone Metadata Limitations:
        -------------------------------
        - Values must be: str, number, bool, or list of strings
        - Keys must be strings
        - Max metadata size: 40KB per vector
        - Filterable types: str, number, bool
        - Lists are NOT filterable (use for display only)

        Parameters:
        -----------
        chunk : ChunkMetadata
            Chunk metadata object

        Returns:
        --------
        dict:
            Pinecone-compatible metadata dictionary

        Notes:
        ------
        - Includes chunk_text in metadata (for retrieval display)
        - Converts timestamps to ISO strings
        - Separates char_range tuple into two fields
        - Handles None values
        """
        # Convert char_range tuple to separate fields
        char_start, char_end = chunk.char_range

        return {
            # Core fields
            "doc_id": chunk.doc_id,
            "domain": chunk.domain,
            "chunk_text": chunk.chunk_text,  # Include for display
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

        Pinecone returns:
        {
            'matches': [
                {
                    'id': 'chunk-uuid-1',
                    'score': 0.95,
                    'metadata': {...}
                },
                ...
            ]
        }

        We standardize to:
        [
            {
                'id': 'chunk-uuid-1',
                'score': 0.95,
                'metadata': {...},
                'document': 'chunk text...'  # Extracted for convenience
            },
            ...
        ]

        Parameters:
        -----------
        query_response : QueryResponse
            Raw response from Pinecone query()

        Returns:
        --------
        List[Dict]:
            Formatted results list
        """
        formatted = []

        for match in query_response['matches']:
            formatted.append({
                'id': match['id'],
                'score': match['score'],
                'metadata': match['metadata'],
                'document': match['metadata'].get('chunk_text', '')  # Extract text
            })

        return formatted

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index.

        Returns:
        --------
        dict:
            Index statistics including:
            - 'total_vector_count': Total vectors in index
            - 'dimension': Vector dimension
            - 'index_fullness': Storage utilization (0.0-1.0)
            - 'namespaces': Per-namespace stats (if using namespaces)

        Example:
        --------
        stats = store.get_index_stats()
        print(f"Total vectors: {stats['total_vector_count']:,}")
        print(f"Dimension: {stats['dimension']}")
        print(f"Fullness: {stats['index_fullness']:.1%}")
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
    Run: python core/vector_stores/pinecone_store.py

    Requirements:
    - Set PINECONE_API_KEY environment variable
    - pip install pinecone
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
        index_name="test-docs-dev",  # Use your index name
        dimension=384,  # Match your embedding model
        cloud="aws",
        region="us-east-1"
    )

    stats = store.get_index_stats()
    print(f"Index: test-docs-dev")
    print(f"Total vectors: {stats['total_vector_count']:,}")
    print(f"Dimension: {stats['dimension']}")

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

    # Generate dummy embedding (in real use, use embedder)
    embeddings = np.random.rand(1, 384).astype(np.float32)

    store.upsert(chunks, embeddings)

    # Wait for eventual consistency
    print("Waiting 2 seconds for eventual consistency...")
    time.sleep(2)

    # Example 3: Search
    print("\n3. Searching")
    print("-" * 70)

    query_embedding = np.random.rand(384).astype(np.float32)
    results = store.search(query_embedding, top_k=5)

    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Text: {result['document'][:60]}...")
        print()

    # Example 4: Filtered search
    print("\n4. Filtered Search")
    print("-" * 70)

    results = store.search(
        query_embedding,
        top_k=10,
        filters={"domain": "hr"}
    )

    print(f"Found {len(results)} results in HR domain")

    # Example 5: Delete
    print("\n5. Deleting Document")
    print("-" * 70)

    store.delete_by_doc_id("sample_doc")

    print("\n" + "=" * 70)
    print("PineconeStore examples completed!")
    print("=" * 70)
