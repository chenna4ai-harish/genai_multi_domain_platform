"""

core/vectorstores/chromadb_store.py

This module implements the ChromaDB vector store adapter (OPTION 1).

What is ChromaDB?
------------------
ChromaDB is an open-source embedding database designed for AI applications.
It's a lightweight, in-process vector store that persists data to disk, making
it perfect for development, testing, and small-to-medium production deployments.

Key Features:
-------------
- **Local & Free**: Runs on your machine, no cloud dependencies
- **Persistent**: Data saved to disk, survives restarts
- **Simple**: Easy setup, no configuration required
- **Fast**: Optimized HNSW index for approximate nearest neighbor search
- **Metadata Filtering**: SQL-like filtering on metadata fields
- **Collections**: Organize vectors into separate collections (like database tables)

Architecture:
-------------
ChromaDB uses:
- HNSW (Hierarchical Navigable Small World) index for similarity search
- SQLite for metadata storage and filtering
- File-based persistence (embeds in your application)

Performance:
------------
- Insert: ~1000-5000 vectors/sec
- Query: ~10-100ms for top-10 results (depends on collection size)
- Memory: ~1-2GB for 100K vectors (384-dim)
- Disk: ~400MB for 100K vectors (384-dim)

When to Use ChromaDB:
----------------------
✅ MVP and development phase (fast iteration)
✅ Single-server deployments
✅ Small-to-medium datasets (<1M vectors)
✅ Budget-constrained projects (free)
✅ Quick prototyping and testing
❌ Distributed systems (use Pinecone/Weaviate)
❌ Very large datasets (>10M vectors)
❌ High-availability requirements
❌ Multi-region deployments

Scaling Considerations:
-----------------------
- Up to 100K vectors: Excellent performance
- 100K-1M vectors: Good performance, monitor memory
- 1M+ vectors: Consider Pinecone or cluster setup

Installation:
-------------
pip install chromadb

Optional (for better performance):
pip install chromadb[server]

References:
-----------
- Documentation: https://docs.trychroma.com/
- GitHub: https://github.com/chroma-core/chroma
- Discord: https://discord.gg/MMeYNTmh3x

"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import chromadb
from chromadb.config import Settings
from core.interfaces.vectorstore_interface import VectorStoreInterface
from models.metadata_models import ChunkMetadata
import logging

# Configure logging
logger = logging.getLogger(__name__)


class ChromaDBStore(VectorStoreInterface):
    """
    ChromaDB vector store adapter with local persistence.

    This implementation provides a clean interface to ChromaDB for storing
    and retrieving document chunks with embeddings and metadata.

    Configuration Parameters:
    -------------------------
    persist_directory : str
        Directory path where ChromaDB will persist data
        Example: "./data/chroma_db"
        The directory will contain:
        - chroma.sqlite3 (metadata database)
        - index/ (HNSW index files)
        Important: Back up this directory regularly!

    collection_name : str
        Name of the ChromaDB collection (like a database table)
        Each domain should have its own collection
        Examples: "hr_collection", "finance_collection"
        Naming rules:
        - 3-63 characters
        - Start/end with alphanumeric
        - Can contain: letters, numbers, underscores, hyphens

    Example Usage:
    --------------
    # Initialize store
    store = ChromaDBStore(
        persist_directory="./data/chroma_db",
        collection_name="hr_collection"
    )

    # Upsert chunks with embeddings
    store.upsert(chunks, embeddings)

    # Search for similar chunks
    query_embedding = embedder.embed_texts(["vacation policy"])
    results = store.search(query_embedding[0], top_k=10)

    # Delete old document
    store.delete_by_doc_id("old_policy_2023")

    # Get all documents for BM25 (Phase 2 hybrid retrieval)
    corpus, doc_ids = store.get_all_documents()
    """

    def __init__(self, persist_directory: str, collection_name: str):
        """
        Initialize ChromaDB store with persistence.

        Parameters:
        -----------
        persist_directory : str
            Path to directory for persisting data
        collection_name : str
            Name of the collection to use/create
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        logger.info(
            f"Initializing ChromaDB store:\n"
            f"  Directory: {persist_directory}\n"
            f"  Collection: {collection_name}"
        )

        try:
            # Create persist directory if it doesn't exist
            from pathlib import Path
            persist_path = Path(persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Persist path: {persist_path.absolute()}")

            # Initialize ChromaDB PersistentClient (0.4.x API)
            self.client = chromadb.PersistentClient(path=str(persist_path))

            # Get or create collection with cosine similarity
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Cosine similarity for embeddings
            )

            # Log collection info
            count = self.collection.count()
            logger.info(
                f"✅ ChromaDB initialized successfully!\n"
                f"   Collection: {collection_name}\n"
                f"   Existing vectors: {count:,}\n"
                f"   Distance metric: cosine\n"
                f"   Persist directory: {persist_path.absolute()}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}", exc_info=True)
            raise RuntimeError(
                f"Could not initialize ChromaDB\n"
                f"Directory: {persist_directory}\n"
                f"Collection: {collection_name}\n"
                f"Error: {str(e)}"
            )

    def upsert(self, chunks: List[ChunkMetadata], embeddings: np.ndarray) -> None:
        """
        Insert or update chunks with embeddings in ChromaDB.

        "Upsert" = Update if exists, Insert if new
        ChromaDB uses chunk_id as the primary key.

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
            If embeddings have wrong shape
        RuntimeError:
            If upsert operation fails
        """
        # Step 1: Validate inputs
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) must match embeddings count ({len(embeddings)})"
            )

        if len(chunks) == 0:
            logger.warning("Upsert called with empty chunks list")
            return

        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be 2D array, got shape: {embeddings.shape}"
            )

        logger.info(f"Upserting {len(chunks)} chunks to ChromaDB...")

        try:
            # Step 2: Extract data from ChunkMetadata objects
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.chunk_text for chunk in chunks]
            metadatas = [self._serialize_metadata(chunk) for chunk in chunks]

            # Step 3: Upsert to ChromaDB
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings.tolist(),  # Convert numpy to list
                metadatas=metadatas
            )

            # Step 4: Log success
            new_count = self.collection.count()
            logger.info(
                f"✅ Upsert successful!\n"
                f"   Upserted: {len(chunks)} chunks\n"
                f"   Total vectors in collection: {new_count:,}\n"
                f"   Collection: {self.collection_name}"
            )

        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            raise RuntimeError(
                f"Failed to upsert chunks to ChromaDB\n"
                f"Collection: {self.collection_name}\n"
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

        Parameters:
        -----------
        query_embedding : np.ndarray
            1D numpy array of query embedding
        top_k : int
            Number of top results to return
        filters : Dict[str, Any], optional
            Metadata filters using ChromaDB's where syntax

        Returns:
        --------
        List[Dict]:
            List of results with id, document, metadata, distance
        """
        # Step 1: Validate inputs
        if query_embedding.ndim != 1:
            raise ValueError(
                f"Query embedding must be 1D array, got shape: {query_embedding.shape}"
            )

        if top_k < 1 or top_k > 1000:
            raise ValueError(f"top_k must be between 1 and 1000, got {top_k}")

        logger.debug(
            f"Searching ChromaDB: top_k={top_k}, "
            f"filters={'Yes' if filters else 'No'}"
        )

        try:
            # Step 2: Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )

            # Step 3: Format results
            formatted_results = self._format_results(results)
            logger.debug(f"Found {len(formatted_results)} results")

            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RuntimeError(
                f"Failed to search ChromaDB\n"
                f"Collection: {self.collection_name}\n"
                f"top_k: {top_k}\n"
                f"Filters: {filters}\n"
                f"Error: {e}"
            )

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
            # Get count before deletion (for logging)
            before_count = self.collection.count()

            # Delete using metadata filter
            self.collection.delete(
                where={"doc_id": doc_id}
            )

            # Get count after deletion
            after_count = self.collection.count()
            deleted_count = before_count - after_count

            logger.info(
                f"✅ Deleted {deleted_count} chunks for doc_id: {doc_id}\n"
                f"   Remaining vectors: {after_count:,}"
            )

        except Exception as e:
            logger.error(f"Deletion failed: {e}")
            raise RuntimeError(
                f"Failed to delete chunks for doc_id: {doc_id}\n"
                f"Error: {e}"
            )

    def get_all_documents(self) -> Tuple[List[str], List[str]]:
        """
        Retrieve all document texts and IDs from the collection.

        This method is REQUIRED for Phase 2 hybrid retrieval to build BM25 indexes.

        Returns:
        --------
        Tuple[List[str], List[str]]:
            Tuple of (corpus, doc_ids)
            - corpus: List of all chunk texts
            - doc_ids: List of corresponding chunk IDs

        Example:
        --------
        # Build BM25 index for hybrid retrieval
        corpus, doc_ids = store.get_all_documents()
        bm25_index = BM25Retrieval(corpus=corpus, doc_ids=doc_ids)
        """
        logger.info(f"Retrieving all documents from collection: {self.collection_name}")

        try:
            # Get total count for logging
            total_count = self.collection.count()

            if total_count == 0:
                logger.warning("Collection is empty, returning empty corpus")
                return [], []

            logger.info(f"Collection has {total_count:,} chunks, fetching all...")

            # ChromaDB API: collection.get() retrieves all documents
            results = self.collection.get(
                include=["documents"]  # Only need documents and IDs
            )

            # Extract data from results
            doc_ids = results['ids']
            corpus = results['documents']

            # Validate data
            if len(corpus) != len(doc_ids):
                raise RuntimeError(
                    f"Data mismatch: got {len(corpus)} documents but {len(doc_ids)} IDs"
                )

            logger.info(
                f"✅ Retrieved {len(corpus):,} documents from ChromaDB\n"
                f"   Collection: {self.collection_name}\n"
                f"   Corpus size: {sum(len(text) for text in corpus):,} characters"
            )

            return corpus, doc_ids

        except Exception as e:
            logger.error(f"Failed to retrieve all documents: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to retrieve documents from ChromaDB\n"
                f"Collection: {self.collection_name}\n"
                f"Error: {e}"
            )

    # =========================================================================
    # HELPER METHODS (Private)
    # =========================================================================

    def _serialize_metadata(self, chunk: ChunkMetadata) -> Dict[str, Any]:
        """
        Convert ChunkMetadata to ChromaDB-compatible metadata dict.

        ChromaDB Metadata Limitations:
        - Values must be: str, int, float, or bool
        - No nested dicts or lists
        - No None values
        """
        char_start, char_end = chunk.char_range

        return {
            # Core fields
            "doc_id": chunk.doc_id,
            "domain": chunk.domain,
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

    def _format_results(self, results: Dict) -> List[Dict]:
        """
        Format ChromaDB query results into standardized structure.

        ChromaDB returns nested lists, we flatten to list of dicts.
        """
        formatted = []

        for i in range(len(results['ids'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })

        return formatted

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
        --------
        dict:
            Collection statistics (name, count, metadata)
        """
        return {
            'name': self.collection.name,
            'count': self.collection.count(),
            'metadata': self.collection.metadata
        }


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of ChromaDBStore usage.
    Run: python core/vectorstores/chromadb_store.py
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("ChromaDBStore Usage Examples")
    print("=" * 70)

    # Example 1: Initialize store
    print("\n1. Initialize ChromaDB Store")
    print("-" * 70)

    store = ChromaDBStore(
        persist_directory="./data/test_chroma_db",
        collection_name="test_collection"
    )

    stats = store.get_collection_stats()
    print(f"Collection: {stats['name']}")
    print(f"Existing vectors: {stats['count']:,}")

    # Example 2: Upsert chunks
    print("\n2. Upserting Chunks with Embeddings")
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
            embedding_model_name="test-model",
            chunking_strategy="recursive"
        ),
        ChunkMetadata(
            doc_id="sample_doc",
            domain="hr",
            chunk_text="Health insurance covers medical and dental",
            char_range=(45, 87),
            source_file_path="./test.pdf",
            source_file_hash="abc123",
            embedding_model_name="test-model",
            chunking_strategy="recursive"
        )
    ]

    embeddings = np.random.rand(2, 384).astype(np.float32)
    store.upsert(chunks, embeddings)

    # Example 3: Search
    print("\n3. Searching for Similar Chunks")
    print("-" * 70)

    query_embedding = np.random.rand(384).astype(np.float32)
    results = store.search(query_embedding, top_k=2)

    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Distance: {result['distance']:.4f}")
        print(f"  Text: {result['document'][:60]}...")
        print(f"  Domain: {result['metadata']['domain']}")
        print()

    # Example 4: Search with filters
    print("\n4. Filtered Search")
    print("-" * 70)

    results = store.search(
        query_embedding,
        top_k=10,
        filters={"domain": "hr"}
    )
    print(f"Found {len(results)} results in HR domain")

    # Example 5: Get all documents (Phase 2 - for BM25)
    print("\n5. Retrieve All Documents (Hybrid Retrieval)")
    print("-" * 70)

    corpus, doc_ids = store.get_all_documents()
    print(f"Corpus size: {len(corpus)} documents")
    print(f"Total characters: {sum(len(text) for text in corpus):,}")

    if corpus:
        print(f"\nSample document 1:")
        print(f"  ID: {doc_ids[0]}")
        print(f"  Text: {corpus[0][:80]}...")

    # Example 6: Delete by doc_id
    print("\n6. Deleting Document")
    print("-" * 70)

    store.delete_by_doc_id("sample_doc")
    final_stats = store.get_collection_stats()
    print(f"Final vector count: {final_stats['count']:,}")

    print("\n" + "=" * 70)
    print("ChromaDBStore examples completed!")
    print("=" * 70)
