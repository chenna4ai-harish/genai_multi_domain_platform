"""

core/interfaces/vectorstore_interface.py

This module defines the abstract interface (contract) for all vector store
providers in the multi-domain document intelligence platform.

Purpose:
--------
Defines a standard interface that ALL vector store implementations must follow.
This enables swapping between different vector databases (ChromaDB, Pinecone,
Weaviate, Qdrant, etc.) without changing any calling code.

What Is a Vector Store?
------------------------
A vector store (vector database) is a specialized database optimized for:
1. Storing high-dimensional vectors (embeddings)
2. Similarity search (find vectors close to a query vector)
3. Metadata filtering (combine vector search with traditional filters)

Think of it as a "semantic search engine" - instead of keyword matching,
it finds conceptually similar content.

Example:
--------
Query: "How many vacation days do employees get?"
Query embedding: [0.2, 0.8, -0.3, ..., 0.5]

Vector store finds documents with similar embeddings:
1. "Employees receive 15 vacation days per year" (0.92 similarity)
2. "Paid time off includes 15 vacation days" (0.89 similarity)
3. "Annual leave allowance is 15 days" (0.87 similarity)

Why Use Abstract Base Classes?
-------------------------------
Allows your application to work with ANY vector store:
- Start with ChromaDB (local, simple, free)
- Upgrade to Pinecone (cloud, scalable, managed)
- Try Weaviate (cloud, advanced features)
- Switch to Qdrant (self-hosted, high performance)
- Use PostgreSQL + pgvector (existing database)

All without changing calling code - just update YAML config!

Design Pattern:
---------------
This follows the Repository Pattern + Adapter Pattern:
- Repository: Abstracts data storage/retrieval operations
- Adapter: Wraps different vector DB APIs into one interface

Example Usage:
--------------
# Factory creates the right vector store based on config
store: VectorStoreInterface = VectorStoreFactory.create_store(config)

# Caller doesn't care if it's ChromaDB or Pinecone!
store.upsert(chunks, embeddings)
results = store.search(query_embedding, top_k=10)
store.delete_by_doc_id("old_doc")

# Phase 2: Build BM25 index for hybrid retrieval
corpus, doc_ids = store.get_all_documents()
bm25_index = BM25Retrieval(corpus=corpus, doc_ids=doc_ids)

References:
-----------
- Repository Pattern: https://martinfowler.com/eaaCatalog/repository.html
- Vector Databases Overview: https://www.pinecone.io/learn/vector-database/

"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from models.metadata_models import ChunkMetadata


class VectorStoreInterface(ABC):
    """
    Abstract base class defining the interface for vector store providers.

    All vector store implementations (ChromaDBStore, PineconeStore, etc.)
    MUST inherit from this class and implement all abstract methods.

    Design Pattern:
    ---------------
    This follows the Repository Pattern, where the repository (vector store)
    abstracts away the details of data storage and retrieval. Calling code
    works with the interface, not concrete implementations.

    Responsibilities:
    -----------------
    A vector store is responsible for:
    1. Storing vectors (embeddings) with their metadata
    2. Searching for similar vectors (similarity/semantic search)
    3. Filtering by metadata (combine semantic + traditional search)
    4. Managing vector lifecycle (create, update, delete)
    5. Providing corpus access for hybrid retrieval (BM25 indexing)

    Core Operations (CRUD):
    -----------------------
    - Create/Update: upsert() - insert or update vectors
    - Read: search() - find similar vectors
    - Read: get_all_documents() - retrieve corpus for BM25
    - Delete: delete_by_doc_id() - remove vectors by document

    Key Concepts:
    -------------
    - **Vector**: Dense array of numbers (e.g., [0.2, 0.8, -0.3, ...])
    - **Embedding**: Vector representation of text (from embedding model)
    - **Similarity**: How close two vectors are (cosine, euclidean, dot product)
    - **Metadata**: Additional information stored with vectors (domain, doc_id, etc.)
    - **Upsert**: Update if exists, Insert if new (idempotent operation)
    - **Corpus**: Collection of all documents (for BM25/hybrid retrieval)

    Similarity Metrics:
    -------------------
    Different vector stores support different metrics:
    - Cosine: Measures angle between vectors (most common for text)
    - Euclidean: Measures straight-line distance
    - Dot Product: Measures alignment and magnitude

    Cosine is recommended for text embeddings (normalized vectors).

    Example Implementations:
    ------------------------
    See:
    - core/vectorstores/chromadb_store.py (OPTION 1: Local, free)
    - core/vectorstores/pinecone_store.py (OPTION 2: Cloud, managed)

    Usage Example:
    --------------
    # This interface allows polymorphic usage:
    def store_and_search(store: VectorStoreInterface, chunks, embeddings, query):
        '''Works with ANY vector store implementation!'''
        # Store chunks
        store.upsert(chunks, embeddings)

        # Search
        query_emb = embedder.embed_texts([query])[0]
        results = store.search(query_emb, top_k=10)
        return results

    # Works with ChromaDB
    chromadb_store = ChromaDBStore(...)
    results = store_and_search(chromadb_store, chunks, embeddings, "vacation policy")

    # Works with Pinecone (same function!)
    pinecone_store = PineconeStore(...)
    results = store_and_search(pinecone_store, chunks, embeddings, "vacation policy")
    """

    @abstractmethod
    def upsert(self, chunks: List[ChunkMetadata], embeddings: np.ndarray) -> None:
        """
        Insert or update chunks with embeddings in the vector store.

        "Upsert" = Update if exists (based on chunk_id), Insert if new.
        This operation is idempotent - running it multiple times with the
        same data produces the same result.

        This is the PRIMARY WRITE operation for the vector store. It combines:
        1. Vector storage (embeddings)
        2. Metadata storage (ChunkMetadata fields)
        3. Index building (for fast similarity search)

        Parameters:
        -----------
        chunks : List[ChunkMetadata]
            List of chunk metadata objects.
            Each chunk contains:
            - chunk_id: Unique identifier (primary key)
            - chunk_text: The actual text content
            - All metadata fields (domain, doc_id, page_num, etc.)

        embeddings : np.ndarray
            2D numpy array of embeddings.
            Shape: (n_chunks, embedding_dim)
            Type: float32 or float64

            Requirements:
            - Length must match chunks length
            - All embeddings must have same dimension
            - Dimension must match vector store configuration

            Example: (100, 384) for 100 chunks with 384-dim embeddings

        Raises:
        -------
        ValueError:
            - If len(chunks) != len(embeddings)
            - If embeddings have wrong shape or dimension
            - If chunk_id is missing or invalid

        RuntimeError:
            - If vector store operation fails
            - If connection is lost
            - If storage quota is exceeded

        Implementation Guidelines:
        --------------------------
        1. **Validate Inputs**:
           - Check lengths match
           - Verify embedding dimensions
           - Validate chunk_ids are present

        2. **Handle Batching**:
           - Most vector stores have batch size limits
           - Split large upserts into batches
           - Log progress for large operations

        3. **Metadata Serialization**:
           - Convert ChunkMetadata to store-specific format
           - Handle type limitations (e.g., no nested dicts)
           - Store chunk_text with metadata for retrieval

        4. **Error Handling**:
           - Provide clear error messages
           - Log failures with context
           - Consider partial success handling

        5. **Idempotency**:
           - Same chunk_id should update, not duplicate
           - Handle duplicate inserts gracefully

        Performance Considerations:
        ---------------------------
        - Batch operations are much faster than individual inserts
        - Large batches (100-1000 vectors) are more efficient
        - Upserts may be asynchronous (eventual consistency)
        - Index building can add latency (seconds to minutes)

        Example Implementation Pattern:
        -------------------------------
        def upsert(self, chunks, embeddings):
            # 1. Validate
            if len(chunks) != len(embeddings):
                raise ValueError("Length mismatch")

            # 2. Prepare data
            ids = [c.chunk_id for c in chunks]
            vectors = embeddings.tolist()
            metadata = [self._serialize(c) for c in chunks]

            # 3. Batch upsert
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_vectors = vectors[i:i+batch_size]
                batch_metadata = metadata[i:i+batch_size]

                self.client.upsert(
                    ids=batch_ids,
                    vectors=batch_vectors,
                    metadata=batch_metadata
                )

            logger.info(f"Upserted {len(chunks)} chunks")

        Example Usage:
        --------------
        # Prepare chunks and embeddings
        chunks = chunker.chunk_text(text, ...)
        chunk_texts = [c.chunk_text for c in chunks]
        embeddings = embedder.embed_texts(chunk_texts)

        # Upsert to vector store
        store.upsert(chunks, embeddings)
        # Chunks are now searchable!
        """
        pass  # Subclasses MUST implement this method

    @abstractmethod
    def search(
            self,
            query_embedding: np.ndarray,
            top_k: int,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """
        Search for similar chunks using vector similarity and optional metadata filters.

        This is the PRIMARY READ operation for the vector store. It performs
        approximate nearest neighbor (ANN) search to find vectors most similar
        to the query embedding.

        Parameters:
        -----------
        query_embedding : np.ndarray
            1D numpy array representing the query
            Shape: (embedding_dim,)
            Type: float32 or float64

            Usually generated by embedding a user query:
            query_emb = embedder.embed_texts(["vacation policy"])[0]

        top_k : int
            Number of top results to return
            Recommended: 5-50 for most use cases
            Range: 1-10000 (depends on vector store)
            More results = better recall but slower

        filters : Dict[str, Any], optional
            Metadata filters to narrow search
            Syntax varies by vector store (see implementation)

            Common filters:
            - Domain: {"domain": "hr"}
            - Authoritative: {"is_authoritative": True}
            - Compound: {"$and": [{"domain": "hr"}, {"page_num": {"$lt": 10}}]}

            Benefits:
            - Faster than filtering after retrieval
            - Enables multi-tenant isolation
            - Combines semantic + traditional search

        Returns:
        --------
        List[Dict]:
            List of result dictionaries, sorted by similarity (best first).
            Each dict should contain:
            - 'id': chunk_id (unique identifier)
            - 'score' or 'distance': similarity metric
              * score: higher = more similar (e.g., 0.95)
              * distance: lower = more similar (e.g., 0.05)
            - 'metadata': metadata dictionary with all fields
            - 'document': chunk_text (extracted for convenience)

            Example:
            [
                {
                    'id': 'chunk-uuid-1',
                    'score': 0.95,
                    'metadata': {'domain': 'hr', 'doc_id': '...', ...},
                    'document': 'Employees receive 15 vacation days...'
                },
                {
                    'id': 'chunk-uuid-2',
                    'score': 0.89,
                    'metadata': {...},
                    'document': 'Paid time off includes 15 days...'
                },
                ...
            ]

        Raises:
        -------
        ValueError:
            - If query_embedding has wrong shape or dimension
            - If top_k is out of valid range
            - If filter syntax is invalid

        RuntimeError:
            - If search operation fails
            - If connection is lost

        Algorithm:
        ----------
        1. Validate query embedding dimension
        2. Apply metadata filters (if provided)
        3. Perform ANN search in filtered set
        4. Retrieve top_k most similar vectors
        5. Fetch associated metadata and text
        6. Format and return results

        Performance:
        ------------
        - Latency: 10ms-200ms (depends on index size and filters)
        - Scales: O(log n) for HNSW, O(sqrt(n)) for IVF
        - Filters add overhead: ~10-50% depending on selectivity

        Approximate vs Exact Search:
        ----------------------------
        Most vector stores use ANN (Approximate Nearest Neighbor):
        - Trades some accuracy for speed
        - 95-99% of exact results found
        - Enables billion-vector scale

        Exact search is too slow for large datasets (O(n) complexity).

        Example Implementation Pattern:
        -------------------------------
        def search(self, query_embedding, top_k, filters=None):
            # 1. Validate
            if query_embedding.ndim != 1:
                raise ValueError("Query must be 1D")

            # 2. Search
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                filter=filters
            )

            # 3. Format
            formatted = []
            for result in results:
                formatted.append({
                    'id': result.id,
                    'score': result.score,
                    'metadata': result.metadata,
                    'document': result.metadata.get('chunk_text', '')
                })

            return formatted

        Example Usage:
        --------------
        # Simple search
        query = "How many vacation days?"
        query_emb = embedder.embed_texts([query])[0]
        results = store.search(query_emb, top_k=10)

        # Search with filters
        results = store.search(
            query_emb,
            top_k=10,
            filters={
                "domain": "hr",
                "is_authoritative": True,
                "deprecated_flag": False
            }
        )

        # Process results
        for result in results:
            print(f"{result['score']:.3f} - {result['document'][:100]}...")
        """
        pass  # Subclasses MUST implement this method

    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> None:
        """
        Delete all chunks belonging to a specific document.

        This is the PRIMARY DELETE operation for managing document lifecycle.
        Used when documents are removed, updated, or marked as obsolete.

        Parameters:
        -----------
        doc_id : str
            Document ID whose chunks should be deleted
            All chunks with matching metadata.doc_id will be removed

        Raises:
        -------
        ValueError:
            If doc_id is empty or invalid

        RuntimeError:
            If deletion operation fails

        Use Cases:
        ----------
        1. **Document Removal**: User deletes a document
           store.delete_by_doc_id("employee_handbook_2023")

        2. **Document Update**: Replace old version with new
           store.delete_by_doc_id("policy_v1")  # Remove old
           store.upsert(new_chunks, new_embeddings)  # Add new

        3. **Cleanup**: Remove test or temporary data
           store.delete_by_doc_id("test_doc_123")

        4. **Deprecation**: Remove outdated content
           for old_doc in deprecated_docs:
               store.delete_by_doc_id(old_doc)

        Implementation Guidelines:
        --------------------------
        1. **Use Metadata Filtering**:
           - Delete via metadata filter: {"doc_id": doc_id}
           - Don't fetch all chunks then delete (inefficient)

        2. **Handle Async Deletion**:
           - Some stores delete asynchronously
           - Document eventual consistency
           - Consider polling for completion if needed

        3. **Log Operations**:
           - Log before and after counts
           - Log if no chunks found (may be expected)
           - Provide clear feedback

        4. **Error Handling**:
           - Don't fail if doc_id not found (idempotent)
           - Provide clear errors for connection issues

        Performance:
        ------------
        - Usually fast: O(log n) for indexed metadata
        - May be asynchronous (eventual consistency)
        - Batch deletions are more efficient than one-by-one

        Example Implementation Pattern:
        -------------------------------
        def delete_by_doc_id(self, doc_id):
            # 1. Validate
            if not doc_id:
                raise ValueError("doc_id cannot be empty")

            # 2. Count before (optional)
            before_count = self.collection.count()

            # 3. Delete using metadata filter
            self.collection.delete(
                where={"doc_id": doc_id}
            )

            # 4. Count after (optional)
            after_count = self.collection.count()
            deleted = before_count - after_count

            logger.info(f"Deleted {deleted} chunks for doc_id: {doc_id}")

        Example Usage:
        --------------
        # Delete old document before uploading new version
        old_doc_id = "employee_handbook_2023"
        new_doc_id = "employee_handbook_2024"

        # Remove old
        store.delete_by_doc_id(old_doc_id)

        # Process and upload new
        new_chunks = chunker.chunk_text(new_text, doc_id=new_doc_id, ...)
        new_embeddings = embedder.embed_texts([c.chunk_text for c in new_chunks])
        store.upsert(new_chunks, new_embeddings)

        # Bulk cleanup
        for old_doc in ["doc1", "doc2", "doc3"]:
            store.delete_by_doc_id(old_doc)

        Notes:
        ------
        - Operation is permanent (no undo)
        - May not be immediate (eventual consistency)
        - Safe to call multiple times (idempotent)
        - Consider backup before bulk deletions
        """
        pass  # Subclasses MUST implement this method

    @abstractmethod
    def get_all_documents(self) -> Tuple[List[str], List[str]]:
        """
        Retrieve all document texts and IDs from the vector store.

        This method is REQUIRED for Phase 2 hybrid retrieval to build BM25 indexes.
        The BM25 (sparse keyword-based) retrieval needs access to the entire corpus
        to calculate term frequencies and document frequencies.

        Purpose:
        --------
        Enables hybrid retrieval (dense + sparse) by providing corpus for BM25:
        1. Retrieve all chunks from vector store
        2. Extract chunk texts and IDs
        3. Build BM25 index from corpus
        4. Use BM25 index alongside vector search (hybrid)

        Returns:
        --------
        Tuple[List[str], List[str]]:
            Tuple of (corpus, doc_ids)

            corpus : List[str]
                List of all chunk texts in the vector store
                Example: ["Employee benefits include...", "Health insurance covers...", ...]

            doc_ids : List[str]
                List of corresponding chunk IDs (same order as corpus)
                Example: ["chunk-uuid-1", "chunk-uuid-2", ...]

            Requirements:
            - Both lists must have the same length
            - Order must match (corpus[i] corresponds to doc_ids[i])
            - Should include ALL chunks (no filtering)

        Raises:
        -------
        RuntimeError:
            - If retrieval operation fails
            - If connection is lost
            - If vector store is empty (optional - can return empty lists)

        Use Cases:
        ----------
        1. **Building BM25 Index for Hybrid Retrieval**:
           corpus, doc_ids = vector_store.get_all_documents()
           bm25_index = BM25Retrieval(corpus=corpus, doc_ids=doc_ids)

           # Now use hybrid retrieval
           hybrid_retriever = HybridRetrieval(
               vector_store=vector_store,
               embedding_model=embedder,
               bm25_index=bm25_index,
               alpha=0.7  # 70% dense, 30% sparse
           )

        2. **Analytics and Monitoring**:
           corpus, _ = vector_store.get_all_documents()
           total_chunks = len(corpus)
           avg_length = sum(len(text) for text in corpus) / total_chunks
           print(f"Total chunks: {total_chunks}, Avg length: {avg_length}")

        3. **Data Export/Backup**:
           corpus, doc_ids = vector_store.get_all_documents()
           # Export to file for backup or analysis

        Implementation Guidelines:
        --------------------------
        1. **Pagination for Large Stores**:
           # Don't load all at once if millions of chunks
           all_texts = []
           all_ids = []
           page_size = 1000
           offset = 0

           while True:
               batch = self.collection.get(limit=page_size, offset=offset)
               if not batch:
                   break
               all_texts.extend([doc['text'] for doc in batch])
               all_ids.extend([doc['id'] for doc in batch])
               offset += page_size

           return all_texts, all_ids

        2. **Extract Chunk Text from Metadata**:
           # Chunk text is usually stored in metadata
           results = self.collection.get()
           corpus = [result.metadata['chunk_text'] for result in results]
           doc_ids = [result.id for result in results]
           return corpus, doc_ids

        3. **Handle Empty Store**:
           if self.collection.count() == 0:
               logger.warning("Vector store is empty")
               return [], []

        4. **Caching for Performance**:
           # If corpus doesn't change often, cache it
           if hasattr(self, '_corpus_cache'):
               return self._corpus_cache

           corpus, doc_ids = self._fetch_all_documents()
           self._corpus_cache = (corpus, doc_ids)
           return corpus, doc_ids

        Performance Considerations:
        ---------------------------
        - This operation can be slow for large datasets (millions of chunks)
        - Consider pagination/batching for large stores
        - May want to cache results if corpus doesn't change frequently
        - BM25 index building is O(N*M) where N=docs, M=avg_terms_per_doc

        Example Implementation Pattern:
        -------------------------------
        def get_all_documents(self) -> Tuple[List[str], List[str]]:
            # 1. Get all chunks from vector store
            all_chunks = self.collection.get()

            # 2. Extract texts and IDs
            corpus = []
            doc_ids = []

            for chunk in all_chunks:
                # Extract chunk_text from metadata
                chunk_text = chunk.metadata.get('chunk_text', '')
                if chunk_text:  # Skip empty chunks
                    corpus.append(chunk_text)
                    doc_ids.append(chunk.id)

            logger.info(f"Retrieved {len(corpus)} documents from vector store")

            return corpus, doc_ids

        Example Usage:
        --------------
        # Initialize hybrid retrieval for Phase 2
        vector_store = VectorStoreFactory.create_store(config, embedding_dim, metadata_fields)
        embedder = EmbeddingFactory.create_embedder(embedding_config)

        # Build BM25 index from vector store corpus
        corpus, doc_ids = vector_store.get_all_documents()
        print(f"Corpus size: {len(corpus)} chunks")

        # Create BM25 index
        bm25_index = BM25Retrieval(corpus=corpus, doc_ids=doc_ids)

        # Create hybrid retriever (combines vector + BM25)
        hybrid_retriever = HybridRetrieval(
            vector_store=vector_store,
            embedding_model=embedder,
            bm25_index=bm25_index,
            alpha=0.7
        )

        # Now queries use both semantic (vector) and keyword (BM25) search!
        results = hybrid_retriever.retrieve("vacation policy", top_k=10)

        Notes:
        ------
        - This method is only called when initializing hybrid retrieval
        - Not called on every query (BM25 index is built once)
        - For frequently changing data, may need to rebuild BM25 index periodically
        - Consider lazy loading - only build BM25 when hybrid retrieval is requested
        """
        pass  # Subclasses MUST implement this method


# =============================================================================
# USAGE NOTES FOR IMPLEMENTERS
# =============================================================================

"""
How to Implement a New Vector Store Provider:
----------------------------------------------

1. Create a new file: core/vectorstores/my_store.py

2. Import the interface:
   from core.interfaces.vector_store_interface import VectorStoreInterface
   from models.metadata_models import ChunkMetadata
   import numpy as np

3. Create your class inheriting from VectorStoreInterface:

   class MyVectorStore(VectorStoreInterface):
       def __init__(self, connection_string: str, collection: str):
           # Initialize your vector store client
           self.client = MyVectorStoreClient(connection_string)
           self.collection = self.client.get_collection(collection)

       def upsert(self, chunks: List[ChunkMetadata], embeddings: np.ndarray):
           # Your upsert implementation
           pass

       def search(self, query_embedding, top_k, filters=None):
           # Your search implementation
           pass

       def delete_by_doc_id(self, doc_id: str):
           # Your deletion implementation
           pass

       def get_all_documents(self) -> Tuple[List[str], List[str]]:
           # Your corpus retrieval implementation (for BM25)
           pass

4. Register in factory: core/factories/vector_store_factory.py

   elif config.provider == "my_store":
       return MyVectorStore(
           connection_string=config.my_store.connection_string,
           collection=config.my_store.collection
       )

5. Add config model: models/domain_config.py

   class MyStoreConfig(BaseModel):
       connection_string: str
       collection: str

   class VectorStoreConfig(BaseModel):
       provider: str = "chromadb"
       my_store: Optional[MyStoreConfig] = None

6. Use in YAML:
   vector_store:
     provider: "my_store"
     my_store:
       connection_string: "http://localhost:8000"
       collection: "my_collection"

7. Test thoroughly:
   - Test upsert with various batch sizes
   - Test search with and without filters
   - Test deletion (single and bulk)
   - Test get_all_documents with empty/small/large stores
   - Test error handling (connection loss, invalid data)

That's it! No changes to calling code required (config-driven architecture).
"""
