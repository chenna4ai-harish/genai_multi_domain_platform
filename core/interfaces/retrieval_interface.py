"""

core/interfaces/retrieval_interface.py

This module defines the abstract interface for all retrieval strategies in the
Multi-Domain Document Intelligence Platform (Phase 2).

What is the Retrieval Interface?
---------------------------------
The RetrievalInterface defines the contract that ALL retrieval strategies must
follow. This enables swapping between different retrieval methods (dense vector,
sparse BM25, hybrid) without changing any calling code.

Phase 2 Retrieval Architecture:
--------------------------------
Phase 2 introduces MULTIPLE retrieval strategies that can be used simultaneously:

1. **Vector Similarity (Dense)**: Semantic search using embeddings
   - Best for: Conceptual queries ("benefits overview")
   - Technology: Cosine similarity on dense vectors

2. **BM25 (Sparse)**: Keyword-based search using term frequencies
   - Best for: Specific terms ("form W-2", "section 401k")
   - Technology: BM25 ranking algorithm

3. **Hybrid (Dense + Sparse)**: Combines both with weighted scoring
   - Best for: General queries (most robust)
   - Technology: Alpha-weighted score fusion

Why Use Abstract Interface?
----------------------------
Allows your application to work with ANY retrieval strategy:
- Start with simple vector similarity
- Upgrade to hybrid for better quality
- A/B test different strategies
- Run multiple strategies simultaneously

All without changing calling code - just update YAML config!

Design Pattern:
---------------
This follows the Strategy Pattern:
- Strategy: Different retrieval algorithms can be swapped
- Each strategy implements the same retrieve() method
- Caller doesn't care which strategy is used

Example Usage:
--------------
# Factory creates retriever based on config
retriever: RetrievalInterface = RetrievalFactory.create_retriever(config)

# Caller doesn't care if it's hybrid, BM25, or vector!
results = retriever.retrieve(
    query_text="vacation policy",
    metadata_filters={"domain": "hr", "deprecated": False},
    top_k=10
)

# Results have consistent format regardless of strategy
for result in results:
    print(f"{result['score']:.3f}: {result['document'][:100]}...")

References:
-----------
- Phase 2 Spec: Section 8 (Hybrid Retrieval Implementation)
- Strategy Pattern: https://refactoring.guru/design-patterns/strategy

"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class RetrievalInterface(ABC):
    """
    Abstract base class defining the interface for retrieval strategies.

    All retrieval implementations (VectorSimilarityRetrieval, BM25Retrieval,
    HybridRetrieval) MUST inherit from this class and implement the
    abstract retrieve() method.

    Design Pattern:
    ---------------
    This interface follows the Strategy Pattern, allowing different retrieval
    algorithms to be swapped at runtime based on configuration.

    Responsibilities:
    -----------------
    A retrieval strategy is responsible for:
    1. Converting query text into search representation (vector, terms, or both)
    2. Executing search against vector store and/or BM25 index
    3. Applying metadata filters (domain, doc_type, etc.)
    4. Ranking and scoring results
    5. Returning top-k results in standardized format

    Key Concepts:
    -------------
    - **Dense Retrieval**: Uses vector embeddings (semantic search)
      Example: Query "employee benefits" matches "compensation package"

    - **Sparse Retrieval**: Uses term matching (keyword search)
      Example: Query "401k" matches exact term "401k"

    - **Hybrid Retrieval**: Combines dense + sparse with weighted fusion
      Example: alpha=0.7 means 70% semantic, 30% keyword

    Retrieval Strategies:
    ---------------------
    1. **VectorSimilarityRetrieval** (Dense-only)
       - Pure semantic search using embeddings
       - Best for: Conceptual queries
       - Speed: Fast (single vector search)
       - Quality: Good for paraphrases, poor for exact terms

    2. **BM25Retrieval** (Sparse-only)
       - Pure keyword search using BM25 algorithm
       - Best for: Specific terms, acronyms, IDs
       - Speed: Very fast (inverted index lookup)
       - Quality: Good for exact matches, poor for synonyms

    3. **HybridRetrieval** (Dense + Sparse)
       - Combines both strategies with alpha weighting
       - Best for: General queries (most robust)
       - Speed: Slower (runs both searches)
       - Quality: Best overall (handles both cases)

    Example Implementations:
    ------------------------
    See:
    - core/retrievals/vector_similarity_retrieval.py (STRATEGY 1)
    - core/retrievals/bm25_retrieval.py (STRATEGY 2)
    - core/retrievals/hybrid_retrieval.py (STRATEGY 3 - RECOMMENDED)

    Usage Example:
    --------------
    # Polymorphic usage - works with ANY retriever!
    def search_documents(query: str, retriever: RetrievalInterface):
        '''Search documents with ANY retrieval strategy.'''
        results = retriever.retrieve(
            query_text=query,
            metadata_filters={"domain": "hr"},
            top_k=5
        )

        for result in results:
            print(f"Score: {result['score']:.3f}")
            print(f"Text: {result['document'][:100]}...")

        return results

    # Works with vector similarity
    vector_retriever = VectorSimilarityRetrieval(vectorstore, embedder)
    search_documents("vacation policy", vector_retriever)

    # Works with hybrid (same calling code!)
    hybrid_retriever = HybridRetrieval(vectorstore, embedder, bm25_index)
    search_documents("vacation policy", hybrid_retriever)
    """

    @abstractmethod
    def retrieve(
            self,
            query_text: str,
            metadata_filters: Optional[Dict[str, Any]] = None,
            top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.

        This is the CORE method that all retrieval strategies must implement.
        Each implementation will use its own algorithm but must return results
        in the same standardized format.

        Parameters:
        -----------
        query_text : str
            Natural language query from the user.
            Examples:
            - "How many vacation days do employees get?"
            - "What is the 401k matching policy?"
            - "expense reimbursement procedure"

            Important:
            - Can be a question, phrase, or keywords
            - Strategy converts this to appropriate search representation:
              * Vector: Embeds query text
              * BM25: Tokenizes into terms
              * Hybrid: Does both

        metadata_filters : Dict[str, Any], optional
            Filters to narrow search by metadata fields.

            Common filters:
            - Domain: {"domain": "hr"}
            - Doc type: {"doc_type": "policy"}
            - Authority: {"is_authoritative": True}
            - Not deprecated: {"deprecated_flag": False}
            - Compound: {"domain": "hr", "doc_type": "policy"}

            Filter syntax depends on vector store backend:
            - ChromaDB: {"field": "value"}
            - Pinecone: {"field": {"$eq": "value"}}

            Benefits:
            - Faster than post-filtering
            - Enables multi-tenant isolation
            - Combines semantic + metadata search

        top_k : int
            Number of top results to return.

            Recommended ranges:
            - 5-10: For precise answers
            - 10-20: For comprehensive retrieval
            - 20-50: For re-ranking pipelines

            Trade-offs:
            - Higher k = Better recall, more context
            - Lower k = Faster, more focused

            Default: 10 (good balance)

        Returns:
        --------
        List[Dict[str, Any]]:
            List of result dictionaries, sorted by relevance (best first).

            Each result MUST contain:
            - 'id': str - Chunk ID (unique identifier)
            - 'score': float - Relevance score (higher = more relevant)
              * Range: 0.0 to 1.0 (normalized)
              * Hybrid: Combined score
              * Vector: Cosine similarity
              * BM25: Normalized BM25 score
            - 'document': str - Chunk text
            - 'metadata': Dict - Chunk metadata (all fields from ChunkMetadata)

            Optional fields (strategy-specific):
            - 'dense_score': float - Vector similarity score (hybrid only)
            - 'sparse_score': float - BM25 score (hybrid only)
            - 'strategy': str - Which strategy produced this result

            Example:
            [
                {
                    'id': 'chunk-uuid-1',
                    'score': 0.95,
                    'document': 'Employees receive 15 vacation days per year...',
                    'metadata': {
                        'domain': 'hr',
                        'doc_id': 'handbook_2025',
                        'page_num': 12,
                        'is_authoritative': True,
                        ...
                    }
                },
                {
                    'id': 'chunk-uuid-2',
                    'score': 0.89,
                    'document': 'Paid time off includes 15 days of vacation...',
                    'metadata': {...}
                },
                ...
            ]

        Raises:
        -------
        ValueError:
            - If query_text is empty
            - If top_k is out of valid range
            - If metadata_filters have invalid syntax

        RuntimeError:
            - If retrieval operation fails
            - If vector store or BM25 index unavailable

        Implementation Guidelines:
        --------------------------
        1. **Validate Input**:
           if not query_text or not query_text.strip():
               raise ValueError("query_text cannot be empty")

           if top_k < 1 or top_k > 1000:
               raise ValueError("top_k must be between 1 and 1000")

        2. **Convert Query** (strategy-specific):
           # Vector: Embed query
           query_embedding = self.embedder.embed_texts([query_text])[0]

           # BM25: Tokenize query
           query_terms = self.tokenize(query_text)

           # Hybrid: Do both

        3. **Execute Search**:
           # Vector: Query vector store
           results = self.vectorstore.search(query_embedding, top_k, filters)

           # BM25: Query BM25 index
           results = self.bm25_index.search(query_terms, top_k)

           # Hybrid: Query both and merge
           dense_results = self.vectorstore.search(...)
           sparse_results = self.bm25_index.search(...)
           results = self.merge_results(dense_results, sparse_results, alpha)

        4. **Normalize Scores** (important!):
           # Ensure all scores are 0.0-1.0 range
           for result in results:
               result['score'] = min(max(result['score'], 0.0), 1.0)

        5. **Apply Metadata Filters** (if not done by backend):
           # Most vector stores handle this natively
           # But you may need to post-filter for complex conditions

        6. **Return Top-K**:
           return results[:top_k]

        Performance Considerations:
        ---------------------------
        - **Vector Search**: 10-100ms for 100K chunks
        - **BM25 Search**: 1-10ms for 100K chunks
        - **Hybrid Search**: Sum of both + merge overhead (~20-150ms)

        - Use metadata filters to reduce search space
        - Cache query embeddings for repeated queries
        - Consider async execution for hybrid (parallel searches)

        Example Implementation Pattern:
        -------------------------------
        def retrieve(self, query_text, metadata_filters=None, top_k=10):
            # 1. Validate
            if not query_text:
                raise ValueError("query_text cannot be empty")

            # 2. Convert query (strategy-specific)
            query_embedding = self.embedder.embed_texts([query_text])[0]

            # 3. Execute search
            results = self.vectorstore.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=metadata_filters
            )

            # 4. Format results (ensure consistent structure)
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': result['id'],
                    'score': result.get('score', 0.0),
                    'document': result.get('document', ''),
                    'metadata': result.get('metadata', {})
                })

            # 5. Return top-k
            return formatted_results[:top_k]

        Testing Your Implementation:
        ----------------------------
        - Test with simple query: "vacation policy"
        - Test with empty query: "" (should raise ValueError)
        - Test with filters: {"domain": "hr"}
        - Test with various top_k values: 1, 10, 100
        - Verify result format (id, score, document, metadata)
        - Verify scores are normalized (0.0-1.0)
        - Verify results sorted by score (descending)

        Example Test:
        -------------
        retriever = HybridRetrieval(...)

        results = retriever.retrieve("vacation policy", top_k=5)

        assert len(results) <= 5
        assert all('id' in r for r in results)
        assert all('score' in r for r in results)
        assert all(0.0 <= r['score'] <= 1.0 for r in results)
        assert results == sorted(results, key=lambda x: x['score'], reverse=True)
        """
        pass  # Subclasses MUST implement this method


# =============================================================================
# USAGE NOTES FOR IMPLEMENTERS
# =============================================================================

"""
How to Implement a New Retrieval Strategy:
-------------------------------------------

1. Create a new file: core/retrievals/my_retrieval.py

2. Import the interface:
   from core.interfaces.retrieval_interface import RetrievalInterface

3. Create your class inheriting from RetrievalInterface:

   class MyRetrieval(RetrievalInterface):
       def __init__(self, vectorstore, embedder, **kwargs):
           # Initialize your retrieval strategy
           self.vectorstore = vectorstore
           self.embedder = embedder

       def retrieve(self, query_text, metadata_filters=None, top_k=10):
           # Your retrieval implementation

           # 1. Convert query to search representation
           query_embedding = self.embedder.embed_texts([query_text])[0]

           # 2. Execute search
           results = self.vectorstore.search(
               query_embedding,
               top_k,
               metadata_filters
           )

           # 3. Format and return
           return results

4. Register in factory: core/factories/retrieval_factory.py

   elif config.strategy == "my_retrieval":
       return MyRetrieval(
           vectorstore=vectorstore,
           embedder=embedding_model
       )

5. Add config model: models/domain_config.py

   class RetrievalConfig(BaseModel):
       strategies: List[str] = ["hybrid", "my_retrieval"]

6. Use in YAML:
   retrieval:
     strategies: ["my_retrieval"]

7. Test thoroughly:
   - Test with various queries
   - Test with and without filters
   - Test edge cases (empty query, top_k=0)
   - Verify result format
   - Verify score normalization

That's it! No changes to calling code required (config-driven architecture).


Common Retrieval Patterns:
---------------------------

Pattern 1: Pure Vector Search
------------------------------
query_emb = embedder.embed_texts([query_text])[0]
results = vectorstore.search(query_emb, top_k, filters)
return results

Pattern 2: Pure BM25 Search
----------------------------
query_terms = tokenize(query_text)
results = bm25_index.search(query_terms, top_k)
results = apply_filters(results, filters)  # Post-filter
return results

Pattern 3: Hybrid (Alpha Weighted)
-----------------------------------
# Get both types of results
query_emb = embedder.embed_texts([query_text])[0]
dense_results = vectorstore.search(query_emb, top_k * 2, filters)
sparse_results = bm25_index.search(tokenize(query_text), top_k * 2)

# Merge with alpha weighting
merged = {}
for result in dense_results:
    merged[result['id']] = {
        **result,
        'score': alpha * result['score']
    }

for result in sparse_results:
    if result['id'] in merged:
        merged[result['id']]['score'] += (1 - alpha) * result['score']
    else:
        merged[result['id']] = {
            **result,
            'score': (1 - alpha) * result['score']
        }

# Sort and return top-k
sorted_results = sorted(merged.values(), key=lambda x: x['score'], reverse=True)
return sorted_results[:top_k]


Performance Tips:
-----------------
1. **Caching**: Cache query embeddings for repeated queries
2. **Async**: Run dense and sparse searches in parallel (hybrid)
3. **Filters**: Apply metadata filters early (before retrieval)
4. **Batch**: Process multiple queries in batches
5. **Index**: Ensure vector store and BM25 indexes are optimized
"""
