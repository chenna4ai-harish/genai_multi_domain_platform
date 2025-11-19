"""
core/factories/retrieval_factory.py

This module implements the Factory Pattern for creating retrieval strategy instances.

What is Retrieval?
------------------
Retrieval is the process of finding relevant documents/chunks for a given query.
Different retrieval strategies combine semantic search (embeddings) with keyword
search (BM25, TF-IDF) in various ways to optimize relevance.

Retrieval Strategies:
---------------------
1. **Hybrid** (Dense + Sparse):
   - Combines semantic similarity (vector search) with keyword matching (BM25)
   - Uses alpha parameter to weight the combination (e.g., 70% semantic, 30% keyword)
   - Best for most use cases - gets benefits of both approaches
   - Example: Query "vacation policy" finds semantically similar AND keyword matches

2. **Dense Only** (Pure Semantic):
   - Only uses embedding similarity (cosine distance)
   - Finds conceptually similar content even without exact keywords
   - Best for: Natural language queries, conceptual searches
   - Example: "time off" can find "vacation days" (different words, same meaning)

3. **Sparse Only** (Pure Keyword):
   - Only uses traditional keyword matching (BM25, TF-IDF)
   - Requires exact or close keyword matches
   - Best for: Technical docs with specific terms, code, API names
   - Example: "GET /api/users" finds exact method names

Why Use a Factory?
------------------
- Enables config-driven strategy selection (change YAML, no code changes)
- Abstracts away strategy initialization complexity
- Makes A/B testing different strategies trivial
- Follows same pattern as other factories (consistency)

Note: Retrieval Strategies (Optional for MVP)
----------------------------------------------
The retrieval strategies are more advanced and optional for basic upsert functionality.
For MVP, you can use simple vector search directly from the vector store.

Retrieval strategies become important when:
- You need to combine multiple search methods
- You want reranking capabilities
- You need advanced filtering logic
- You're optimizing for specific query types

This factory is included for completeness but can be implemented later.

Example Usage:
--------------
from models.domain_config import RetrievalConfig

# Hybrid retrieval (semantic + keyword)
config = RetrievalConfig(
    strategy="hybrid",
    alpha=0.7,  # 70% semantic, 30% keyword
    top_k=10
)

retriever = RetrievalFactory.create_retriever(
    config,
    vector_store=store,
    embedder=embedder
)

# Search
results = retriever.retrieve(query="vacation policy")
"""

from typing import Optional , Any
from core.interfaces.embedding_interface import EmbeddingInterface
from core.interfaces.vector_store_interface import VectorStoreInterface
# from core.retrieval.hybrid_retriever import HybridRetriever  # To be implemented
# from core.retrieval.dense_retriever import DenseRetriever      # To be implemented
# from core.retrieval.sparse_retriever import SparseRetriever    # To be implemented
from models.domain_config import RetrievalConfig
import logging

# Configure logging
logger = logging.getLogger(__name__)


class RetrievalFactory:
    """
    Factory for creating retrieval strategy instances based on configuration.

    NOTE: This is a PLACEHOLDER factory for future retrieval implementations.
    For MVP, you can use vector store's search() method directly.

    Supported Strategies (Future):
    -------------------------------
    - "hybrid": Dense (semantic) + Sparse (keyword) with alpha weighting
    - "dense_only": Pure semantic/vector search
    - "sparse_only": Pure keyword/BM25 search

    When to Implement:
    ------------------
    Implement retrieval strategies when you need:
    1. Combining multiple search methods (hybrid)
    2. Reranking results (e.g., cross-encoder reranking)
    3. Query expansion (synonyms, related terms)
    4. Advanced filtering beyond metadata
    5. Custom scoring functions

    For MVP:
    --------
    You can skip this and use vector store search directly:

    # Simple approach (no retrieval factory needed)
    query_emb = embedder.embed_texts([query])[0]
    results = vector_store.search(query_emb, top_k=10, filters={"domain": "hr"})

    Design Pattern:
    ---------------
    Similar to other factories, but more complex because retrieval strategies
    need access to both vector store and embedder.

    Example Usage (Future):
    -----------------------
    # Initialize factories
    embedder = EmbeddingFactory.create_embedder(config.embeddings)
    vector_store = VectorStoreFactory.create_store(config.vector_store)

    # Create retriever
    retriever = RetrievalFactory.create_retriever(
        config.retrieval,
        vector_store=vector_store,
        embedder=embedder
    )

    # Use retriever (abstracted from implementation details)
    results = retriever.retrieve(
        query="How many vacation days?",
        top_k=10,
        filters={"domain": "hr"}
    )
    """

    @staticmethod
    def create_retriever(
            config: RetrievalConfig,
            vector_store: VectorStoreInterface,
            embedder: EmbeddingInterface,
            sparse_index: Optional[Any] = None  # For BM25/keyword search (future)
    ):
        """
        Create and return a retriever instance based on configuration.

        NOTE: This is a PLACEHOLDER implementation. For MVP, use vector store
        search directly. Implement full retrieval strategies later when needed.

        Parameters:
        -----------
        config : RetrievalConfig
            Retrieval configuration from domain YAML
            Contains:
            - strategy: "hybrid", "dense_only", or "sparse_only"
            - alpha: Weight for hybrid (0.0-1.0)
            - top_k: Number of results to retrieve
            - enable_metadata_filtering: Whether to allow filters
            - normalize_scores: Whether to normalize scores

        vector_store : VectorStoreInterface
            Vector store for semantic search

        embedder : EmbeddingInterface
            Embedder for query encoding

        sparse_index : Optional, future
            BM25 or keyword index for sparse search (not implemented yet)

        Returns:
        --------
        RetrieverInterface (future):
            Concrete retriever implementation

        Raises:
        -------
        NotImplementedError:
            Currently raises this for all strategies (placeholder)
        ValueError:
            If config.strategy is not recognized

        Future Implementation:
        ----------------------
        if config.strategy == "hybrid":
            return HybridRetriever(
                vector_store=vector_store,
                embedder=embedder,
                sparse_index=sparse_index,
                alpha=config.alpha,
                top_k=config.top_k,
                normalize_scores=config.normalize_scores
            )

        elif config.strategy == "dense_only":
            return DenseRetriever(
                vector_store=vector_store,
                embedder=embedder,
                top_k=config.top_k
            )

        elif config.strategy == "sparse_only":
            return SparseRetriever(
                sparse_index=sparse_index,
                top_k=config.top_k
            )
        """
        strategy = config.strategy.lower()

        logger.warning(
            f"RetrievalFactory is a placeholder for future implementation.\n"
            f"Requested strategy: '{strategy}'\n"
            f"For MVP, use vector store search directly:\n"
            f"  query_emb = embedder.embed_texts([query])[0]\n"
            f"  results = vector_store.search(query_emb, top_k={config.top_k})"
        )

        # Placeholder - to be implemented in Phase 2
        raise NotImplementedError(
            f"Retrieval strategies not yet implemented.\n"
            f"Requested: {strategy}\n\n"
            f"For MVP, use this pattern instead:\n\n"
            f"# Embed query\n"
            f"query_embedding = embedder.embed_texts([query])[0]\n\n"
            f"# Search vector store\n"
            f"results = vector_store.search(\n"
            f"    query_embedding,\n"
            f"    top_k={config.top_k},\n"
            f"    filters={{'domain': 'your_domain'}}\n"
            f")\n\n"
            f"# Process results\n"
            f"for result in results:\n"
            f"    print(result['document'])\n\n"
            f"Retrieval strategies (hybrid, reranking, etc.) will be added in Phase 2."
        )


# =============================================================================
# FUTURE: Retrieval Strategy Interfaces
# =============================================================================

"""
When implementing retrieval strategies, create these files:

1. core/interfaces/retrieval_interface.py
   ----------------------------------------
   from abc import ABC, abstractmethod
   from typing import List, Dict, Any, Optional

   class RetrieverInterface(ABC):
       '''Abstract interface for retrieval strategies.'''

       @abstractmethod
       def retrieve(
           self,
           query: str,
           top_k: int,
           filters: Optional[Dict[str, Any]] = None
       ) -> List[Dict]:
           '''Retrieve relevant documents for query.'''
           pass

2. core/retrieval/hybrid_retriever.py
   -----------------------------------
   class HybridRetriever(RetrieverInterface):
       '''Combines dense (semantic) and sparse (keyword) search.'''

       def __init__(self, vector_store, embedder, sparse_index, alpha=0.7, ...):
           self.vector_store = vector_store
           self.embedder = embedder
           self.sparse_index = sparse_index
           self.alpha = alpha  # Weight for dense vs sparse

       def retrieve(self, query, top_k, filters=None):
           # 1. Dense search (semantic)
           query_emb = self.embedder.embed_texts([query])[0]
           dense_results = self.vector_store.search(query_emb, top_k*2, filters)

           # 2. Sparse search (keyword/BM25)
           sparse_results = self.sparse_index.search(query, top_k*2, filters)

           # 3. Combine with alpha weighting
           combined = self._alpha_fusion(dense_results, sparse_results, self.alpha)

           # 4. Return top_k
           return combined[:top_k]

       def _alpha_fusion(self, dense, sparse, alpha):
           # Weighted combination: alpha * dense + (1-alpha) * sparse
           # Normalize scores, merge, re-rank
           ...

3. core/retrieval/dense_retriever.py
   ----------------------------------
   class DenseRetriever(RetrieverInterface):
       '''Pure semantic/vector search.'''

       def __init__(self, vector_store, embedder, top_k=10):
           self.vector_store = vector_store
           self.embedder = embedder
           self.top_k = top_k

       def retrieve(self, query, top_k, filters=None):
           query_emb = self.embedder.embed_texts([query])[0]
           return self.vector_store.search(query_emb, top_k, filters)

4. core/retrieval/sparse_retriever.py
   -----------------------------------
   class SparseRetriever(RetrieverInterface):
       '''Pure keyword/BM25 search.'''

       def __init__(self, sparse_index, top_k=10):
           self.sparse_index = sparse_index  # BM25Index, ElasticsearchIndex, etc.
           self.top_k = top_k

       def retrieve(self, query, top_k, filters=None):
           return self.sparse_index.search(query, top_k, filters)

5. core/retrieval/bm25_index.py (for sparse search)
   -------------------------------------------------
   from rank_bm25 import BM25Okapi

   class BM25Index:
       '''BM25 keyword search index.'''

       def __init__(self, documents, metadata):
           # Tokenize documents
           tokenized_docs = [doc.split() for doc in documents]

           # Build BM25 index
           self.bm25 = BM25Okapi(tokenized_docs)
           self.documents = documents
           self.metadata = metadata

       def search(self, query, top_k, filters=None):
           # Tokenize query
           query_tokens = query.split()

           # BM25 scoring
           scores = self.bm25.get_scores(query_tokens)

           # Get top_k
           top_indices = scores.argsort()[-top_k:][::-1]

           # Format results
           results = []
           for idx in top_indices:
               results.append({
                   'document': self.documents[idx],
                   'metadata': self.metadata[idx],
                   'score': scores[idx]
               })

           return results

6. Update models/domain_config.py
   -------------------------------
   Add to RetrievalConfig:

   class RetrievalConfig(BaseModel):
       strategy: str = "hybrid"
       alpha: float = 0.7
       top_k: int = 10
       enable_metadata_filtering: bool = True
       normalize_scores: bool = True

       # Advanced options (future)
       enable_reranking: bool = False
       reranker_model: Optional[str] = None
       enable_query_expansion: bool = False
       max_query_expansions: int = 3
"""

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of RetrievalFactory (placeholder implementation).
    Run: python core/factories/retrieval_factory.py
    """

    import logging
    from models.domain_config import RetrievalConfig

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("RetrievalFactory Usage Examples (Placeholder)")
    print("=" * 70)

    # Example 1: Attempting to create retriever (will show error)
    print("\n1. Attempting to Create Retriever")
    print("-" * 70)

    config = RetrievalConfig(
        strategy="hybrid",
        alpha=0.7,
        top_k=10
    )

    print(f"Config: strategy={config.strategy}, alpha={config.alpha}, top_k={config.top_k}")
    print("\nAttempting to create retriever...")

    try:
        retriever = RetrievalFactory.create_retriever(
            config,
            vector_store=None,  # Placeholder
            embedder=None  # Placeholder
        )
    except NotImplementedError as e:
        print(f"\n⚠️  Expected NotImplementedError (placeholder):")
        print(f"\n{str(e)}")

    # Example 2: MVP Pattern (how to do retrieval without factory)
    print("\n2. MVP Pattern: Direct Vector Store Search")
    print("-" * 70)
