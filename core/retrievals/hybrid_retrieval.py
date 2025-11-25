"""

core/retrievals/hybrid_retrieval.py

Hybrid retrieval combining dense vector + sparse BM25 search for Phase 2.

What is Hybrid Retrieval?
--------------------------
Hybrid retrieval combines TWO complementary search methods:
1. **Dense (Vector)**: Semantic similarity using embeddings
2. **Sparse (BM25)**: Keyword matching using term frequencies

Why is this the BEST approach? Each method handles what the other misses!

The Power of Hybrid:
--------------------
Vector Search excels at:
✅ Paraphrases: "vacation" ↔ "time off"
✅ Synonyms: "benefits" ↔ "perks"
✅ Concepts: "how to request leave"
❌ BUT misses: Exact terms like "401k", "Form W-2"

BM25 Search excels at:
✅ Exact terms: "401k", "Form W-2", "section 5.2"
✅ Acronyms: "PTO", "HR", "ISO-9001"
✅ IDs/codes: "employee-12345"
❌ BUT misses: Paraphrases and synonyms

Hybrid = BOTH ✅✅
Handles semantic queries AND exact term matching!

How Hybrid Works (Alpha Weighting):
------------------------------------
1. Run BOTH searches in parallel (or sequentially)
2. Normalize scores to 0.0-1.0 range
3. Combine scores with alpha weighting:

   final_score = α × dense_score + (1 - α) × sparse_score

   Where:
   - α (alpha) = weight for dense/vector search (0.0 to 1.0)
   - Common values: 0.5 (equal), 0.7 (favor semantic), 0.3 (favor keywords)

4. Re-rank results by combined score
5. Return top-k

Example with α = 0.7 (70% semantic, 30% keywords):
---------------------------------------------------
Query: "401k matching policy"

Dense results:
- Doc A: "401k employer contribution" → 0.95
- Doc B: "retirement benefits overview" → 0.85

Sparse results:
- Doc A: "401k employer contribution" → 0.90
- Doc C: "401k plan details" → 0.80

Combined (α=0.7):
- Doc A: 0.7×0.95 + 0.3×0.90 = 0.935 ⭐ BEST
- Doc B: 0.7×0.85 + 0.3×0.00 = 0.595
- Doc C: 0.7×0.00 + 0.3×0.80 = 0.240

Doc A wins because it scores high in BOTH searches!

When to Use Hybrid:
-------------------
✅ General-purpose search (RECOMMENDED for production)
✅ Unknown query types (handles both semantic and exact)
✅ Enterprise search (documents with IDs, codes, concepts)
✅ Multi-domain systems (HR policies + technical docs)

Performance:
------------
- Speed: Slower than single method (~2x time)
  * Dense search: 50-100ms
  * Sparse search: 10-20ms
  * Merge overhead: 5-10ms
  * Total: ~65-130ms (still very fast!)

- Quality: BEST (handles all query types)

Phase 2 Recommendation:
-----------------------
Use Hybrid as your PRIMARY retrieval strategy!
Alpha = 0.7 is a good default (favor semantic understanding).

Installation:
-------------
No additional dependencies! Uses existing components:
- VectorSimilarityRetrieval (for dense search)
- BM25Retrieval (for sparse search)

Example Usage:
--------------
from core.retrievals.hybrid_retrieval import HybridRetrieval

# Initialize hybrid retriever
hybrid = HybridRetrieval(
    vector_store=vector_store,
    embedding_model=embedder,
    bm25_index=bm25_retrieval,
    alpha=0.7,  # 70% semantic, 30% keywords
    normalize_scores=True
)

# Search (handles BOTH semantic and exact queries)
results = hybrid.retrieve(
    query_text="What is the 401k matching policy?",
    metadata_filters={"domain": "hr"},
    top_k=10
)

References:
-----------
- Phase 2 Spec: Section 8.3 (Hybrid Retrieval Implementation)
- Research: "Hybrid Search: https://www.pinecone.io/learn/hybrid-search/

"""

from typing import List, Dict, Any, Optional
import logging
from collections import defaultdict

# Import interface and components
from core.interfaces.retrieval_interface import RetrievalInterface
from core.interfaces.vectorstore_interface import VectorStoreInterface
from core.interfaces.embedding_interface import EmbeddingInterface
from core.retrievals.bm25_retrieval import BM25Retrieval

# Configure logging
logger = logging.getLogger(__name__)


class HybridRetrieval(RetrievalInterface):
    """
    Hybrid retrieval combining dense vector + sparse BM25 search.

    **RECOMMENDED STRATEGY FOR PHASE 2 PRODUCTION USE**

    Combines the strengths of both retrieval methods:
    - Dense (Vector): Semantic similarity, handles paraphrases
    - Sparse (BM25): Keyword matching, handles exact terms

    Uses alpha-weighted score fusion to merge results from both searches.

    How It Works:
    -------------
    1. **Parallel Search**: Execute both dense and sparse searches
    2. **Normalize Scores**: Ensure both score ranges are 0.0-1.0
    3. **Alpha Weighting**: Combine scores with configurable weight
       final_score = α × dense_score + (1-α) × sparse_score
    4. **Merge**: Combine results from both searches
    5. **Re-rank**: Sort by combined score (descending)
    6. **Return**: Top-k results

    Alpha Parameter:
    ----------------
    Controls the balance between semantic and keyword search:

    - α = 1.0: Pure dense/semantic (same as VectorSimilarityRetrieval)
    - α = 0.7: Favor semantic (70% vector, 30% BM25) ← RECOMMENDED
    - α = 0.5: Equal weight (50% vector, 50% BM25)
    - α = 0.3: Favor keywords (30% vector, 70% BM25)
    - α = 0.0: Pure sparse/keywords (same as BM25Retrieval)

    Recommended: α = 0.7 for most use cases

    Architecture:
    -------------
    HybridRetrieval delegates to:
    - VectorStoreInterface + EmbeddingInterface (dense search)
    - BM25Retrieval (sparse search)

    Benefits:
    ---------
    ✅ **Robust**: Handles all query types (semantic + exact)
    ✅ **Best Quality**: Combines strengths of both methods
    ✅ **Flexible**: Tune alpha to favor semantic or keywords
    ✅ **Production-Ready**: Recommended for enterprise use

    Trade-offs:
    -----------
    ⚠️ **Speed**: ~2x slower than single method (still fast ~100ms)
    ⚠️ **Complexity**: Requires both vector store and BM25 index

    Parameters:
    -----------
    vector_store : VectorStoreInterface
        Vector store for dense search
    embedding_model : EmbeddingInterface
        Embedding model for query encoding
    bm25_index : BM25Retrieval
        BM25 index for sparse search
    alpha : float
        Dense search weight (0.0 to 1.0), default: 0.7
        Higher = favor semantic, Lower = favor keywords
    normalize_scores : bool
        Whether to normalize scores before merging (default: True)
        Recommended: True for fair combination

    Example:
    --------
    # Get corpus for BM25
    corpus, doc_ids = vector_store.get_all_documents()

    # Create BM25 index
    bm25 = BM25Retrieval(corpus=corpus, doc_ids=doc_ids)

    # Create hybrid retriever
    hybrid = HybridRetrieval(
        vector_store=vector_store,
        embedding_model=embedder,
        bm25_index=bm25,
        alpha=0.7,
        normalize_scores=True
    )

    # Search (handles both semantic and exact queries!)
    results = hybrid.retrieve(
        query_text="What is the 401k employer matching?",
        metadata_filters={"domain": "hr", "deprecated": False},
        top_k=10
    )

    # Results include BOTH:
    # - Semantically similar: "retirement benefits", "pension plan"
    # - Exact matches: "401k", "employer contribution"
    """

    def __init__(
            self,
            vector_store: VectorStoreInterface,
            embedding_model: EmbeddingInterface,
            bm25_index: BM25Retrieval,
            alpha: float = 0.7,
            normalize_scores: bool = True
    ):
        """
        Initialize hybrid retrieval.

        Parameters:
        -----------
        vector_store : VectorStoreInterface
            Vector store for dense search
        embedding_model : EmbeddingInterface
            Embedding model for query encoding
        bm25_index : BM25Retrieval
            BM25 index for sparse search
        alpha : float
            Dense search weight (0.0-1.0), default: 0.7
        normalize_scores : bool
            Normalize scores before merging (default: True)

        Raises:
        -------
        ValueError:
            If alpha is not between 0.0 and 1.0
        TypeError:
            If components don't implement required interfaces
        """
        # Validate alpha
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")

        # Validate interfaces
        if not isinstance(vector_store, VectorStoreInterface):
            raise TypeError(
                f"vector_store must implement VectorStoreInterface, "
                f"got {type(vector_store)}"
            )

        if not isinstance(embedding_model, EmbeddingInterface):
            raise TypeError(
                f"embedding_model must implement EmbeddingInterface, "
                f"got {type(embedding_model)}"
            )

        if not isinstance(bm25_index, BM25Retrieval):
            raise TypeError(
                f"bm25_index must be BM25Retrieval instance, "
                f"got {type(bm25_index)}"
            )

        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.bm25_index = bm25_index
        self.alpha = alpha
        self.normalize_scores = normalize_scores

        logger.info(
            f"HybridRetrieval initialized:\n"
            f"  Alpha: {alpha:.2f} (dense={alpha:.0%}, sparse={1 - alpha:.0%})\n"
            f"  Normalize scores: {normalize_scores}\n"
            f"  Embedding model: {embedding_model.get_model_name()}\n"
            f"  BM25 corpus: {len(bm25_index.corpus):,} documents"
        )

    def retrieve(
            self,
            query_text: str,
            metadata_filters: Optional[Dict[str, Any]] = None,
            top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid (dense + sparse) search.

        Implements RetrievalInterface.retrieve() using alpha-weighted
        combination of dense vector and sparse BM25 search.

        Workflow:
        ---------
        1. Validate input
        2. Execute dense search (vector similarity)
        3. Execute sparse search (BM25)
        4. Normalize scores (if enabled)
        5. Merge results with alpha weighting
        6. Re-rank by combined score
        7. Return top-k

        Parameters:
        -----------
        query_text : str
            Natural language query
        metadata_filters : Dict[str, Any], optional
            Metadata filters (applied to dense search, post-filtered for sparse)
        top_k : int
            Number of results to return

        Returns:
        --------
        List[Dict[str, Any]]:
            List of results sorted by hybrid score (descending)
            Each result contains:
            - 'id': str - Chunk ID
            - 'score': float - Combined hybrid score (0.0-1.0)
            - 'dense_score': float - Dense/vector score
            - 'sparse_score': float - Sparse/BM25 score
            - 'document': str - Chunk text
            - 'metadata': Dict - Chunk metadata

        Example:
        --------
        results = hybrid.retrieve(
            query_text="What is the 401k matching policy?",
            metadata_filters={"domain": "hr", "deprecated": False},
            top_k=10
        )

        for result in results:
            print(f"Score: {result['score']:.3f} "
                  f"(dense={result['dense_score']:.3f}, "
                  f"sparse={result['sparse_score']:.3f})")
            print(f"Text: {result['document'][:100]}...")
            print()
        """
        # Step 1: Validate input
        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty")

        if top_k < 1 or top_k > 1000:
            raise ValueError(f"top_k must be between 1 and 1000, got {top_k}")

        logger.info(
            f"Hybrid search: '{query_text}'\n"
            f"  Alpha: {self.alpha:.2f}\n"
            f"  Top-k: {top_k}\n"
            f"  Filters: {metadata_filters}"
        )

        # Step 2: Execute dense search (vector similarity)
        logger.debug("Executing dense search...")
        try:
            # Embed query
            query_embeddings = self.embedding_model.embed_texts([query_text])
            query_embedding = query_embeddings[0]

            # Search vector store
            # Retrieve more than top_k for better merging
            dense_top_k = top_k * 2
            dense_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=dense_top_k,
                filters=metadata_filters
            )

            logger.debug(f"Dense search returned {len(dense_results)} results")

        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            # Fallback to BM25 only if dense fails
            logger.warning("Falling back to BM25-only search")
            return self.bm25_index.retrieve(query_text, metadata_filters, top_k)

        # Step 3: Execute sparse search (BM25)
        logger.debug("Executing sparse search...")
        try:
            # BM25 search (post-filters metadata)
            sparse_top_k = top_k * 2
            sparse_results = self.bm25_index.retrieve(
                query_text=query_text,
                metadata_filters=metadata_filters,
                top_k=sparse_top_k
            )

            logger.debug(f"Sparse search returned {len(sparse_results)} results")

        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            # Fallback to dense only if sparse fails
            logger.warning("Falling back to dense-only search")
            return dense_results[:top_k]

        # Step 4: Normalize scores (if enabled)
        if self.normalize_scores:
            dense_results = self._normalize_scores(dense_results)
            sparse_results = self._normalize_scores(sparse_results)

        # Step 5: Merge results with alpha weighting
        logger.debug("Merging results...")
        merged_results = self._merge_results(dense_results, sparse_results)

        # Step 6: Sort by combined score (descending)
        merged_results.sort(key=lambda x: x['score'], reverse=True)

        # Step 7: Return top-k
        final_results = merged_results[:top_k]

        logger.info(
            f"✅ Hybrid retrieval complete:\n"
            f"   Dense results: {len(dense_results)}\n"
            f"   Sparse results: {len(sparse_results)}\n"
            f"   Merged results: {len(merged_results)}\n"
            f"   Returned: {len(final_results)}\n"
            f"   Top score: {final_results[0]['score']:.3f if final_results else 0:.3f}"
        )

        return final_results

    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        """
        Normalize scores to 0.0-1.0 range using min-max normalization.

        This ensures fair combination of dense and sparse scores which
        may have different ranges.

        Formula:
        --------
        normalized = (score - min_score) / (max_score - min_score)

        Parameters:
        -----------
        results : List[Dict]
            Results with 'score' field

        Returns:
        --------
        List[Dict]:
            Results with normalized scores (0.0-1.0)
        """
        if not results:
            return results

        # Extract scores
        scores = [r['score'] for r in results]

        # Get min and max
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        # Normalize (handle edge case where all scores are equal)
        if score_range > 0:
            for result in results:
                result['score'] = (result['score'] - min_score) / score_range
        else:
            # All scores equal, set to 1.0
            for result in results:
                result['score'] = 1.0

        return results

    def _merge_results(
            self,
            dense_results: List[Dict],
            sparse_results: List[Dict]
    ) -> List[Dict]:
        """
        Merge dense and sparse results using alpha-weighted scoring.

        Merging Strategy:
        -----------------
        1. Create dict keyed by document ID
        2. For each result, add weighted score:
           - Dense score gets weight alpha
           - Sparse score gets weight (1 - alpha)
        3. If document appears in both, scores are summed
        4. Return merged list

        Formula:
        --------
        If doc in BOTH dense and sparse:
            final_score = α × dense_score + (1-α) × sparse_score

        If doc in dense ONLY:
            final_score = α × dense_score

        If doc in sparse ONLY:
            final_score = (1-α) × sparse_score

        Parameters:
        -----------
        dense_results : List[Dict]
            Results from dense search
        sparse_results : List[Dict]
            Results from sparse search

        Returns:
        --------
        List[Dict]:
            Merged results with combined scores
        """
        # Create merged dict keyed by document ID
        merged = {}

        # Add dense results (weighted by alpha)
        for result in dense_results:
            doc_id = result['id']
            dense_score = result['score']

            merged[doc_id] = {
                'id': doc_id,
                'score': self.alpha * dense_score,  # Weighted dense score
                'dense_score': dense_score,
                'sparse_score': 0.0,
                'document': result.get('document', ''),
                'metadata': result.get('metadata', {})
            }

        # Add/merge sparse results (weighted by 1-alpha)
        for result in sparse_results:
            doc_id = result['id']
            sparse_score = result['score']

            if doc_id in merged:
                # Document in BOTH - add weighted sparse score
                merged[doc_id]['score'] += (1 - self.alpha) * sparse_score
                merged[doc_id]['sparse_score'] = sparse_score
            else:
                # Document in sparse ONLY
                merged[doc_id] = {
                    'id': doc_id,
                    'score': (1 - self.alpha) * sparse_score,  # Weighted sparse score
                    'dense_score': 0.0,
                    'sparse_score': sparse_score,
                    'document': result.get('document', ''),
                    'metadata': result.get('metadata', {})
                }

        # Convert to list
        merged_list = list(merged.values())

        logger.debug(
            f"Merged {len(dense_results)} dense + {len(sparse_results)} sparse "
            f"→ {len(merged_list)} unique results"
        )

        return merged_list

    def set_alpha(self, alpha: float) -> None:
        """
        Update alpha parameter (for experimentation/tuning).

        Parameters:
        -----------
        alpha : float
            New alpha value (0.0-1.0)

        Example:
        --------
        # Start with balanced search
        hybrid.set_alpha(0.5)

        # Favor semantic understanding
        hybrid.set_alpha(0.7)

        # Favor keyword matching
        hybrid.set_alpha(0.3)
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha}")

        old_alpha = self.alpha
        self.alpha = alpha

        logger.info(
            f"Alpha updated: {old_alpha:.2f} → {alpha:.2f}\n"
            f"  Dense weight: {alpha:.0%}\n"
            f"  Sparse weight: {1 - alpha:.0%}"
        )


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of HybridRetrieval usage.
    Run: python core/retrievals/hybrid_retrieval.py
    """
    import logging

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("Hybrid Retrieval - Dense + Sparse Search (RECOMMENDED)")
    print("=" * 70)

    print("""
Hybrid Retrieval: The Best of Both Worlds
------------------------------------------

Why Hybrid?
-----------
Combines two complementary search methods:
1. Dense (Vector): Semantic similarity
2. Sparse (BM25): Keyword matching

Each handles what the other misses!

Query Examples:
---------------

Query 1: "What is the 401k matching policy?"
  ✅ Dense finds: "retirement benefits", "pension plan"
  ✅ Sparse finds: "401k", "matching", "contribution"
  ✅ Hybrid finds: BOTH semantic + exact matches!

Query 2: "How do I request time off?"
  ✅ Dense finds: "vacation request", "PTO application"
  ✅ Sparse finds: "time off", "request procedure"
  ✅ Hybrid ranks documents high that match BOTH semantically and literally!

Alpha Parameter:
----------------
Controls balance between dense and sparse:

α = 1.0: Pure dense (semantic only)
α = 0.7: Favor semantic (RECOMMENDED) ← Most robust
α = 0.5: Equal weight
α = 0.3: Favor keywords
α = 0.0: Pure sparse (keywords only)

Formula:
final_score = α × dense_score + (1-α) × sparse_score

Integration:
------------

# Step 1: Get corpus for BM25 (from vector store)
from core.vectorstores.chromadb_store import ChromaDBStore

vector_store = ChromaDBStore(
    persist_directory="./data/chroma_db",
    collection_name="hr_collection"
)

corpus, doc_ids = vector_store.get_all_documents()

# Step 2: Create BM25 index
from core.retrievals.bm25_retrieval import BM25Retrieval

bm25 = BM25Retrieval(corpus=corpus, doc_ids=doc_ids)

# Step 3: Create embedder
from core.embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddings

embedder = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Step 4: Create hybrid retriever
from core.retrievals.hybrid_retrieval import HybridRetrieval

hybrid = HybridRetrieval(
    vector_store=vector_store,
    embedding_model=embedder,
    bm25_index=bm25,
    alpha=0.7,  # 70% semantic, 30% keywords
    normalize_scores=True
)

# Step 5: Search!
results = hybrid.retrieve(
    query_text="What is the 401k employer matching?",
    metadata_filters={"domain": "hr", "deprecated": False},
    top_k=10
)

# Display results
for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"  Dense: {result['dense_score']:.3f}")
    print(f"  Sparse: {result['sparse_score']:.3f}")
    print(f"  Text: {result['document'][:100]}...")
    print()

Performance:
------------
- Dense search: ~50-100ms
- Sparse search: ~10-20ms
- Merge overhead: ~5-10ms
- Total: ~65-130ms (still very fast!)

Quality: BEST (handles all query types)

Use Cases:
----------
✅ General-purpose search (most robust)
✅ Unknown query types (handles both semantic + exact)
✅ Enterprise search (policies + technical docs)
✅ Multi-domain systems
✅ Production deployments (RECOMMENDED)

Phase 2 Recommendation:
-----------------------
Use Hybrid as your PRIMARY retrieval strategy!
- Alpha = 0.7 is a good default
- Adjust based on your specific use case:
  * More semantic queries? → Higher alpha (0.8)
  * More exact term queries? → Lower alpha (0.5)

Configuration:
--------------
In your domain YAML config:

retrieval:
  strategies: ["hybrid"]
  top_k: 10
  hybrid:
    alpha: 0.7
    normalize_scores: true

The factory will create HybridRetrieval automatically!
    """)

    print("\n" + "=" * 70)
    print("✅ Hybrid Retrieval ready for production use!")
    print("=" * 70)
