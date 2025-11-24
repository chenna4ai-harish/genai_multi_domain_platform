"""

core/retrievals/bm25_retrieval.py

BM25 (Best Matching 25) sparse retrieval implementation for Phase 2.

What is BM25?
-------------
BM25 is a ranking algorithm based on term frequency (TF) and inverse document
frequency (IDF). It's a SPARSE retrieval method that uses keyword matching
rather than semantic embeddings.

How BM25 Works:
---------------
1. **Tokenize**: Break query and documents into terms/words
2. **Term Frequency (TF)**: How often does a term appear in a document?
3. **Inverse Document Frequency (IDF)**: How rare is the term across all documents?
4. **Score**: Combine TF and IDF with tuning parameters (k1, b)

BM25 vs Vector Search:
----------------------
| Aspect          | BM25 (Sparse)           | Vector (Dense)           |
|-----------------|-------------------------|--------------------------|
| Match Type      | Exact term matching     | Semantic similarity      |
| Best For        | Specific terms, IDs     | Conceptual queries       |
| Example Query   | "Form W-2", "401k"      | "employee benefits"      |
| Speed           | Very fast (inverted idx)| Fast (ANN search)        |
| Quality         | Poor for paraphrases    | Poor for exact terms     |
| Index Size      | Small (terms only)      | Large (dense vectors)    |

When to Use BM25:
-----------------
✅ Queries with specific terms: "section 401k", "form W-2"
✅ Acronyms and IDs: "ISO-9001", "SOC-2"
✅ Proper names: "John Smith", "HR Department"
✅ Exact phrase matching: "as per company policy"
❌ Paraphrased queries: "time off" vs "vacation days"
❌ Semantic queries: "how to request leave"

Phase 2 Recommendation:
-----------------------
Use BM25 as part of HYBRID retrieval (combines BM25 + Vector).
Hybrid gives you the best of both worlds!

Installation:
-------------
pip install rank-bm25

Example Usage:
--------------
from core.retrievals.bm25_retrieval import BM25Retrieval

# Get corpus from vector store (Phase 2)
corpus, doc_ids = vector_store.get_all_documents()

# Initialize BM25 index
bm25 = BM25Retrieval(corpus=corpus, doc_ids=doc_ids)

# Search
results = bm25.retrieve(
    query_text="401k matching policy",
    top_k=10,
    metadata_filters={"domain": "hr"}  # Post-filtering
)

References:
-----------
- Phase 2 Spec: Section 8.2 (BM25 Implementation)
- BM25 Paper: https://en.wikipedia.org/wiki/Okapi_BM25
- rank-bm25 Library: https://github.com/dorianbrown/rank_bm25

"""

from typing import List, Dict, Any, Optional
import logging
import re

# Import BM25 library
try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# Import interface
from core.interfaces.retrieval_interface import RetrievalInterface

# Configure logging
logger = logging.getLogger(__name__)


class BM25Retrieval(RetrievalInterface):
    """
    BM25 sparse retrieval strategy (keyword-based search).

    Implements the RetrievalInterface using BM25 algorithm for term-based
    document ranking. Excellent for exact term matching and specific queries.

    How It Works:
    -------------
    1. **Build Index**: Tokenize all documents into terms, build inverted index
    2. **Query**: Tokenize query, score documents using BM25 formula
    3. **Rank**: Return top-k documents sorted by BM25 score
    4. **Filter**: Apply metadata filters (post-filtering, not pre-filtering)

    BM25 Formula:
    -------------
    For each term in query:
        IDF(term) = log((N - df(term) + 0.5) / (df(term) + 0.5))
        TF(term, doc) = (f(term, doc) * (k1 + 1)) /
                        (f(term, doc) + k1 * (1 - b + b * |doc| / avgdl))
        Score(doc) = Σ IDF(term) * TF(term, doc)

    Where:
    - N: Total number of documents
    - df(term): Documents containing term
    - f(term, doc): Term frequency in document
    - |doc|: Document length
    - avgdl: Average document length
    - k1: Term frequency saturation (default: 1.5)
    - b: Length normalization (default: 0.75)

    Characteristics:
    ----------------
    - **Speed**: Very fast (inverted index lookup)
    - **Quality**: Excellent for exact terms, poor for paraphrases
    - **Index Size**: Small (only term counts)
    - **Filtering**: Post-filtering (not as efficient as vector store)

    Parameters:
    -----------
    corpus : List[str]
        List of all document texts (from vector_store.get_all_documents())
    doc_ids : List[str]
        List of corresponding document IDs (same order as corpus)
    metadata : List[Dict], optional
        List of metadata dicts for post-filtering
    k1 : float
        BM25 term frequency saturation parameter (default: 1.5)
        Higher = more weight to term frequency
    b : float
        BM25 length normalization parameter (default: 0.75)
        Higher = more penalty for long documents

    Example:
    --------
    # Get corpus from vector store
    corpus, doc_ids = vector_store.get_all_documents()

    # Get metadata for filtering (optional but recommended)
    metadata = [{'domain': 'hr', 'doc_type': 'policy'}, ...]

    # Initialize BM25
    bm25 = BM25Retrieval(
        corpus=corpus,
        doc_ids=doc_ids,
        metadata=metadata,
        k1=1.5,
        b=0.75
    )

    # Search
    results = bm25.retrieve(
        query_text="401k matching contribution",
        metadata_filters={"domain": "hr"},
        top_k=10
    )
    """

    def __init__(
            self,
            corpus: List[str],
            doc_ids: List[str],
            metadata: Optional[List[Dict[str, Any]]] = None,
            k1: float = 1.5,
            b: float = 0.75
    ):
        """
        Initialize BM25 retrieval with corpus.

        Parameters:
        -----------
        corpus : List[str]
            List of all document texts
        doc_ids : List[str]
            List of corresponding document IDs (same order)
        metadata : List[Dict], optional
            List of metadata dicts for filtering (same order)
        k1 : float
            BM25 k1 parameter (term frequency saturation)
        b : float
            BM25 b parameter (length normalization)

        Raises:
        -------
        ImportError:
            If rank-bm25 not installed
        ValueError:
            If corpus and doc_ids lengths don't match
        """
        if not BM25_AVAILABLE:
            raise ImportError(
                "rank-bm25 is not installed.\n"
                "Install with: pip install rank-bm25"
            )

        if len(corpus) != len(doc_ids):
            raise ValueError(
                f"Corpus length ({len(corpus)}) must match doc_ids length ({len(doc_ids)})"
            )

        if metadata is not None and len(metadata) != len(corpus):
            raise ValueError(
                f"Metadata length ({len(metadata)}) must match corpus length ({len(corpus)})"
            )

        logger.info(
            f"Initializing BM25Retrieval:\n"
            f"  Corpus size: {len(corpus):,} documents\n"
            f"  Parameters: k1={k1}, b={b}"
        )

        self.corpus = corpus
        self.doc_ids = doc_ids
        self.metadata = metadata or [{}] * len(corpus)
        self.k1 = k1
        self.b = b

        # Tokenize corpus
        logger.info("Tokenizing corpus...")
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]

        # Build BM25 index
        logger.info("Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

        logger.info(
            f"✅ BM25 index built:\n"
            f"   Documents: {len(corpus):,}\n"
            f"   Avg doc length: {sum(len(doc.split()) for doc in corpus) / len(corpus):.1f} terms"
        )

    def retrieve(
            self,
            query_text: str,
            metadata_filters: Optional[Dict[str, Any]] = None,
            top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using BM25 keyword search.

        Implements RetrievalInterface.retrieve() using BM25 algorithm.

        Parameters:
        -----------
        query_text : str
            Natural language query
        metadata_filters : Dict[str, Any], optional
            Metadata filters (applied as post-filtering)
        top_k : int
            Number of results to return

        Returns:
        --------
        List[Dict[str, Any]]:
            List of results sorted by BM25 score (descending)
            Each result contains:
            - 'id': Document ID
            - 'score': Normalized BM25 score (0.0-1.0)
            - 'document': Document text
            - 'metadata': Document metadata

        Example:
        --------
        results = bm25.retrieve(
            query_text="401k employer matching",
            metadata_filters={"domain": "hr", "doc_type": "policy"},
            top_k=5
        )

        for result in results:
            print(f"{result['score']:.3f}: {result['document'][:100]}...")
        """
        # Validate input
        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty")

        if top_k < 1 or top_k > 1000:
            raise ValueError(f"top_k must be between 1 and 1000, got {top_k}")

        logger.debug(f"BM25 search: '{query_text}' (top_k={top_k})")

        # Tokenize query
        query_tokens = self._tokenize(query_text)
        logger.debug(f"Query tokens: {query_tokens}")

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Create (score, index) pairs
        scored_docs = [(score, idx) for idx, score in enumerate(scores)]

        # Sort by score (descending)
        scored_docs.sort(reverse=True, key=lambda x: x[0])

        # Apply metadata filters (post-filtering)
        if metadata_filters:
            logger.debug(f"Applying metadata filters: {metadata_filters}")
            filtered_docs = []

            for score, idx in scored_docs:
                if self._matches_filters(self.metadata[idx], metadata_filters):
                    filtered_docs.append((score, idx))

            scored_docs = filtered_docs
            logger.debug(f"Filtered to {len(scored_docs)} documents")

        # Take top-k
        top_docs = scored_docs[:top_k]

        # Normalize scores to 0.0-1.0 range
        if top_docs:
            max_score = max(score for score, _ in top_docs)
            min_score = min(score for score, _ in top_docs)
            score_range = max_score - min_score if max_score > min_score else 1.0
        else:
            max_score = 1.0
            min_score = 0.0
            score_range = 1.0

        # Format results
        results = []
        for score, idx in top_docs:
            # Normalize score to 0.0-1.0
            if score_range > 0:
                normalized_score = (score - min_score) / score_range
            else:
                normalized_score = 1.0 if score > 0 else 0.0

            results.append({
                'id': self.doc_ids[idx],
                'score': normalized_score,
                'document': self.corpus[idx],
                'metadata': self.metadata[idx]
            })

        logger.debug(f"BM25 retrieved {len(results)} results")

        return results

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms for BM25 index.

        Simple tokenization strategy:
        1. Lowercase
        2. Split on whitespace and punctuation
        3. Remove empty tokens

        For production, consider:
        - Stemming (Porter stemmer)
        - Lemmatization (spaCy)
        - Stopword removal
        - N-grams for phrases

        Parameters:
        -----------
        text : str
            Text to tokenize

        Returns:
        --------
        List[str]:
            List of tokens

        Example:
        --------
        tokens = self._tokenize("The 401k matching is 5%")
        # Returns: ['the', '401k', 'matching', 'is', '5']
        """
        # Convert to lowercase
        text = text.lower()

        # Split on whitespace and punctuation (keep alphanumeric)
        # This regex keeps numbers and letters together (e.g., "401k")
        tokens = re.findall(r'\b\w+\b', text)

        # Remove empty tokens
        tokens = [t for t in tokens if t]

        return tokens

    def _matches_filters(
            self,
            metadata: Dict[str, Any],
            filters: Dict[str, Any]
    ) -> bool:
        """
        Check if metadata matches filters (post-filtering).

        Simple exact matching strategy. For production, consider:
        - Range queries: {"page_num": {"$gt": 10}}
        - Array contains: {"tags": {"$contains": "important"}}
        - Regex matching: {"title": {"$regex": "policy.*2025"}}

        Parameters:
        -----------
        metadata : Dict[str, Any]
            Document metadata
        filters : Dict[str, Any]
            Filter conditions

        Returns:
        --------
        bool:
            True if metadata matches all filters

        Example:
        --------
        metadata = {"domain": "hr", "doc_type": "policy"}
        filters = {"domain": "hr"}
        # Returns: True

        metadata = {"domain": "finance", "doc_type": "faq"}
        filters = {"domain": "hr"}
        # Returns: False
        """
        for key, value in filters.items():
            # Check if key exists in metadata
            if key not in metadata:
                return False

            # Check if values match (exact match)
            if metadata[key] != value:
                return False

        return True

    def update_corpus(
            self,
            corpus: List[str],
            doc_ids: List[str],
            metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Update BM25 index with new corpus.

        Call this when documents are added/removed/updated in vector store.
        This rebuilds the entire BM25 index.

        For incremental updates, consider using Whoosh or Elasticsearch.

        Parameters:
        -----------
        corpus : List[str]
            New list of all document texts
        doc_ids : List[str]
            New list of corresponding document IDs
        metadata : List[Dict], optional
            New list of metadata dicts

        Example:
        --------
        # Get updated corpus from vector store
        corpus, doc_ids = vector_store.get_all_documents()

        # Rebuild BM25 index
        bm25.update_corpus(corpus, doc_ids)
        """
        logger.info(f"Updating BM25 index with {len(corpus):,} documents...")

        self.__init__(
            corpus=corpus,
            doc_ids=doc_ids,
            metadata=metadata,
            k1=self.k1,
            b=self.b
        )


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of BM25Retrieval usage.
    Run: python core/retrievals/bm25_retrieval.py
    """
    import logging

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("BM25 Retrieval - Sparse Keyword-Based Search")
    print("=" * 70)

    if not BM25_AVAILABLE:
        print("\n❌ rank-bm25 not installed!")
        print("Install with: pip install rank-bm25")
        exit(1)

    # Example corpus (simulating vector store corpus)
    print("\n1. Sample Corpus")
    print("-" * 70)

    corpus = [
        "Employees are entitled to 15 vacation days per year. Unused vacation days can be carried over.",
        "The 401k employer matching contribution is 5% of salary up to the IRS limit.",
        "Health insurance coverage includes medical, dental, and vision benefits for employees and dependents.",
        "To submit an expense report, use Form W-2 and attach all receipts.",
        "Paid time off (PTO) policy includes vacation, sick leave, and personal days totaling 20 days annually.",
        "The company matches 401k contributions dollar-for-dollar up to 5% of compensation.",
        "Annual performance reviews are conducted in December for all employees.",
        "Remote work policy allows employees to work from home up to 3 days per week."
    ]

    doc_ids = [f"doc-{i + 1}" for i in range(len(corpus))]

    metadata = [
        {"domain": "hr", "doc_type": "policy", "page_num": 5},
        {"domain": "hr", "doc_type": "policy", "page_num": 12},
        {"domain": "hr", "doc_type": "policy", "page_num": 8},
        {"domain": "finance", "doc_type": "faq", "page_num": 3},
        {"domain": "hr", "doc_type": "policy", "page_num": 5},
        {"domain": "hr", "doc_type": "policy", "page_num": 13},
        {"domain": "hr", "doc_type": "policy", "page_num": 20},
        {"domain": "hr", "doc_type": "policy", "page_num": 15}
    ]

    print(f"Corpus: {len(corpus)} documents")

    # Initialize BM25
    print("\n2. Initialize BM25")
    print("-" * 70)

    bm25 = BM25Retrieval(corpus=corpus, doc_ids=doc_ids, metadata=metadata)

    # Example 1: Search for specific term
    print("\n3. Example 1: Specific Term Search ('401k')")
    print("-" * 70)

    results = bm25.retrieve(query_text="401k matching", top_k=3)

    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.3f}")
        print(f"   ID: {result['id']}")
        print(f"   Text: {result['document'][:80]}...")
        print()

    # Example 2: Search with metadata filter
    print("\n4. Example 2: Search with Metadata Filter")
    print("-" * 70)

    results = bm25.retrieve(
        query_text="vacation days",
        metadata_filters={"domain": "hr", "doc_type": "policy"},
        top_k=3
    )

    print(f"Query: 'vacation days' (filtered: domain=hr, doc_type=policy)")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.3f}")
        print(f"   Text: {result['document'][:80]}...")
        print()

    # Example 3: Comparison - BM25 vs Dense
    print("\n5. BM25 Strengths")
    print("-" * 70)
    print("""
BM25 excels at:
✅ Exact term matching: "401k", "Form W-2"
✅ Acronyms and IDs: "PTO", "IRS"
✅ Numbers: "5%", "15 days"
✅ Specific phrases: "employer matching contribution"

BM25 struggles with:
❌ Paraphrases: "vacation" vs "time off"
❌ Synonyms: "benefits" vs "perks"
❌ Semantic queries: "how to request leave"

Recommendation: Use BM25 as part of HYBRID retrieval!
    """)

    print("\n" + "=" * 70)
    print("✅ BM25 Retrieval ready to use!")
    print("=" * 70)
