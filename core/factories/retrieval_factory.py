"""

core/factories/retrieval_factory.py

Factory for creating retrieval strategy instances based on configuration.

What is This Factory?
----------------------
The RetrievalFactory creates the appropriate retrieval strategy based on your
domain configuration. It's part of the Factory Pattern that enables swapping
retrieval methods WITHOUT changing any calling code.

Phase 2 Enhancement:
--------------------
Phase 2 adds support for MULTIPLE retrieval strategies:
1. **vector_similarity**: Pure dense/semantic search (Vector only)
2. **bm25**: Pure sparse/keyword search (BM25 only)
3. **hybrid**: Dense + Sparse combined (RECOMMENDED) ⭐

Why Use a Factory?
------------------
- **Config-Driven**: Change retrieval strategy via YAML, no code changes
- **Clean Code**: Hides complex initialization logic
- **Testing**: Easy to mock different strategies
- **Flexibility**: Add new strategies without changing callers

Example YAML Configuration:
---------------------------
retrieval:
  strategies: ["hybrid"]           # List of enabled strategies
  top_k: 10
  similarity: "cosine"
  hybrid:
    alpha: 0.7                     # 70% semantic, 30% keywords
    normalize_scores: true

How It Works:
-------------
1. Load domain config from YAML
2. RetrievalFactory reads config.retrieval.strategies
3. For each strategy, factory creates appropriate retriever
4. Returns dict of {strategy_name: retriever_instance}
5. Pipeline uses retrievers based on config

Example Usage:
--------------
# In DocumentPipeline
from core.factories.retrieval_factory import RetrievalFactory

# Create retrievers from config
retrievers = RetrievalFactory.create_retrievers(
    config=domain_config,
    vector_store=vector_store,
    embedding_model=embedding_model
)

# Use hybrid retriever (recommended)
hybrid_retriever = retrievers['hybrid']
results = hybrid_retriever.retrieve(query_text, filters, top_k)

References:
-----------
- Phase 2 Spec: Section 9 (Factory Layer Enhancements)
- Factory Pattern: https://refactoring.guru/design-patterns/factory-method

"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent  # Go up two levels from config_manager.py
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))



from typing import Dict, Any, Optional
import logging

# Import interfaces
from core.interfaces.retrieval_interface import RetrievalInterface
from core.interfaces.vectorstore_interface import VectorStoreInterface
from core.interfaces.embedding_interface import EmbeddingInterface

# Import retrieval implementations
from core.retrievals.vector_similarity_retrieval import VectorSimilarityRetrieval
from core.retrievals.bm25_retrieval import BM25Retrieval
from core.retrievals.hybrid_retrieval import HybridRetrieval

# Import config models
from models.domain_config import DomainConfig

# Configure logging
logger = logging.getLogger(__name__)


class RetrievalFactory:
    """
    Factory for creating retrieval strategy instances.

    Creates appropriate retrieval strategy based on domain configuration.
    Supports multiple strategies simultaneously (Phase 2 feature).

    Supported Strategies:
    ---------------------
    1. **vector_similarity** (Dense/Semantic)
       - Uses: VectorSimilarityRetrieval
       - Best for: Paraphrases, concepts, synonyms
       - Speed: Fast (~50-100ms)
       - Dependencies: vector_store, embedding_model

    2. **bm25** (Sparse/Keywords)
       - Uses: BM25Retrieval
       - Best for: Exact terms, acronyms, IDs
       - Speed: Very fast (~10-20ms)
       - Dependencies: vector_store (for corpus)

    3. **hybrid** (Dense + Sparse) ⭐ RECOMMENDED
       - Uses: HybridRetrieval
       - Best for: All query types (most robust)
       - Speed: Moderate (~100-150ms)
       - Dependencies: vector_store, embedding_model

    Methods:
    --------
    - create_retrievers(): Create multiple retrievers from config
    - create_retriever(): Create single retriever by name

    Example:
    --------
    # Create all configured retrievers
    retrievers = RetrievalFactory.create_retrievers(
        config=domain_config,
        vector_store=chromadb_store,
        embedding_model=sentence_transformer
    )

    # Use hybrid retriever
    results = retrievers['hybrid'].retrieve("vacation policy", top_k=10)
    """

    @staticmethod
    def create_retrievers(
            config: DomainConfig,
            vector_store: VectorStoreInterface,
            embedding_model: EmbeddingInterface
    ) -> Dict[str, RetrievalInterface]:
        """
        Create all retrieval strategies specified in config.

        This is the PRIMARY factory method for Phase 2.
        Creates multiple retrievers based on config.retrieval.strategies list.

        Parameters:
        -----------
        config : DomainConfig
            Domain configuration with retrieval settings
        vector_store : VectorStoreInterface
            Vector store instance (ChromaDB, Pinecone, etc.)
        embedding_model : EmbeddingInterface
            Embedding model instance (Sentence-Transformers, Gemini, etc.)

        Returns:
        --------
        Dict[str, RetrievalInterface]:
            Dictionary mapping strategy name to retriever instance
            Example: {
                'hybrid': HybridRetrieval(...),
                'vector_similarity': VectorSimilarityRetrieval(...),
                'bm25': BM25Retrieval(...)
            }

        Raises:
        -------
        ValueError:
            If no strategies specified in config
            If unknown strategy name in config

        Example:
        --------
        # Config YAML:
        # retrieval:
        #   strategies: ["hybrid", "vector_similarity"]
        #   hybrid:
        #     alpha: 0.7

        retrievers = RetrievalFactory.create_retrievers(
            config=domain_config,
            vector_store=vector_store,
            embedding_model=embedder
        )

        # Use hybrid
        hybrid_results = retrievers['hybrid'].retrieve(query_text, top_k=10)

        # Use vector_similarity
        dense_results = retrievers['vector_similarity'].retrieve(query_text, top_k=10)
        """
        logger.info(f"Creating retrievers for strategies: {config.retrieval.strategies}")

        # Validate strategies list
        if not config.retrieval.strategies:
            raise ValueError(
                "No retrieval strategies specified in config.retrieval.strategies"
            )

        retrievers = {}

        # Create each configured strategy
        for strategy_name in config.retrieval.strategies:
            logger.info(f"Creating retriever: {strategy_name}")

            try:
                retriever = RetrievalFactory.create_retriever(
                    strategy_name=strategy_name,
                    config=config,
                    vector_store=vector_store,
                    embedding_model=embedding_model
                )

                retrievers[strategy_name] = retriever
                logger.info(f"✅ Created {strategy_name} retriever")

            except Exception as e:
                logger.error(f"Failed to create {strategy_name} retriever: {e}")
                raise

        logger.info(f"✅ Created {len(retrievers)} retrievers: {list(retrievers.keys())}")

        return retrievers

    @staticmethod
    def create_retriever(
            strategy_name: str,
            config: DomainConfig,
            vector_store: VectorStoreInterface,
            embedding_model: EmbeddingInterface
    ) -> RetrievalInterface:
        """
        Create a single retrieval strategy by name.

        Parameters:
        -----------
        strategy_name : str
            Retrieval strategy name: "vector_similarity", "bm25", or "hybrid"
        config : DomainConfig
            Domain configuration
        vector_store : VectorStoreInterface
            Vector store instance
        embedding_model : EmbeddingInterface
            Embedding model instance

        Returns:
        --------
        RetrievalInterface:
            Retriever instance

        Raises:
        -------
        ValueError:
            If unknown strategy name

        Example:
        --------
        # Create hybrid retriever
        hybrid = RetrievalFactory.create_retriever(
            strategy_name="hybrid",
            config=domain_config,
            vector_store=vector_store,
            embedding_model=embedder
        )
        """
        logger.debug(f"Creating retriever: {strategy_name}")

        # Strategy 1: Vector Similarity (Dense/Semantic)
        if strategy_name == "vector_similarity":
            return RetrievalFactory._create_vector_similarity_retrieval(
                vector_store=vector_store,
                embedding_model=embedding_model
            )

        # Strategy 2: BM25 (Sparse/Keywords)
        elif strategy_name == "bm25":
            return RetrievalFactory._create_bm25_retrieval(
                vector_store=vector_store
            )

        # Strategy 3: Hybrid (Dense + Sparse) - RECOMMENDED
        elif strategy_name == "hybrid":
            return RetrievalFactory._create_hybrid_retrieval(
                config=config,
                vector_store=vector_store,
                embedding_model=embedding_model
            )

        else:
            raise ValueError(
                f"Unknown retrieval strategy: '{strategy_name}'\n"
                f"Supported strategies: vector_similarity, bm25, hybrid"
            )

    # =========================================================================
    # PRIVATE FACTORY METHODS (Strategy-Specific)
    # =========================================================================

    @staticmethod
    def _create_vector_similarity_retrieval(
            vector_store: VectorStoreInterface,
            embedding_model: EmbeddingInterface
    ) -> VectorSimilarityRetrieval:
        """
        Create vector similarity retrieval (dense/semantic search).

        Simple strategy - delegates to existing vector store and embedder.
        No additional configuration needed.

        Returns:
        --------
        VectorSimilarityRetrieval:
            Dense retrieval instance
        """
        logger.debug("Creating VectorSimilarityRetrieval...")

        retriever = VectorSimilarityRetrieval(
            vector_store=vector_store,
            embedding_model=embedding_model
        )

        logger.debug(
            f"✅ VectorSimilarityRetrieval created: "
            f"model={embedding_model.get_model_name()}"
        )

        return retriever

    @staticmethod
    def _create_bm25_retrieval(
            vector_store: VectorStoreInterface
    ) -> BM25Retrieval:
        """
        Create BM25 retrieval (sparse/keyword search).

        Workflow:
        ---------
        1. Get corpus from vector store (all document texts)
        2. Build BM25 index from corpus
        3. Return BM25Retrieval instance

        Note: This is called once at initialization. For large corpora,
        consider caching or incremental updates.

        Returns:
        --------
        BM25Retrieval:
            Sparse retrieval instance

        Raises:
        -------
        RuntimeError:
            If vector store doesn't support get_all_documents()
        """
        logger.debug("Creating BM25Retrieval...")

        # Get corpus from vector store (Phase 2 requirement)
        try:
            logger.info("Fetching corpus from vector store for BM25 index...")
            corpus, doc_ids = vector_store.get_all_documents()
            logger.info(f"Retrieved {len(corpus):,} documents for BM25 index")

        except Exception as e:
            logger.error(f"Failed to get corpus from vector store: {e}")
            raise RuntimeError(
                f"Vector store must implement get_all_documents() for BM25.\n"
                f"Error: {e}"
            )

        # Create BM25 retrieval
        retriever = BM25Retrieval(
            corpus=corpus,
            doc_ids=doc_ids,
            k1=1.5,  # BM25 term frequency saturation
            b=0.75  # BM25 length normalization
        )

        logger.debug(
            f"✅ BM25Retrieval created: "
            f"corpus_size={len(corpus):,}, k1=1.5, b=0.75"
        )

        return retriever

    @staticmethod
    def _create_hybrid_retrieval(
            config: DomainConfig,
            vector_store: VectorStoreInterface,
            embedding_model: EmbeddingInterface
    ) -> HybridRetrieval:
        """
        Create hybrid retrieval (dense + sparse combined).

        This is the RECOMMENDED strategy for Phase 2 production use.

        Workflow:
        ---------
        1. Get corpus from vector store
        2. Create BM25 index from corpus
        3. Create HybridRetrieval combining vector + BM25
        4. Use alpha from config (default: 0.7)

        Returns:
        --------
        HybridRetrieval:
            Hybrid retrieval instance

        Raises:
        -------
        RuntimeError:
            If corpus retrieval or BM25 creation fails
        """
        logger.debug("Creating HybridRetrieval...")

        # Step 1: Get corpus from vector store
        try:
            logger.info("Fetching corpus from vector store for hybrid index...")
            corpus, doc_ids = vector_store.get_all_documents()
            logger.info(f"Retrieved {len(corpus):,} documents for hybrid index")

        except Exception as e:
            logger.error(f"Failed to get corpus from vector store: {e}")
            raise RuntimeError(
                f"Vector store must implement get_all_documents() for hybrid.\n"
                f"Error: {e}"
            )

        # Step 2: Create BM25 index
        logger.info("Building BM25 index for hybrid retrieval...")
        bm25_index = BM25Retrieval(
            corpus=corpus,
            doc_ids=doc_ids,
            k1=1.5,
            b=0.75
        )

        # Step 3: Get hybrid config (alpha, normalize)
        hybrid_config = config.retrieval.hybrid
        alpha = hybrid_config.alpha if hybrid_config else 0.7
        normalize_scores = hybrid_config.normalize_scores if hybrid_config else True

        # Step 4: Create hybrid retrieval
        retriever = HybridRetrieval(
            vector_store=vector_store,
            embedding_model=embedding_model,
            bm25_index=bm25_index,
            alpha=alpha,
            normalize_scores=normalize_scores
        )

        logger.debug(
            f"✅ HybridRetrieval created:\n"
            f"   Alpha: {alpha:.2f} (dense={alpha:.0%}, sparse={1 - alpha:.0%})\n"
            f"   Normalize: {normalize_scores}\n"
            f"   Corpus: {len(corpus):,} documents"
        )

        return retriever


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of RetrievalFactory usage.
    Run: python core/factories/retrieval_factory.py
    """
    import logging

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("RetrievalFactory - Create Retrieval Strategies from Config")
    print("=" * 70)

    print("""
RetrievalFactory Overview:
--------------------------

Purpose:
Creates appropriate retrieval strategy based on domain configuration.
Enables swapping retrieval methods via YAML config, no code changes!

Supported Strategies (Phase 2):
--------------------------------

1. vector_similarity (Dense/Semantic)
   - Pure vector search using embeddings
   - Best for: Paraphrases, concepts, synonyms
   - Speed: Fast (~50-100ms)
   - Config:
     retrieval:
       strategies: ["vector_similarity"]

2. bm25 (Sparse/Keywords)
   - Pure keyword search using BM25
   - Best for: Exact terms, acronyms, IDs
   - Speed: Very fast (~10-20ms)
   - Config:
     retrieval:
       strategies: ["bm25"]

3. hybrid (Dense + Sparse) ⭐ RECOMMENDED
   - Combines vector + BM25 with alpha weighting
   - Best for: All query types (most robust)
   - Speed: Moderate (~100-150ms)
   - Config:
     retrieval:
       strategies: ["hybrid"]
       hybrid:
         alpha: 0.7              # 70% semantic, 30% keywords
         normalize_scores: true

Multi-Strategy Support:
-----------------------
Phase 2 allows MULTIPLE strategies simultaneously:

retrieval:
  strategies: ["hybrid", "vector_similarity", "bm25"]
  hybrid:
    alpha: 0.7

Factory creates all 3 retrievers!
You can compare results or use different strategies for different query types.

Usage Pattern:
--------------

# Step 1: Load domain config
from core.config_manager import ConfigManager

config_mgr = ConfigManager()
domain_config = config_mgr.load_domain_config("hr")

# Step 2: Create vector store and embedder
from core.factories.vector_store_factory import VectorStoreFactory
from core.factories.embedding_factory import EmbeddingFactory

vector_store = VectorStoreFactory.create_vector_store(domain_config)
embedder = EmbeddingFactory.create_embedding_model(domain_config)

# Step 3: Create retrievers
from core.factories.retrieval_factory import RetrievalFactory

retrievers = RetrievalFactory.create_retrievers(
    config=domain_config,
    vector_store=vector_store,
    embedding_model=embedder
)

# Step 4: Use retrievers
hybrid = retrievers['hybrid']  # Recommended!
results = hybrid.retrieve(
    query_text="What is the vacation policy?",
    metadata_filters={"domain": "hr"},
    top_k=10
)

# Or use specific strategy
dense_only = retrievers.get('vector_similarity')
if dense_only:
    results = dense_only.retrieve(query_text, top_k=10)

Benefits:
---------
✅ Config-driven (change strategy via YAML)
✅ No code changes when switching strategies
✅ Easy A/B testing (run multiple strategies)
✅ Clean separation of concerns
✅ Testable (mock different strategies)

Phase 2 Integration:
---------------------
The factory is used by:
- DocumentPipeline: Creates retrievers at initialization
- DocumentService: Uses pipeline's retrievers for queries
- CLI Tools: Allows strategy selection via command-line

Configuration Examples:
-----------------------

# Development (fast, simple)
retrieval:
  strategies: ["vector_similarity"]
  top_k: 10

# Production (robust, recommended)
retrieval:
  strategies: ["hybrid"]
  top_k: 10
  hybrid:
    alpha: 0.7
    normalize_scores: true

# Experimentation (compare all)
retrieval:
  strategies: ["hybrid", "vector_similarity", "bm25"]
  top_k: 10
  hybrid:
    alpha: 0.7
    normalize_scores: true

Factory automatically creates all configured strategies!
    """)

    print("\n" + "=" * 70)
    print("✅ RetrievalFactory ready to use!")
    print("=" * 70)
