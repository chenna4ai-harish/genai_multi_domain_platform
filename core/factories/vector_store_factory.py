"""

core/factories/vector_store_factory.py

Factory for creating vector store instances based on configuration (Phase 2).

Supported Vector Stores:
------------------------
✅ ChromaDB - Local, free, simple (IMPLEMENTED)
✅ Pinecone - Cloud, scalable, production (IMPLEMENTED)
⏳ Qdrant - Future (optional)
⏳ FAISS - Future (optional)

Phase 2 MVP requires only ChromaDB and Pinecone.
Additional stores can be added later without changing calling code.

"""

from typing import Dict, Any
import logging

# Import interfaces
from core.interfaces.vectorstore_interface import VectorStoreInterface

# Import IMPLEMENTED vector stores only
from core.vectorstores.chromadb_store import ChromaDBStore
from core.vectorstores.pinecone_store import PineconeStore

# Import config models
from models.domain_config import DomainConfig

# Configure logging
logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """
    Factory for creating vector store instances.

    Phase 2 Implementation:
    -----------------------
    ✅ ChromaDB - Local development and MVP
    ✅ Pinecone - Production cloud deployment

    Future Enhancements (Optional):
    --------------------------------
    - Qdrant: Self-hosted or cloud, privacy-focused
    - FAISS: In-memory, research/prototyping
    - Weaviate: GraphQL API, hybrid search
    - Milvus: Distributed, high-scale

    All can be added without changing calling code (config-driven).
    """

    @staticmethod
    def create_vector_store(
            config: DomainConfig
    ) -> VectorStoreInterface:
        """
        Create vector store based on configuration.

        Parameters:
        -----------
        config : DomainConfig
            Domain configuration with vector_store settings

        Returns:
        --------
        VectorStoreInterface:
            Vector store instance (ChromaDB or Pinecone)

        Raises:
        -------
        ValueError:
            If vector store provider not supported or not implemented

        Example:
        --------
        # From YAML config:
        # vector_store:
        #   provider: "chromadb"
        #   collection_name: "hr_collection"

        vector_store = VectorStoreFactory.create_vector_store(config)
        """
        provider = config.vector_store.provider.lower()

        logger.info(f"Creating vector store: {provider}")

        # ChromaDB - Local vector database
        if provider == "chromadb":
            return VectorStoreFactory._create_chromadb(config)

        # Pinecone - Cloud vector database
        elif provider == "pinecone":
            return VectorStoreFactory._create_pinecone(config)

        # Unsupported or not yet implemented
        else:
            raise ValueError(
                f"Unsupported vector store provider: '{provider}'\n"
                f"Implemented providers: chromadb, pinecone\n"
                f"Future providers: qdrant, faiss (not yet implemented)"
            )

    # =========================================================================
    # PRIVATE FACTORY METHODS (Provider-Specific)
    # =========================================================================

    @staticmethod
    def _create_chromadb(config: DomainConfig) -> ChromaDBStore:
        """
        Create ChromaDB vector store.

        ChromaDB is the recommended option for:
        - Local development
        - MVP/prototype
        - Small to medium datasets (< 1M vectors)
        - Free tier
        """
        logger.debug("Creating ChromaDB store...")

        vs_config = config.vector_store

        vector_store = ChromaDBStore(
            persist_directory=vs_config.persist_directory,
            collection_name=vs_config.collection_name
        )

        logger.info(
            f"✅ ChromaDB store created:\n"
            f"   Collection: {vs_config.collection_name}\n"
            f"   Directory: {vs_config.persist_directory}"
        )

        return vector_store

    @staticmethod
    def _create_pinecone(config: DomainConfig) -> PineconeStore:
        """
        Create Pinecone vector store.

        Pinecone is recommended for:
        - Production deployments
        - Large datasets (> 1M vectors)
        - Global distribution
        - Enterprise features
        """
        logger.debug("Creating Pinecone store...")

        vs_config = config.vector_store

        vector_store = PineconeStore(
            index_name=vs_config.collection_name,  # In Pinecone, it's called index_name
            api_key=vs_config.api_key,
            environment=vs_config.region,  # Pinecone calls it environment
            dimension=vs_config.dimension
        )

        logger.info(
            f"✅ Pinecone store created:\n"
            f"   Index: {vs_config.collection_name}\n"
            f"   Environment: {vs_config.region}"
        )

        return vector_store


# =============================================================================
# FUTURE IMPLEMENTATION STUBS (Optional)
# =============================================================================

"""
To add Qdrant support in future:

1. Install: pip install qdrant-client

2. Create: core/vector_stores/qdrant_store.py

3. Add to factory:
   elif provider == "qdrant":
       return VectorStoreFactory._create_qdrant(config)

4. Implement _create_qdrant():
   @staticmethod
   def _create_qdrant(config: DomainConfig) -> QdrantStore:
       from core.vector_stores.qdrant_store import QdrantStore

       vs_config = config.vector_store

       return QdrantStore(
           host=vs_config.host or "localhost",
           port=vs_config.port or 6333,
           collection_name=vs_config.collection_name,
           api_key=vs_config.api_key
       )

Same pattern for FAISS, Weaviate, Milvus, etc.
"""

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of VectorStoreFactory usage.
    Run: python core/factories/vector_store_factory.py
    """
    import logging

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("VectorStoreFactory - Create Vector Stores from Config")
    print("=" * 70)

    print("""
VectorStoreFactory Overview:
----------------------------

Phase 2 Implementation Status:
✅ ChromaDB - Local, free, simple (READY)
✅ Pinecone - Cloud, scalable, production (READY)

Future Enhancements (Optional):
⏳ Qdrant - Self-hosted or cloud
⏳ FAISS - In-memory, fast
⏳ Weaviate - GraphQL API
⏳ Milvus - Distributed

Configuration Examples:
------------------------

# ChromaDB (Local)
vector_store:
  provider: "chromadb"
  collection_name: "hr_collection"
  persist_directory: "./data/chroma_db"

# Pinecone (Cloud)
vector_store:
  provider: "pinecone"
  collection_name: "hr-docs-prod"
  api_key: ${PINECONE_API_KEY}  # From environment
  region: "us-east-1"
  dimension: 384

Usage Pattern:
--------------

from core.config_manager import ConfigManager
from core.factories.vector_store_factory import VectorStoreFactory

# Load config
config_mgr = ConfigManager()
hr_config = config_mgr.load_domain_config("hr")

# Create vector store (factory selects based on config)
vector_store = VectorStoreFactory.create_vector_store(hr_config)

# Use vector store (works with any provider!)
vector_store.add(chunks, embeddings)
results = vector_store.search(query_embedding, top_k=10)

Adding New Providers:
---------------------
To add Qdrant/FAISS/Weaviate later:
1. Create implementation: core/vector_stores/qdrant_store.py
2. Add to factory elif chain
3. Update config model
4. No changes to calling code needed!

Current Status:
---------------
Your Phase 2 core is COMPLETE with ChromaDB + Pinecone.
Additional vector stores are optional enhancements.
    """)
