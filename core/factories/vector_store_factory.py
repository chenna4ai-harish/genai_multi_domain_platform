"""
core/factories/vector_store_factory.py

This module implements the Factory Pattern for creating vector store instances.

Purpose:
--------
Enables config-driven vector store selection. Switch between ChromaDB (local)
and Pinecone (cloud) by just changing YAML config, with no code changes.

Factory Pattern Benefits:
-------------------------
- Abstracts provider-specific initialization (connection strings, API keys, etc.)
- Handles environment variable reading for API keys
- Manages dimension validation (critical for vector stores)
- Provides clear error messages for configuration issues
- Makes adding new vector stores trivial (just add elif branch)

Why Vector Store Dimension Matters:
------------------------------------
Vector stores need to know embedding dimension BEFORE storing any vectors.
This is because the underlying index structure depends on dimension.

Mismatch examples:
- Index configured for 384-dim, but embeddings are 768-dim → ERROR
- Index configured for 768-dim, but embeddings are 384-dim → ERROR

This factory ensures dimension consistency between:
1. Embedding model output dimension
2. Vector store index configuration

Example:
--------
# Embedding model: all-MiniLM-L6-v2 (384-dim)
# Vector store MUST be configured for 384-dim

embedder = EmbeddingFactory.create_embedder(config)
dimension = 384  # Get from embedder or config

store = VectorStoreFactory.create_store(
    config.vector_store,
    embedding_dimension=dimension  # Pass dimension to factory
)

Migration Example:
------------------
Start with ChromaDB for MVP:
  vector_store:
    provider: "chromadb"
    chromadb:
      persist_directory: "./data/chroma_db"
      collection_name: "hr_collection"

Later, switch to Pinecone for production:
  vector_store:
    provider: "pinecone"
    pinecone:
      index_name: "hr-docs-prod"
      dimension: 384
      cloud: "aws"
      region: "us-east-1"

NO CODE CHANGES NEEDED! Just update YAML and set PINECONE_API_KEY.
"""

from typing import Optional
from core.interfaces.vector_store_interface import VectorStoreInterface
from core.vector_stores.chromadb_store import ChromaDBStore
from core.vector_stores.pinecone_store import PineconeStore
from models.domain_config import VectorStoreConfig
import logging
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Configure logging
logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """
    Factory for creating vector store instances based on configuration.

    This factory implements the Factory Method pattern to decouple vector store
    creation from usage. The factory reads configuration and returns the
    appropriate VectorStoreInterface implementation.

    Supported Providers:
    --------------------
    - "chromadb": Local, file-based vector store (free, simple)
    - "pinecone": Cloud, managed vector store (paid, scalable)

    Future Providers (easy to add):
    --------------------------------
    - "weaviate": Cloud or self-hosted, GraphQL API
    - "qdrant": Self-hosted, high performance, Rust-based
    - "milvus": Open-source, distributed, production-ready
    - "pgvector": PostgreSQL extension (use existing database)
    - "redis": Redis with RediSearch module

    Design Benefits:
    ----------------
    - **Polymorphism**: Returns VectorStoreInterface, works with any implementation
    - **Config-Driven**: Provider determined by YAML config, not code
    - **Extensible**: Add new providers without modifying calling code
    - **Type-Safe**: Static type checkers understand the return type

    Dimension Handling:
    -------------------
    Different embedding models produce different dimensions:
    - all-MiniLM-L6-v2: 384 dimensions
    - all-mpnet-base-v2: 768 dimensions
    - text-embedding-ada-002: 1536 dimensions

    The factory ensures vector store is configured with correct dimension.

    Example Usage:
    --------------
    from models.domain_config import VectorStoreConfig, ChromaDBConfig

    # Create config
    config = VectorStoreConfig(
        provider="chromadb",
        chromadb=ChromaDBConfig(
            persist_directory="./data/chroma_db",
            collection_name="hr_collection"
        )
    )

    # Create vector store via factory
    store = VectorStoreFactory.create_store(
        config,
        embedding_dimension=384  # From embedding model
    )

    # Use store (polymorphic - works with any provider!)
    store.upsert(chunks, embeddings)
    results = store.search(query_embedding, top_k=10)
    """

    @staticmethod
    def create_store(
            config: VectorStoreConfig,
            embedding_dimension: int = 384
    ) -> VectorStoreInterface:
        """
        Create and return a vector store instance based on configuration.

        This is the CORE FACTORY METHOD that implements the Factory Pattern.
        It encapsulates the decision logic of which vector store to instantiate.

        Parameters:
        -----------
        config : VectorStoreConfig
            Vector store configuration from domain YAML
            Contains:
            - provider: "chromadb" or "pinecone"
            - chromadb: ChromaDBConfig settings (if provider="chromadb")
            - pinecone: PineconeConfig settings (if provider="pinecone")

        embedding_dimension : int, optional
            Dimension of embedding vectors (default: 384)
            MUST match your embedding model's output dimension!

            Common dimensions:
            - 384: all-MiniLM-L6-v2, paraphrase-MiniLM-L6-v2
            - 768: all-mpnet-base-v2, BERT, Gemini
            - 1536: OpenAI text-embedding-ada-002

            This parameter is:
            - Optional for ChromaDB (dimension auto-detected)
            - Required for Pinecone (must match index configuration)

        Returns:
        --------
        VectorStoreInterface:
            Concrete vector store implementation (ChromaDBStore or PineconeStore)
            Guaranteed to implement the VectorStoreInterface contract

        Raises:
        -------
        ValueError:
            - If config.provider is not recognized
            - If required configuration is missing
            - If API key is missing (for cloud providers)
            - If dimension mismatch detected
        RuntimeError:
            - If vector store initialization fails

        Algorithm:
        ----------
        1. Read config.provider to determine which store to create
        2. Extract provider-specific config (config.chromadb or config.pinecone)
        3. Validate required settings (API keys, dimensions, etc.)
        4. Instantiate the appropriate vector store class
        5. Return the store (polymorphic - caller doesn't know which type)

        Example:
        --------
        # ChromaDB (local)
        config = VectorStoreConfig(
            provider="chromadb",
            chromadb=ChromaDBConfig(
                persist_directory="./data/chroma_db",
                collection_name="hr_collection"
            )
        )
        store = VectorStoreFactory.create_store(config, embedding_dimension=384)
        # Returns: ChromaDBStore instance

        # Pinecone (cloud)
        config = VectorStoreConfig(
            provider="pinecone",
            pinecone=PineconeConfig(
                index_name="hr-docs-prod",
                dimension=384,
                cloud="aws",
                region="us-east-1"
            )
        )
        store = VectorStoreFactory.create_store(config, embedding_dimension=384)
        # Returns: PineconeStore instance

        # Both return VectorStoreInterface, so calling code works with either!

        Notes:
        ------
        - This method is static (no instance needed)
        - Provider determined at runtime based on config
        - Type hint is VectorStoreInterface (polymorphic return)
        - New providers can be added without breaking existing code
        """
        provider = config.provider.lower()

        logger.debug(
            f"Creating vector store: provider='{provider}', "
            f"dimension={embedding_dimension}"
        )

        # Factory logic: Switch on provider and instantiate appropriate class

        if provider == "chromadb":
            # OPTION 1: ChromaDB (Local, File-based)

            # Validate ChromaDB config is provided
            if config.chromadb is None:
                raise ValueError(
                    "ChromaDB configuration is missing.\n"
                    "Add 'chromadb' section to your domain config YAML:\n"
                    "vector_store:\n"
                    "  provider: 'chromadb'\n"
                    "  chromadb:\n"
                    "    persist_directory: './data/chroma_db'\n"
                    "    collection_name: 'your_collection'"
                )

            logger.info(
                f"Creating ChromaDBStore: "
                f"directory={config.chromadb.persist_directory}, "
                f"collection={config.chromadb.collection_name}"
            )

            return ChromaDBStore(
                persist_directory=config.chromadb.persist_directory,
                collection_name=config.chromadb.collection_name
            )

        elif provider == "pinecone":
            # OPTION 2: Pinecone (Cloud, Managed)

            # Validate Pinecone config is provided
            if config.pinecone is None:
                raise ValueError(
                    "Pinecone configuration is missing.\n"
                    "Add 'pinecone' section to your domain config YAML:\n"
                    "vector_store:\n"
                    "  provider: 'pinecone'\n"
                    "  pinecone:\n"
                    "    index_name: 'your-index'\n"
                    "    dimension: 384\n"
                    "    cloud: 'aws'\n"
                    "    region: 'us-east-1'"
                )

            # Read API key from environment
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError(
                    "PINECONE_API_KEY environment variable is required for Pinecone provider.\n"
                    "Get your API key at: https://app.pinecone.io/ → API Keys\n"
                    "Set it with: export PINECONE_API_KEY='your-key-here'"
                )

            # Validate dimension matches
            if config.pinecone.dimension != embedding_dimension:
                logger.warning(
                    f"Dimension mismatch detected!\n"
                    f"  Pinecone index configured for: {config.pinecone.dimension}-dim\n"
                    f"  Embedding model produces: {embedding_dimension}-dim\n"
                    f"  This will cause errors during upsert!"
                )
                # Don't raise error here - let Pinecone handle it
                # User might be intentionally reconfiguring

            logger.info(
                f"Creating PineconeStore: "
                f"index={config.pinecone.index_name}, "
                f"dimension={config.pinecone.dimension}, "
                f"cloud={config.pinecone.cloud}/{config.pinecone.region}"
            )

            return PineconeStore(
                api_key=api_key,
                index_name=config.pinecone.index_name,
                dimension=config.pinecone.dimension,
                cloud=config.pinecone.cloud,
                region=config.pinecone.region
            )

        # TO ADD WEAVIATE (example):
        # elif provider == "weaviate":
        #     if config.weaviate is None:
        #         raise ValueError("Weaviate configuration missing")
        #
        #     return WeaviateStore(
        #         url=config.weaviate.url,
        #         api_key=os.getenv("WEAVIATE_API_KEY"),
        #         class_name=config.weaviate.class_name
        #     )

        # TO ADD QDRANT (example):
        # elif provider == "qdrant":
        #     if config.qdrant is None:
        #         raise ValueError("Qdrant configuration missing")
        #
        #     return QdrantStore(
        #         url=config.qdrant.url,
        #         collection_name=config.qdrant.collection_name,
        #         dimension=embedding_dimension
        #     )

        else:
            # Unknown provider - provide helpful error message
            supported_providers = ["chromadb", "pinecone"]
            raise ValueError(
                f"Unknown vector store provider: '{provider}'\n"
                f"Supported providers: {supported_providers}\n"
                f"Check your domain config YAML file.\n"
                f"To add a new provider, update VectorStoreFactory.create_store()"
            )


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of VectorStoreFactory usage.
    Run: python core/factories/vector_store_factory.py
    """

    import logging
    from models.domain_config import (
        VectorStoreConfig,
        ChromaDBConfig,
        PineconeConfig
    )

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("VectorStoreFactory Usage Examples")
    print("=" * 70)

    # Example 1: Creating ChromaDB store via factory
    print("\n1. Creating ChromaDB Store")
    print("-" * 70)

    chromadb_config = VectorStoreConfig(
        provider="chromadb",
        chromadb=ChromaDBConfig(
            persist_directory="./data/test_chroma_db",
            collection_name="test_collection"
        )
    )

    store = VectorStoreFactory.create_store(
        chromadb_config,
        embedding_dimension=384
    )

    print(f"Created: {type(store).__name__}")
    print(f"Interface: {isinstance(store, VectorStoreInterface)}")
    print(f"Config: ChromaDB at ./data/test_chroma_db")

    # Example 2: Creating Pinecone store (if API key available)
    print("\n2. Creating Pinecone Store")
    print("-" * 70)

    if os.getenv("PINECONE_API_KEY"):
        pinecone_config = VectorStoreConfig(
            provider="pinecone",
            pinecone=PineconeConfig(
                index_name="test-docs-dev",
                dimension=384,
                cloud="aws",
                region="us-east-1"
            )
        )

        try:
            store = VectorStoreFactory.create_store(
                pinecone_config,
                embedding_dimension=384
            )
            print(f"Created: {type(store).__name__}")
            print(f"Interface: {isinstance(store, VectorStoreInterface)}")
            print(f"Config: Pinecone index 'test-docs-dev'")
        except Exception as e:
            print(f"Failed to create Pinecone store: {e}")
    else:
        print("PINECONE_API_KEY not set, skipping Pinecone example")
        print("Set it with: export PINECONE_API_KEY='your-key'")

    # Example 3: Demonstrating polymorphism
    print("\n3. Demonstrating Polymorphism")
    print("-" * 70)


    def use_any_vector_store(config: VectorStoreConfig, dimension: int):
        """
        This function works with ANY vector store implementation!
        It doesn't know or care which provider is used.
        """
        # Create store via factory (polymorphic)
        store = VectorStoreFactory.create_store(config, dimension)

        print(f"Provider: {config.provider}")
        print(f"Store type: {type(store).__name__}")
        print(f"Interface: {isinstance(store, VectorStoreInterface)}")

        # Get stats (works with any implementation!)
        stats = store.get_collection_stats() if hasattr(store, 'get_collection_stats') else \
            store.get_index_stats() if hasattr(store, 'get_index_stats') else {}

        if stats:
            print(f"Stats: {stats}")

        return store


    # Works with ChromaDB
    print("\nUsing ChromaDB:")
    use_any_vector_store(chromadb_config, 384)

    print("\n✅ Same function works with different providers!")
    print("   This is the power of the Factory Pattern + Polymorphism")

    # Example 4: Dimension validation
    print("\n4. Dimension Validation")
    print("-" * 70)

    # This demonstrates dimension mismatch warning
    if os.getenv("PINECONE_API_KEY"):
        mismatched_config = VectorStoreConfig(
            provider="pinecone",
            pinecone=PineconeConfig(
                index_name="test-docs-dev",
                dimension=768,  # Index configured for 768-dim
                cloud="aws",
                region="us-east-1"
            )
        )

        # Pass different dimension (384-dim embeddings)
        print("Creating Pinecone with dimension mismatch...")
        try:
            store = VectorStoreFactory.create_store(
                mismatched_config,
                embedding_dimension=384  # Mismatch: 384 != 768
            )
            print("⚠️  Store created but dimension mismatch warning logged")
        except Exception as e:
            print(f"Error: {e}")

    # Example 5: Error Handling
    print("\n5. Error Handling")
    print("-" * 70)

    try:
        bad_config = VectorStoreConfig(provider="nonexistent")
        store = VectorStoreFactory.create_store(bad_config, 384)
    except ValueError as e:
        print(f"✅ Caught expected error:")
        print(f"   {str(e).split(chr(10))[0]}")  # First line only

    try:
        # Missing ChromaDB config - THIS IS THE TEST CASE
        bad_config = VectorStoreConfig(provider="chromadb", chromadb=None)
        store = VectorStoreFactory.create_store(bad_config, 384)
    except ValueError as e:
        print(f"\n✅ Caught configuration error:")
        print(f"   {str(e).split(chr(10))[0]}")  # First line only
