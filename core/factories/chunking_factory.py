"""
core/factories/chunking_factory.py

This module implements the Factory Pattern for creating chunking strategy instances.

What is the Factory Pattern?
-----------------------------
The Factory Pattern is a creational design pattern that provides an interface
for creating objects without specifying their exact class. Instead of calling
constructors directly, you ask a factory to create the object for you based
on configuration or parameters.

Why Use Factories?
------------------
1. **Separation of Concerns**: Object creation logic is isolated from usage
2. **Config-Driven**: Create different implementations based on YAML config
3. **Single Responsibility**: Factory handles creation, calling code handles usage
4. **Open/Closed Principle**: Add new chunking strategies without modifying existing code
5. **Testability**: Easy to mock factories in unit tests

How This Enables Config-Driven Architecture:
---------------------------------------------
Instead of:
    if config.strategy == "recursive":
        chunker = RecursiveChunker(config.chunk_size, config.overlap)
    elif config.strategy == "semantic":
        chunker = SemanticChunker(config.threshold, config.max_size)

You write:
    chunker = ChunkingFactory.create_chunker(config, model_name)

Benefits:
- Calling code doesn't know about concrete chunker classes
- Adding new chunking strategies only requires updating the factory
- Configuration changes don't require code changes
- Type hints work correctly (returns ChunkerInterface)

Example Usage:
--------------
# Load config
config_mgr = ConfigManager()
domain_config = config_mgr.load_domain_config("hr")

# Create chunker based on config (no if/else needed!)
chunker = ChunkingFactory.create_chunker(
    domain_config.chunking,
    domain_config.embeddings.model_name
)

# Use chunker (works with any implementation!)
chunks = chunker.chunk_text(text, doc_id, domain, ...)

References:
-----------
- Factory Pattern: https://refactoring.guru/design-patterns/factory-method
- Real Python Guide: https://realpython.com/factory-method-python/
"""


from core.interfaces.chunking_interface import ChunkerInterface
from core.chunking.recursive_chunker import RecursiveChunker
from core.chunking.semantic_chunker import SemanticChunker
from models.domain_config import ChunkingConfig
import logging

# Configure logging
logger = logging.getLogger(__name__)


class ChunkingFactory:
    """
    Factory for creating chunking strategy instances based on configuration.

    This factory implements the Factory Method pattern to decouple chunking
    strategy creation from usage. The factory reads the configuration and
    returns the appropriate ChunkerInterface implementation.

    Supported Strategies:
    ---------------------
    - "recursive": Fixed-size chunking with overlap (RecursiveChunker)
    - "semantic": Embedding-based topical grouping (SemanticChunker)

    Adding New Strategies:
    ----------------------
    1. Implement new chunker class inheriting from ChunkerInterface
    2. Add elif branch in create_chunker() method
    3. Update ChunkingConfig to include new strategy settings
    4. That's it! No other code changes needed.

    Design Benefits:
    ----------------
    - **Polymorphism**: Returns ChunkerInterface, works with any implementation
    - **Config-Driven**: Strategy determined by YAML config, not code
    - **Extensible**: Easy to add new strategies without modifying calling code
    - **Type-Safe**: Static type checkers understand the return type

    Example Usage:
    --------------
    from models.domain_config import ChunkingConfig, RecursiveChunkingConfig

    # Create config
    config = ChunkingConfig(
        strategy="recursive",
        recursive=RecursiveChunkingConfig(chunk_size=500, overlap=50)
    )

    # Create chunker via factory
    chunker = ChunkingFactory.create_chunker(config, "all-MiniLM-L6-v2")

    # Use chunker (polymorphic - works with any strategy!)
    chunks = chunker.chunk_text(
        text="Your document text...",
        doc_id="doc123",
        domain="hr",
        source_file_path="./doc.pdf",
        file_hash="abc123"
    )
    """

    @staticmethod
    def create_chunker(config: ChunkingConfig, embedding_model_name: str) -> ChunkerInterface:
        """
        Create and return a chunker instance based on configuration.

        This is the CORE FACTORY METHOD that implements the Factory Pattern.
        It encapsulates the decision logic of which chunker class to instantiate.

        Parameters:
        -----------
        config : ChunkingConfig
            Chunking configuration from domain YAML
            Contains:
            - strategy: "recursive" or "semantic"
            - recursive: RecursiveChunkingConfig settings
            - semantic: SemanticChunkingConfig settings

        embedding_model_name : str
            Name of the embedding model (for metadata tracking)
            This is stored in chunk metadata for provenance

        Returns:
        --------
        ChunkerInterface:
            Concrete chunker implementation (RecursiveChunker or SemanticChunker)
            Guaranteed to implement the ChunkerInterface contract

        Raises:
        -------
        ValueError:
            If config.strategy is not recognized
            If required configuration is missing

        Algorithm:
        ----------
        1. Read config.strategy to determine which chunker to create
        2. Extract strategy-specific config (config.recursive or config.semantic)
        3. Instantiate the appropriate chunker class with its config
        4. Return the chunker (polymorphic - caller doesn't know which type)

        Example:
        --------
        # Recursive chunking
        config = ChunkingConfig(strategy="recursive")
        chunker = ChunkingFactory.create_chunker(config, "model-name")
        # Returns: RecursiveChunker instance

        # Semantic chunking
        config = ChunkingConfig(strategy="semantic")
        chunker = ChunkingFactory.create_chunker(config, "model-name")
        # Returns: SemanticChunker instance

        # Both return ChunkerInterface, so calling code works with either!

        Notes:
        ------
        - This method is static (no instance needed)
        - Strategy determined at runtime based on config
        - Type hint is ChunkerInterface (polymorphic return)
        - New strategies can be added without breaking existing code
        """
        strategy = config.strategy.lower()

        logger.debug(
            f"Creating chunker: strategy='{strategy}', "
            f"embedding_model='{embedding_model_name}'"
        )

        # Factory logic: Switch on strategy and instantiate appropriate class

        if strategy == "recursive":
            # OPTION 1: Recursive (Fixed-size) Chunking
            logger.info(
                f"Creating RecursiveChunker: "
                f"chunk_size={config.recursive.chunk_size}, "
                f"overlap={config.recursive.overlap}"
            )

            return RecursiveChunker(
                config=config.recursive,
                embedding_model_name=embedding_model_name
            )

        elif strategy == "semantic":
            # OPTION 2: Semantic (Similarity-based) Chunking
            logger.info(
                f"Creating SemanticChunker: "
                f"threshold={config.semantic.similarity_threshold}, "
                f"max_size={config.semantic.max_chunk_size}"
            )

            return SemanticChunker(
                config=config.semantic,
                embedding_model_name=embedding_model_name
            )

        # TO ADD A NEW STRATEGY:
        # elif strategy == "paragraph":
        #     return ParagraphChunker(
        #         config=config.paragraph,
        #         embedding_model_name=embedding_model_name
        #     )

        else:
            # Unknown strategy - provide helpful error message
            supported_strategies = ["recursive", "semantic"]
            raise ValueError(
                f"Unknown chunking strategy: '{strategy}'\n"
                f"Supported strategies: {supported_strategies}\n"
                f"Check your domain config YAML file."
            )


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of ChunkingFactory usage.
    Run: python core/factories/chunking_factory.py
    """

    import logging
    from models.domain_config import (
        ChunkingConfig,
        RecursiveChunkingConfig,
        SemanticChunkingConfig
    )

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("ChunkingFactory Usage Examples")
    print("=" * 70)

    # Example 1: Creating recursive chunker via factory
    print("\n1. Creating Recursive Chunker")
    print("-" * 70)

    recursive_config = ChunkingConfig(
        strategy="recursive",
        recursive=RecursiveChunkingConfig(chunk_size=500, overlap=50)
    )

    chunker = ChunkingFactory.create_chunker(
        recursive_config,
        embedding_model_name="all-MiniLM-L6-v2"
    )

    print(f"Created: {type(chunker).__name__}")
    print(f"Interface: {isinstance(chunker, ChunkerInterface)}")
    print(f"Config: chunk_size=500, overlap=50")

    # Example 2: Creating semantic chunker via factory
    print("\n2. Creating Semantic Chunker")
    print("-" * 70)

    semantic_config = ChunkingConfig(
        strategy="semantic",
        semantic=SemanticChunkingConfig(
            similarity_threshold=0.7,
            max_chunk_size=1000
        )
    )

    chunker = ChunkingFactory.create_chunker(
        semantic_config,
        embedding_model_name="all-MiniLM-L6-v2"
    )

    print(f"Created: {type(chunker).__name__}")
    print(f"Interface: {isinstance(chunker, ChunkerInterface)}")
    print(f"Config: threshold=0.7, max_size=1000")

    # Example 3: Demonstrating polymorphism
    print("\n3. Demonstrating Polymorphism")
    print("-" * 70)


    def process_with_any_chunker(config: ChunkingConfig, text: str):
        """
        This function works with ANY chunker implementation!
        It doesn't know or care which strategy is used.
        """
        # Create chunker via factory (polymorphic)
        chunker = ChunkingFactory.create_chunker(config, "test-model")

        # Use chunker (works with any implementation!)
        chunks = chunker.chunk_text(
            text=text,
            doc_id="test_doc",
            domain="test",
            source_file_path="test.txt",
            file_hash="test123"
        )

        print(f"Strategy: {config.strategy}")
        print(f"Chunker: {type(chunker).__name__}")
        print(f"Chunks created: {len(chunks)}")
        return chunks


    test_text = "This is a test document. " * 50

    # Works with recursive
    recursive_chunks = process_with_any_chunker(recursive_config, test_text)
    print()

    # Works with semantic (same function!)
    semantic_chunks = process_with_any_chunker(semantic_config, test_text)

    print("\n✅ Same function works with different strategies!")
    print("   This is the power of the Factory Pattern + Polymorphism")

    # Example 4: Error handling
    print("\n4. Error Handling")
    print("-" * 70)

    try:
        bad_config = ChunkingConfig(strategy="nonexistent")
        chunker = ChunkingFactory.create_chunker(bad_config, "model")
    except ValueError as e:
        print(f"✅ Caught expected error:")
        print(f"   {e}")

    print("\n" + "=" * 70)
    print("ChunkingFactory examples completed!")
    print("=" * 70)
