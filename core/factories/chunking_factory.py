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

from typing import Union, Dict, Any
import logging

# Import chunker implementations
from core.chunking.recursive_chunker import RecursiveChunker
from core.chunking.semantic_chunker import SemanticChunker
from core.interfaces.chunking_interface import ChunkerInterface

# Import config models
from models.domain_config import RecursiveChunkingConfig, SemanticChunkingConfig

# Configure logging
logger = logging.getLogger(__name__)


class ChunkingFactory:
    """
    Factory for creating chunker instances based on configuration.

    This factory supports:
    - Recursive (fixed-size) chunking
    - Semantic (similarity-based) chunking
    - Easy extension for new chunking strategies

    All chunkers implement ChunkerInterface for consistency.
    """

    # Registry of available chunking strategies
    _available_chunkers = {
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
        # Add new strategies here as you implement them:
        # "sliding_window": SlidingWindowChunker,
        # "paragraph": ParagraphChunker,
    }

    @staticmethod
    def create_chunker(
            config: Union[RecursiveChunkingConfig, SemanticChunkingConfig, Dict[str, Any]],
            embedding_model_name: str
    ) -> ChunkerInterface:
        """
        Create a chunker instance based on configuration.

        This method:
        1. Extracts the strategy from config
        2. Validates the strategy is supported
        3. Creates appropriate config object if needed
        4. Instantiates the chunker with proper parameters

        Parameters:
        -----------
        config : Union[RecursiveChunkingConfig, SemanticChunkingConfig, Dict]
            Configuration object or dict with chunking parameters
            Required fields:
            - strategy: str ("recursive" or "semantic")

            For recursive strategy:
            - chunk_size: int (default: 500)
            - overlap: int (default: 50)

            For semantic strategy:
            - similarity_threshold: float (default: 0.7)
            - max_chunk_size: int (default: 1000)

        embedding_model_name : str
            Name of the embedding model (for metadata tracking)
            Example: "all-MiniLM-L6-v2"

        Returns:
        --------
        ChunkerInterface:
            Instantiated chunker implementing ChunkerInterface

        Raises:
        -------
        ValueError:
            If strategy is missing or unsupported
            If required config parameters are missing

        Example:
        --------
        # Using Pydantic config
        config = RecursiveChunkingConfig(
            strategy="recursive",
            chunk_size=500,
            overlap=50
        )
        chunker = ChunkingFactory.create_chunker(config, "all-MiniLM-L6-v2")

        # Using dict config
        config = {
            "strategy": "recursive",
            "chunk_size": 500,
            "overlap": 50
        }
        chunker = ChunkingFactory.create_chunker(config, "all-MiniLM-L6-v2")
        """
        # Step 1: Extract strategy from config (support both Pydantic and dict)
        if isinstance(config, dict):
            strategy = config.get("strategy")
        else:
            strategy = getattr(config, "strategy", None)

        if not strategy:
            raise ValueError(
                "Chunking config must specify a 'strategy' field. "
                f"Available strategies: {list(ChunkingFactory._available_chunkers.keys())}"
            )

        strategy = strategy.lower()
        logger.info(f"Creating chunker with strategy: {strategy}")

        # Step 2: Validate strategy is supported
        chunker_cls = ChunkingFactory._available_chunkers.get(strategy)
        if not chunker_cls:
            raise ValueError(
                f"Unknown chunking strategy: '{strategy}'. "
                f"Available strategies: {list(ChunkingFactory._available_chunkers.keys())}"
            )

        # Step 3: Instantiate chunker based on strategy
        try:
            if strategy == "recursive":
                # Create RecursiveChunker with proper config
                if isinstance(config, dict):
                    # Convert dict to Pydantic model
                    chunking_config = RecursiveChunkingConfig(
                        strategy=strategy,
                        chunk_size=config.get("chunk_size", 500),
                        overlap=config.get("overlap", 50)
                    )
                else:
                    # Already a Pydantic model
                    chunking_config = config

                chunker = RecursiveChunker(
                    config=chunking_config,
                    embedding_model_name=embedding_model_name
                )

                logger.info(
                    f"Created RecursiveChunker: "
                    f"chunk_size={chunking_config.chunk_size}, "
                    f"overlap={chunking_config.overlap}, "
                    f"model={embedding_model_name}"
                )

            elif strategy == "semantic":
                # Create SemanticChunker with proper config
                if isinstance(config, dict):
                    # Convert dict to Pydantic model
                    chunking_config = SemanticChunkingConfig(
                        strategy=strategy,
                        similarity_threshold=config.get("similarity_threshold", 0.7),
                        max_chunk_size=config.get("max_chunk_size", 1000)
                    )
                else:
                    # Already a Pydantic model
                    chunking_config = config

                chunker = SemanticChunker(
                    config=chunking_config,
                    embedding_model_name=embedding_model_name
                )

                logger.info(
                    f"Created SemanticChunker: "
                    f"threshold={chunking_config.similarity_threshold}, "
                    f"max_size={chunking_config.max_chunk_size}, "
                    f"model={embedding_model_name}"
                )

            else:
                # For future custom chunkers
                # Assume they follow the same pattern: __init__(config, embedding_model_name)
                if isinstance(config, dict):
                    # Cannot create Pydantic model for unknown strategy
                    # Pass dict directly and let chunker handle it
                    chunker = chunker_cls(config, embedding_model_name)
                else:
                    chunker = chunker_cls(config, embedding_model_name)

                logger.info(f"Created custom chunker: {chunker_cls.__name__}")

            return chunker

        except Exception as e:
            logger.error(
                f"Failed to create chunker with strategy '{strategy}': {e}"
            )
            raise ValueError(
                f"Failed to instantiate chunker for strategy '{strategy}'. "
                f"Error: {e}"
            )

    @staticmethod
    def get_available_strategies() -> list:
        """
        Get list of available chunking strategies.

        Useful for:
        - Validation
        - UI dropdowns
        - Documentation

        Returns:
        --------
        list:
            List of strategy names
            Example: ["recursive", "semantic"]
        """
        return list(ChunkingFactory._available_chunkers.keys())

    @staticmethod
    def register_strategy(name: str, chunker_class: type):
        """
        Register a new chunking strategy (for extensibility).

        This allows adding custom chunking strategies at runtime
        without modifying the factory code.

        Parameters:
        -----------
        name : str
            Strategy name (will be lowercased)
        chunker_class : type
            Chunker class implementing ChunkerInterface

        Example:
        --------
        # Register custom chunker
        ChunkingFactory.register_strategy("paragraph", ParagraphChunker)

        # Now can use it
        config = {"strategy": "paragraph", "min_length": 100}
        chunker = ChunkingFactory.create_chunker(config, "model-name")
        """
        name = name.lower()
        if name in ChunkingFactory._available_chunkers:
            logger.warning(f"Overwriting existing strategy: {name}")

        ChunkingFactory._available_chunkers[name] = chunker_class
        logger.info(f"Registered chunking strategy: {name}")


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of ChunkingFactory usage.
    Run: python core/factories/chunking_factory.py
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("ChunkingFactory Usage Examples")
    print("=" * 70)

    # Example 1: Create recursive chunker from dict config
    print("\n1. Recursive Chunker from Dict Config")
    print("-" * 70)

    recursive_config = {
        "strategy": "recursive",
        "chunk_size": 500,
        "overlap": 50
    }

    recursive_chunker = ChunkingFactory.create_chunker(
        config=recursive_config,
        embedding_model_name="all-MiniLM-L6-v2"
    )

    print(f"Created: {recursive_chunker.__class__.__name__}")
    print(f"Chunk size: {recursive_chunker.chunk_size}")
    print(f"Overlap: {recursive_chunker.overlap}")

    # Example 2: Create semantic chunker from dict config
    print("\n2. Semantic Chunker from Dict Config")
    print("-" * 70)

    semantic_config = {
        "strategy": "semantic",
        "similarity_threshold": 0.7,
        "max_chunk_size": 1000
    }

    semantic_chunker = ChunkingFactory.create_chunker(
        config=semantic_config,
        embedding_model_name="all-MiniLM-L6-v2"
    )

    print(f"Created: {semantic_chunker.__class__.__name__}")
    print(f"Threshold: {semantic_chunker.similarity_threshold}")
    print(f"Max size: {semantic_chunker.max_chunk_size}")

    # Example 3: Create from Pydantic config
    print("\n3. Recursive Chunker from Pydantic Config")
    print("-" * 70)

    from models.domain_config import RecursiveChunkingConfig

    pydantic_config = RecursiveChunkingConfig(
        strategy="recursive",
        chunk_size=800,
        overlap=80
    )

    pydantic_chunker = ChunkingFactory.create_chunker(
        config=pydantic_config,
        embedding_model_name="all-mpnet-base-v2"
    )

    print(f"Created: {pydantic_chunker.__class__.__name__}")
    print(f"Chunk size: {pydantic_chunker.chunk_size}")

    # Example 4: List available strategies
    print("\n4. Available Strategies")
    print("-" * 70)

    strategies = ChunkingFactory.get_available_strategies()
    print(f"Available: {strategies}")

    # Example 5: Error handling
    print("\n5. Error Handling")
    print("-" * 70)

    try:
        invalid_config = {"strategy": "unknown_strategy"}
        ChunkingFactory.create_chunker(invalid_config, "model")
    except ValueError as e:
        print(f"✅ Caught expected error: {e}")

    print("\n" + "=" * 70)
    print("ChunkingFactory examples completed!")
    print("=" * 70)
