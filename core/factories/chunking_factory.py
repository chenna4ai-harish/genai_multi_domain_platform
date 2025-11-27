"""
core/factories/chunking_factory.py
"""

# CORRECTED IMPORTS - Add Optional
from typing import Union, Dict, Any, Optional
import logging

# Import chunker implementations (LAZY - moved inside methods)
from core.interfaces.chunking_interface import ChunkerInterface
from core.config_manager import ChunkingConfig

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

    Usage:
        # No instantiation needed - use static methods directly
        chunker = ChunkingFactory.create_chunker(config, model_name)
    """

    # Registry of available chunking strategies (class-level attribute)
    _available_chunkers = {
        "recursive": "RecursiveChunker",
        "semantic": "SemanticChunker",
    }

    @staticmethod
    def create_chunker(config: ChunkingConfig, embedding_model_name: Optional[str] = None):
        """
        Create a chunker instance based on the chunking configuration.

        Args:
            config: ChunkingConfig object with strategy and parameters
            embedding_model_name: Embedding model name (required for both recursive and semantic!)

        Returns:
            Chunker instance
        """
        strategy = config.strategy.lower()

        logger.info(f"Creating chunker for strategy: {strategy}")

        # Validate embedding_model_name is provided
        if not embedding_model_name:
            # Use a default if not provided
            embedding_model_name = "all-MiniLM-L6-v2"
            logger.warning(f"No embedding_model_name provided, using default: {embedding_model_name}")

        try:
            if strategy == "recursive" or strategy == "fixed":
                from core.chunking.recursive_chunker import RecursiveChunker

                if strategy == "fixed":
                    logger.warning("'fixed' strategy is deprecated, using 'recursive' instead")

                # Get recursive config
                recursive_config = config.recursive
                if not recursive_config:
                    from core.config_manager import RecursiveChunkingConfig
                    recursive_config = RecursiveChunkingConfig()

                # Create RecursiveChunker with BOTH required arguments
                chunker = RecursiveChunker(
                    config=recursive_config,
                    embedding_model_name=embedding_model_name  # ← THIS WAS MISSING!
                )

                logger.info(
                    f"Created RecursiveChunker: chunk_size={recursive_config.chunk_size}, "
                    f"overlap={recursive_config.overlap}, model={embedding_model_name}"
                )
                return chunker

            elif strategy == "semantic":
                from core.chunking.semantic_chunker import SemanticChunker

                semantic_config = config.semantic
                if not semantic_config:
                    raise ValueError("Semantic chunking config is missing")

                # Create SemanticChunker
                chunker = SemanticChunker(
                    config=semantic_config,
                    embedding_model_name=embedding_model_name
                )

                logger.info(f"Created SemanticChunker with model={embedding_model_name}")
                return chunker

            else:
                supported = ["recursive", "semantic", "fixed"]
                raise ValueError(
                    f"Unsupported chunking strategy: '{strategy}'. "
                    f"Supported strategies: {supported}"
                )

        except Exception as e:
            logger.error(f"Failed to create chunker with strategy '{strategy}': {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(
                f"Failed to instantiate chunker for strategy '{strategy}'. Error: {e}"
            )

    @staticmethod
    def get_chunker(config: ChunkingConfig, embedding_model_name: Optional[str] = None):
        """
        Alias for create_chunker for backward compatibility.
        """
        return ChunkingFactory.create_chunker(config, embedding_model_name)

    @staticmethod
    def get_available_strategies() -> list:
        """Get list of available chunking strategies."""
        return list(ChunkingFactory._available_chunkers.keys())

    @staticmethod
    def register_strategy(name: str, chunker_class: type):
        """Register a new chunking strategy."""
        name = name.lower()
        if name in ChunkingFactory._available_chunkers:
            logger.warning(f"Overwriting existing strategy: {name}")
        ChunkingFactory._available_chunkers[name] = chunker_class.__name__
        logger.info(f"Registered chunking strategy: {name}")

    @staticmethod
    def register_strategy(name: str, chunker_class: type):
        """
        Register a new chunking strategy (for extensibility).

        Args:
            name: Strategy name
            chunker_class: Chunker class implementing ChunkerInterface
        """
        name = name.lower()
        if name in ChunkingFactory._available_chunkers:
            logger.warning(f"Overwriting existing strategy: {name}")
        ChunkingFactory._available_chunkers[name] = chunker_class.__name__
        logger.info(f"Registered chunking strategy: {name}")


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of ChunkingFactory usage.
    Run: python core/factories/chunking_factory.py
    """
    import logging

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("ChunkingFactory Usage Examples")
    print("=" * 70)

    # Example 1: Create recursive chunker
    print("\n1. Recursive Chunker")
    print("-" * 70)

    from core.config_manager import ChunkingConfig, RecursiveChunkingConfig

    config = ChunkingConfig(
        strategy="recursive",
        recursive=RecursiveChunkingConfig(chunk_size=500, overlap=50)
    )

    chunker = ChunkingFactory.create_chunker(config)
    print(f"✅ Created: {chunker.__class__.__name__}")

    # Test chunking
    text = "This is a test sentence. " * 100
    chunks = chunker.chunk_text(text)
    print(f"✅ Created {len(chunks)} chunks from {len(text)} chars")

    print("\n" + "=" * 70)
    print("ChunkingFactory examples completed!")
    print("=" * 70)
