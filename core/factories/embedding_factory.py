"""
core/factories/embedding_factory.py

This module implements the Factory Pattern for creating embedding provider instances.

Purpose:
--------
Enables config-driven embedding provider selection. Switch between local
Sentence-Transformers and cloud APIs (Gemini, OpenAI) by just changing YAML config.

Factory Pattern Benefits:
-------------------------
- Abstracts away provider-specific instantiation logic
- Handles environment variable reading (API keys)
- Provides clear error messages for missing dependencies
- Makes adding new providers easy (just add elif branch)

Example:
--------
Instead of:
    if config.provider == "sentence_transformers":
        embedder = SentenceTransformerEmbeddings(...)
    elif config.provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("...")
        embedder = GeminiEmbeddings(api_key, ...)

You write:
    embedder = EmbeddingFactory.create_embedder(config)

Much cleaner! And adding a new provider only requires updating the factory.
"""
from typing import List
from core.interfaces.embedding_interface import EmbeddingInterface
from core.embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddings
from core.embeddings.gemini_embeddings import GeminiEmbeddings
from models.domain_config import EmbeddingConfig
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """
    Factory for creating embedding provider instances based on configuration.

    Supported Providers:
    --------------------
    - "sentence_transformers": Local Sentence-Transformers models (free, no API key)
    - "gemini": Google Gemini API embeddings (requires GEMINI_API_KEY)

    Future Providers (easy to add):
    --------------------------------
    - "openai": OpenAI embeddings (requires OPENAI_API_KEY)
    - "cohere": Cohere embeddings (requires COHERE_API_KEY)
    - "azure_openai": Azure OpenAI embeddings
    - "custom": Your own custom embedding endpoint

    API Key Management:
    -------------------
    The factory automatically reads API keys from environment variables:
    - GEMINI_API_KEY for Google Gemini
    - OPENAI_API_KEY for OpenAI (when implemented)
    - etc.

    This keeps secrets out of code and config files (security best practice).

    Example Usage:
    --------------
    from models.domain_config import EmbeddingConfig
    import os

    # Sentence-Transformers (no API key needed)
    config = EmbeddingConfig(
        provider="sentence_transformers",
        model_name="all-MiniLM-L6-v2"
    )
    embedder = EmbeddingFactory.create_embedder(config)

    # Gemini (reads GEMINI_API_KEY from environment)
    os.environ["GEMINI_API_KEY"] = "your-key-here"
    config = EmbeddingConfig(
        provider="gemini",
        model_name="models/embedding-001"
    )
    embedder = EmbeddingFactory.create_embedder(config)
    """

    @staticmethod
    def create_embedder(config: EmbeddingConfig) -> EmbeddingInterface:
        """
        Create and return an embedder instance based on configuration.

        Parameters:
        -----------
        config : EmbeddingConfig
            Embedding configuration from domain YAML
            Contains:
            - provider: "sentence_transformers" or "gemini"
            - model_name: Model identifier
            - device: "cpu", "cuda", or "mps" (for local models)
            - batch_size: Batch size for encoding
            - normalize_embeddings: Whether to normalize

        Returns:
        --------
        EmbeddingInterface:
            Concrete embedder implementation
            Guaranteed to implement embed_texts() and get_model_name()

        Raises:
        -------
        ValueError:
            - If provider is not recognized
            - If required API key is missing (for cloud providers)
            - If required configuration is missing
        RuntimeError:
            - If embedder initialization fails

        Example:
        --------
        config = EmbeddingConfig(provider="sentence_transformers")
        embedder = EmbeddingFactory.create_embedder(config)

        # Use embedder (works with any provider!)
        texts = ["Hello", "World"]
        embeddings = embedder.embed_texts(texts)
        """
        provider = config.provider.lower()

        logger.debug(
            f"Creating embedder: provider='{provider}', "
            f"model='{config.model_name}'"
        )

        # Factory logic: Switch on provider and instantiate appropriate class

        if provider == "sentence_transformers":
            # OPTION 1: Local Sentence-Transformers (free, no API key)
            logger.info(
                f"Creating SentenceTransformerEmbeddings: "
                f"model={config.model_name}, device={config.device}"
            )

            return SentenceTransformerEmbeddings(
                model_name=config.model_name,
                device=config.device,
                batch_size=config.batch_size,
                normalize=config.normalize_embeddings
            )

        elif provider == "gemini":
            # OPTION 2: Google Gemini API (requires API key)
            logger.info(
                f"Creating GeminiEmbeddings: model={config.model_name}"
            )

            # Read API key from environment
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY environment variable is required for Gemini provider.\n"
                    "Get your API key at: https://makersuite.google.com/app/apikey\n"
                    "Set it with: export GEMINI_API_KEY='your-key-here'"
                )

            return GeminiEmbeddings(
                api_key=api_key,
                model_name=config.model_name,
                batch_size=config.batch_size
            )

        # TO ADD OpenAI (example):
        # elif provider == "openai":
        #     api_key = os.getenv("OPENAI_API_KEY")
        #     if not api_key:
        #         raise ValueError("OPENAI_API_KEY required")
        #     return OpenAIEmbeddings(
        #         api_key=api_key,
        #         model_name=config.model_name
        #     )

        else:
            # Unknown provider - provide helpful error
            supported_providers = ["sentence_transformers", "gemini"]
            raise ValueError(
                f"Unknown embedding provider: '{provider}'\n"
                f"Supported providers: {supported_providers}\n"
                f"Check your domain config YAML file.\n"
                f"To add a new provider, update EmbeddingFactory.create_embedder()"
            )


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of EmbeddingFactory usage.
    Run: python core/factories/embedding_factory.py
    """

    import logging
    from models.domain_config import EmbeddingConfig
    import numpy as np

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("EmbeddingFactory Usage Examples")
    print("=" * 70)

    # Example 1: Creating Sentence-Transformers embedder
    print("\n1. Creating Sentence-Transformers Embedder")
    print("-" * 70)

    st_config = EmbeddingConfig(
        provider="sentence_transformers",
        model_name="all-MiniLM-L6-v2",
        device="cpu",
        batch_size=32,
        normalize_embeddings=True
    )

    embedder = EmbeddingFactory.create_embedder(st_config)

    print(f"Created: {type(embedder).__name__}")
    print(f"Model: {embedder.get_model_name()}")
    print(f"Interface: {isinstance(embedder, EmbeddingInterface)}")

    # Test embedding
    texts = ["Hello world", "Factory pattern rocks"]
    embeddings = embedder.embed_texts(texts)
    print(f"Embeddings shape: {embeddings.shape}")

    # Example 2: Creating Gemini embedder (if API key available)
    print("\n2. Creating Gemini Embedder")
    print("-" * 70)

    if os.getenv("GEMINI_API_KEY"):
        gemini_config = EmbeddingConfig(
            provider="gemini",
            model_name="models/embedding-001",
            batch_size=32
        )

        try:
            embedder = EmbeddingFactory.create_embedder(gemini_config)
            print(f"Created: {type(embedder).__name__}")
            print(f"Model: {embedder.get_model_name()}")

            # Test embedding
            embeddings = embedder.embed_texts(texts)
            print(f"Embeddings shape: {embeddings.shape}")
        except Exception as e:
            print(f"Failed to create Gemini embedder: {e}")
    else:
        print("GEMINI_API_KEY not set, skipping Gemini example")
        print("Set it with: export GEMINI_API_KEY='your-key'")

    # Example 3: Polymorphism
    print("\n3. Demonstrating Polymorphism")
    print("-" * 70)


    def embed_with_any_provider(config: EmbeddingConfig, texts: List[str]):
        """Works with ANY embedding provider!"""
        embedder = EmbeddingFactory.create_embedder(config)
        embeddings = embedder.embed_texts(texts)
        print(f"Provider: {config.provider}")
        print(f"Model: {embedder.get_model_name()}")
        print(f"Shape: {embeddings.shape}")
        return embeddings


    test_texts = ["Python is great", "AI is powerful"]

    # Works with Sentence-Transformers
    print("Using Sentence-Transformers:")
    embed_with_any_provider(st_config, test_texts)

    print("\n✅ Same function works with different providers!")

    # Example 4: Error handling
    print("\n4. Error Handling")
    print("-" * 70)

    try:
        bad_config = EmbeddingConfig(provider="nonexistent")
        embedder = EmbeddingFactory.create_embedder(bad_config)
    except ValueError as e:
        print(f"✅ Caught expected error:")
        print(f"   {str(e).split(chr(10))[0]}")  # First line only

    print("\n" + "=" * 70)
    print("EmbeddingFactory examples completed!")
    print("=" * 70)
