"""

core/factories/embedding_factory.py

This module implements the Factory Pattern for creating embedding provider instances.

What is the Embedding Factory?
-------------------------------
The EmbeddingFactory creates embedding provider instances (e.g., Sentence-Transformers,
Gemini, OpenAI) based on configuration without requiring calling code to know about
specific implementations.

Why Use a Factory for Embeddings?
----------------------------------
1. **Provider Abstraction**: Switch embedding providers via config, not code changes
2. **Configuration-Driven**: All embedding settings in YAML config files
3. **Environment Handling**: Automatically handles API keys from environment variables
4. **Easy Testing**: Mock factories in tests without touching real providers
5. **Extensibility**: Add new providers by registering, no changes to existing code

How This Enables Multi-Provider Architecture:
----------------------------------------------
Instead of:
    if config.provider == "sentence_transformers":
        embedder = SentenceTransformerEmbeddings(model, device, batch_size)
    elif config.provider == "gemini":
        embedder = GeminiEmbeddings(api_key, model)
    # ... many more providers

You write:
    embedder = EmbeddingFactory.create_embedder(config)

Benefits:
- Calling code doesn't know which provider is used
- Adding new providers only requires updating the factory
- Configuration changes don't require code deployment
- All providers implement EmbeddingInterface (consistent API)

Example Usage:
--------------
# From domain config
config_mgr = ConfigManager()
domain_config = config_mgr.load_domain_config("hr")
embedder = EmbeddingFactory.create_embedder(domain_config.embeddings)

# From dict config
config = {
    "provider": "sentence_transformers",
    "model_name": "all-MiniLM-L6-v2",
    "device": "cuda",
    "batch_size": 32
}
embedder = EmbeddingFactory.create_embedder(config)

# Use embedder (same interface regardless of provider!)
texts = ["Hello world", "AI is powerful"]
embeddings = embedder.embed_texts(texts)

References:
-----------
- Factory Pattern: https://refactoring.guru/design-patterns/factory-method
- Embedding Providers: See core/embeddings/ directory

"""

import os
import logging
from typing import Union, Dict, Any

# Import embedding implementations
from core.embeddings.sentence_transformer_embeddings import SentenceTransformerEmbeddings
from core.embeddings.gemini_embeddings import GeminiEmbeddings
from core.interfaces.embedding_interface import EmbeddingInterface

# Configure logging
logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """
    Factory for creating embedding provider instances based on configuration.

    Supported Providers:
    - sentence_transformers: Local embeddings (HuggingFace Sentence-Transformers)
    - gemini: Google Gemini API embeddings

    All providers implement EmbeddingInterface for consistency.
    """

    # Registry of available embedding providers
    _available_providers = {
        "sentence_transformers": SentenceTransformerEmbeddings,
        "gemini": GeminiEmbeddings,
        # Add new providers here as you implement them:
        # "openai": OpenAIEmbeddings,
        # "huggingface": HuggingFaceEmbeddings,
        # "cohere": CohereEmbeddings,
    }

    @staticmethod
    def create_embedder(
            config: Union[Dict[str, Any], Any]
    ) -> EmbeddingInterface:
        """
        Create an embedding provider instance based on configuration.

        This method:
        1. Extracts the provider from config
        2. Validates the provider is supported
        3. Extracts provider-specific parameters
        4. Handles API keys from environment variables
        5. Instantiates the embedder with correct parameters

        Parameters:
        -----------
        config : Union[Dict, Any]
            Configuration object or dict with embedding parameters
            Required fields:
            - provider: str ("sentence_transformers", "gemini", etc.)

            Common fields:
            - model_name: str (model identifier)

            Provider-specific fields:
            For sentence_transformers:
            - device: str ("cpu", "cuda", "mps") - default: "cpu"
            - batch_size: int - default: 32
            - normalize: bool - default: True

            For gemini:
            - api_key: str (optional, reads from GEMINI_API_KEY env var)
            - batch_size: int - default: 32
            - task_type: str - default: "RETRIEVAL_DOCUMENT"

        Returns:
        --------
        EmbeddingInterface:
            Instantiated embedder implementing EmbeddingInterface

        Raises:
        -------
        ValueError:
            If provider is missing or unsupported
            If required parameters are missing (e.g., API keys)
            If parameters are invalid

        Example:
        --------
        # Sentence-Transformers (local)
        config = {
            "provider": "sentence_transformers",
            "model_name": "all-MiniLM-L6-v2",
            "device": "cuda",
            "batch_size": 32,
            "normalize": True
        }
        embedder = EmbeddingFactory.create_embedder(config)

        # Gemini (cloud API)
        config = {
            "provider": "gemini",
            "model_name": "models/embedding-001"
        }
        embedder = EmbeddingFactory.create_embedder(config)
        """
        # Step 1: Extract provider from config (support both Pydantic and dict)
        if isinstance(config, dict):
            provider = config.get("provider")
        else:
            provider = getattr(config, "provider", None)

        if not provider:
            raise ValueError(
                "Embedding config must specify a 'provider' field. "
                f"Available providers: {list(EmbeddingFactory._available_providers.keys())}"
            )

        provider = provider.lower()
        logger.info(f"Creating embedder with provider: {provider}")

        # Step 2: Validate provider is supported
        embedder_cls = EmbeddingFactory._available_providers.get(provider)
        if not embedder_cls:
            raise ValueError(
                f"Unknown embedding provider: '{provider}'. "
                f"Available providers: {list(EmbeddingFactory._available_providers.keys())}"
            )

        # Step 3: Extract common parameters
        if isinstance(config, dict):
            model_name = config.get("model_name")
        else:
            model_name = getattr(config, "model_name", None)

        if not model_name:
            raise ValueError(
                f"Embedding config must specify 'model_name' for provider '{provider}'"
            )

        # Step 4: Instantiate embedder based on provider
        try:
            if provider == "sentence_transformers":
                # Extract Sentence-Transformers specific parameters
                if isinstance(config, dict):
                    device = config.get("device", "cpu")
                    batch_size = config.get("batch_size", 32)
                    normalize = config.get("normalize", True)
                else:
                    device = getattr(config, "device", "cpu")
                    batch_size = getattr(config, "batch_size", 32)
                    normalize = getattr(config, "normalize", True)

                embedder = SentenceTransformerEmbeddings(
                    model_name=model_name,
                    device=device,
                    batch_size=batch_size,
                    normalize=normalize
                )

                logger.info(
                    f"Created SentenceTransformerEmbeddings: "
                    f"model={model_name}, device={device}, "
                    f"batch_size={batch_size}, normalize={normalize}"
                )

            elif provider == "gemini":
                # Get API key from environment or config
                if isinstance(config, dict):
                    api_key = config.get("api_key") or os.getenv("GEMINI_API_KEY")
                    batch_size = config.get("batch_size", 32)
                    task_type = config.get("task_type", "RETRIEVAL_DOCUMENT")
                else:
                    api_key = getattr(config, "api_key", None) or os.getenv("GEMINI_API_KEY")
                    batch_size = getattr(config, "batch_size", 32)
                    task_type = getattr(config, "task_type", "RETRIEVAL_DOCUMENT")

                if not api_key:
                    raise ValueError(
                        "GEMINI_API_KEY is required for Gemini embeddings. "
                        "Set environment variable: export GEMINI_API_KEY='your-key' "
                        "or provide 'api_key' in config. "
                        "Get key at: https://makersuite.google.com/app/apikey"
                    )

                embedder = GeminiEmbeddings(
                    api_key=api_key,
                    model_name=model_name,
                    batch_size=batch_size,
                    task_type=task_type
                )

                logger.info(
                    f"Created GeminiEmbeddings: "
                    f"model={model_name}, batch_size={batch_size}, "
                    f"task_type={task_type}"
                )

            # Add more providers here as you implement them
            # elif provider == "openai":
            #     api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
            #     if not api_key:
            #         raise ValueError("OPENAI_API_KEY required")
            #     embedder = OpenAIEmbeddings(api_key=api_key, model_name=model_name)

            else:
                # For future custom providers
                # Assume they follow pattern: __init__(model_name, **kwargs)
                raise ValueError(
                    f"Provider '{provider}' is registered but not yet implemented. "
                    f"Please add instantiation logic in EmbeddingFactory.create_embedder()"
                )

            return embedder

        except Exception as e:
            logger.error(
                f"Failed to create embedder with provider '{provider}': {e}"
            )
            raise ValueError(
                f"Failed to instantiate embedder for provider '{provider}'. "
                f"Error: {e}"
            )

    @staticmethod
    def get_available_providers() -> list:
        """
        Get list of available embedding providers.

        Useful for:
        - Validation
        - UI dropdowns
        - Documentation
        - CLI help text

        Returns:
        --------
        list:
            List of provider names
            Example: ["sentence_transformers", "gemini"]
        """
        return list(EmbeddingFactory._available_providers.keys())

    @staticmethod
    def register_provider(name: str, embedder_class: type):
        """
        Register a new embedding provider (for extensibility).

        This allows adding custom embedding providers at runtime
        without modifying the factory code.

        Parameters:
        -----------
        name : str
            Provider name (will be lowercased)
        embedder_class : type
            Embedder class implementing EmbeddingInterface

        Raises:
        -------
        ValueError:
            If embedder_class doesn't implement required methods

        Example:
        --------
        # Register custom embedder
        EmbeddingFactory.register_provider("custom", CustomEmbeddings)

        # Now can use it
        config = {"provider": "custom", "model_name": "custom-model"}
        embedder = EmbeddingFactory.create_embedder(config)
        """
        name = name.lower()

        # Validate that class implements EmbeddingInterface methods
        required_methods = ["embed_texts", "get_model_name"]
        for method in required_methods:
            if not hasattr(embedder_class, method):
                raise ValueError(
                    f"Embedder class must implement '{method}' method "
                    f"to be registered (EmbeddingInterface)"
                )

        if name in EmbeddingFactory._available_providers:
            logger.warning(f"Overwriting existing provider: {name}")

        EmbeddingFactory._available_providers[name] = embedder_class
        logger.info(f"Registered embedding provider: {name}")


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of EmbeddingFactory usage.
    Run: python core/factories/embedding_factory.py
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("EmbeddingFactory Usage Examples")
    print("=" * 70)

    # Example 1: Create Sentence-Transformers embedder from dict config
    print("\n1. Sentence-Transformers Embedder from Dict Config")
    print("-" * 70)

    st_config = {
        "provider": "sentence_transformers",
        "model_name": "all-MiniLM-L6-v2",
        "device": "cpu",
        "batch_size": 32,
        "normalize": True
    }

    st_embedder = EmbeddingFactory.create_embedder(st_config)

    print(f"Created: {st_embedder.__class__.__name__}")
    print(f"Model: {st_embedder.get_model_name()}")
    print(f"Dimension: {st_embedder.get_embedding_dimension()}")

    # Test embedding generation
    texts = ["Hello world", "AI is powerful"]
    embeddings = st_embedder.embed_texts(texts)
    print(f"Generated embeddings: shape={embeddings.shape}")

    # Example 2: Create Gemini embedder (if API key is set)
    print("\n2. Gemini Embedder from Dict Config")
    print("-" * 70)

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        gemini_config = {
            "provider": "gemini",
            "model_name": "models/embedding-001",
            "batch_size": 32
        }

        gemini_embedder = EmbeddingFactory.create_embedder(gemini_config)

        print(f"Created: {gemini_embedder.__class__.__name__}")
        print(f"Model: {gemini_embedder.get_model_name()}")
        print(f"Dimension: {gemini_embedder.get_embedding_dimension()}")

        # Test embedding generation
        embeddings = gemini_embedder.embed_texts(texts)
        print(f"Generated embeddings: shape={embeddings.shape}")
    else:
        print("⚠️  GEMINI_API_KEY not set. Skipping Gemini example.")
        print("Set it with: export GEMINI_API_KEY='your-key'")

    # Example 3: List available providers
    print("\n3. Available Providers")
    print("-" * 70)

    providers = EmbeddingFactory.get_available_providers()
    print(f"Available providers: {providers}")

    # Example 4: Error handling - unknown provider
    print("\n4. Error Handling - Unknown Provider")
    print("-" * 70)

    try:
        invalid_config = {
            "provider": "unknown_provider",
            "model_name": "some-model"
        }
        EmbeddingFactory.create_embedder(invalid_config)
    except ValueError as e:
        print(f"✅ Caught expected error: {e}")

    # Example 5: Error handling - missing model_name
    print("\n5. Error Handling - Missing Model Name")
    print("-" * 70)

    try:
        invalid_config = {
            "provider": "sentence_transformers"
            # Missing model_name
        }
        EmbeddingFactory.create_embedder(invalid_config)
    except ValueError as e:
        print(f"✅ Caught expected error: {e}")

    # Example 6: Provider-agnostic code
    print("\n6. Provider-Agnostic Usage Pattern")
    print("-" * 70)

    # This code works with ANY provider!
    configs_to_test = [
        {
            "provider": "sentence_transformers",
            "model_name": "all-MiniLM-L6-v2",
            "device": "cpu"
        }
    ]

    # Add Gemini if API key is available
    if gemini_api_key:
        configs_to_test.append({
            "provider": "gemini",
            "model_name": "models/embedding-001"
        })

    test_text = ["Machine learning transforms industries"]

    for config in configs_to_test:
        embedder = EmbeddingFactory.create_embedder(config)
        embedding = embedder.embed_texts(test_text)

        print(f"\nProvider: {config['provider']}")
        print(f"  Model: {embedder.get_model_name()}")
        print(f"  Dimension: {embedder.get_embedding_dimension()}")
        print(f"  Embedding shape: {embedding.shape}")

    print("\n" + "=" * 70)
    print("EmbeddingFactory examples completed!")
    print("=" * 70)
