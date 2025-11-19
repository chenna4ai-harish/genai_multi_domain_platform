"""
core/interfaces/embedding_interface.py

This module defines the abstract interfaces (contract) for all embedding providers
in the multi-domain document intelligence platform.

Purpose:
--------
Defines a standard interfaces that ALL embedding implementations must follow.
This enables swapping between different embedding providers (Sentence-Transformers,
Google Gemini, OpenAI, etc.) without changing any calling code.

What Are Embeddings?
--------------------
Embeddings are dense vector representations of text that capture semantic meaning.
Similar texts produce similar vectors (measured by cosine similarity).

Example:
"cat" → [0.2, 0.8, 0.1, ...]  (384 or 768 dimensions)
"kitten" → [0.21, 0.79, 0.11, ...]  (very similar vector)
"car" → [-0.5, 0.1, 0.9, ...]  (very different vector)

Why Use Abstract Base Classes?
-------------------------------
Allows your application to work with ANY embedding provider:
- Start with free Sentence-Transformers (local)
- Upgrade to Google Gemini (cloud API)
- Switch to OpenAI embeddings (best quality)
- Try custom fine-tuned models

All without changing a single line of calling code!

Design Pattern:
---------------
This follows the Strategy Pattern + Adapter Pattern:
- Strategy: Different embedding algorithms can be swapped
- Adapter: Wraps different APIs (HuggingFace, Google, OpenAI) into one interfaces

Example Usage:
--------------
# Factory creates the right embedder based on config
embedder: EmbeddingInterface = EmbeddingFactory.create_embedder(config)

# Caller doesn't care if it's Sentence-Transformers or Gemini!
texts = ["Hello world", "Python is great"]
embeddings = embedder.embed_texts(texts)
# Returns: numpy array of shape (2, embedding_dim)
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class EmbeddingInterface(ABC):
    """
    Abstract base class defining the interfaces for embedding providers.

    All embedding implementations (SentenceTransformerEmbeddings,
    GeminiEmbeddings, OpenAIEmbeddings, etc.) MUST inherit from this
    class and implement all abstract methods.

    Responsibilities:
    -----------------
    An embedder is responsible for:
    1. Converting text strings into dense vector embeddings
    2. Handling batching for efficiency
    3. Returning consistent embedding dimensions
    4. Providing model name/version information

    Key Concepts:
    -------------
    - **Embedding Dimension**: Number of values in each vector
      Common dimensions: 384 (MiniLM), 768 (BERT, Gemini), 1536 (OpenAI)

    - **Batching**: Processing multiple texts at once (faster than one-by-one)
      Example: 100 texts in batches of 32 = 4 API calls instead of 100

    - **Normalization**: Scaling vectors to unit length (L2 norm = 1)
      Benefit: Cosine similarity = dot product (faster computation)

    Provider Options:
    -----------------
    1. **Sentence-Transformers** (Local, Free)
       - Runs on your CPU/GPU
       - No API keys or costs
       - Models: all-MiniLM-L6-v2 (384-dim), all-mpnet-base-v2 (768-dim)
       - Speed: ~1000 texts/sec on GPU, ~50 texts/sec on CPU
       - Use case: MVP, budget projects, privacy-sensitive data

    2. **Google Gemini** (Cloud API, Premium)
       - Requires API key and billing
       - Models: models/embedding-001 (768-dim)
       - Speed: Rate limited by API (typically 1000 requests/min)
       - Use case: Production, when quality matters more than cost

    3. **OpenAI** (Cloud API, Premium)
       - Requires API key and billing (most expensive)
       - Models: text-embedding-ada-002 (1536-dim), text-embedding-3-small
       - Speed: Rate limited by API
       - Use case: When you need the absolute best quality

    Example Implementations:
    ------------------------
    See:
    - core/embeddings/sentence_transformer_embeddings.py (OPTION 1)
    - core/embeddings/gemini_embeddings.py (OPTION 2)
    - core/embeddings/openai_embeddings.py (OPTION 3 - future)

    Usage Example:
    --------------
    # Polymorphic usage - works with ANY embedder!

    def embed_documents(texts: List[str], embedder: EmbeddingInterface):
        '''Embed documents with ANY embedding provider.'''
        embeddings = embedder.embed_texts(texts)
        model_name = embedder.get_model_name()
        print(f"Embedded {len(texts)} texts using {model_name}")
        print(f"Embedding shape: {embeddings.shape}")
        return embeddings

    # Works with Sentence-Transformers
    st_embedder = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
    embed_documents(["Hello", "World"], st_embedder)

    # Works with Gemini (same calling code!)
    gemini_embedder = GeminiEmbeddings(api_key="your-key")
    embed_documents(["Hello", "World"], gemini_embedder)
    """

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        This is the CORE method that all embedding providers must implement.
        Each implementation will use its own model/API but must return
        embeddings in the same numpy array format.

        Parameters:
        -----------
        texts : List[str]
            List of text strings to embed.
            Can be single sentence or multi-sentence chunks.

            Example:
            [
                "Employee benefits include 15 vacation days.",
                "Health insurance covers medical and dental.",
                "401k matching up to 6% of salary."
            ]

            Important:
            - Empty strings should be handled gracefully (return zero vector or skip)
            - Very long texts may need truncation (check model max length)
            - Special characters should be preserved (don't strip unnecessarily)

        Returns:
        --------
        np.ndarray:
            2D numpy array of embeddings with shape (n_texts, embedding_dim).

            Example for 3 texts with 384-dim embeddings:
            array([[0.2, 0.1, -0.3, ..., 0.5],   # Text 1 embedding (384 values)
                   [0.1, 0.2, -0.1, ..., 0.4],   # Text 2 embedding (384 values)
                   [0.3, 0.0, -0.2, ..., 0.6]])  # Text 3 embedding (384 values)

            Shape: (3, 384)

            Requirements:
            - dtype should be float32 or float64
            - All embeddings must have the same dimension
            - If normalized, each row should have L2 norm ≈ 1.0

        Raises:
        -------
        ValueError:
            - If texts list is empty
            - If texts contain invalid data (e.g., all None)

        RuntimeError:
            - If embedding model fails to load
            - If API request fails (for cloud providers)
            - If rate limit is exceeded

        Implementation Guidelines:
        --------------------------
        1. **Validate Input**:
           if not texts:
               raise ValueError("texts cannot be empty")

        2. **Handle Empty Strings**:
           # Option 1: Skip empty strings
           valid_texts = [t for t in texts if t.strip()]

           # Option 2: Replace with placeholder
           texts = [t if t.strip() else "[EMPTY]" for t in texts]

        3. **Implement Batching** (for efficiency):
           all_embeddings = []
           for i in range(0, len(texts), batch_size):
               batch = texts[i:i + batch_size]
               batch_embeddings = self._embed_batch(batch)
               all_embeddings.append(batch_embeddings)
           return np.vstack(all_embeddings)

        4. **Handle Errors Gracefully**:
           try:
               embeddings = model.encode(texts)
           except Exception as e:
               logger.error(f"Embedding failed: {e}")
               raise RuntimeError(f"Failed to generate embeddings: {e}")

        5. **Normalize If Configured**:
           if self.normalize:
               embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        6. **Return Correct Shape**:
           assert embeddings.shape == (len(texts), self.embedding_dim)
           return embeddings

        Performance Considerations:
        ---------------------------
        - **Batching**: Process 16-64 texts at once (not one-by-one)
        - **GPU Acceleration**: Use CUDA if available (5-10x speedup)
        - **Caching**: Cache embeddings for frequently used texts
        - **Async Processing**: Use async APIs for cloud providers

        Example Implementation Pattern:
        -------------------------------
        def embed_texts(self, texts: List[str]) -> np.ndarray:
            # 1. Validate
            if not texts:
                raise ValueError("texts cannot be empty")

            # 2. Batch processing
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]

                # 3. Generate embeddings (provider-specific)
                batch_emb = self.model.encode(batch)  # Your implementation
                all_embeddings.append(batch_emb)

            # 4. Combine batches
            embeddings = np.vstack(all_embeddings)

            # 5. Normalize if configured
            if self.normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms

            return embeddings

        Testing Your Implementation:
        ----------------------------
        - Test with single text: ["Hello world"]
        - Test with batch: ["Text 1", "Text 2", "Text 3"]
        - Test with empty list: [] (should raise ValueError)
        - Test with very long text (1000+ words)
        - Verify output shape: (n_texts, embedding_dim)
        - Verify normalized vectors have L2 norm ≈ 1.0
        - Verify similar texts have high cosine similarity (>0.8)
        """
        pass  # Subclasses MUST implement this method

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the name/identifier of the embedding model being used.

        This information is stored in chunk metadata (ChunkMetadata.embedding_model_name)
        and is critical for:
        1. Debugging retrieval quality issues
        2. Tracking which model generated which embeddings
        3. Managing migrations between embedding models
        4. Audit trails and provenance

        Returns:
        --------
        str:
            Model name or identifier.

            Examples:
            - Sentence-Transformers: "all-MiniLM-L6-v2", "all-mpnet-base-v2"
            - Google Gemini: "models/embedding-001"
            - OpenAI: "text-embedding-ada-002", "text-embedding-3-small"
            - Custom: "company-finetuned-bert-v2"

        Example Usage:
        --------------
        embedder = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
        model_name = embedder.get_model_name()

        # Store in metadata
        chunk_metadata = ChunkMetadata(
            ...,
            embedding_model_name=model_name,  # "all-MiniLM-L6-v2"
            ...
        )

        # Later, when debugging:
        print(f"This chunk was embedded with: {chunk_metadata.embedding_model_name}")

        Why This Matters:
        -----------------
        If you change embedding models, old embeddings become incompatible!
        You need to track which model was used to know when to re-embed.

        Example Scenario:
        all-MiniLM-L6-v2 (384-dim) → all-mpnet-base-v2 (768-dim)

        Without tracking model names:
        - You'd get dimension mismatch errors
        - You wouldn't know which chunks need re-embedding

        With tracking:
        - Query chunks where embedding_model_name != "all-mpnet-base-v2"
        - Re-embed only those chunks
        - System continues working

        Implementation Example:
        -----------------------
        class SentenceTransformerEmbeddings(EmbeddingInterface):
            def __init__(self, model_name: str):
                self.model_name = model_name
                self.model = SentenceTransformer(model_name)

            def get_model_name(self) -> str:
                return self.model_name  # Return what was configured

        class GeminiEmbeddings(EmbeddingInterface):
            def __init__(self, api_key: str, model_name: str = "models/embedding-001"):
                self.model_name = model_name
                genai.configure(api_key=api_key)

            def get_model_name(self) -> str:
                return self.model_name  # Return Google's model identifier
        """
        pass  # Subclasses MUST implement this method


# =============================================================================
# USAGE NOTES FOR IMPLEMENTERS
# =============================================================================

"""
How to Implement a New Embedding Provider:
-------------------------------------------

1. Create a new file: core/embeddings/my_embeddings.py

2. Import the interfaces:
   from core.interfaces.embedding_interface import EmbeddingInterface
   import numpy as np

3. Create your class inheriting from EmbeddingInterface:
   class MyEmbeddings(EmbeddingInterface):
       def __init__(self, api_key: str, model_name: str, **kwargs):
           self.api_key = api_key
           self.model_name = model_name
           self.batch_size = kwargs.get('batch_size', 32)
           # Initialize your API client / model here

       def embed_texts(self, texts: List[str]) -> np.ndarray:
           # Your implementation using your API/model
           embeddings = your_api.embed(texts)
           return np.array(embeddings)

       def get_model_name(self) -> str:
           return self.model_name

4. Register in factory: core/factories/embedding_factory.py
   elif config.provider == "my_provider":
       return MyEmbeddings(
           api_key=os.getenv("MY_API_KEY"),
           model_name=config.model_name
       )

5. Update config: models/domain_config.py
   Update EmbeddingConfig field validator to allow "my_provider"

6. Use in YAML:
   embeddings:
     provider: "my_provider"
     model_name: "my-model-v1"
     batch_size: 32

7. Set environment variable:
   export MY_API_KEY="your-api-key-here"

That's it! Config-driven architecture means no changes to calling code.
"""
