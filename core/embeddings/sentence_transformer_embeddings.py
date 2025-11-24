"""

core/embeddings/sentence_transformer_embeddings.py

This module implements the Sentence-Transformers embedding provider (OPTION 1).

What are Sentence-Transformers?
--------------------------------
Sentence-Transformers is a Python library that provides an easy way to compute
dense vector representations (embeddings) for sentences, paragraphs, and images.
It's built on top of HuggingFace Transformers and PyTorch.

Key Features:
-------------
- **Local Execution**: Runs on your CPU/GPU, no API calls needed
- **Free**: No API keys or usage costs
- **Fast**: Optimized for inference, especially on GPU
- **High Quality**: State-of-the-art models trained on large datasets
- **Easy to Use**: Simple API, automatic batching, progress bars

Popular Models:
---------------
1. all-MiniLM-L6-v2 (RECOMMENDED FOR MVP):
   - Dimension: 384
   - Speed: Very fast (~1000 sentences/sec on GPU)
   - Quality: Good for general use cases
   - Size: 80MB
   - Use case: Default choice for most applications

2. all-mpnet-base-v2:
   - Dimension: 768
   - Speed: Fast (~500 sentences/sec on GPU)
   - Quality: Better than MiniLM, best overall performance
   - Size: 420MB
   - Use case: When quality matters more than speed

3. paraphrase-multilingual-MiniLM-L12-v2:
   - Dimension: 384
   - Speed: Fast
   - Quality: Good for 50+ languages
   - Size: 420MB
   - Use case: Multilingual documents

When to Use Sentence-Transformers:
-----------------------------------
✅ MVP and development phase (fast iteration)
✅ Budget-constrained projects (free)
✅ Privacy-sensitive data (stays on your servers)
✅ High-volume applications (no API rate limits)
✅ Offline deployments (no internet required)
❌ When you need absolute best quality (use Gemini/OpenAI)
❌ When you don't have GPU and need speed (cloud APIs are faster on CPU)

Performance:
------------
- CPU: ~50-200 sentences/sec (depends on model size)
- GPU (NVIDIA): ~1000-5000 sentences/sec (10-50x faster)
- Memory: ~1-2GB RAM for model + embeddings

Installation:
-------------
pip install sentence-transformers

For GPU support (10-50x speedup):
pip install sentence-transformers torch torchvision --index-url https://download.pytorch.org/whl/cu118

References:
-----------
- Documentation: https://www.sbert.net/
- Models: https://www.sbert.net/docs/pretrained_models.html
- GitHub: https://github.com/UKPLab/sentence-transformers

"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from core.interfaces.embedding_interface import EmbeddingInterface
import logging
import torch

# Configure logging
logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddings(EmbeddingInterface):
    """
    Sentence-Transformers embedding provider (local, free).

    This implementation uses the sentence-transformers library to generate
    embeddings locally on your CPU or GPU, with no API calls or costs.

    Configuration Parameters:
    -------------------------
    model_name : str
        Name of the Sentence-Transformers model to use
        Popular choices:
        - "all-MiniLM-L6-v2" (384-dim, fast, recommended default)
        - "all-mpnet-base-v2" (768-dim, better quality)
        - "paraphrase-multilingual-MiniLM-L12-v2" (384-dim, multilingual)
        See full list: https://www.sbert.net/docs/pretrained_models.html

    device : str
        Compute device to use: "cpu", "cuda", or "mps" (Apple Silicon)
        - "cpu": Works everywhere, slower (50-200 sent/sec)
        - "cuda": NVIDIA GPU, much faster (1000-5000 sent/sec)
        - "mps": Apple M1/M2, faster than CPU
        The library auto-detects if CUDA is available

    batch_size : int
        Number of texts to process in one batch
        Larger = faster but more memory
        Recommended: 16-32 for CPU, 64-128 for GPU

    normalize : bool
        Whether to normalize embeddings to unit length (L2 norm = 1)
        Recommended: True (default)
        Why: Enables cosine similarity = dot product (faster)

    Example Usage:
    --------------
    # Initialize embedder
    embedder = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        device="cuda",  # or "cpu"
        batch_size=32,
        normalize=True
    )

    # Generate embeddings
    texts = ["Hello world", "Python is great", "AI is powerful"]
    embeddings = embedder.embed_texts(texts)

    # Result: numpy array of shape (3, 384)
    print(embeddings.shape)  # (3, 384)
    print(embeddings[0][:5])  # First 5 values of first embedding
    """

    def __init__(
            self,
            model_name: str = "all-MiniLM-L6-v2",
            device: str = "cpu",
            batch_size: int = 32,
            normalize: bool = True
    ):
        """
        Initialize the Sentence-Transformers embedder.

        Parameters:
        -----------
        model_name : str
            Name of the pre-trained model to load
        device : str
            Device to run on: "cpu", "cuda", or "mps"
        batch_size : int
            Batch size for encoding
        normalize : bool
            Whether to normalize embeddings to unit length

        Notes:
        ------
        Model loading takes 2-10 seconds depending on model size and disk speed.
        The model is loaded into memory once and reused for all subsequent calls.

        Memory usage:
        - all-MiniLM-L6-v2: ~300MB RAM
        - all-mpnet-base-v2: ~1.5GB RAM
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize

        # Validate device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available. Falling back to CPU. "
                "Install PyTorch with CUDA support for GPU acceleration."
            )
            self.device = "cpu"

        # Load pre-trained model
        logger.info(f"Loading Sentence-Transformer model: {model_name} on {self.device}...")
        try:
            self.model = SentenceTransformer(model_name, device=self.device)

            # Get embedding dimension for logging
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            logger.info(
                f"✅ Model loaded successfully!\n"
                f"   Model: {model_name}\n"
                f"   Dimension: {self.embedding_dim}\n"
                f"   Device: {self.device}\n"
                f"   Batch size: {batch_size}\n"
                f"   Normalize: {normalize}"
            )

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise RuntimeError(
                f"Could not load Sentence-Transformer model: {model_name}\n"
                f"Error: {e}\n"
                f"Try installing: pip install sentence-transformers"
            )

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        This method processes texts in batches for efficiency and returns
        dense vector representations.

        Parameters:
        -----------
        texts : List[str]
            List of text strings to embed
            Can be sentences, paragraphs, or short documents

        Returns:
        --------
        np.ndarray:
            2D numpy array of shape (n_texts, embedding_dim)
            dtype: float32
            If normalize=True, each row has L2 norm = 1.0

        Raises:
        -------
        ValueError:
            If texts is empty
        RuntimeError:
            If embedding generation fails

        Performance:
        ------------
        The method uses automatic batching and shows a progress bar for
        large batches (>100 texts). Typical speeds:
        - CPU (all-MiniLM-L6-v2): ~50-100 texts/sec
        - GPU (all-MiniLM-L6-v2): ~1000-2000 texts/sec
        - GPU (all-mpnet-base-v2): ~500-1000 texts/sec

        Example:
        --------
        texts = ["Hello world", "AI is amazing", "Python rocks"]
        embeddings = embedder.embed_texts(texts)

        # Check shape and normalization
        print(embeddings.shape)  # (3, 384)
        print(np.linalg.norm(embeddings[0]))  # ~1.0 if normalized

        # Compute similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        print(f"Similarity: {similarity:.3f}")
        """
        # Step 1: Validate input
        if not texts:
            raise ValueError("texts cannot be empty")

        if not isinstance(texts, list):
            raise ValueError(f"texts must be a list, got {type(texts)}")

        # Filter out empty strings (encode would fail)
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("All texts are empty after filtering")

        if len(valid_texts) < len(texts):
            logger.warning(
                f"Filtered out {len(texts) - len(valid_texts)} empty texts. "
                f"Embedding {len(valid_texts)} valid texts."
            )

        logger.debug(f"Embedding {len(valid_texts)} texts using {self.model_name}...")

        try:
            # Step 2: Generate embeddings (batched internally by library)
            embeddings = self.model.encode(
                valid_texts,
                batch_size=self.batch_size,
                show_progress_bar=len(valid_texts) > 100,  # Show progress for large batches
                convert_to_numpy=True,  # Return numpy array (not torch tensor)
                normalize_embeddings=self.normalize,  # L2 normalization
                device=self.device  # Explicitly specify device
            )

            # Step 3: Validate output
            expected_shape = (len(valid_texts), self.embedding_dim)
            if embeddings.shape != expected_shape:
                raise RuntimeError(
                    f"Unexpected embedding shape: {embeddings.shape}, "
                    f"expected {expected_shape}"
                )

            logger.debug(
                f"✅ Generated embeddings: shape={embeddings.shape}, "
                f"dtype={embeddings.dtype}"
            )

            # Step 4: Verify normalization (if enabled)
            if self.normalize:
                # Check first embedding's L2 norm (should be ~1.0)
                norm = np.linalg.norm(embeddings[0])
                if not (0.99 <= norm <= 1.01):
                    logger.warning(
                        f"Normalization may have failed. L2 norm = {norm:.4f} (expected ~1.0)"
                    )

            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(
                f"Failed to generate embeddings using {self.model_name}\n"
                f"Error: {e}\n"
                f"Texts sample: {valid_texts[:2]}"
            )

    def get_model_name(self) -> str:
        """
        Return the name of the embedding model.

        This is stored in chunk metadata for provenance tracking.

        Returns:
        --------
        str:
            Model name/identifier
            Example: "all-MiniLM-L6-v2"
        """
        return self.model_name

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.

        Useful for:
        - Configuring vector stores (Pinecone needs dimension upfront)
        - Validating embedding shapes
        - Documentation and logging

        Returns:
        --------
        int:
            Embedding dimension
            Examples: 384 (MiniLM), 768 (MPNet, BERT)
        """
        return self.embedding_dim


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of SentenceTransformerEmbeddings usage.
    Run this file: python core/embeddings/sentence_transformer_embeddings.py
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("SentenceTransformerEmbeddings Usage Examples")
    print("=" * 70)

    # Example 1: Basic embedding generation
    print("\n1. Basic Embedding Generation")
    print("-" * 70)

    embedder = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2",
        device="cpu",  # Change to "cuda" if you have GPU
        batch_size=32,
        normalize=True
    )

    texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language"
    ]

    embeddings = embedder.embed_texts(texts)

    print(f"Input: {len(texts)} texts")
    print(f"Output: {embeddings.shape} (texts × dimensions)")
    print(f"Data type: {embeddings.dtype}")
    print(f"First embedding (first 5 values): {embeddings[0][:5]}")
    print(f"L2 norm of first embedding: {np.linalg.norm(embeddings[0]):.4f}")

    # Example 2: Computing similarity
    print("\n2. Computing Semantic Similarity")
    print("-" * 70)

    from sklearn.metrics.pairwise import cosine_similarity

    # Embed query and documents
    query = "What is machine learning?"
    documents = [
        "Machine learning is a branch of AI",
        "Python is a programming language",
        "Deep learning uses neural networks"
    ]

    query_embedding = embedder.embed_texts([query])
    doc_embeddings = embedder.embed_texts(documents)

    # Compute similarities
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    print(f"Query: '{query}'\n")
    for doc, sim in zip(documents, similarities):
        print(f"  {sim:.3f} - {doc}")

    print(f"\nMost relevant: {documents[similarities.argmax()]}")

    # Example 3: Batch processing performance
    print("\n3. Batch Processing Performance")
    print("-" * 70)

    import time

    # Generate test texts
    test_texts = [f"This is test sentence number {i}" for i in range(100)]

    start = time.time()
    batch_embeddings = embedder.embed_texts(test_texts)
    batch_time = time.time() - start

    print(f"Batch processing: {len(test_texts)} texts in {batch_time:.3f}s")
    print(f"Throughput: {len(test_texts) / batch_time:.0f} texts/second")
    print(f"Output shape: {batch_embeddings.shape}")

    # Example 4: Device comparison (if CUDA available)
    print("\n4. Device Comparison")
    print("-" * 70)

    if torch.cuda.is_available():
        print("CUDA is available! Comparing CPU vs GPU speed...")

        # CPU
        cpu_embedder = SentenceTransformerEmbeddings(device="cpu", batch_size=32)
        start = time.time()
        cpu_embeddings = cpu_embedder.embed_texts(test_texts)
        cpu_time = time.time() - start

        # GPU
        gpu_embedder = SentenceTransformerEmbeddings(device="cuda", batch_size=64)
        start = time.time()
        gpu_embeddings = gpu_embedder.embed_texts(test_texts)
        gpu_time = time.time() - start

        print(f"CPU: {cpu_time:.3f}s ({len(test_texts) / cpu_time:.0f} texts/sec)")
        print(f"GPU: {gpu_time:.3f}s ({len(test_texts) / gpu_time:.0f} texts/sec)")
        print(f"Speedup: {cpu_time / gpu_time:.1f}x faster on GPU")
    else:
        print("CUDA not available. Install PyTorch with CUDA support for GPU acceleration:")
        print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

    # Example 5: Model comparison
    print("\n5. Model Comparison")
    print("-" * 70)

    models_to_test = [
        ("all-MiniLM-L6-v2", "Fast, 384-dim"),
        ("all-mpnet-base-v2", "Better quality, 768-dim")
    ]

    sample_text = ["Machine learning transforms industries"]

    for model_name, description in models_to_test:
        try:
            embedder = SentenceTransformerEmbeddings(model_name=model_name, device="cpu")
            embedding = embedder.embed_texts(sample_text)
            print(f"{model_name}:")
            print(f"  Description: {description}")
            print(f"  Dimension: {embedding.shape[1]}")
            print(f"  First 3 values: {embedding[0][:3]}")
        except Exception as e:
            print(f"{model_name}: Failed to load - {e}")

    print("\n" + "=" * 70)
    print("SentenceTransformerEmbeddings examples completed!")
    print("=" * 70)
