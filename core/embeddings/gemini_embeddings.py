"""

core/embeddings/gemini_embeddings.py

This module implements the Google Gemini embedding provider (OPTION 2).

What is Google Gemini Embeddings?
----------------------------------
Gemini is Google's latest AI model family, offering state-of-the-art text
embeddings through a cloud API. The embedding model converts text into 768-
dimensional vectors optimized for semantic search and retrieval tasks.

Key Features:
-------------
- **High Quality**: State-of-the-art performance on benchmark datasets
- **Managed Service**: No model hosting or maintenance required
- **Scalable**: Handles high request volumes automatically
- **Task-Specific**: Optimized for different tasks (retrieval, similarity, classification)
- **Multilingual**: Supports 100+ languages

Model Information:
------------------
- Model: gemini-embedding-001 (also called "models/embedding-001")
- Dimension: 768
- Context Length: 2048 tokens (~1500 words)
- Languages: 100+ languages
- Output: Dense vectors optimized for cosine similarity

Task Types:
-----------
1. RETRIEVAL_DOCUMENT: For indexing documents (what we use)
2. RETRIEVAL_QUERY: For search queries
3. SEMANTIC_SIMILARITY: For general similarity tasks
4. CLASSIFICATION: For text classification
5. CLUSTERING: For clustering tasks

Pricing (as of 2025):
---------------------
- Free tier: 60 requests/minute
- Paid tier: $0.00025 per 1K characters (~$0.0025 per 1K tokens)
- Example: 1 million chunks (~500K tokens) ≈ $1.25

See: https://ai.google.dev/pricing

When to Use Gemini Embeddings:
-------------------------------
✅ Production deployments (managed, scalable)
✅ When quality matters more than cost
✅ Multilingual applications (100+ languages)
✅ When you don't want to manage infrastructure
✅ Cloud-native applications
❌ MVP/development (use free Sentence-Transformers)
❌ Privacy-sensitive data (stays on Google servers)
❌ Very high volume with tight budgets
❌ Offline deployments (requires internet)

API Setup:
----------
1. Go to https://makersuite.google.com/app/apikey
2. Create a new API key
3. Set environment variable:
   export GEMINI_API_KEY="your-api-key-here"

Installation:
-------------
pip install google-generativeai

References:
-----------
- Documentation: https://ai.google.dev/docs/embeddings
- API Reference: https://ai.google.dev/api/python/google/generativeai
- Pricing: https://ai.google.dev/pricing

"""

from typing import List
import numpy as np
import google.generativeai as genai
from core.interfaces.embedding_interface import EmbeddingInterface
import logging
import time

# Configure logging
logger = logging.getLogger(__name__)


class GeminiEmbeddings(EmbeddingInterface):
    """
    Google Gemini embedding provider (cloud API, premium).

    This implementation uses Google's Gemini API to generate high-quality
    embeddings through a managed cloud service.

    Configuration Parameters:
    -------------------------
    api_key : str
        Google API key for Gemini API
        Get one at: https://makersuite.google.com/app/apikey
        Best practice: Store in environment variable
            export GEMINI_API_KEY="your-key"
        Then: api_key=os.getenv("GEMINI_API_KEY")

    model_name : str
        Gemini embedding model to use
        Options:
        - "models/embedding-001" (768-dim, latest)
        - "gemini-embedding-001" (same as above, alias)
        Default: "models/embedding-001"

    batch_size : int
        Number of texts to process in one API call
        API limit: 100 texts per request
        Recommended: 32-64 for balance of speed and reliability

    task_type : str
        Optimization hint for the model
        Options:
        - "RETRIEVAL_DOCUMENT": For indexing (default for our use case)
        - "RETRIEVAL_QUERY": For search queries
        - "SEMANTIC_SIMILARITY": For similarity tasks
        Default: "RETRIEVAL_DOCUMENT"

    Example Usage:
    --------------
    import os

    # Initialize embedder with API key
    embedder = GeminiEmbeddings(
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name="models/embedding-001",
        batch_size=32
    )

    # Generate embeddings
    texts = ["Hello world", "AI is powerful"]
    embeddings = embedder.embed_texts(texts)

    # Result: numpy array of shape (2, 768)
    print(embeddings.shape)  # (2, 768)
    """

    def __init__(
            self,
            api_key: str,
            model_name: str = "models/embedding-001",
            batch_size: int = 32,
            task_type: str = "RETRIEVAL_DOCUMENT"
    ):
        """
        Initialize the Gemini embedder.

        Parameters:
        -----------
        api_key : str
            Google Gemini API key
        model_name : str
            Embedding model name
        batch_size : int
            Batch size for API calls (max 100)
        task_type : str
            Task optimization hint

        Raises:
        -------
        ValueError:
            If API key is missing or invalid
        RuntimeError:
            If API initialization fails
        """
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required. Get one at: "
                "https://makersuite.google.com/app/apikey\n"
                "Set environment variable: export GEMINI_API_KEY='your-key'"
            )

        self.api_key = api_key
        self.model_name = model_name
        self.batch_size = min(batch_size, 100)  # API limit: 100 per request
        self.task_type = task_type
        self.embedding_dim = 768  # Gemini embedding dimension

        # Configure Gemini API
        logger.info(f"Initializing Gemini API with model: {model_name}...")
        try:
            genai.configure(api_key=self.api_key)

            logger.info(
                f"✅ Gemini API initialized!\n"
                f"   Model: {model_name}\n"
                f"   Dimension: {self.embedding_dim}\n"
                f"   Batch size: {self.batch_size}\n"
                f"   Task type: {task_type}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            raise RuntimeError(
                f"Could not initialize Gemini API\n"
                f"Error: {e}\n"
                f"Check API key and internet connection"
            )

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using Gemini API.

        This method sends texts to Google's Gemini API in batches and
        returns dense vector representations.

        Parameters:
        -----------
        texts : List[str]
            List of text strings to embed

        Returns:
        --------
        np.ndarray:
            2D numpy array of shape (n_texts, 768)
            dtype: float32

        Raises:
        -------
        ValueError:
            If texts is empty
        RuntimeError:
            If API request fails

        API Limits:
        -----------
        - Free tier: 60 requests/minute
        - Paid tier: 1000 requests/minute
        - Max texts per request: 100
        - Max tokens per text: 2048 (~1500 words)

        Performance:
        ------------
        - Latency: ~200-500ms per API call
        - Throughput: ~60-300 texts/sec (depends on batch size and tier)

        Example:
        --------
        texts = ["Hello world", "AI is amazing"]
        embeddings = embedder.embed_texts(texts)
        print(embeddings.shape)  # (2, 768)
        """
        # Step 1: Validate input
        if not texts:
            raise ValueError("texts cannot be empty")

        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("All texts are empty after filtering")

        if len(valid_texts) < len(texts):
            logger.warning(
                f"Filtered out {len(texts) - len(valid_texts)} empty texts"
            )

        logger.debug(f"Embedding {len(valid_texts)} texts using Gemini API...")

        # Step 2: Process in batches (API limit: 100 texts per request)
        all_embeddings = []

        for i in range(0, len(valid_texts), self.batch_size):
            batch = valid_texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(valid_texts) + self.batch_size - 1) // self.batch_size

            logger.debug(
                f"Processing batch {batch_num}/{total_batches} "
                f"({len(batch)} texts)..."
            )

            try:
                # Call Gemini API
                start = time.time()
                result = genai.embed_content(
                    model=self.model_name,
                    content=batch,
                    task_type=self.task_type
                )
                api_time = time.time() - start

                # Extract embeddings from response
                # API returns dict with 'embedding' key containing list of values
                batch_embeddings = [emb['values'] for emb in result['embedding']]
                all_embeddings.extend(batch_embeddings)

                logger.debug(
                    f"Batch {batch_num} completed in {api_time:.2f}s "
                    f"({len(batch) / api_time:.0f} texts/sec)"
                )

                # Rate limiting: Add small delay between batches to avoid hitting limits
                if i + self.batch_size < len(valid_texts):
                    time.sleep(0.1)  # 100ms delay between batches

            except Exception as e:
                logger.error(f"Gemini API call failed for batch {batch_num}: {e}")
                raise RuntimeError(
                    f"Failed to generate embeddings using Gemini API\n"
                    f"Batch: {batch_num}/{total_batches}\n"
                    f"Error: {e}\n"
                    f"Batch sample: {batch[:2]}"
                )

        # Step 3: Convert to numpy array
        embeddings = np.array(all_embeddings, dtype=np.float32)

        # Step 4: Validate output
        expected_shape = (len(valid_texts), self.embedding_dim)
        if embeddings.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected embedding shape: {embeddings.shape}, "
                f"expected {expected_shape}"
            )

        logger.debug(
            f"✅ Generated {len(embeddings)} embeddings "
            f"(shape: {embeddings.shape})"
        )

        return embeddings

    def get_model_name(self) -> str:
        """
        Return the Gemini model name.

        This is stored in chunk metadata for provenance tracking.

        Returns:
        --------
        str:
            Model name/identifier
            Example: "models/embedding-001"
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
            Embedding dimension (768 for Gemini)
        """
        return self.embedding_dim


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of GeminiEmbeddings usage.
    Run: python core/embeddings/gemini_embeddings.py

    Requirements:
    - Set GEMINI_API_KEY environment variable
    - pip install google-generativeai
    """
    import os

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("GeminiEmbeddings Usage Examples")
    print("=" * 70)

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n❌ GEMINI_API_KEY not set!")
        print("Get API key: https://makersuite.google.com/app/apikey")
        print("Then run: export GEMINI_API_KEY='your-key'")
        exit(1)

    # Example 1: Basic embedding generation
    print("\n1. Basic Embedding Generation")
    print("-" * 70)

    embedder = GeminiEmbeddings(
        api_key=api_key,
        model_name="models/embedding-001",
        batch_size=32
    )

    texts = [
        "Machine learning powers modern AI applications",
        "Python is the most popular language for data science",
        "Cloud computing enables scalable applications"
    ]

    embeddings = embedder.embed_texts(texts)

    print(f"Input: {len(texts)} texts")
    print(f"Output: {embeddings.shape} (texts × dimensions)")
    print(f"Data type: {embeddings.dtype}")
    print(f"First embedding (first 5 values): {embeddings[0][:5]}")
    print(f"Embedding dimension: {embedder.get_embedding_dimension()}")

    # Example 2: Semantic similarity with Gemini
    print("\n2. Computing Similarity with Gemini Embeddings")
    print("-" * 70)

    from sklearn.metrics.pairwise import cosine_similarity

    query = "What is artificial intelligence?"
    documents = [
        "AI is the simulation of human intelligence by machines",
        "Python is a programming language",
        "Machine learning is a subset of AI"
    ]

    query_emb = embedder.embed_texts([query])
    doc_embs = embedder.embed_texts(documents)

    similarities = cosine_similarity(query_emb, doc_embs)[0]

    print(f"Query: '{query}'\n")
    for doc, sim in zip(documents, similarities):
        print(f"  {sim:.3f} - {doc}")

    print(f"\nMost relevant: {documents[similarities.argmax()]}")

    # Example 3: Batch processing
    print("\n3. Batch Processing Performance")
    print("-" * 70)

    import time

    test_texts = [f"Sample document number {i} about various topics" for i in range(50)]

    start = time.time()
    batch_embeddings = embedder.embed_texts(test_texts)
    total_time = time.time() - start

    print(f"Processed: {len(test_texts)} texts")
    print(f"Time: {total_time:.2f}s")
    print(f"Throughput: {len(test_texts) / total_time:.0f} texts/sec")
    print(f"Output shape: {batch_embeddings.shape}")

    # Example 4: Different task types
    print("\n4. Task-Specific Embeddings")
    print("-" * 70)

    # For document indexing
    doc_embedder = GeminiEmbeddings(
        api_key=api_key,
        task_type="RETRIEVAL_DOCUMENT"
    )

    # For queries
    query_embedder = GeminiEmbeddings(
        api_key=api_key,
        task_type="RETRIEVAL_QUERY"
    )

    document = ["The Python programming language was created in 1991"]
    query = ["Who created Python?"]

    doc_emb = doc_embedder.embed_texts(document)
    query_emb = query_embedder.embed_texts(query)

    similarity = cosine_similarity(query_emb, doc_emb)[0][0]

    print(f"Document (RETRIEVAL_DOCUMENT): {document[0]}")
    print(f"Query (RETRIEVAL_QUERY): {query[0]}")
    print(f"Similarity: {similarity:.3f}")

    print("\n" + "=" * 70)
    print("Gemini Embeddings examples completed!")
    print("=" * 70)
