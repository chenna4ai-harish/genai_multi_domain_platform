"""

core/chunking/semantic_chunker.py

This module implements the Semantic Chunking strategy.

What is Semantic Chunking?
---------------------------
Semantic chunking groups sentences together based on their semantic similarity
(measured using embeddings). Unlike fixed-size chunking, semantic chunking
creates variable-sized chunks that are topically coherent.

How It Works:
-------------
1. Split document into sentences
2. Generate embeddings for each sentence
3. Compare consecutive sentences using cosine similarity
4. Group similar sentences (above threshold) into same chunk
5. Start new chunk when similarity drops below threshold
6. Respect max_chunk_size as safety limit

Example:
--------
Text:
"Python is a programming language. It's easy to learn. Python has great libraries.
The weather is nice today. It's sunny outside."

Semantic chunks (threshold=0.7):
Chunk 1: "Python is a programming language. It's easy to learn. Python has great libraries."
(High similarity - all about Python)

Chunk 2: "The weather is nice today. It's sunny outside."
(High similarity - both about weather, but different topic from Chunk 1)

When to Use Semantic Chunking:
-------------------------------
✅ Technical documentation with distinct topics
✅ Documents with clear section boundaries
✅ When coherence matters more than size uniformity
✅ Academic papers, research documents
❌ Very long documents (slow due to embedding computation)
❌ When processing speed is critical
❌ Documents without clear topical structure

Advantages:
-----------
+ Creates topically coherent chunks
+ Respects semantic boundaries
+ Better retrieval quality (chunks are self-contained topics)
+ No awkward mid-sentence splits

Disadvantages:
--------------
- Slower (requires embedding every sentence)
- Variable chunk sizes (harder to predict token usage)
- Higher memory footprint
- Requires embedding model loaded in memory

"""

from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from core.interfaces.chunking_interface import ChunkerInterface
from models.metadata_models import ChunkMetadata
from models.domain_config import SemanticChunkingConfig
import logging
import re

# Configure logging
logger = logging.getLogger(__name__)


class SemanticChunker(ChunkerInterface):
    """
    Semantic chunker that groups sentences by topical similarity.

    This implementation uses sentence embeddings to determine which sentences
    belong together based on semantic similarity rather than fixed size.

    Configuration Parameters:
    -------------------------
    similarity_threshold : float
        Cosine similarity threshold (0.0-1.0) for grouping sentences
        Higher = stricter grouping (more, smaller chunks)
        Lower = looser grouping (fewer, larger chunks)
        Recommended: 0.6-0.8 for most documents
        Default: 0.7 (sentences must be 70% similar)

    max_chunk_size : int
        Maximum characters allowed in a semantic chunk
        Acts as safety limit to prevent very long chunks
        Range: 200-3000 characters
        Recommended: 800-1500
        Default: 1000

    embedding_model_name : str
        Name of the embedding model (for metadata tracking)

    Internal Model:
    ---------------
    Uses 'all-MiniLM-L6-v2' (384-dim) for sentence embeddings.
    This is a lightweight, fast model good for sentence-level similarity.
    Could be made configurable for advanced use cases.

    Example Usage:
    --------------
    # Initialize with config
    config = SemanticChunkingConfig(
        similarity_threshold=0.7,
        max_chunk_size=1000
    )
    chunker = SemanticChunker(config, embedding_model_name="all-MiniLM-L6-v2")

    # Chunk a document
    text = "Your document text here..."
    chunks = chunker.chunk_text(
        text=text,
        doc_id="technical_spec_v2",
        domain="engineering",
        source_file_path="./docs/api_spec.pdf",
        file_hash="xyz789..."
    )

    # Result: Variable-sized, topically coherent chunks
    for chunk in chunks:
        print(f"{chunk.char_range}: {len(chunk.chunk_text)} chars")
    """

    def __init__(self, config: SemanticChunkingConfig, embedding_model_name: str):
        """
        Initialize the semantic chunker with configuration.

        Parameters:
        -----------
        config : SemanticChunkingConfig
            Configuration with similarity_threshold and max_chunk_size
        embedding_model_name : str
            Name of the embedding model (stored in metadata)

        Notes:
        ------
        The __init__ method loads the sentence embedding model into memory.
        This happens once per chunker instance. Model loading takes ~2-5 seconds
        but subsequent embeddings are fast.
        """
        self.similarity_threshold = config.similarity_threshold
        self.max_chunk_size = config.max_chunk_size
        self.embedding_model_name = embedding_model_name

        # Load sentence embedding model for similarity computation
        # Using all-MiniLM-L6-v2: lightweight (80MB), fast, good quality
        logger.info("Loading sentence embedding model for semantic chunking...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        logger.info(
            f"Initialized SemanticChunker: threshold={self.similarity_threshold}, "
            f"max_size={self.max_chunk_size}, model={self.embedding_model_name}"
        )

    def get_strategy_name(self) -> str:
        """
        Return the name of this chunking strategy.

        Required by ChunkingInterface.

        Returns:
        --------
        str:
            Strategy name: "semantic"
        """
        return "semantic"

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk text by semantic similarity (Phase 2 interface).

        This method implements the ChunkingInterface.chunk() signature.

        Parameters:
        -----------
        text : str
            Text to chunk
        metadata : Dict[str, Any]
            Metadata dict with required fields

        Returns:
        --------
        List[Dict[str, Any]]:
            List of dicts with 'text' and 'metadata' keys
        """
        # Call the existing chunk_text method
        chunk_metadata_list = self.chunk_text(
            text=text,
            doc_id=metadata.get('doc_id', 'unknown'),
            domain=metadata.get('domain', 'unknown'),
            source_file_path=metadata.get('source_file_path', ''),
            file_hash=metadata.get('source_file_hash', 'a' * 64),
            uploader_id=metadata.get('uploader_id'),
            page_num=metadata.get('page_num')
        )

        # Convert ChunkMetadata objects to dicts
        chunks = []
        for chunk_meta in chunk_metadata_list:
            chunks.append({
                'text': chunk_meta.chunk_text,
                'metadata': chunk_meta
            })

        return chunks


    def chunk_text(
            self,
            text: str,
            doc_id: str,
            domain: str,
            source_file_path: str,
            file_hash: str,
            uploader_id: str = None,
            page_num: int = None
    ) -> List[ChunkMetadata]:
        """
        Split text by semantic similarity between sentences.

        Algorithm:
        ----------
        1. Split text into sentences
        2. Embed all sentences (batch processing for speed)
        3. Start first chunk with first sentence
        4. For each subsequent sentence:
           a. Compute similarity with previous sentence
           b. If similar enough (>= threshold) AND chunk not too large:
              - Add to current chunk
           c. Else:
              - Finalize current chunk
              - Start new chunk
        5. Create ChunkMetadata for each semantic chunk

        Parameters:
        -----------
        See ChunkerInterface.chunk_text() for parameter documentation.

        Returns:
        --------
        List[ChunkMetadata]:
            List of variable-sized, topically coherent chunks.

        Performance Notes:
        ------------------
        - Embedding generation is the slowest part (~10-50ms per sentence)
        - For 100-sentence document: ~1-5 seconds on CPU, ~0.2-0.5s on GPU
        - Batch embedding is much faster than one-by-one

        Example:
        --------
        text = "Python is great. It's easy. Python has libraries. " \
               "Weather is nice. It's sunny."

        With threshold=0.7:
        Chunk 1: "Python is great. It's easy. Python has libraries."
        (High intra-chunk similarity - all about Python)

        Chunk 2: "Weather is nice. It's sunny."
        (High intra-chunk similarity - both about weather)
        """
        # Step 1: Validate inputs
        if not text or not text.strip():
            logger.warning(f"Empty text for doc_id={doc_id}")
            raise ValueError(f"Text cannot be empty for document: {doc_id}")

        logger.info(
            f"Semantic chunking: doc_id={doc_id}, text_length={len(text)} chars"
        )

        # Step 2: Split into sentences
        sentences = self._split_sentences(text)
        if not sentences:
            logger.warning(f"No sentences found in doc_id={doc_id}")
            return []

        logger.debug(f"Split into {len(sentences)} sentences")

        # Step 3: Generate embeddings for all sentences (batch for speed)
        logger.debug("Generating sentence embeddings...")
        embeddings = self.model.encode(
            sentences,
            batch_size=32,  # Process 32 sentences at once
            show_progress_bar=False,  # Disable progress bar (avoid log spam)
            convert_to_numpy=True
        )
        logger.debug(f"Generated embeddings: shape={embeddings.shape}")

        # Step 4: Group sentences by semantic similarity
        chunks: List[ChunkMetadata] = []
        current_chunk_sentences = [sentences[0]]  # Start with first sentence
        current_chunk_start = 0  # Character position in original text

        for i in range(1, len(sentences)):
            # Compute similarity between current and previous sentence
            similarity = cosine_similarity(
                embeddings[i - 1:i],  # Previous sentence embedding
                embeddings[i:i + 1]  # Current sentence embedding
            )[0][0]

            # Check if we should add to current chunk or start new one
            current_text = ' '.join(current_chunk_sentences + [sentences[i]])
            should_add_to_current = (
                    similarity >= self.similarity_threshold  # Similar enough
                    and len(current_text) <= self.max_chunk_size  # Not too large
            )

            if should_add_to_current:
                # Add to current chunk
                current_chunk_sentences.append(sentences[i])
                logger.debug(
                    f"Added sentence {i} to current chunk "
                    f"(similarity={similarity:.3f}, size={len(current_text)})"
                )
            else:
                # Finalize current chunk and start new one
                chunk_metadata = self._create_chunk_metadata(
                    sentences=current_chunk_sentences,
                    text=text,
                    doc_id=doc_id,
                    domain=domain,
                    source_file_path=source_file_path,
                    file_hash=file_hash,
                    uploader_id=uploader_id,
                    page_num=page_num,
                    start_pos=current_chunk_start
                )
                chunks.append(chunk_metadata)

                logger.debug(
                    f"Finalized chunk {len(chunks)} "
                    f"(similarity drop: {similarity:.3f} < {self.similarity_threshold} "
                    f"or size limit reached)"
                )

                # Start new chunk with current sentence
                current_chunk_sentences = [sentences[i]]
                current_chunk_start = text.find(sentences[i], current_chunk_start)

        # Step 5: Add final chunk
        if current_chunk_sentences:
            chunk_metadata = self._create_chunk_metadata(
                sentences=current_chunk_sentences,
                text=text,
                doc_id=doc_id,
                domain=domain,
                source_file_path=source_file_path,
                file_hash=file_hash,
                uploader_id=uploader_id,
                page_num=page_num,
                start_pos=current_chunk_start
            )
            chunks.append(chunk_metadata)

        logger.info(
            f"Created {len(chunks)} semantic chunks from {len(sentences)} sentences "
            f"for doc_id={doc_id}"
        )

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex.

        This is a simple sentence splitter that works for most English text.
        For production use, consider more sophisticated libraries:
        - spaCy: pip install spacy (best quality, slower)
        - NLTK: pip install nltk (good quality, medium speed)
        - pySBD: pip install pysbd (fast, good quality)

        Parameters:
        -----------
        text : str
            Text to split into sentences

        Returns:
        --------
        List[str]:
            List of sentences (whitespace stripped)

        Example:
        --------
        text = "Hello world. How are you? I'm fine!"
        sentences = _split_sentences(text)
        # Returns: ["Hello world.", "How are you?", "I'm fine!"]

        Notes:
        ------
        - Handles periods, exclamation marks, question marks
        - Preserves punctuation in sentences
        - Filters out empty sentences
        - Does NOT handle edge cases like:
          * Abbreviations (Dr. Smith)
          * Decimals (3.14)
          * Ellipsis (...)
        """
        # Split on sentence-ending punctuation followed by space/newline
        # Pattern: . or ! or ? followed by whitespace or end of string
        sentence_pattern = r'(?<=[.!?])\s+'

        # Split and clean
        sentences = re.split(sentence_pattern, text)

        # Filter out empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _create_chunk_metadata(
            self,
            sentences: List[str],
            text: str,
            doc_id: str,
            domain: str,
            source_file_path: str,
            file_hash: str,
            uploader_id: str,
            page_num: int,
            start_pos: int
    ) -> ChunkMetadata:
        """
        Create ChunkMetadata object for a semantic chunk.

        Parameters:
        -----------
        sentences : List[str]
            List of sentences in this chunk
        text : str
            Original full document text (for position calculation)
        start_pos : int
            Starting position in original text
        ... (other params same as chunk_text)

        Returns:
        --------
        ChunkMetadata:
            Metadata object for this semantic chunk
        """
        # Combine sentences into chunk text
        chunk_text = ' '.join(sentences).strip()

        # Calculate character range in original text
        # Find where this chunk appears in the original text
        chunk_start = text.find(chunk_text, start_pos)
        if chunk_start == -1:
            # Fallback if exact match not found (edge case)
            chunk_start = start_pos
        chunk_end = chunk_start + len(chunk_text)

        # Create metadata
        metadata = ChunkMetadata(
            doc_id=doc_id,
            domain=domain,
            chunk_text=chunk_text,
            char_range=(chunk_start, chunk_end),
            page_num=page_num,
            uploader_id=uploader_id,
            source_file_path=source_file_path,
            source_file_hash=file_hash,
            embedding_model_name=self.embedding_model_name,
            chunking_strategy="semantic",  # Identifier for this strategy
            chunk_type="text"
        )

        return metadata


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of SemanticChunker usage.
    Run this file directly: python core/chunking/semantic_chunker.py
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("SemanticChunker Usage Examples")
    print("=" * 70)

    # Example 1: Basic semantic chunking
    print("\n1. Basic Semantic Chunking")
    print("-" * 70)

    sample_text = """
Python is a high-level programming language. It was created by Guido van Rossum.
Python emphasizes code readability. The language provides constructs for clear programming.
Python has a large standard library. It supports multiple programming paradigms.

The weather today is sunny and warm. Temperature is around 75 degrees.
It's a perfect day for outdoor activities. Many people are enjoying the parks.

Machine learning is a subset of artificial intelligence. It enables systems to learn from data.
Deep learning uses neural networks. These networks can recognize complex patterns.
"""

    config = SemanticChunkingConfig(
        similarity_threshold=0.7,
        max_chunk_size=1000
    )
    chunker = SemanticChunker(config, embedding_model_name="all-MiniLM-L6-v2")

    chunks = chunker.chunk_text(
        text=sample_text,
        doc_id="tech_article_2025",
        domain="engineering",
        source_file_path="./docs/article.txt",
        file_hash="xyz789abc123",
        uploader_id="tech_writer@company.com"
    )

    print(f"Created {len(chunks)} semantic chunks from text\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"  Size: {len(chunk.chunk_text)} characters")
        print(f"  Range: {chunk.char_range}")
        print(f"  Preview: {chunk.chunk_text[:100]}...")
        print(f"  Topic: {chunk.chunk_text.split('.')[0][:50]}...")  # First sentence
        print()

    print("\n" + "=" * 70)
    print("SemanticChunker examples completed!")
    print("=" * 70)
