"""

core/chunking/recursive_chunker.py

This module implements the Recursive (Fixed-Size) Chunking strategy.

What is Recursive Chunking?
----------------------------
Recursive chunking splits text into fixed-size chunks with a configurable overlap
between consecutive chunks. The "overlap" ensures that context at chunk boundaries
is preserved, preventing information loss.

Example:
--------
Text: "ABCDEFGHIJKLMNOP"
Chunk size: 8 characters
Overlap: 2 characters

Chunks:
1. "ABCDEFGH" (position 0-8)
2. "GHIJKLMN" (position 6-14) - overlaps "GH" with chunk 1
3. "MNOP" (position 12-16) - overlaps "MN" with chunk 2

Why Use Overlapping Chunks?
----------------------------
Without overlap, a sentence split at the boundary loses context:
❌ Chunk 1: "Employee benefits include 15 vacation d"
❌ Chunk 2: "ays per year and unlimited sick leave."

With overlap, both chunks get full sentences:
✅ Chunk 1: "Employee benefits include 15 vacation days per year."
✅ Chunk 2: "15 vacation days per year and unlimited sick leave."

When to Use Recursive Chunking:
--------------------------------
✅ Default strategy for most documents
✅ Structured documents (policies, manuals, reports)
✅ When processing speed matters
✅ When you want predictable chunk sizes
❌ When documents have clear topical boundaries (use semantic chunking instead)
❌ When chunk coherence matters more than size uniformity

Advantages:
-----------
+ Simple and fast (no embedding computation needed)
+ Predictable chunk sizes (good for token budgets)
+ Works well with any content type
+ Low memory footprint

Disadvantages:
--------------
- May split sentences/paragraphs awkwardly
- No awareness of semantic boundaries
- Fixed size may not align with natural text structure

"""



import sys
from pathlib import Path
project_root = Path(__file__).parent.parent  # Go up two levels from config_manager.py
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import List, Dict, Any
from core.interfaces.chunking_interface import ChunkerInterface
from models.metadata_models import ChunkMetadata
from models.domain_config import RecursiveChunkingConfig
import logging
import re

# Configure logging for this module
logger = logging.getLogger(__name__)


class RecursiveChunker(ChunkerInterface):
    """
    Recursive text chunker with configurable size and overlap.

    This implementation splits text into fixed-size chunks (by character count)
    with a specified overlap between consecutive chunks to preserve context.

    Configuration Parameters:
    -------------------------
    chunk_size : int
        Number of characters per chunk (default: 500)
        Range: 100-2000 characters
        Recommendation: 300-800 for most use cases

    overlap : int
        Number of characters that overlap between chunks (default: 50)
        Range: 0-500 characters
        Recommendation: 10-20% of chunk_size (e.g., 50 for chunk_size=500)

    embedding_model_name : str
        Name of the embedding model (for metadata tracking)
        This is stored in chunk metadata for provenance

    Example Usage:
    --------------
    # Initialize with config
    config = RecursiveChunkingConfig(chunk_size=500, overlap=50)
    chunker = RecursiveChunker(config, embedding_model_name="all-MiniLM-L6-v2")

    # Chunk a document
    text = "Your document text here..."
    chunks = chunker.chunk_text(
        text=text,
        doc_id="employee_handbook_2025",
        domain="hr",
        source_file_path="./docs/handbook.pdf",
        file_hash="abc123...",
        uploader_id="admin@company.com",
        page_num=12
    )

    # Result: List of ChunkMetadata objects with overlapping chunks
    print(f"Created {len(chunks)} chunks")
    for chunk in chunks[:3]:  # Show first 3
        print(f"  {chunk.char_range}: {chunk.chunk_text[:50]}...")
    """

    def __init__(self, config: RecursiveChunkingConfig, embedding_model_name: str):
        """
        Initialize the recursive chunker with configuration.

        Parameters:
        -----------
        config : RecursiveChunkingConfig
            Configuration object with chunk_size and overlap settings
        embedding_model_name : str
            Name of the embedding model (stored in chunk metadata)

        Raises:
        -------
        ValueError:
            If chunk_size < 100 or overlap >= chunk_size
        """
        self.chunk_size = config.chunk_size
        self.overlap = config.overlap
        self.embedding_model_name = embedding_model_name

        # Validate configuration
        if self.chunk_size < 100:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be at least 100 characters"
            )

        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be less than chunk_size ({self.chunk_size})"
            )

        logger.info(
            f"Initialized RecursiveChunker: chunk_size={self.chunk_size}, "
            f"overlap={self.overlap}, model={self.embedding_model_name}"
        )

    def get_strategy_name(self) -> str:
        """
        Return the name of this chunking strategy.

        Required by ChunkingInterface.

        Returns:
        --------
        str:
            Strategy name: "recursive"
        """
        return "recursive"

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk text into fixed-size pieces with overlap (Phase 2 interface).

        This method implements the ChunkingInterface.chunk() signature.

        Parameters:
        -----------
        text : str
            Text to chunk
        metadata : Dict[str, Any]
            Metadata dict with required fields:
            - doc_id, domain, source_file_path, source_file_hash

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
            file_hash=metadata.get('source_file_hash', 'a' * 64),  # Default hash
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
        Split text into overlapping fixed-size chunks with metadata.

        Algorithm:
        ----------
        1. Start at position 0
        2. Extract chunk of size chunk_size
        3. Move forward by (chunk_size - overlap) characters
        4. Repeat until end of text
        5. Create ChunkMetadata for each chunk

        Parameters:
        -----------
        See ChunkerInterface.chunk_text() for parameter documentation.

        Returns:
        --------
        List[ChunkMetadata]:
            List of chunk metadata objects, one per chunk.
            Each chunk has accurate char_range for source location.

        Raises:
        -------
        ValueError:
            If text is empty or None
            If required parameters are missing

        Example:
        --------
        text = "Employee benefits include 15 vacation days per year. " \
               "Health insurance covers medical and dental. " \
               "401k matching up to 6% of salary."

        With chunk_size=50, overlap=10:
        Chunk 1 (0-50): "Employee benefits include 15 vacation days per ye"
        Chunk 2 (40-90): "per year. Health insurance covers medical and de"
        Chunk 3 (80-130): "and dental. 401k matching up to 6% of salary."

        Note: Overlap preserves context at boundaries!
        """
        # Step 1: Validate inputs
        if not text or not text.strip():
            logger.warning(f"Empty text for doc_id={doc_id}")
            raise ValueError(
                f"Text cannot be empty for document: {doc_id}\n"
                f"Please provide non-empty text content."
            )

        if not doc_id or not domain or not source_file_path or not file_hash:
            raise ValueError(
                "Required parameters missing: doc_id, domain, source_file_path, file_hash"
            )

        logger.info(
            f"Chunking document: doc_id={doc_id}, domain={domain}, "
            f"text_length={len(text)} chars"
        )

        # Step 2: Initialize chunking variables
        chunks: List[ChunkMetadata] = []
        start = 0  # Starting position for current chunk
        chunk_index = 0  # For logging/debugging

        # Step 3: Sliding window chunking with overlap
        while start < len(text):
            # Calculate end position for this chunk
            end = min(start + self.chunk_size, len(text))

            # Extract chunk text
            chunk_text = text[start:end]

            # Clean up chunk (remove leading/trailing whitespace)
            chunk_text = chunk_text.strip()

            # Skip empty chunks (can happen with whitespace-heavy documents)
            if not chunk_text:
                logger.debug(
                    f"Skipping empty chunk at position {start}-{end} for doc_id={doc_id}"
                )
                # Move to next position
                start += self.chunk_size - self.overlap
                continue

            # Step 4: Create comprehensive metadata for this chunk
            try:
                metadata = ChunkMetadata(
                    # Core fields
                    doc_id=doc_id,
                    domain=domain,
                    chunk_text=chunk_text,
                    char_range=(start, end),  # Exact position in original text
                    page_num=page_num,  # Optional: for paginated documents
                    # Provenance fields
                    uploader_id=uploader_id,
                    source_file_path=source_file_path,
                    source_file_hash=file_hash,
                    # Processing fields
                    embedding_model_name=self.embedding_model_name,
                    chunking_strategy="recursive",  # Identifier for this strategy
                    chunk_type="text"  # Could be "text", "code", "list", etc.
                )

                chunks.append(metadata)
                chunk_index += 1

                # Log every 100 chunks (avoid log spam)
                if chunk_index % 100 == 0:
                    logger.debug(
                        f"Created {chunk_index} chunks for doc_id={doc_id} "
                        f"(position: {start}/{len(text)})"
                    )

            except Exception as e:
                logger.error(
                    f"Failed to create metadata for chunk at position {start}-{end}: {e}"
                )
                # Continue processing other chunks instead of failing entirely
                pass

            # Step 5: Move sliding window forward
            # Move by (chunk_size - overlap) to create overlap between chunks
            start += self.chunk_size - self.overlap

        # Step 6: Validate and return results
        if not chunks:
            logger.warning(
                f"No chunks created for doc_id={doc_id}. "
                f"Text length: {len(text)}, chunk_size: {self.chunk_size}"
            )
            # Return empty list rather than raising error (let caller decide)

        logger.info(
            f"Successfully created {len(chunks)} chunks for doc_id={doc_id} "
            f"using recursive strategy (chunk_size={self.chunk_size}, overlap={self.overlap})"
        )

        return chunks

    def _smart_split_at_sentence_boundary(self, text: str, max_pos: int) -> int:
        """
        Helper method to find optimal split point at sentence boundary.

        This is an OPTIONAL enhancement to avoid splitting mid-sentence.
        Not used in basic implementation but can be added for better quality.

        Parameters:
        -----------
        text : str
            Text to find boundary in
        max_pos : int
            Maximum position to search up to

        Returns:
        --------
        int:
            Position of sentence boundary, or max_pos if no boundary found

        Example:
        --------
        text = "Hello world. This is a test. More text here."
        max_pos = 20
        Returns: 13 (position after "Hello world.")

        Usage:
        ------
        # In chunk_text(), instead of:
        end = min(start + self.chunk_size, len(text))

        # Use:
        ideal_end = min(start + self.chunk_size, len(text))
        end = self._smart_split_at_sentence_boundary(text, ideal_end)
        """
        # Look for sentence-ending punctuation followed by space or newline
        sentence_enders = r'[.!?]\s+'

        # Search backwards from max_pos to find last sentence boundary
        search_text = text[max(0, max_pos - 100):max_pos]  # Look back up to 100 chars
        matches = list(re.finditer(sentence_enders, search_text))

        if matches:
            # Return position of last match (end of sentence)
            last_match = matches[-1]
            return max(0, max_pos - 100) + last_match.end()

        # No sentence boundary found, return original position
        return max_pos


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    import logging
    from utils.hashing import compute_string_hash  # Use your hashing utility

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

    # Example text
    text = """
    Employees receive 15 vacation days per year. Unused vacation days 
    can be carried over to the next year. The 401k employer matching 
    contribution is 5% of salary up to the IRS limit.
    """

    # ✅ Generate valid hash from text
    file_hash = compute_string_hash(text)

    # Create metadata with valid hash
    metadata = {
        "doc_id": "employee_benefits_2025",
        "domain": "hr",
        "source_file_path": "./data/benefits.txt",
        "source_file_hash": file_hash,  # ✅ Valid SHA-256 hash
        "embedding_model_name": "all-MiniLM-L6-v2",
        "chunking_strategy": "recursive"
    }

    # Create chunker and chunk
    config = RecursiveChunkingConfig(chunk_size=100, overlap=20)
    chunker = RecursiveChunker(config, embedding_model_name="all-MiniLM-L6-v2")

    chunks = chunker.chunk(text, metadata)
    print(f"✅ Created {len(chunks)} chunks with valid hash")
