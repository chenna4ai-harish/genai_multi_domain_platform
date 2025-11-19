"""
core/interfaces/chunking_interface.py

This module defines the abstract interfaces (contract) for all chunking strategies
in the multi-domain document intelligence platform.

Purpose:
--------
Defines a standard interfaces that ALL chunking implementations must follow.
This enables the Factory Pattern and config-driven architecture, allowing you
to swap chunking strategies without changing any calling code.

Why Use Abstract Base Classes (ABCs)?
--------------------------------------
1. **Contract Enforcement**: Forces all chunking implementations to provide
   the required methods (chunk_text).

2. **Type Safety**: Allows type hints like `ChunkerInterface` to work with
   any concrete implementation (RecursiveChunker, SemanticChunker, etc.).

3. **Documentation**: Serves as living documentation of what a chunker must do.

4. **IDE Support**: Enables autocomplete and type checking in IDEs.

5. **Polymorphism**: Allows code to work with "any chunker" without knowing
   which specific implementation is being used.

Example Usage:
--------------
# Factory creates the right chunker based on config
chunker: ChunkerInterface = ChunkingFactory.create_chunker(config)

# Caller doesn't care if it's recursive or semantic - interfaces is the same!
chunks = chunker.chunk_text(text, doc_id, domain, file_path, file_hash)

Alternative: Why Not Just Use Duck Typing?
-------------------------------------------
Python supports "duck typing" (if it walks like a duck and quacks like a duck,
it's a duck). You could skip ABCs and just have classes with matching methods.

However, ABCs provide:
- Explicit contracts (clear what methods are required)
- Runtime validation (can't instantiate incomplete implementations)
- Better IDE support and type checking
- Self-documenting code

References:
-----------
- Python ABC documentation: https://docs.python.org/3/library/abc.html
- PEP 3119 (Introducing ABCs): https://www.python.org/dev/peps/pep-3119/
"""

from abc import ABC, abstractmethod
from typing import List
from models.metadata_models import ChunkMetadata


class ChunkerInterface(ABC):
    """
    Abstract base class defining the interfaces for document chunking strategies.

    All chunking implementations (RecursiveChunker, SemanticChunker, etc.)
    MUST inherit from this class and implement all abstract methods.

    Design Pattern:
    ---------------
    This follows the Strategy Pattern, where different algorithms (chunking
    strategies) can be swapped at runtime based on configuration.

    Responsibilities:
    -----------------
    A chunker is responsible for:
    1. Taking a full document text as input
    2. Splitting it into meaningful chunks (with configurable strategy)
    3. Creating comprehensive metadata for each chunk
    4. Returning a list of ChunkMetadata objects

    What Makes a Good Chunk?
    ------------------------
    - Contains enough context to be meaningful standalone
    - Not too large (fits in embedding model context window)
    - Not too small (preserves semantic coherence)
    - Has clear boundaries (sentence/paragraph breaks)

    Example Implementations:
    ------------------------
    1. RecursiveChunker: Fixed-size chunks with overlap
       - Chunk size: 500 characters, overlap: 50 characters
       - Fast, predictable, works well for most documents

    2. SemanticChunker: Embedding-based topical grouping
       - Groups sentences by semantic similarity
       - Slower but creates more coherent chunks
       - Best for documents with clear topic boundaries

    Usage Example:
    --------------
    # This interfaces allows polymorphic usage:

    def process_document(text: str, chunker: ChunkerInterface) -> List[ChunkMetadata]:
        '''Process document with ANY chunking strategy.'''
        chunks = chunker.chunk_text(
            text=text,
            doc_id="my_doc",
            domain="hr",
            source_file_path="./docs/handbook.pdf",
            file_hash="abc123"
        )
        return chunks

    # Works with ANY chunker implementation!
    recursive_chunker = RecursiveChunker(chunk_size=500, overlap=50)
    chunks1 = process_document(text, recursive_chunker)

    semantic_chunker = SemanticChunker(similarity_threshold=0.7)
    chunks2 = process_document(text, semantic_chunker)
    """

    @abstractmethod
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
        Split document text into chunks with comprehensive metadata.

        This is the CORE method that all chunking strategies must implement.
        Each implementation will use its own algorithm (recursive, semantic, etc.)
        but must return the same ChunkMetadata structure.

        Parameters:
        -----------
        text : str
            The full document text to be chunked.
            Could be from PDF, DOCX, TXT, etc. (already extracted).
            Example: "Employee benefits include 15 vacation days per year..."

        doc_id : str
            Unique identifier for the source document.
            Used to group chunks from the same document.
            Example: "employee_handbook_2025", "finance_policy_v3"

        domain : str
            Domain/department this document belongs to.
            Used for filtering queries to specific departments.
            Example: "hr", "finance", "engineering", "legal"

        source_file_path : str
            Original file path or URL of the source document.
            Used for audit trails and locating original files.
            Example: "./data/raw_documents/hr/handbook.pdf"

        file_hash : str
            SHA256 hash of the source file content.
            Used for:
            1) Detecting file changes (idempotency)
            2) File integrity verification
            3) Deduplication
            Example: "abc123def456789..."

        uploader_id : str, optional
            User/system identifier who uploaded this document.
            Used for audit trails and access control.
            Example: "admin@company.com", "hr_team"
            Default: None

        page_num : int, optional
            Page number in source document (for PDFs).
            Used for citations in UI.
            Example: If chunk is from page 12, citation shows [doc_id:12]
            Default: None (for non-paginated documents)

        Returns:
        --------
        List[ChunkMetadata]:
            List of chunk metadata objects, one per chunk.
            Each ChunkMetadata contains:
            - The chunk text itself
            - Position information (char_range, page_num)
            - Provenance information (uploader_id, timestamps, file_hash)
            - Processing information (chunking_strategy, embedding_model_name)
            - Quality information (is_authoritative, confidence_score)

        Raises:
        -------
        ValueError:
            If text is empty or None
            If required parameters are missing or invalid

        Implementation Guidelines:
        --------------------------
        1. Validate input parameters (check for empty text, invalid doc_id, etc.)
        2. Apply your chunking algorithm (recursive, semantic, etc.)
        3. Create ChunkMetadata for each chunk with ALL required fields
        4. Ensure char_range is accurate (enables precise source location)
        5. Handle edge cases (very short documents, empty chunks, etc.)
        6. Log warnings for unusual cases (no chunks created, very large chunks)

        Example Implementation Pattern:
        -------------------------------
        def chunk_text(self, text, doc_id, domain, source_file_path,
                       file_hash, uploader_id=None, page_num=None):
            # 1. Validate inputs
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")

            # 2. Apply chunking algorithm (implementation-specific)
            raw_chunks = self._split_text(text)  # Your algorithm here

            # 3. Create metadata for each chunk
            chunks = []
            for i, chunk_text in enumerate(raw_chunks):
                metadata = ChunkMetadata(
                    doc_id=doc_id,
                    domain=domain,
                    chunk_text=chunk_text,
                    char_range=self._calculate_position(text, chunk_text),
                    page_num=page_num,
                    uploader_id=uploader_id,
                    source_file_path=source_file_path,
                    source_file_hash=file_hash,
                    embedding_model_name=self.embedding_model_name,
                    chunking_strategy="your_strategy_name"
                )
                chunks.append(metadata)

            # 4. Return list of metadata objects
            return chunks

        Performance Considerations:
        ---------------------------
        - For large documents (100+ pages), consider progress logging
        - For semantic chunking, batching embeddings improves speed
        - Avoid creating chunks smaller than 50 characters (too granular)
        - Avoid creating chunks larger than 2000 characters (may exceed model limits)

        Testing Your Implementation:
        ----------------------------
        - Test with empty text (should raise ValueError)
        - Test with very short text (single sentence)
        - Test with very long text (100+ pages)
        - Test with special characters and encoding issues
        - Verify char_range is correct (slice text[start:end] == chunk_text)
        - Verify all metadata fields are populated
        """
        pass  # Subclasses MUST implement this method


# =============================================================================
# USAGE NOTES FOR IMPLEMENTERS
# =============================================================================

"""
How to Implement a New Chunking Strategy:
------------------------------------------

1. Create a new file: core/chunking/my_chunker.py

2. Import the interfaces:
   from core.interfaces.chunking_interface import ChunkerInterface
   from models.metadata_models import ChunkMetadata

3. Create your class inheriting from ChunkerInterface:
   class MyChunker(ChunkerInterface):
       def __init__(self, config: MyChunkingConfig):
           # Initialize your chunker with config
           pass

       def chunk_text(self, text, doc_id, domain, source_file_path,
                      file_hash, uploader_id=None, page_num=None):
           # Your implementation here
           chunks = []
           # ... your chunking logic ...
           return chunks

4. Register in factory: core/factories/chunking_factory.py
   elif config.strategy == "my_strategy":
       return MyChunker(config.my_strategy)

5. Add config model: models/domain_config.py
   class MyChunkingConfig(BaseModel):
       # Your config parameters
       pass

6. Update ChunkingConfig:
   class ChunkingConfig(BaseModel):
       strategy: str = "recursive"  # Add "my_strategy" to options
       my_strategy: MyChunkingConfig = Field(default_factory=MyChunkingConfig)

7. Use in YAML:
   chunking:
     strategy: "my_strategy"
     my_strategy:
       param1: value1
       param2: value2

That's it! No changes to calling code required (config-driven architecture).
"""
