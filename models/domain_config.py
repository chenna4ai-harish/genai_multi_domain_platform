"""
models/domain_config.py

This module defines the Pydantic models for domain-specific configuration
in the multi-domain document intelligence platform.

Purpose:
--------
This file enables the "config-driven" architecture where you can:
- Define different configurations per domain (HR, Finance, Engineering)
- Switch between providers (ChromaDB ↔ Pinecone, SentenceTransformers ↔ Gemini)
- Change strategies (Recursive ↔ Semantic chunking)
- Add new domains WITHOUT code changes (just add a YAML file)

The configuration hierarchy works as:
Global Config (global_config.yaml)
    ↓ (merged with)
Domain Config (hr_domain.yaml, finance_domain.yaml, etc.)
    ↓ (validated by)
Pydantic Models (this file)
    ↓ (used by)
Factory Classes (to instantiate components)

Example:
--------
# In hr_domain.yaml:
embeddings:
  provider: "sentence_transformers"
  model_name: "all-MiniLM-L6-v2"

# This gets validated by EmbeddingConfig class below
# Then EmbeddingFactory uses it to create the right embedder
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


# =============================================================================
# CHUNKING CONFIGURATION - How to split documents into chunks
# =============================================================================

class RecursiveChunkingConfig(BaseModel):
    """
    Configuration for recursive (fixed-size) chunking strategy.

    This strategy splits text into fixed-size chunks with overlap, ensuring
    no semantic context is lost at chunk boundaries.

    Use Case:
    ---------
    - Default strategy for most documents
    - Works well for structured documents (policies, manuals)
    - Simple, predictable, and fast

    Example:
    --------
    Text: "ABCDEFGHIJ" with chunk_size=5, overlap=2
    Chunks: ["ABCDE", "DEFGH", "GHIJ"]
    Notice "DE" and "GH" overlap to preserve context
    """

    chunk_size: int = Field(
        default=500,
        ge=100,  # Minimum 100 characters
        le=2000,  # Maximum 2000 characters
        description="Number of characters per chunk. "
                    "Smaller = more granular retrieval but more chunks. "
                    "Larger = more context but less precise retrieval. "
                    "Recommended: 300-800 for most use cases."
    )

    overlap: int = Field(
        default=50,
        ge=0,  # No overlap minimum
        le=500,  # Maximum 500 characters overlap
        description="Number of characters that overlap between consecutive chunks. "
                    "Prevents information loss at chunk boundaries. "
                    "Recommended: 10-20% of chunk_size (e.g., 50 for chunk_size=500)."
    )

    @field_validator('overlap')
    @classmethod
    def overlap_must_be_less_than_chunk_size(cls, v, info):
        """
        Ensure overlap is smaller than chunk_size.

        Why: If overlap >= chunk_size, you'd create duplicate or invalid chunks.
        Example: chunk_size=100, overlap=100 would create identical chunks.
        """
        if 'chunk_size' in info.data and v >= info.data['chunk_size']:
            raise ValueError(
                f"overlap ({v}) must be less than chunk_size ({info.data['chunk_size']})"
            )
        return v


class SemanticChunkingConfig(BaseModel):
    """
    Configuration for semantic chunking strategy.

    This strategy groups sentences by semantic similarity (using embeddings),
    creating chunks that are topically coherent rather than fixed-size.

    Use Case:
    ---------
    - Technical documentation with distinct topics
    - Documents with clear section boundaries
    - When preserving semantic coherence is more important than size uniformity

    How It Works:
    -------------
    1. Split document into sentences
    2. Embed each sentence
    3. Compare consecutive sentences for similarity
    4. Group similar sentences (above threshold) into same chunk
    5. Start new chunk when similarity drops below threshold
    """

    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,  # Minimum similarity (completely different)
        le=1.0,  # Maximum similarity (identical)
        description="Cosine similarity threshold (0.0 to 1.0) for grouping sentences. "
                    "Higher = stricter grouping (more, smaller chunks). "
                    "Lower = looser grouping (fewer, larger chunks). "
                    "Recommended: 0.6-0.8 for most documents. "
                    "0.7 = sentences must be 70% similar to be in same chunk."
    )

    max_chunk_size: int = Field(
        default=1000,
        ge=200,  # Minimum 200 characters
        le=3000,  # Maximum 3000 characters
        description="Maximum characters allowed in a semantic chunk. "
                    "Acts as a safety limit to prevent very long chunks. "
                    "Even if sentences are similar, chunk will split at this limit. "
                    "Recommended: 800-1500 for most use cases."
    )


class ChunkingConfig(BaseModel):
    """
    Top-level chunking configuration with strategy selection.

    This model allows you to:
    1. Choose which chunking strategy to use ("recursive" or "semantic")
    2. Configure both strategies (only the chosen one will be used)

    Config-Driven Benefit:
    ----------------------
    Change chunking strategy by editing YAML, no code changes needed!

    Example YAML:
    -------------
    chunking:
      strategy: "recursive"  # Switch to "semantic" to use different strategy
      recursive:
        chunk_size: 500
        overlap: 50
      semantic:
        similarity_threshold: 0.7
        max_chunk_size: 1000
    """

    strategy: str = Field(
        default="recursive",
        description="Chunking strategy to use. Options: 'recursive', 'semantic'. "
                    "Recursive = fixed-size with overlap (default, simple, fast). "
                    "Semantic = embedding-based topical grouping (slower, more coherent)."
    )

    recursive: RecursiveChunkingConfig = Field(
        default_factory=RecursiveChunkingConfig,
        description="Configuration for recursive chunking strategy. "
                    "Used only if strategy='recursive'."
    )

    semantic: SemanticChunkingConfig = Field(
        default_factory=SemanticChunkingConfig,
        description="Configuration for semantic chunking strategy. "
                    "Used only if strategy='semantic'."
    )

    @field_validator('strategy')
    @classmethod
    def validate_strategy(cls, v):
        """Ensure strategy is one of the supported values."""
        allowed = ["recursive", "semantic"]
        if v not in allowed:
            raise ValueError(f"strategy must be one of {allowed}, got '{v}'")
        return v


# =============================================================================
# VECTOR STORE CONFIGURATION - Where to store embeddings
# =============================================================================

class ChromaDBConfig(BaseModel):
    """
    Configuration for ChromaDB vector store (OPTION 1).

    ChromaDB:
    ---------
    - Local, file-based vector database
    - No external dependencies or API keys
    - Great for development, testing, and small-scale deployments
    - Data persisted to disk (survives restarts)

    Use Case:
    ---------
    - MVP/development phase
    - Single-server deployments
    - Budget-constrained projects
    - Quick prototyping
    """

    persist_directory: str = Field(
        default="./data/chroma_db",
        description="Directory path where ChromaDB will persist vector data. "
                    "Data is stored as files in this location. "
                    "Example: './data/chroma_db' or '/var/lib/chroma'. "
                    "Make sure this directory is backed up regularly!"
    )

    collection_name: str = Field(
        ...,  # Required field
        description="Name of the ChromaDB collection for this domain. "
                    "Each domain should have its own collection. "
                    "Example: 'hr_collection', 'finance_collection'. "
                    "Collections are like database tables - they isolate data."
    )


class PineconeConfig(BaseModel):
    """
    Configuration for Pinecone vector store (OPTION 2).

    Pinecone:
    ---------
    - Cloud-based, managed vector database
    - Requires API key (paid service after free tier)
    - Highly scalable, production-ready
    - Serverless architecture (auto-scaling)

    Use Case:
    ---------
    - Production deployments
    - Multi-server/distributed systems
    - High-volume applications (millions of vectors)
    - When you need managed infrastructure
    """

    index_name: str = Field(
        ...,  # Required field
        description="Name of the Pinecone index for this domain. "
                    "Index name must be globally unique in your Pinecone project. "
                    "Example: 'hr-docs-prod', 'finance-policies-v2'. "
                    "Lowercase with hyphens recommended."
    )

    cloud: str = Field(
        default="aws",
        description="Cloud provider for Pinecone serverless deployment. "
                    "Options: 'aws' (Amazon Web Services), 'gcp' (Google Cloud), "
                    "'azure' (Microsoft Azure). "
                    "Choose based on your other infrastructure location."
    )

    region: str = Field(
        default="us-east-1",
        description="Cloud region for Pinecone index. "
                    "Choose closest to your application servers for low latency. "
                    "Examples: 'us-east-1' (AWS), 'us-central1' (GCP), 'eastus' (Azure)."
    )

    dimension: int = Field(
        default=384,
        ge=1,
        le=20000,
        description="Dimensionality of embedding vectors. "
                    "MUST match your embedding model's output dimension! "
                    "Examples: 384 (all-MiniLM-L6-v2), 768 (BERT, Gemini), "
                    "1536 (OpenAI text-embedding-ada-002). "
                    "Cannot be changed after index creation!"
    )


class VectorStoreConfig(BaseModel):
    """
    Top-level vector store configuration with provider selection.

    This model enables the config-driven architecture for vector stores:
    - Switch between ChromaDB (local) and Pinecone (cloud) via YAML
    - No code changes needed to migrate between providers

    Config-Driven Benefit:
    ----------------------
    Start with ChromaDB for MVP, switch to Pinecone for production!

    Example YAML:
    -------------
    vector_store:
      provider: "chromadb"  # Change to "pinecone" for cloud deployment
      chromadb:
        persist_directory: "./data/chroma_db"
        collection_name: "hr_collection"
      pinecone:  # Not used if provider="chromadb", but must be defined
        index_name: "hr-docs-prod"
        cloud: "aws"
        region: "us-east-1"
        dimension: 384
    """

    provider: str = Field(
        default="chromadb",
        description="Vector store provider to use. Options: 'chromadb', 'pinecone'. "
                    "ChromaDB = local, free, simple (default for MVP). "
                    "Pinecone = cloud, scalable, production-ready."
    )

    chromadb: Optional[ChromaDBConfig] = Field(
        default=None,
        description="ChromaDB configuration. Required if provider='chromadb'."
    )

    pinecone: Optional[PineconeConfig] = Field(
        default=None,
        description="Pinecone configuration. Required if provider='pinecone'."
    )

    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v):
        """Ensure provider is one of the supported values."""
        allowed = ["chromadb", "pinecone"]
        if v not in allowed:
            raise ValueError(f"provider must be one of {allowed}, got '{v}'")
        return v


# =============================================================================
# EMBEDDING CONFIGURATION - How to convert text to vectors
# =============================================================================

class EmbeddingConfig(BaseModel):
    """
    Configuration for embedding providers.

    Embeddings:
    -----------
    Convert text into dense vector representations (numbers) that capture
    semantic meaning. Similar text → similar vectors.

    Provider Options:
    -----------------
    1. sentence_transformers (OPTION 1):
       - Local models (run on your server/laptop)
       - Free, no API keys required
       - Fast inference (GPU optional)
       - Best for: MVP, budget projects, privacy-sensitive data

    2. gemini (OPTION 2):
       - Google's cloud-based embeddings API
       - Requires API key (paid after free tier)
       - High quality, constantly improving
       - Best for: Production, when quality > cost

    Config-Driven Benefit:
    ----------------------
    Try different embedding models by just changing YAML config!

    Example YAML:
    -------------
    embeddings:
      provider: "sentence_transformers"  # or "gemini"
      model_name: "all-MiniLM-L6-v2"
      device: "cpu"  # or "cuda" for GPU
      batch_size: 32
      normalize_embeddings: true
    """

    provider: str = Field(
        default="sentence_transformers",
        description="Embedding provider to use. "
                    "Options: 'sentence_transformers' (local, free), 'gemini' (cloud, premium). "
                    "Sentence-Transformers = runs on your hardware (CPU/GPU). "
                    "Gemini = Google API (requires GEMINI_API_KEY environment variable)."
    )

    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Name of the embedding model. "
                    "For sentence_transformers: 'all-MiniLM-L6-v2' (384-dim, fast), "
                    "'all-mpnet-base-v2' (768-dim, better quality), "
                    "'paraphrase-multilingual-MiniLM-L12-v2' (multilingual). "
                    "For gemini: 'models/embedding-001' (768-dim). "
                    "See: https://www.sbert.net/docs/pretrained_models.html"
    )

    device: str = Field(
        default="cpu",
        description="Compute device for sentence_transformers models. "
                    "Options: 'cpu' (slower, works everywhere), 'cuda' (GPU, faster), "
                    "'mps' (Apple Silicon). Ignored for cloud providers like Gemini. "
                    "Use 'cuda' if you have NVIDIA GPU for 5-10x speedup!"
    )

    batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Number of texts to embed in a single batch. "
                    "Larger = faster throughput but more memory usage. "
                    "Recommended: 16-32 for CPU, 64-128 for GPU. "
                    "For large documents (1000+ chunks), batching is crucial!"
    )

    normalize_embeddings: bool = Field(
        default=True,
        description="Whether to normalize embeddings to unit length (L2 norm). "
                    "Recommended: True (default). Normalization enables: "
                    "1) Cosine similarity = dot product (faster computation) "
                    "2) Consistent similarity scores across models "
                    "3) Better clustering and retrieval performance"
    )

    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v):
        """Ensure provider is one of the supported values."""
        allowed = ["sentence_transformers", "gemini"]
        if v not in allowed:
            raise ValueError(f"provider must be one of {allowed}, got '{v}'")
        return v

    @field_validator('device')
    @classmethod
    def validate_device(cls, v):
        """Ensure device is one of the supported values."""
        allowed = ["cpu", "cuda", "mps"]
        if v not in allowed:
            raise ValueError(f"device must be one of {allowed}, got '{v}'")
        return v


# =============================================================================
# RETRIEVAL CONFIGURATION - How to search for relevant chunks
# =============================================================================

class RetrievalConfig(BaseModel):
    """
    Configuration for retrieval strategies.

    Retrieval Strategies:
    ---------------------
    1. hybrid (OPTION 1 - RECOMMENDED):
       - Combines dense (semantic/vector) and sparse (keyword/BM25) search
       - Alpha parameter controls the balance (0.7 = 70% semantic, 30% keyword)
       - Best of both worlds: semantic understanding + exact keyword matching
       - Use case: Most production applications

    2. dense_only (OPTION 2):
       - Pure semantic/vector search (cosine similarity)
       - Finds conceptually similar content even without exact keywords
       - Use case: When users ask questions in natural language

    3. sparse_only (OPTION 3):
       - Pure keyword/BM25 search (like traditional search engines)
       - Requires exact or close keyword matches
       - Use case: Technical docs with specific terms (APIs, code, commands)

    Config-Driven Benefit:
    ----------------------
    A/B test different retrieval strategies to find what works best!

    Example YAML:
    -------------
    retrieval:
      strategy: "hybrid"  # or "dense_only" or "sparse_only"
      alpha: 0.7  # Only used for hybrid (70% semantic, 30% keyword)
      top_k: 10  # Return top 10 most relevant chunks
      enable_metadata_filtering: true
      normalize_scores: true
    """

    strategy: str = Field(
        default="hybrid",
        description="Retrieval strategy to use. "
                    "Options: 'hybrid' (semantic + keyword), 'dense_only' (semantic), "
                    "'sparse_only' (keyword). "
                    "Hybrid recommended for most use cases (best accuracy)."
    )

    alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for hybrid retrieval (0.0 to 1.0). "
                    "Only used when strategy='hybrid'. "
                    "alpha=1.0 → pure semantic (same as dense_only) "
                    "alpha=0.0 → pure keyword (same as sparse_only) "
                    "alpha=0.7 → 70% semantic + 30% keyword (recommended) "
                    "Tune based on your use case: more semantic for natural language, "
                    "more keyword for technical/exact matching."
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of top results to retrieve. "
                    "More results = better recall but slower and more LLM tokens. "
                    "Recommended: 5-15 for most applications. "
                    "If results are always poor, try increasing to 20-30."
    )

    enable_metadata_filtering: bool = Field(
        default=True,
        description="Whether to allow filtering results by metadata. "
                    "Example: 'Only show authoritative HR docs from 2025'. "
                    "Recommended: True (default). Enables domain isolation and filtering."
    )

    normalize_scores: bool = Field(
        default=True,
        description="Whether to normalize retrieval scores to 0-1 range. "
                    "Recommended: True (default). Makes scores comparable across "
                    "different retrieval strategies and models."
    )

    @field_validator('strategy')
    @classmethod
    def validate_strategy(cls, v):
        """Ensure strategy is one of the supported values."""
        allowed = ["hybrid", "dense_only", "sparse_only"]
        if v not in allowed:
            raise ValueError(f"strategy must be one of {allowed}, got '{v}'")
        return v


# =============================================================================
# SECURITY CONFIGURATION - File upload restrictions
# =============================================================================

class SecurityConfig(BaseModel):
    """
    Configuration for security and file upload restrictions.

    Purpose:
    --------
    Prevent malicious or problematic file uploads by:
    1. Restricting allowed file types (only PDF, DOCX, TXT, etc.)
    2. Limiting file sizes (prevent DoS attacks)
    3. Future: Role-based access control, encryption settings

    Example YAML:
    -------------
    security:
      allowed_file_types: ["pdf", "docx", "txt"]
      max_file_size_mb: 50
    """

    allowed_file_types: List[str] = Field(
        default=["pdf", "docx", "txt", "csv"],
        description="List of allowed file extensions (without dot). "
                    "Only files with these extensions can be uploaded. "
                    "Recommended: ['pdf', 'docx', 'txt', 'csv']. "
                    "Add 'pptx' for presentations, 'xlsx' for spreadsheets. "
                    "Never allow: 'exe', 'sh', 'bat' (security risk!)."
    )

    max_file_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum file size allowed in megabytes (MB). "
                    "Prevents users from uploading very large files that could: "
                    "1) Slow down processing 2) Fill up disk space "
                    "3) Cause out-of-memory errors. "
                    "Recommended: 10-50 MB for most documents. "
                    "Increase to 100-200 MB if handling large PDFs/presentations."
    )


# =============================================================================
# DOMAIN CONFIGURATION - Top-level config for a single domain
# =============================================================================

class DomainConfig(BaseModel):
    """
    Complete configuration for a single domain (e.g., HR, Finance, Engineering).

    This is the TOP-LEVEL model that combines all configuration sections.
    Each domain gets its own YAML file (e.g., hr_domain.yaml) that follows
    this schema.

    Config-Driven Architecture:
    ---------------------------
    To add a new domain:
    1. Create new YAML file: configs/domains/legal_domain.yaml
    2. Fill in configuration following this schema
    3. NO CODE CHANGES NEEDED!
    4. System automatically loads and validates with this Pydantic model

    Example YAML Structure:
    -----------------------
    name: "hr"
    display_name: "Human Resources"
    description: "HR policies, benefits, leave guidelines"

    vector_store:
      provider: "chromadb"
      chromadb:
        persist_directory: "./data/chroma_db"
        collection_name: "hr_collection"

    embeddings:
      provider: "sentence_transformers"
      model_name: "all-MiniLM-L6-v2"

    chunking:
      strategy: "recursive"
      recursive:
        chunk_size: 500
        overlap: 50

    retrieval:
      strategy: "hybrid"
      alpha: 0.7
      top_k: 10

    security:
      allowed_file_types: ["pdf", "docx", "txt"]
      max_file_size_mb: 50
    """

    name: str = Field(
        ...,  # Required field
        description="Internal identifier for this domain (lowercase, no spaces). "
                    "Examples: 'hr', 'finance', 'engineering', 'legal'. "
                    "Used in: collection names, file paths, metadata filtering."
    )

    display_name: str = Field(
        ...,  # Required field
        description="Human-readable name for this domain (for UI display). "
                    "Examples: 'Human Resources', 'Finance & Accounting', "
                    "'Engineering Documentation'. Shown to end users."
    )

    description: str = Field(
        ...,  # Required field
        description="Brief description of what this domain contains. "
                    "Examples: 'HR policies, benefits, leave guidelines', "
                    "'Financial policies, expense rules, accounting procedures'. "
                    "Helps users understand what they can find in this domain."
    )

    vector_store: VectorStoreConfig = Field(
        ...,  # Required field
        description="Vector store configuration for this domain. "
                    "Defines WHERE embeddings are stored (ChromaDB or Pinecone)."
    )

    embeddings: EmbeddingConfig = Field(
        ...,  # Required field
        description="Embedding configuration for this domain. "
                    "Defines HOW text is converted to vectors (model, provider, etc.)."
    )

    chunking: ChunkingConfig = Field(
        ...,  # Required field
        description="Chunking configuration for this domain. "
                    "Defines HOW documents are split into chunks (strategy, size, etc.)."
    )

    retrieval: RetrievalConfig = Field(
        ...,  # Required field
        description="Retrieval configuration for this domain. "
                    "Defines HOW relevant chunks are found (strategy, top_k, etc.)."
    )

    security: SecurityConfig = Field(
        ...,  # Required field
        description="Security configuration for this domain. "
                    "Defines WHAT files can be uploaded (file types, size limits)."
    )


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of DomainConfig usage and validation.
    Run this file directly to see examples: python models/domain_config.py
    """

    # Example 1: Valid HR domain configuration
    hr_config = DomainConfig(
        name="hr",
        display_name="Human Resources",
        description="HR policies, benefits, leave guidelines",
        vector_store=VectorStoreConfig(
            provider="chromadb",
            chromadb=ChromaDBConfig(
                persist_directory="./data/chroma_db",
                collection_name="hr_collection"
            )
        ),
        embeddings=EmbeddingConfig(
            provider="sentence_transformers",
            model_name="all-MiniLM-L6-v2",
            device="cpu",
            batch_size=32,
            normalize_embeddings=True
        ),
        chunking=ChunkingConfig(
            strategy="recursive",
            recursive=RecursiveChunkingConfig(
                chunk_size=500,
                overlap=50
            )
        ),
        retrieval=RetrievalConfig(
            strategy="hybrid",
            alpha=0.7,
            top_k=10
        ),
        security=SecurityConfig(
            allowed_file_types=["pdf", "docx", "txt"],
            max_file_size_mb=50
        )
    )

    print("Example 1: HR Domain Configuration")
    print("=" * 70)
    print(hr_config.model_dump_json(indent=2))
    print("\n")

    # Example 2: Finance domain with Pinecone and Gemini
    finance_config = DomainConfig(
        name="finance",
        display_name="Finance & Accounting",
        description="Financial policies, expense rules, accounting procedures",
        vector_store=VectorStoreConfig(
            provider="pinecone",
            pinecone=PineconeConfig(
                index_name="finance-docs-prod",
                cloud="aws",
                region="us-east-1",
                dimension=768
            )
        ),
        embeddings=EmbeddingConfig(
            provider="gemini",
            model_name="models/embedding-001",
            batch_size=32
        ),
        chunking=ChunkingConfig(
            strategy="semantic",
            semantic=SemanticChunkingConfig(
                similarity_threshold=0.75,
                max_chunk_size=1000
            )
        ),
        retrieval=RetrievalConfig(
            strategy="dense_only",
            alpha=1.0,
            top_k=15
        ),
        security=SecurityConfig(
            allowed_file_types=["pdf", "xlsx", "csv"],
            max_file_size_mb=100
        )
    )

    print("Example 2: Finance Domain Configuration (Pinecone + Gemini)")
    print("=" * 70)
    print(finance_config.model_dump_json(indent=2))
    print("\n")

    # Example 3: Validation error demonstration
    print("Example 3: Pydantic Validation Errors")
    print("=" * 70)

    try:
        # Invalid: overlap >= chunk_size
        bad_chunking = RecursiveChunkingConfig(
            chunk_size=100,
            overlap=100  # Should be < chunk_size
        )
    except Exception as e:
        print(f"❌ Error 1: {e}\n")

    try:
        # Invalid: unknown strategy
        bad_config = ChunkingConfig(strategy="invalid_strategy")
    except Exception as e:
        print(f"❌ Error 2: {e}\n")

    try:
        # Invalid: alpha out of range
        bad_retrieval = RetrievalConfig(alpha=1.5)  # Must be 0.0-1.0
    except Exception as e:
        print(f"❌ Error 3: {e}\n")

    print("=" * 70)
    print("✅ Config models loaded successfully!")
    print("Pydantic validation prevents invalid configurations!")
