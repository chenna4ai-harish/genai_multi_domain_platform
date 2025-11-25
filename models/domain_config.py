"""

models/domain_config.py

This module defines the Pydantic models for domain-specific configuration
in the Multi-Domain Document Intelligence Platform (Phase 2).

What is This File?
-------------------
This file contains the **data models** (schemas) that define what a valid
configuration looks like. Think of it as a contract: any configuration
YAML file must conform to these models.

Purpose:
--------
Enables "configuration-driven" architecture where you can:
- Define different configurations per domain (HR, Finance, Engineering)
- Switch providers (ChromaDB ↔ Pinecone, Sentence-Transformers ↔ Gemini)
- Change strategies (Recursive ↔ Semantic chunking, Vector ↔ Hybrid retrieval)
- Add new domains WITHOUT code changes (just add a YAML file!)

Why Pydantic?
-------------
Pydantic provides:
1. **Type Safety**: Automatic validation of data types
2. **Data Validation**: Range checks, enum validation, custom validators
3. **IDE Support**: Autocomplete and type hints in your IDE
4. **Error Messages**: Clear, actionable validation errors
5. **Serialization**: Easy conversion to/from JSON, YAML, dict

Configuration Flow:
-------------------
1. Write YAML config (configs/domains/hr.yaml)
2. ConfigManager loads YAML
3. Pydantic validates against these models
4. If valid → DomainConfig object created
5. If invalid → ValidationError with helpful message
6. Factories use DomainConfig to instantiate components

Phase 2 Requirements:
---------------------
- Multi-strategy retrieval (hybrid, vector_similarity, bm25)
- Multi-provider support (embeddings, vector stores)
- Comprehensive metadata tracking
- Security and validation settings

References:
-----------
- Phase 2 Spec: Section 12 (Configuration Management)
- Pydantic Docs: https://docs.pydantic.dev/

"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================

class RecursiveChunkingConfig(BaseModel):
    """
    Configuration for recursive (fixed-size) chunking.

    Splits text into fixed-size chunks with overlap to preserve context
    across chunk boundaries.

    Use Cases:
    - General documents (policies, manuals)
    - When predictable chunk sizes are important
    - Fast processing required

    Example:
    --------
    Text: "ABCDEFGHIJ"
    chunk_size=5, overlap=2
    Chunks: ["ABCDE", "DEFGH", "GHIJ"]
    Notice: "DE" and "GH" overlap
    """
    strategy: str = Field(default="recursive", description="Must be 'recursive'")
    chunk_size: int = Field(
        default=500,
        ge=100,
        le=5000,
        description="Characters per chunk (100-5000)"
    )
    overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between chunks (0-500)"
    )

    @field_validator('overlap')
    @classmethod
    def overlap_less_than_chunk_size(cls, v, info):
        if 'chunk_size' in info.data and v >= info.data['chunk_size']:
            raise ValueError(f"overlap ({v}) must be < chunk_size ({info.data['chunk_size']})")
        return v


class SemanticChunkingConfig(BaseModel):
    """
    Configuration for semantic (similarity-based) chunking.

    Groups sentences by semantic similarity to create topically coherent chunks.

    Use Cases:
    - Technical documentation with clear topics
    - When semantic coherence > size uniformity
    - Documents with distinct sections
    """
    strategy: str = Field(default="semantic", description="Must be 'semantic'")
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for grouping (0.0-1.0)"
    )
    max_chunk_size: int = Field(
        default=1000,
        ge=200,
        le=5000,
        description="Maximum chunk size in characters"
    )


class ChunkingConfig(BaseModel):
    """
    Top-level chunking configuration with strategy selection.

    Phase 2: Supports multiple strategies selectable via config.
    """
    strategy: str = Field(..., description="Chunking strategy: recursive or semantic")
    recursive: Optional[RecursiveChunkingConfig] = Field(default_factory=RecursiveChunkingConfig)
    semantic: Optional[SemanticChunkingConfig] = Field(default_factory=SemanticChunkingConfig)

    @field_validator('strategy')
    @classmethod
    def validate_strategy(cls, v):
        allowed = ["recursive", "semantic"]
        if v not in allowed:
            raise ValueError(f"Invalid chunking strategy: {v}. Allowed: {allowed}")
        return v


# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================

class EmbeddingConfig(BaseModel):
    """
    Configuration for embedding providers.

    Phase 2: Supports multiple providers (sentence_transformers, gemini, openai).

    Provider Options:
    -----------------
    - sentence_transformers: Local, free, GPU-optional
    - gemini: Google Cloud API, high quality
    - openai: OpenAI API, highest quality (future)
    """
    provider: str = Field(..., description="Provider: sentence_transformers, gemini, openai")
    model_name: str = Field(..., description="Model identifier")
    device: Optional[str] = Field(default="cpu", description="Device: cpu, cuda, mps")
    batch_size: Optional[int] = Field(default=32, ge=1, le=1000, description="Batch size")
    normalize: Optional[bool] = Field(default=True, description="Normalize embeddings (L2 norm)")
    api_key: Optional[str] = Field(default=None, description="API key (from environment)")
    task_type: Optional[str] = Field(default="RETRIEVAL_DOCUMENT", description="Gemini task type")

    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v):
        allowed = ["sentence_transformers", "gemini", "openai"]
        if v not in allowed:
            raise ValueError(f"Invalid embedding provider: {v}. Allowed: {allowed}")
        return v

    @field_validator('device')
    @classmethod
    def validate_device(cls, v):
        allowed = ["cpu", "cuda", "mps"]
        if v not in allowed:
            raise ValueError(f"Invalid device: {v}. Allowed: {allowed}")
        return v


# =============================================================================
# VECTOR STORE CONFIGURATION
# =============================================================================

class VectorStoreConfig(BaseModel):
    """
    Configuration for vector store providers.

    Phase 2: Supports multiple providers (chromadb, pinecone, qdrant, faiss).

    Provider Options:
    -----------------
    - chromadb: Local, free, simple (MVP)
    - pinecone: Cloud, scalable, production
    - qdrant: Self-hosted or cloud, privacy-focused
    - faiss: In-memory, research/prototyping
    """
    provider: str = Field(..., description="Provider: chromadb, pinecone, qdrant, faiss")
    collection_name: str = Field(..., description="Collection/index name")
    index_type: Optional[str] = Field(default="hnsw", description="Index algorithm")

    # ChromaDB fields
    persist_directory: Optional[str] = Field(default="./data/chroma_db", description="ChromaDB directory")

    # Pinecone/Qdrant fields
    cloud: Optional[str] = Field(default="aws", description="Cloud provider: aws, gcp, azure")
    region: Optional[str] = Field(default="us-east-1", description="Cloud region")
    api_key: Optional[str] = Field(default=None, description="API key (from environment)")

    # General fields
    dimension: Optional[int] = Field(default=None, description="Embedding dimension (auto-detected)")

    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v):
        allowed = ["chromadb", "pinecone", "qdrant", "faiss"]
        if v not in allowed:
            raise ValueError(f"Invalid vector store provider: {v}. Allowed: {allowed}")
        return v


# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================

class HybridRetrievalConfig(BaseModel):
    """
    Configuration for hybrid (dense + sparse) retrieval.

    Phase 2 Primary Strategy: Combines semantic and keyword search.
    """
    alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Dense weight (1-alpha = sparse weight). 0.7 = 70% semantic, 30% keyword"
    )
    normalize_scores: bool = Field(
        default=True,
        description="Normalize scores before combining"
    )


class RetrievalConfig(BaseModel):
    """
    Configuration for retrieval strategies.

    Phase 2: Supports multiple strategies simultaneously!

    Strategy Options:
    -----------------
    - hybrid: Dense + Sparse with alpha weighting (RECOMMENDED)
    - vector_similarity: Pure semantic/dense search
    - bm25: Pure keyword/sparse search

    Example YAML:
    -------------
    retrieval:
      strategies: ["hybrid", "vector_similarity"]  # Use both!
      top_k: 10
      similarity: "cosine"
      hybrid:
        alpha: 0.7
        normalize_scores: true
    """
    strategies: List[str] = Field(
        default=["hybrid"],
        description="List of enabled strategies: hybrid, vector_similarity, bm25"
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    similarity: str = Field(default="cosine", description="Similarity metric")
    hybrid: Optional[HybridRetrievalConfig] = Field(default_factory=HybridRetrievalConfig)

    @field_validator('strategies')
    @classmethod
    def validate_strategies(cls, v):
        allowed = ["hybrid", "vector_similarity", "bm25"]
        for strategy in v:
            if strategy not in allowed:
                raise ValueError(f"Invalid retrieval strategy: {strategy}. Allowed: {allowed}")
        return v


# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================

class SecurityConfig(BaseModel):
    """
    Configuration for security and file validation.

    Phase 2: Validates file uploads to prevent security issues.
    """
    allowed_file_types: List[str] = Field(
        default=["pdf", "docx", "txt"],
        description="Allowed file extensions (without dot)"
    )
    max_file_size_mb: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum file size in megabytes"
    )
    require_authentication: bool = Field(
        default=False,
        description="Require user authentication"
    )


# =============================================================================
# METADATA CONFIGURATION
# =============================================================================

class MetadataConfig(BaseModel):
    """
    Configuration for metadata tracking and management.

    Phase 2: Comprehensive metadata for provenance and filtering.
    """
    track_versions: bool = Field(default=True, description="Track document versions")
    enable_deprecation: bool = Field(default=True, description="Enable deprecation workflow")
    compute_file_hash: bool = Field(default=True, description="Compute file hashes")
    extract_page_numbers: bool = Field(default=True, description="Extract page numbers from PDFs")
    required_fields: List[str] = Field(
        default=["doc_id", "title", "domain", "doc_type", "uploader_id"],
        description="Required metadata fields for uploads"
    )


# =============================================================================
# DOMAIN CONFIGURATION (ROOT MODEL)
# =============================================================================

class DomainConfig(BaseModel):
    """
    Complete Phase 2 domain configuration.

    This is the ROOT configuration model that combines all sections.

    **CRITICAL**: Field names MUST match what code expects:
    - embeddings (not embedding)
    - vector_store (not vectorstore)
    - chunking
    - retrieval

    Example YAML Structure:
    -----------------------
    domain_id: "hr"
    name: "Human Resources"
    description: "HR policies and procedures"

    chunking:
      strategy: "recursive"
      recursive:
        chunk_size: 500
        overlap: 50

    embeddings:
      provider: "sentence_transformers"
      model_name: "all-MiniLM-L6-v2"
      normalize: true

    vector_store:
      provider: "chromadb"
      collection_name: "hr_collection"
      persist_directory: "./data/chroma_db"

    retrieval:
      strategies: ["hybrid"]
      top_k: 10
      hybrid:
        alpha: 0.7

    security:
      allowed_file_types: ["pdf", "docx", "txt"]
      max_file_size_mb: 20
    """
    # Identity
    domain_id: str = Field(..., description="Unique domain identifier")
    name: str = Field(..., description="Human-readable domain name")
    description: Optional[str] = Field(default=None, description="Domain description")

    # Component configurations (MUST match code field names!)
    chunking: ChunkingConfig = Field(..., description="Chunking configuration")
    embeddings: EmbeddingConfig = Field(..., description="Embedding configuration")
    vector_store: VectorStoreConfig = Field(..., description="Vector store configuration")
    retrieval: RetrievalConfig = Field(..., description="Retrieval configuration")

    # Optional configurations
    security: Optional[SecurityConfig] = Field(default_factory=SecurityConfig)
    metadata: Optional[MetadataConfig] = Field(default_factory=MetadataConfig)

    class Config:
        extra = "allow"  # Allow extra fields for future extensibility


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of DomainConfig validation.
    Run: python models/domain_config.py
    """
    print("=" * 70)
    print("Domain Configuration Examples (Phase 2)")
    print("=" * 70)

    # Example 1: Valid HR domain configuration
    print("\n1. HR Domain (ChromaDB + Sentence-Transformers)")
    print("-" * 70)

    hr_config = DomainConfig(
        domain_id="hr",
        name="Human Resources",
        description="HR policies, benefits, leave guidelines",
        chunking=ChunkingConfig(
            strategy="recursive",
            recursive=RecursiveChunkingConfig(chunk_size=500, overlap=50)
        ),
        embeddings=EmbeddingConfig(
            provider="sentence_transformers",
            model_name="all-MiniLM-L6-v2",
            device="cpu",
            normalize=True
        ),
        vector_store=VectorStoreConfig(
            provider="chromadb",
            collection_name="hr_collection",
            persist_directory="./data/chroma_db"
        ),
        retrieval=RetrievalConfig(
            strategies=["hybrid"],
            top_k=10,
            hybrid=HybridRetrievalConfig(alpha=0.7)
        ),
        security=SecurityConfig(
            allowed_file_types=["pdf", "docx", "txt"],
            max_file_size_mb=20
        )
    )

    print(f"✅ Valid config for domain: {hr_config.domain_id}")
    print(f"   Chunking: {hr_config.chunking.strategy}")
    print(f"   Embeddings: {hr_config.embeddings.provider}")
    print(f"   Vector Store: {hr_config.vector_store.provider}")
    print(f"   Retrieval: {', '.join(hr_config.retrieval.strategies)}")

    # Example 2: Finance domain (Pinecone + Gemini)
    print("\n2. Finance Domain (Pinecone + Gemini)")
    print("-" * 70)

    finance_config = DomainConfig(
        domain_id="finance",
        name="Finance & Accounting",
        description="Financial policies and procedures",
        chunking=ChunkingConfig(
            strategy="semantic",
            semantic=SemanticChunkingConfig(similarity_threshold=0.75)
        ),
        embeddings=EmbeddingConfig(
            provider="gemini",
            model_name="models/embedding-001"
        ),
        vector_store=VectorStoreConfig(
            provider="pinecone",
            collection_name="finance-docs-prod",
            cloud="aws",
            region="us-east-1"
        ),
        retrieval=RetrievalConfig(
            strategies=["vector_similarity", "hybrid"],  # Multiple strategies!
            top_k=15
        )
    )

    print(f"✅ Valid config for domain: {finance_config.domain_id}")
    print(f"   Strategies: {', '.join(finance_config.retrieval.strategies)}")

    # Example 3: Validation errors
    print("\n3. Pydantic Validation Errors")
    print("-" * 70)

    try:
        bad_config = ChunkingConfig(strategy="invalid_strategy")
    except Exception as e:
        print(f"❌ Error 1: {e}\n")

    try:
        bad_config = EmbeddingConfig(provider="invalid_provider", model_name="test")
    except Exception as e:
        print(f"❌ Error 2: {e}\n")

    try:
        bad_config = RetrievalConfig(strategies=["invalid_strategy"])
    except Exception as e:
        print(f"❌ Error 3: {e}\n")

    print("=" * 70)
    print("✅ All Phase 2 config models loaded successfully!")
    print("=" * 70)
