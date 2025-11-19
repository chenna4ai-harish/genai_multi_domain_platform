"""
models/metadata_models.py

This module defines the comprehensive metadata schema for document chunks in the
multi-domain document intelligence platform.

Purpose:
--------
Every chunk of text stored in the vector database needs rich metadata for:
- Provenance tracking (who, when, where)
- Version control and conflict resolution
- Debugging and quality assurance
- Citations and source verification
- Domain isolation and access control

The metadata enables features like:
- "Show me only authoritative HR policies from 2025"
- "Which embedding model was used for this chunk?"
- "Has this document been superseded by a newer version?"
- Proper citations: [employee_handbook_2025.pdf:page 12]
"""

from pydantic import BaseModel, Field
from typing import Optional, Tuple
from datetime import datetime
import uuid


class ChunkMetadata(BaseModel):
    """
    Enhanced metadata schema for document chunks.

    This class defines ALL information tracked for each chunk of text
    stored in the vector database. Using Pydantic ensures:
    - Type safety (wrong types raise validation errors)
    - Required field enforcement
    - Easy serialization to/from JSON
    - Automatic documentation

    Example Usage:
    --------------
    metadata = ChunkMetadata(
        doc_id="employee_handbook_2025",
        domain="hr",
        chunk_text="Employees receive 15 vacation days per year...",
        char_range=(1200, 1450),
        page_num=12,
        source_file_path="./data/raw_documents/hr/handbook.pdf",
        source_file_hash="abc123def456...",
        embedding_model_name="all-MiniLM-L6-v2"
    )
    """

    # =========================================================================
    # CORE FIELDS - Basic chunk identification and content
    # =========================================================================

    chunk_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this chunk (UUID). Auto-generated if not provided."
    )

    doc_id: str = Field(
        ...,  # Required field (no default value)
        description="Parent document identifier. Used to group chunks from same document. "
                    "Example: 'employee_handbook_2025' or 'finance_policy_v3'"
    )

    domain: str = Field(
        ...,  # Required field
        description="Domain/department this chunk belongs to. Used for filtering queries "
                    "to specific departments. Example: 'hr', 'finance', 'engineering'"
    )

    chunk_text: str = Field(
        ...,  # Required field
        description="The actual text content of this chunk. This is what gets embedded "
                    "and searched. Should be meaningful, self-contained text."
    )

    char_range: Tuple[int, int] = Field(
        ...,  # Required field
        description="Character position in original document (start, end). "
                    "Enables precise location tracking. Example: (1200, 1450) means "
                    "this chunk spans characters 1200-1450 in the source document."
    )

    page_num: Optional[int] = Field(
        default=None,
        description="Page number in source document (for PDFs). Used for citations. "
                    "Example: If chunk is from page 12, citations show [doc_id:12]. "
                    "None for non-paginated documents (TXT, DOCX without page breaks)."
    )

    # =========================================================================
    # PROVENANCE FIELDS - Who uploaded, when, and from where
    # =========================================================================

    uploader_id: Optional[str] = Field(
        default=None,
        description="User/system identifier who uploaded this document. "
                    "Example: 'admin@company.com' or 'hr_team'. "
                    "Used for audit trails and access control."
    )

    upload_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this document was first uploaded (UTC). Auto-set to current time. "
                    "Used for filtering recent documents and audit logging."
    )

    document_version: str = Field(
        default="1.0",
        description="Version number of the source document. Increment when document is updated. "
                    "Example: '1.0', '2.1', '3.0'. Helps track document evolution."
    )

    last_updated_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this chunk was last modified (UTC). Auto-set to current time. "
                    "Updates when document is reprocessed or metadata changes."
    )

    source_file_path: str = Field(
        ...,  # Required field
        description="Original file path or URL of the source document. "
                    "Example: './data/raw_documents/hr/employee_handbook.pdf' or "
                    "'s3://bucket/docs/policy.docx'. Used to locate original file."
    )

    source_file_hash: str = Field(
        ...,  # Required field
        description="SHA256 hash of the source file content. Used for: "
                    "1) Detecting if file changed (idempotency check) "
                    "2) File integrity verification "
                    "3) Deduplication across uploads"
    )

    # =========================================================================
    # PROCESSING FIELDS - How this chunk was created and embedded
    # =========================================================================

    embedding_version: str = Field(
        default="1.0",
        description="Version of the embedding model or process. Increment when you "
                    "change embedding models or parameters. Helps track which embeddings "
                    "need regeneration. Example: '1.0', '2.0-finetuned'"
    )

    embedding_model_name: str = Field(
        ...,  # Required field
        description="Name of the embedding model used to create vector embeddings. "
                    "Examples: 'all-MiniLM-L6-v2' (Sentence-Transformers), "
                    "'models/embedding-001' (Google Gemini). "
                    "Critical for debugging retrieval quality and migration."
    )

    chunking_strategy: str = Field(
        ...,  # Required field
        description="Strategy used to split document into chunks. "
                    "Possible values: 'recursive' (fixed-size with overlap) or "
                    "'semantic' (embedding-based topical grouping). "
                    "Helps diagnose chunking quality issues."
    )

    chunk_type: str = Field(
        default="text",
        description="Type of content in this chunk. "
                    "Possible values: 'text' (regular paragraph), 'code' (code snippet), "
                    "'list' (bullet/numbered list). Future: 'table', 'image_caption'. "
                    "Enables specialized processing per content type."
    )

    # =========================================================================
    # QUALITY & AUTHORITY FIELDS - Trust and deprecation management
    # =========================================================================

    is_authoritative: bool = Field(
        default=False,
        description="Whether this chunk comes from an authoritative/official source. "
                    "Example: Official HR policy = True, Draft proposal = False. "
                    "Can be used to boost ranking of authoritative content in search. "
                    "Manually marked by administrators."
    )

    confidence_score: float = Field(
        default=1.0,
        ge=0.0,  # Greater than or equal to 0
        le=1.0,  # Less than or equal to 1
        description="Confidence score for this chunk (0.0 to 1.0). "
                    "Default: 1.0 (fully confident). "
                    "Future use: ML model scores for quality, relevance, or accuracy. "
                    "Can be used for ranking or filtering low-quality chunks."
    )

    deprecated_flag: bool = Field(
        default=False,
        description="Whether this chunk is outdated and should not be shown to users. "
                    "Example: When a new policy version is uploaded, old chunks are "
                    "marked deprecated=True. Used to hide stale information without "
                    "deleting it (for audit purposes)."
    )

    superseded_by_chunk_id: Optional[str] = Field(
        default=None,
        description="If this chunk is deprecated, this points to the chunk_id that "
                    "replaced it. Creates a chain of document versions. "
                    "Example: Old policy chunk points to new policy chunk. "
                    "Enables 'show me what changed' features."
    )

    # =========================================================================
    # PYDANTIC CONFIGURATION
    # =========================================================================

    class Config:
        """
        Pydantic configuration for the model.

        json_encoders: Custom serialization for datetime objects.
        When converting to JSON for storage, datetime objects are
        converted to ISO format strings (e.g., "2025-11-18T17:30:00").
        """
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of ChunkMetadata usage and validation.
    Run this file directly to see examples: python models/metadata_models.py
    """

    # Example 1: Creating metadata for an HR policy chunk
    hr_metadata = ChunkMetadata(
        doc_id="employee_handbook_2025",
        domain="hr",
        chunk_text="Employees are entitled to 15 vacation days per year...",
        char_range=(5000, 5250),
        page_num=12,
        uploader_id="hr_admin@company.com",
        source_file_path="./data/raw_documents/hr/handbook.pdf",
        source_file_hash="abc123def456789",
        embedding_model_name="all-MiniLM-L6-v2",
        chunking_strategy="recursive",
        is_authoritative=True,
        document_version="3.0"
    )

    print("Example 1: HR Policy Chunk Metadata")
    print("=" * 70)
    print(hr_metadata.model_dump_json(indent=2))
    print("\n")

    # Example 2: Creating metadata for a deprecated chunk
    deprecated_metadata = ChunkMetadata(
        doc_id="old_expense_policy_2024",
        domain="finance",
        chunk_text="Expense reimbursement limit is $500 per month...",
        char_range=(1000, 1200),
        page_num=5,
        source_file_path="./data/raw_documents/finance/old_policy.pdf",
        source_file_hash="xyz789abc123",
        embedding_model_name="all-MiniLM-L6-v2",
        chunking_strategy="semantic",
        deprecated_flag=True,  # This is outdated
        superseded_by_chunk_id="chunk-uuid-new-policy-2025",  # Points to new chunk
        document_version="2.0"
    )

    print("Example 2: Deprecated Finance Chunk Metadata")
    print("=" * 70)
    print(deprecated_metadata.model_dump_json(indent=2))
    print("\n")

    # Example 3: Validation error demonstration
    print("Example 3: Pydantic Validation")
    print("=" * 70)
    try:
        # This will fail because required fields are missing
        invalid_metadata = ChunkMetadata(
            domain="hr",
            chunk_text="Some text"
            # Missing: doc_id, source_file_path, source_file_hash, embedding_model_name
        )
    except Exception as e:
        print(f"Validation Error: {e}")
        print("\nPydantic caught missing required fields!")

    print("\n" + "=" * 70)
    print("Metadata model loaded successfully!")
