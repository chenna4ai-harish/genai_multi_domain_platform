"""

models/metadata_models.py

This module defines the comprehensive Phase 2 metadata schema for document chunks.

What is This File?
-------------------
Every chunk of text stored in the vector database needs **rich metadata** for:
- Provenance tracking (who uploaded, when, from where)
- Version control and deprecation management
- Debugging and quality assurance
- Citations and source verification
- Domain isolation and access control
- Authority and review workflows

Phase 2 Enhancement:
--------------------
Phase 2 adds comprehensive lifecycle, quality, and authority tracking:
- Enhanced provenance (upload_timestamp, source_file_hash)
- Document versioning (version, document_version, previous_version_id)
- Deprecation workflow (deprecated, deprecated_date, deprecation_reason)
- Authority levels (official, approved, draft, archived)
- Review status (approved, pending, rejected, in_review)
- Complete processing metadata (embedding model, chunking strategy, params)

Why So Much Metadata?
----------------------
Enables powerful features:
- "Show me only official HR policies from 2025 that aren't deprecated"
- "Which embedding model was used for this chunk?" (for re-embedding)
- "Has this document been superseded?" (deprecation tracking)
- Proper citations: [employee_handbook_2025.pdf:page 12]
- Quality ranking (boost authoritative content)

References:
-----------
- Phase 2 Spec: Section 4 (Enhanced Metadata Schema)
- Pydantic Docs: https://docs.pydantic.dev/

"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


# =============================================================================
# ENUMS FOR CONTROLLED VOCABULARIES
# =============================================================================

class AuthorityLevel(str, Enum):
    """
    Authority level of document source.

    Controls trust and ranking in search results.
    """
    OFFICIAL = "official"  # Official company policy
    APPROVED = "approved"  # Reviewed and approved
    DRAFT = "draft"  # Work in progress
    ARCHIVED = "archived"  # Historical record
    DEPRECATED = "deprecated"  # Outdated, replaced


class ReviewStatus(str, Enum):
    """
    Review/approval status of document.

    Tracks document through approval workflow.
    """
    APPROVED = "approved"  # Fully reviewed and approved
    PENDING = "pending"  # Awaiting review
    REJECTED = "rejected"  # Review rejected
    IN_REVIEW = "in_review"  # Currently being reviewed


# =============================================================================
# CHUNK METADATA MODEL (PHASE 2 COMPLETE SCHEMA)
# =============================================================================

class ChunkMetadata(BaseModel):
    """
    Complete Phase 2 metadata schema for document chunks.

    This class defines ALL information tracked for each chunk stored
    in the vector database. Using Pydantic ensures:
    - Type safety (wrong types raise validation errors)
    - Required field enforcement (missing fields caught early)
    - Easy serialization to/from JSON/dict
    - Automatic validation and documentation

    Field Categories:
    -----------------
    1. Core/Identity: chunk_id, doc_id, domain, chunk_text
    2. Content: title, page_num, char_range, doc_type, tags
    3. Provenance: uploader_id, upload_timestamp, source file info
    4. Versioning: version, document_version, last_updated_timestamp
    5. Processing: embedding model, chunking strategy, processing timestamp
    6. Lifecycle: deprecated, deprecated_date, deprecation_reason
    7. Quality: confidence_score, authority_level, review status

    Example Usage:
    --------------
    metadata = ChunkMetadata(
        doc_id="employee_handbook_2025",
        domain="hr",
        chunk_text="Employees receive 15 vacation days per year...",
        title="Employee Handbook 2025",
        doc_type="policy",
        char_range=(1200, 1450),
        page_num=12,
        source_file_path="./data/raw_documents/hr/handbook.pdf",
        source_file_hash="abc123def456...",
        embedding_model_name="all-MiniLM-L6-v2",
        chunking_strategy="recursive",
        uploader_id="admin@company.com",
        authority_level=AuthorityLevel.OFFICIAL,
        review_status=ReviewStatus.APPROVED
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

    # =========================================================================
    # CONTENT FIELDS - Document structure and classification
    # =========================================================================

    title: Optional[str] = Field(
        default=None,
        description="Document title. Example: 'Employee Handbook 2025'. "
                    "Used for display and citations."
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

    doc_type: str = Field(
        default="document",
        description="Type of document. Examples: 'policy', 'faq', 'manual', 'guideline', 'form'. "
                    "Used for filtering and categorization."
    )

    tags: List[str] = Field(
        default_factory=list,
        description="User-defined tags for classification. "
                    "Example: ['benefits', 'leave', '2025']. "
                    "Used for advanced filtering."
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

    source_file_path: str = Field(
        ...,  # Required field
        description="Original file path or URL of the source document. "
                    "Example: './data/raw_documents/hr/employee_handbook.pdf' or "
                    "'s3://bucket/docs/policy.docx'. Used to locate original file."
    )

    source_file_hash: str = Field(
        ...,  # Required field
        description="SHA-256 hash of the source file content. Used for: "
                    "1) Detecting if file changed (idempotency check) "
                    "2) File integrity verification "
                    "3) Deduplication across uploads"
    )

    # =========================================================================
    # VERSIONING FIELDS - Document evolution and history
    # =========================================================================

    version: str = Field(
        default="1.0",
        description="Chunk/processing version. Increment when reprocessing. "
                    "Example: '1.0', '2.0', '2.1'"
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

    # =========================================================================
    # PROCESSING FIELDS - How this chunk was created and embedded
    # =========================================================================

    embedding_model_name: str = Field(
        ...,  # Required field
        description="Name of the embedding model used to create vector embeddings. "
                    "Examples: 'all-MiniLM-L6-v2' (Sentence-Transformers), "
                    "'models/embedding-001' (Google Gemini). "
                    "Critical for debugging retrieval quality and migration."
    )

    embedding_dimension: Optional[int] = Field(
        default=None,
        description="Dimension of embedding vectors. Examples: 384, 768, 1536. "
                    "Useful for validation and index configuration."
    )

    embedding_version: str = Field(
        default="1.0",
        description="Version of the embedding model or process. Increment when you "
                    "change embedding models or parameters. Helps track which embeddings "
                    "need regeneration. Example: '1.0', '2.0-finetuned'"
    )

    chunking_strategy: str = Field(
        ...,  # Required field
        description="Strategy used to split document into chunks. "
                    "Possible values: 'recursive' (fixed-size with overlap) or "
                    "'semantic' (embedding-based topical grouping). "
                    "Helps diagnose chunking quality issues."
    )

    chunking_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used for chunking. "
                    "Example: {'chunk_size': 500, 'overlap': 50} for recursive. "
                    "Useful for debugging and reproducing chunking."
    )

    chunk_type: str = Field(
        default="text",
        description="Type of content in this chunk. "
                    "Possible values: 'text' (regular paragraph), 'code' (code snippet), "
                    "'list' (bullet/numbered list). Future: 'table', 'image_caption'. "
                    "Enables specialized processing per content type."
    )

    processing_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When chunk was processed (chunked + embedded). "
                    "Used for tracking processing pipeline performance."
    )

    # =========================================================================
    # LIFECYCLE FIELDS - Deprecation and supersession
    # =========================================================================

    deprecated_flag: bool = Field(
        default=False,
        description="Whether this chunk is outdated and should not be shown to users. "
                    "Example: When a new policy version is uploaded, old chunks are "
                    "marked deprecated=True. Used to hide stale information without "
                    "deleting it (for audit purposes)."
    )

    deprecated_date: Optional[datetime] = Field(
        default=None,
        description="When this chunk was marked as deprecated. "
                    "Set automatically when deprecated_flag=True."
    )

    deprecation_reason: Optional[str] = Field(
        default=None,
        description="Human-readable reason for deprecation. "
                    "Example: 'Superseded by Employee Handbook 2026' or "
                    "'Policy no longer in effect as of 2025-06-01'."
    )

    superseded_by_chunk_id: Optional[str] = Field(
        default=None,
        description="If this chunk is deprecated, this points to the chunk_id that "
                    "replaced it. Creates a chain of document versions. "
                    "Example: Old policy chunk points to new policy chunk. "
                    "Enables 'show me what changed' features."
    )

    # =========================================================================
    # QUALITY & AUTHORITY FIELDS - Trust and review management
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

    authority_level: AuthorityLevel = Field(
        default=AuthorityLevel.DRAFT,
        description="Authority level of document source. Controls trust and ranking. "
                    "Values: official, approved, draft, archived, deprecated"
    )

    review_status: ReviewStatus = Field(
        default=ReviewStatus.PENDING,
        description="Review/approval status. Tracks document through approval workflow. "
                    "Values: approved, pending, rejected, in_review"
    )

    reviewed_by: Optional[str] = Field(
        default=None,
        description="User ID of reviewer who approved/rejected this document. "
                    "Example: 'senior_hr@company.com'"
    )

    reviewed_date: Optional[datetime] = Field(
        default=None,
        description="When this document was reviewed. "
                    "Set when review_status changes to approved/rejected."
    )

    # =========================================================================
    # CUSTOM/EXTENSIBLE FIELDS
    # =========================================================================

    custom_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific custom metadata fields. "
                    "Allows flexibility without changing core schema. "
                    "Example: {'compliance_id': 'COMP-2025-001', 'cost_center': 'HR-01'}"
    )

    # =========================================================================
    # VALIDATORS
    # =========================================================================

    @field_validator('source_file_hash')
    @classmethod
    def validate_hash(cls, v):
        """Ensure hash is valid SHA-256 format (64 hex characters)."""
        if len(v) != 64 or not all(c in '0123456789abcdef' for c in v.lower()):
            raise ValueError("Invalid SHA-256 hash format. Must be 64 hex characters.")
        return v.lower()

    # =========================================================================
    # PYDANTIC CONFIGURATION
    # =========================================================================

    class Config:
        """
        Pydantic configuration for the model.

        json_encoders: Custom serialization for datetime and enum objects.
        use_enum_values: Store enum values as strings in JSON/dict.
        """
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        use_enum_values = True  # Serialize enums as their values


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of ChunkMetadata usage and validation.
    Run: python models/metadata_models.py
    """
    print("=" * 70)
    print("Phase 2 ChunkMetadata Examples")
    print("=" * 70)

    # Example 1: Creating metadata for an HR policy chunk
    print("\n1. Official HR Policy Chunk")
    print("-" * 70)

    hr_metadata = ChunkMetadata(
        doc_id="employee_handbook_2025",
        domain="hr",
        chunk_text="Employees are entitled to 15 vacation days per year...",
        title="Employee Handbook 2025",
        doc_type="policy",
        tags=["benefits", "leave", "vacation", "2025"],
        char_range=(5000, 5250),
        page_num=12,
        uploader_id="hr_admin@company.com",
        source_file_path="./data/raw_documents/hr/handbook.pdf",
        source_file_hash="abc123def456789" * 4,  # 64 chars
        embedding_model_name="all-MiniLM-L6-v2",
        embedding_dimension=384,
        chunking_strategy="recursive",
        chunking_params={"chunk_size": 500, "overlap": 50},
        is_authoritative=True,
        authority_level=AuthorityLevel.OFFICIAL,
        review_status=ReviewStatus.APPROVED,
        reviewed_by="senior_hr@company.com",
        document_version="3.0"
    )

    print(f"✅ Created metadata for: {hr_metadata.title}")
    print(f"   Doc ID: {hr_metadata.doc_id}")
    print(f"   Domain: {hr_metadata.domain}")
    print(f"   Authority: {hr_metadata.authority_level}")
    print(f"   Review Status: {hr_metadata.review_status}")
    print(f"   Tags: {', '.join(hr_metadata.tags)}")

    # Example 2: Deprecated chunk pointing to replacement
    print("\n2. Deprecated Finance Policy Chunk")
    print("-" * 70)

    deprecated_metadata = ChunkMetadata(
        doc_id="old_expense_policy_2024",
        domain="finance",
        chunk_text="Expense reimbursement limit is $500 per month...",
        title="Expense Policy 2024 (Deprecated)",
        doc_type="policy",
        tags=["expenses", "reimbursement", "2024", "deprecated"],
        char_range=(1000, 1200),
        page_num=5,
        source_file_path="./data/raw_documents/finance/old_policy.pdf",
        source_file_hash="xyz789abc123" + "0" * 52,  # 64 chars
        embedding_model_name="all-MiniLM-L6-v2",
        embedding_dimension=384,
        chunking_strategy="semantic",
        deprecated_flag=True,  # This is outdated
        deprecated_date=datetime.utcnow(),
        deprecation_reason="Superseded by Expense Policy 2025 with updated limits",
        superseded_by_chunk_id="chunk-uuid-new-policy-2025",  # Points to new chunk
        authority_level=AuthorityLevel.DEPRECATED,
        document_version="2.0"
    )

    print(f"✅ Created deprecated metadata for: {deprecated_metadata.title}")
    print(f"   Deprecated: {deprecated_metadata.deprecated_flag}")
    print(f"   Reason: {deprecated_metadata.deprecation_reason}")
    print(f"   Superseded by: {deprecated_metadata.superseded_by_chunk_id}")

    # Example 3: Validation error demonstration
    print("\n3. Pydantic Validation Examples")
    print("-" * 70)

    try:
        # This will fail because required fields are missing
        invalid_metadata = ChunkMetadata(
            domain="hr",
            chunk_text="Some text"
            # Missing: doc_id, source_file_path, source_file_hash, embedding_model_name, chunking_strategy
        )
    except Exception as e:
        print(f"❌ Validation Error (missing required fields):")
        print(f"   {str(e)[:100]}...")

    try:
        # This will fail because hash is invalid
        invalid_hash = ChunkMetadata(
            doc_id="test",
            domain="hr",
            chunk_text="Text",
            char_range=(0, 4),
            source_file_path="test.pdf",
            source_file_hash="invalid_hash",  # Not 64 hex chars
            embedding_model_name="test-model",
            chunking_strategy="recursive"
        )
    except Exception as e:
        print(f"\n❌ Validation Error (invalid hash):")
        print(f"   {e}")

    print("\n" + "=" * 70)
    print("✅ Phase 2 Metadata Model loaded successfully!")
    print("=" * 70)
