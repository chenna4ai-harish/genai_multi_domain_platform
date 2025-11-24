"""

core/utils/validation.py

Input validation utilities for Phase 2 document processing.

What is This Module?
--------------------
Provides validation functions for file uploads, metadata, and user inputs.
Used by DocumentService to enforce security policies and data quality.

Phase 2 Security:
-----------------
Validation is CRITICAL for security:
- Prevent malicious file uploads (executable disguised as PDF)
- Enforce file size limits (prevent DoS attacks)
- Validate metadata completeness (ensure data quality)
- Sanitize user inputs (prevent injection attacks)

Validation Layers:
------------------
1. **File Type Validation**: Check file extensions
2. **File Size Validation**: Enforce size limits
3. **Metadata Validation**: Check required fields
4. **Field Format Validation**: Email, dates, etc.

Example Usage:
--------------
from core.utils.validation import (
    validate_file_type,
    validate_file_size,
    validate_metadata_fields
)

# Validate file upload
validate_file_type("document.pdf", allowed_types=["pdf", "docx"])
validate_file_size(file_obj, max_size_mb=20)
validate_metadata_fields(metadata, required_fields=["doc_id", "title"])

References:
-----------
- Phase 2 Spec: Section 10 (File Processing & Validation)
- OWASP File Upload: https://owasp.org/www-community/vulnerabilities/Unrestricted_File_Upload

"""

from typing import List, Dict, Any, BinaryIO
from pathlib import Path
import re


class ValidationError(Exception):
    """
    Raised when validation fails.

    Use this exception for all validation errors to distinguish
    from other errors (RuntimeError, ValueError, etc.)
    """
    pass


# =============================================================================
# FILE VALIDATION
# =============================================================================

def validate_file_type(
        filename: str,
        allowed_types: List[str]
) -> None:
    """
    Validate file extension against allowed types.

    Security Note:
    --------------
    This is a BASIC check. Malicious users can rename files!
    For production, also validate file content (magic bytes).

    Parameters:
    -----------
    filename : str
        Filename with extension (e.g., "document.pdf")
    allowed_types : List[str]
        List of allowed extensions (without dot)
        Example: ["pdf", "docx", "txt"]

    Raises:
    -------
    ValidationError:
        If file type not allowed or no extension found

    Example:
    --------
    # Valid
    validate_file_type("handbook.pdf", ["pdf", "docx"])  # OK

    # Invalid
    validate_file_type("script.exe", ["pdf", "docx"])  # Raises ValidationError
    validate_file_type("no_extension", ["pdf"])  # Raises ValidationError
    """
    # Check if filename has extension
    if '.' not in filename:
        raise ValidationError(
            f"File '{filename}' has no extension. "
            f"Allowed types: {', '.join(allowed_types)}"
        )

    # Extract extension (convert to lowercase)
    ext = filename.rsplit('.', 1)[-1].lower()

    # Normalize allowed_types to lowercase
    allowed_types_lower = [t.lower() for t in allowed_types]

    # Validate extension
    if ext not in allowed_types_lower:
        raise ValidationError(
            f"File type '.{ext}' not allowed. "
            f"Allowed types: {', '.join(sorted(allowed_types_lower))}"
        )


def validate_file_size(
        file_obj: BinaryIO,
        max_size_mb: int
) -> None:
    """
    Validate file size against maximum allowed.

    Security Note:
    --------------
    Prevents DoS attacks via huge file uploads.
    Also prevents out-of-memory errors during processing.

    Parameters:
    -----------
    file_obj : BinaryIO
        File object opened in binary mode
        Must support .seek() and .tell() methods
    max_size_mb : int
        Maximum file size in megabytes

    Raises:
    -------
    ValidationError:
        If file size exceeds limit

    Example:
    --------
    with open("large_file.pdf", "rb") as f:
        validate_file_size(f, max_size_mb=20)  # OK if < 20MB
    """
    # Get file size
    file_obj.seek(0, 2)  # Seek to end
    file_size_bytes = file_obj.tell()
    file_obj.seek(0)  # Reset to beginning

    # Convert to MB
    file_size_mb = file_size_bytes / (1024 * 1024)

    # Validate size
    if file_size_mb > max_size_mb:
        raise ValidationError(
            f"File size {file_size_mb:.2f}MB exceeds maximum allowed "
            f"size of {max_size_mb}MB"
        )


# =============================================================================
# METADATA VALIDATION
# =============================================================================

def validate_metadata_fields(
        metadata: Dict[str, Any],
        required_fields: List[str]
) -> None:
    """
    Validate that required metadata fields are present and non-empty.

    Data Quality:
    -------------
    Ensures all required fields are provided before processing.
    Catches errors early (fail fast principle).

    Parameters:
    -----------
    metadata : Dict[str, Any]
        Metadata dictionary to validate
    required_fields : List[str]
        List of required field names

    Raises:
    -------
    ValidationError:
        If required fields are missing or empty

    Example:
    --------
    metadata = {
        "doc_id": "handbook_2025",
        "title": "Employee Handbook",
        "doc_type": "policy",
        "uploader_id": "admin@company.com"
    }

    # OK
    validate_metadata_fields(
        metadata,
        required_fields=["doc_id", "title", "doc_type"]
    )

    # Error: missing field
    validate_metadata_fields(
        metadata,
        required_fields=["doc_id", "author"]  # 'author' missing
    )  # Raises ValidationError
    """
    missing = []
    empty = []

    for field in required_fields:
        # Check if field exists
        if field not in metadata:
            missing.append(field)
        # Check if field is non-empty
        elif not metadata[field]:  # Catches None, "", [], {}
            empty.append(field)

    # Raise error if any fields missing
    if missing:
        raise ValidationError(
            f"Missing required metadata fields: {', '.join(missing)}"
        )

    # Raise error if any fields empty
    if empty:
        raise ValidationError(
            f"Empty required metadata fields: {', '.join(empty)}"
        )


def validate_email(email: str) -> None:
    """
    Validate email format (basic check).

    Note: This is a SIMPLE validation. For production, consider:
    - email-validator library
    - DNS verification
    - Actual email delivery test

    Parameters:
    -----------
    email : str
        Email address to validate

    Raises:
    -------
    ValidationError:
        If email format is invalid

    Example:
    --------
    validate_email("admin@company.com")  # OK
    validate_email("invalid.email")  # Raises ValidationError
    """
    # Basic email regex (not comprehensive!)
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if not re.match(email_pattern, email):
        raise ValidationError(f"Invalid email format: {email}")


def validate_doc_id(doc_id: str) -> None:
    """
    Validate document ID format.

    Rules:
    - Only alphanumeric, underscores, hyphens
    - 3-100 characters
    - No spaces or special characters

    Parameters:
    -----------
    doc_id : str
        Document ID to validate

    Raises:
    -------
    ValidationError:
        If doc_id format is invalid

    Example:
    --------
    validate_doc_id("employee_handbook_2025")  # OK
    validate_doc_id("policy-hr-001")  # OK
    validate_doc_id("invalid doc id!")  # Raises ValidationError (spaces)
    """
    # Check length
    if not 3 <= len(doc_id) <= 100:
        raise ValidationError(
            f"doc_id must be 3-100 characters, got {len(doc_id)}"
        )

    # Check characters (alphanumeric, underscore, hyphen only)
    if not re.match(r'^[a-zA-Z0-9_-]+$', doc_id):
        raise ValidationError(
            f"doc_id must contain only alphanumeric, underscore, and hyphen characters. "
            f"Got: '{doc_id}'"
        )


# =============================================================================
# QUERY VALIDATION
# =============================================================================

def validate_query_text(query_text: str, min_length: int = 1, max_length: int = 1000) -> None:
    """
    Validate query text.

    Parameters:
    -----------
    query_text : str
        Query text to validate
    min_length : int
        Minimum length (default: 1)
    max_length : int
        Maximum length (default: 1000)

    Raises:
    -------
    ValidationError:
        If query text is invalid

    Example:
    --------
    validate_query_text("vacation policy")  # OK
    validate_query_text("")  # Raises ValidationError (too short)
    validate_query_text("x" * 2000)  # Raises ValidationError (too long)
    """
    # Check if empty or whitespace-only
    if not query_text or not query_text.strip():
        raise ValidationError("Query text cannot be empty")

    # Check length
    if len(query_text) < min_length:
        raise ValidationError(
            f"Query text too short (minimum: {min_length} characters)"
        )

    if len(query_text) > max_length:
        raise ValidationError(
            f"Query text too long (maximum: {max_length} characters)"
        )


def validate_top_k(top_k: int, min_value: int = 1, max_value: int = 100) -> None:
    """
    Validate top_k parameter.

    Parameters:
    -----------
    top_k : int
        Number of results to return
    min_value : int
        Minimum allowed value (default: 1)
    max_value : int
        Maximum allowed value (default: 100)

    Raises:
    -------
    ValidationError:
        If top_k is out of range

    Example:
    --------
    validate_top_k(10)  # OK
    validate_top_k(0)  # Raises ValidationError (too small)
    validate_top_k(1000)  # Raises ValidationError (too large)
    """
    if not min_value <= top_k <= max_value:
        raise ValidationError(
            f"top_k must be between {min_value} and {max_value}, got {top_k}"
        )


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of validation utilities.
    Run: python core/utils/validation.py
    """
    import tempfile

    print("=" * 70)
    print("Validation Utilities - Input Validation & Security")
    print("=" * 70)

    # Example 1: File type validation
    print("\n1. File Type Validation")
    print("-" * 70)

    try:
        validate_file_type("document.pdf", ["pdf", "docx", "txt"])
        print("✅ Valid: document.pdf")
    except ValidationError as e:
        print(f"❌ {e}")

    try:
        validate_file_type("script.exe", ["pdf", "docx", "txt"])
        print("✅ Valid: script.exe")
    except ValidationError as e:
        print(f"❌ {e}")

    # Example 2: File size validation
    print("\n2. File Size Validation")
    print("-" * 70)

    # Create small file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_file = f.name
        f.write(b"x" * (5 * 1024 * 1024))  # 5MB

    with open(temp_file, 'rb') as f:
        try:
            validate_file_size(f, max_size_mb=20)
            print("✅ Valid: 5MB file (limit: 20MB)")
        except ValidationError as e:
            print(f"❌ {e}")

    with open(temp_file, 'rb') as f:
        try:
            validate_file_size(f, max_size_mb=2)
            print("✅ Valid: 5MB file (limit: 2MB)")
        except ValidationError as e:
            print(f"❌ {e}")

    import os

    os.unlink(temp_file)

    # Example 3: Metadata validation
    print("\n3. Metadata Validation")
    print("-" * 70)

    metadata_valid = {
        "doc_id": "handbook_2025",
        "title": "Employee Handbook",
        "doc_type": "policy",
        "uploader_id": "admin@company.com"
    }

    try:
        validate_metadata_fields(
            metadata_valid,
            required_fields=["doc_id", "title", "doc_type"]
        )
        print("✅ Valid metadata: All required fields present")
    except ValidationError as e:
        print(f"❌ {e}")

    metadata_invalid = {"doc_id": "handbook_2025"}

    try:
        validate_metadata_fields(
            metadata_invalid,
            required_fields=["doc_id", "title", "doc_type"]
        )
        print("✅ Valid metadata")
    except ValidationError as e:
        print(f"❌ {e}")

    # Example 4: Email validation
    print("\n4. Email Validation")
    print("-" * 70)

    emails = [
        ("admin@company.com", True),
        ("user.name+tag@example.co.uk", True),
        ("invalid.email", False),
        ("@company.com", False),
        ("user@", False)
    ]

    for email, should_be_valid in emails:
        try:
            validate_email(email)
            result = "✅ Valid"
        except ValidationError:
            result = "❌ Invalid"

        expected = "✅" if should_be_valid else "❌"
        print(f"{result}: {email} (expected: {expected})")

    # Example 5: Query validation
    print("\n5. Query Validation")
    print("-" * 70)

    queries = [
        ("vacation policy", True),
        ("", False),
        ("   ", False),
        ("x" * 2000, False)
    ]

    for query, should_be_valid in queries:
        try:
            validate_query_text(query)
            result = "✅ Valid"
        except ValidationError as e:
            result = f"❌ Invalid: {str(e)[:50]}"

        display_query = query[:30] + "..." if len(query) > 30 else query
        print(f"{result}: '{display_query}'")

    print("\n" + "=" * 70)
    print("✅ Validation utilities ready to use!")
    print("=" * 70)
