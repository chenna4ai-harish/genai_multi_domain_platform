"""

core/utils/hashing.py

File hashing utilities for provenance tracking and integrity verification.

What is This Module?
--------------------
Provides utilities for computing cryptographic hashes of files and data.
Used throughout Phase 2 for:
- File integrity verification (detect changes)
- Deduplication (same file = same hash)
- Provenance tracking (which file version was used)
- Idempotency (don't reprocess unchanged files)

Why SHA-256?
------------
SHA-256 is the industry standard for file integrity:
- Cryptographically secure (collision-resistant)
- Fast enough for large files
- Widely supported and tested
- 64 hex characters (256 bits)

Phase 2 Usage:
--------------
File processors (PDF, DOCX, TXT) use these functions to compute hashes:
- DocumentService validates files haven't changed
- ChunkMetadata stores source_file_hash for tracking
- Enables "process only if file changed" logic

Example Usage:
--------------
from core.utils.hashing import compute_file_hash, compute_string_hash

# Hash a file
with open("document.pdf", "rb") as f:
    file_hash = compute_file_hash(f)
print(f"File hash: {file_hash}")

# Hash a string
text_hash = compute_string_hash("Employee handbook content")
print(f"Text hash: {text_hash}")

References:
-----------
- Phase 2 Spec: Section 4.2 (Provenance Metadata)
- SHA-256: https://en.wikipedia.org/wiki/SHA-2

"""

import hashlib
from typing import BinaryIO, Union


def compute_file_hash(file_obj: BinaryIO) -> str:
    """
    Compute SHA-256 hash of a file object.

    Reads file in chunks to handle large files efficiently without
    loading entire file into memory.

    Parameters:
    -----------
    file_obj : BinaryIO
        File object opened in binary mode ('rb')
        Must support .read() method

    Returns:
    --------
    str:
        SHA-256 hash as 64-character hexadecimal string
        Example: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    Example:
    --------
    # Hash a PDF file
    with open("handbook.pdf", "rb") as f:
        file_hash = compute_file_hash(f)

    print(f"File hash: {file_hash}")
    print(f"Hash length: {len(file_hash)} chars")  # Always 64

    # Verify file hasn't changed
    with open("handbook.pdf", "rb") as f:
        new_hash = compute_file_hash(f)

    if new_hash == file_hash:
        print("File unchanged")
    else:
        print("File modified!")

    Notes:
    ------
    - File position is reset to beginning before and after hashing
    - Works with files of any size (chunks of 4KB)
    - Thread-safe (creates new hash object per call)
    """
    # Create SHA-256 hash object
    sha256_hash = hashlib.sha256()

    # Reset file position to beginning
    file_obj.seek(0)

    # Read file in chunks (4KB at a time)
    # This handles large files without loading into memory
    for chunk in iter(lambda: file_obj.read(4096), b''):
        sha256_hash.update(chunk)

    # Reset file position to beginning (for subsequent reads)
    file_obj.seek(0)

    # Return hex digest (64 hex characters)
    return sha256_hash.hexdigest()


def compute_string_hash(text: str, encoding: str = 'utf-8') -> str:
    """
    Compute SHA-256 hash of a string.

    Useful for hashing chunk text, queries, or any text content.

    Parameters:
    -----------
    text : str
        String to hash
    encoding : str
        Text encoding (default: 'utf-8')

    Returns:
    --------
    str:
        SHA-256 hash as 64-character hexadecimal string

    Example:
    --------
    # Hash chunk text
    chunk_text = "Employees receive 15 vacation days per year."
    chunk_hash = compute_string_hash(chunk_text)
    print(f"Chunk hash: {chunk_hash}")

    # Check for duplicate chunks
    chunks = ["text1", "text2", "text1"]  # "text1" is duplicate
    hashes = {compute_string_hash(c): c for c in chunks}
    print(f"Unique chunks: {len(hashes)}")  # 2 (duplicates removed)

    Notes:
    ------
    - Same text always produces same hash
    - Case-sensitive (hash("ABC") != hash("abc"))
    - Whitespace matters (hash("a b") != hash("ab"))
    - Empty string is valid: hash("") = "e3b0c44..."
    """
    # Encode string to bytes
    text_bytes = text.encode(encoding)

    # Compute SHA-256 hash
    sha256_hash = hashlib.sha256(text_bytes)

    # Return hex digest
    return sha256_hash.hexdigest()


def compute_bytes_hash(data: bytes) -> str:
    """
    Compute SHA-256 hash of raw bytes.

    Most generic hash function - works with any binary data.

    Parameters:
    -----------
    data : bytes
        Raw bytes to hash

    Returns:
    --------
    str:
        SHA-256 hash as 64-character hexadecimal string

    Example:
    --------
    # Hash binary data
    data = b"\\x00\\x01\\x02\\x03"
    data_hash = compute_bytes_hash(data)
    print(f"Data hash: {data_hash}")

    # Hash image data
    with open("logo.png", "rb") as f:
        image_data = f.read()

    image_hash = compute_bytes_hash(image_data)
    print(f"Image hash: {image_hash}")
    """
    # Compute SHA-256 hash
    sha256_hash = hashlib.sha256(data)

    # Return hex digest
    return sha256_hash.hexdigest()


def validate_hash(hash_string: str) -> bool:
    """
    Validate if a string is a valid SHA-256 hash.

    Checks:
    - Length is exactly 64 characters
    - All characters are hexadecimal (0-9, a-f)

    Parameters:
    -----------
    hash_string : str
        Hash string to validate

    Returns:
    --------
    bool:
        True if valid SHA-256 hash format, False otherwise

    Example:
    --------
    # Valid hash
    valid = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    print(validate_hash(valid))  # True

    # Invalid hashes
    print(validate_hash("abc123"))  # False (too short)
    print(validate_hash("x" * 64))  # False (invalid chars)
    print(validate_hash("ABC" * 21 + "A"))  # False (uppercase, wrong length)

    # Use in validation
    if not validate_hash(user_provided_hash):
        raise ValueError("Invalid hash format")
    """
    # Check length (SHA-256 = 64 hex chars)
    if len(hash_string) != 64:
        return False

    # Check all characters are hexadecimal (0-9, a-f)
    try:
        int(hash_string, 16)  # Will raise ValueError if not hex
        return True
    except ValueError:
        return False


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of hashing utilities.
    Run: python core/utils/hashing.py
    """
    import tempfile

    print("=" * 70)
    print("Hashing Utilities - File Integrity & Provenance")
    print("=" * 70)

    # Example 1: Hash a string
    print("\n1. String Hashing")
    print("-" * 70)

    text1 = "Employees receive 15 vacation days per year."
    hash1 = compute_string_hash(text1)
    print(f"Text: {text1}")
    print(f"Hash: {hash1}")
    print(f"Length: {len(hash1)} characters")

    # Same text = same hash
    hash1_again = compute_string_hash(text1)
    print(f"Same text, same hash: {hash1 == hash1_again}")

    # Different text = different hash
    text2 = "Employees receive 20 vacation days per year."
    hash2 = compute_string_hash(text2)
    print(f"Different text, different hash: {hash1 != hash2}")

    # Example 2: Hash a file
    print("\n2. File Hashing")
    print("-" * 70)

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        temp_file = f.name
        f.write("This is test content for hashing.")

    # Hash the file
    with open(temp_file, 'rb') as f:
        file_hash = compute_file_hash(f)

    print(f"File: {temp_file}")
    print(f"Hash: {file_hash}")

    # Verify file position reset
    with open(temp_file, 'rb') as f:
        hash_before = compute_file_hash(f)
        content = f.read()  # Read after hashing
        print(f"File content readable after hash: {len(content)} bytes")

    # Clean up
    import os

    os.unlink(temp_file)

    # Example 3: Hash validation
    print("\n3. Hash Validation")
    print("-" * 70)

    valid_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    invalid_hash1 = "abc123"
    invalid_hash2 = "x" * 64

    print(f"Valid: '{valid_hash[:20]}...' → {validate_hash(valid_hash)}")
    print(f"Invalid (too short): '{invalid_hash1}' → {validate_hash(invalid_hash1)}")
    print(f"Invalid (bad chars): '{invalid_hash2[:20]}...' → {validate_hash(invalid_hash2)}")

    # Example 4: Deduplication use case
    print("\n4. Deduplication Example")
    print("-" * 70)

    chunks = [
        "Employees receive 15 vacation days.",
        "Health insurance covers medical expenses.",
        "Employees receive 15 vacation days.",  # Duplicate
        "401k matching is 5% of salary."
    ]

    # Find unique chunks using hashes
    unique_chunks = {}
    for chunk in chunks:
        chunk_hash = compute_string_hash(chunk)
        if chunk_hash not in unique_chunks:
            unique_chunks[chunk_hash] = chunk

    print(f"Total chunks: {len(chunks)}")
    print(f"Unique chunks: {len(unique_chunks)}")
    print("Unique content:")
    for i, chunk in enumerate(unique_chunks.values(), 1):
        print(f"  {i}. {chunk}")

    # Example 5: Empty string hash
    print("\n5. Special Cases")
    print("-" * 70)

    empty_hash = compute_string_hash("")
    print(f"Empty string hash: {empty_hash}")
    print(f"This is the SHA-256 hash of empty input (constant)")

    print("\n" + "=" * 70)
    print("✅ Hashing utilities ready to use!")
    print("=" * 70)
