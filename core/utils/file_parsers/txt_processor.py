"""

utils/file_parsers/txt_processor.py

Phase 2 Enhanced plain text file extraction with encoding detection and structured output.

What is This File?
-------------------
TXT processor that extracts text from plain text files with:
- Automatic encoding detection (UTF-8, Latin-1, Windows-1252, etc.)
- Line-level tracking with character positions
- File hash computation for provenance
- Standardized Phase 2 output format
- Comprehensive error handling

Why Enhanced for Phase 2?
--------------------------
Phase 2 requires comprehensive metadata for:
- Citations with line numbers: [notes.txt:Line 42]
- Character range tracking for precise source location
- Provenance tracking via file hash
- Per-line data with position information

Features:
---------
- Auto-detect encoding (UTF-8, Latin-1, Windows-1252, etc.)
- Handle different line endings (Windows, Unix, Mac)
- Line-level metadata with character positions
- File hash computation
- No external dependencies (chardet optional)

Installation:
-------------
pip install chardet  # Optional but recommended for encoding detection

Example Usage (Phase 2):
------------------------
processor = TXTProcessor()

# Extract with full Phase 2 metadata
result = processor.extract("notes.txt")

if result['success']:
    print(f"Text: {result['text'][:100]}...")
    print(f"Lines: {result['metadata']['line_count']}")
    print(f"Encoding: {result['metadata']['encoding']}")
    print(f"Hash: {result['metadata']['file_hash']}")
else:
    print(f"Errors: {result['errors']}")

References:
-----------
- Phase 2 Spec: Section 10.3 (Enhanced File Processors)

"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import hashlib

# Try importing optional chardet for better encoding detection
try:
    import chardet

    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# PHASE 2 FILE HASHING UTILITY
# =============================================================================

def compute_file_hash(file_path: str) -> str:
    """
    Compute SHA-256 hash of file for provenance tracking (Phase 2).

    Parameters:
    -----------
    file_path : str
        Path to file

    Returns:
    --------
    str:
        SHA-256 hash (64 hex characters)
    """
    sha256_hash = hashlib.sha256()

    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


# =============================================================================
# PHASE 2 TXT PROCESSOR - MAIN CLASS
# =============================================================================

class TXTProcessor:
    """
    Phase 2 enhanced plain text processor with standardized output format.

    Handles plain text files (.txt) with:
    - Automatic encoding detection
    - Line-level tracking with character positions
    - Different line ending support (Windows, Unix, Mac)
    - File hash computation
    - Standardized Phase 2 return format

    Returns structured Phase 2 format with metadata, line data, and file hash.

    Features:
    ---------
    - Auto-detect encoding (UTF-8, Latin-1, Windows-1252, etc.)
    - Handle different line endings (\\r\\n, \\n, \\r)
    - Per-line metadata with char ranges
    - File hash for integrity checking
    - Complete error collection
    - No required dependencies (chardet optional)

    Example:
    --------
    # Initialize processor
    processor = TXTProcessor()

    # Extract with Phase 2 format
    result = processor.extract("document.txt")

    # Check success
    if result['success']:
        print(f"Extracted: {len(result['text'])} characters")
        print(f"Lines: {len(result['pages'])}")
        print(f"Encoding: {result['metadata']['encoding']}")
        print(f"Hash: {result['metadata']['file_hash']}")
    else:
        print(f"Errors: {result['errors']}")
    """

    def __init__(self):
        """
        Initialize TXT processor.

        Notes:
        ------
        No external dependencies required, but chardet is recommended
        for better encoding detection.
        """
        logger.info(
            f"TXTProcessor initialized\n"
            f"  chardet: {'Available' if CHARDET_AVAILABLE else 'Not available (optional)'}"
        )

    def extract(
            self,
            file_path: str,
            encoding: Optional[str] = None,
            strip_whitespace: bool = True,
            compute_hash: bool = True
    ) -> Dict[str, Any]:
        """
        Extract text and metadata from TXT file (Phase 2 standardized format).

        This is the PRIMARY extraction method for Phase 2 service layer.
        Returns structured data with comprehensive metadata and error handling.

        Parameters:
        -----------
        file_path : str
            Path to TXT file
        encoding : str, optional
            Force specific encoding (default: auto-detect)
            Examples: 'utf-8', 'latin-1', 'windows-1252', 'ascii'
        strip_whitespace : bool
            Whether to strip leading/trailing whitespace from lines (default: True)
        compute_hash : bool
            Whether to compute file hash (default: True)

        Returns:
        --------
        Dict[str, Any]:
            {
                'text': str,                    # Full extracted text
                'metadata': {
                    'line_count': int,
                    'file_hash': str,           # SHA-256 hash
                    'file_size_bytes': int,
                    'encoding': str,            # Detected/used encoding
                    'extraction_method': str    # 'chardet' or 'fallback'
                },
                'pages': List[Dict],            # Per-line info (called 'pages' for API compatibility)
                'success': bool,                # Extraction successful
                'errors': List[str]             # Any warnings/errors
            }

        Page/Line structure:
        --------------------
        'pages': [
            {
                'page_num': int,                # Line number (1-indexed)
                'text': str,                    # Line text
                'char_range': (start, end),     # Character positions in full text
                'length': int,                  # Line text length
                'type': str                     # 'line'
            },
            ...
        ]

        Raises:
        -------
        FileNotFoundError:
            If TXT file doesn't exist
        RuntimeError:
            If extraction completely fails (all encodings fail)

        Example:
        --------
        processor = TXTProcessor()

        # Auto-detect encoding
        result = processor.extract("document.txt")

        # Force specific encoding
        result = processor.extract("document.txt", encoding="utf-8")

        if result['success']:
            print(f"Lines: {result['metadata']['line_count']}")
            print(f"Encoding: {result['metadata']['encoding']}")

            for line in result['pages'][:5]:
                print(f"Line {line['page_num']}: {line['text'][:50]}...")
        else:
            print(f"Errors: {result['errors']}")
        """
        logger.info(f"Extracting TXT: {file_path}")

        errors = []
        text = ""
        lines_data = []
        detected_encoding = encoding or "unknown"
        extraction_method = "unknown"

        # Validate file exists
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"TXT file not found: {file_path}")

        # Get file size
        try:
            file_size = file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Could not get file size: {e}")
            file_size = 0

        # Detect encoding if not specified
        if encoding is None:
            detected_encoding, extraction_method = self._detect_encoding(str(file_path))
            logger.debug(f"Detected encoding: {detected_encoding} (method: {extraction_method})")
        else:
            detected_encoding = encoding
            extraction_method = "manual"

        # Extract text with detected/specified encoding
        try:
            with open(file_path, 'r', encoding=detected_encoding, errors='replace') as f:
                lines = f.readlines()

            # Process lines with position tracking
            char_position = 0

            for line_num, line in enumerate(lines, 1):
                # Strip whitespace if requested
                if strip_whitespace:
                    line_text = line.strip()
                else:
                    line_text = line.rstrip('\n\r')

                # Skip empty lines if stripping whitespace
                if strip_whitespace and not line_text:
                    continue

                # Track character range for this line
                char_start = char_position
                char_end = char_position + len(line_text)

                lines_data.append({
                    'page_num': line_num,
                    'text': line_text,
                    'char_range': (char_start, char_end),
                    'length': len(line_text),
                    'type': 'line'
                })

                char_position = char_end + 1  # +1 for newline separator

            # Join lines
            text = '\n'.join([line['text'] for line in lines_data])

            logger.info(f"✅ Extraction successful: {len(lines_data)} lines")

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            errors.append(f"Extraction failed: {str(e)}")
            raise RuntimeError(f"Failed to extract TXT: {e}")

        # Compute file hash (Phase 2 requirement)
        file_hash = None
        if compute_hash:
            try:
                file_hash = compute_file_hash(str(file_path))
                logger.debug(f"Computed file hash: {file_hash[:16]}...")
            except Exception as e:
                logger.warning(f"Failed to compute file hash: {e}")
                errors.append(f"Hash computation failed: {str(e)}")
                file_hash = "HASH_FAILED"

        # Build Phase 2 standardized result
        result = {
            'text': text,
            'metadata': {
                'line_count': len(lines_data),
                'file_hash': file_hash,
                'file_size_bytes': file_size,
                'encoding': detected_encoding,
                'extraction_method': extraction_method
            },
            'pages': lines_data,  # Called 'pages' for API compatibility
            'success': bool(text) or len(lines_data) > 0,
            'errors': errors
        }

        logger.info(
            f"✅ TXT extraction complete:\n"
            f"   Encoding: {detected_encoding}\n"
            f"   Lines: {len(lines_data)}\n"
            f"   Text length: {len(text):,} chars\n"
            f"   Errors: {len(errors)}"
        )

        return result

    def extract_text(
            self,
            file_path: str,
            encoding: Optional[str] = None,
            strip_whitespace: bool = True
    ) -> str:
        """
        Extract plain text from TXT file (backward compatibility).

        For Phase 2, use extract() method instead which returns structured data.

        Parameters:
        -----------
        file_path : str
            Path to TXT file
        encoding : str, optional
            Force specific encoding (default: auto-detect)
        strip_whitespace : bool
            Whether to strip whitespace

        Returns:
        --------
        str:
            Extracted text
        """
        result = self.extract(file_path, encoding, strip_whitespace, compute_hash=False)
        return result['text']

    def extract_lines(
            self,
            file_path: str,
            encoding: Optional[str] = None,
            strip_whitespace: bool = True
    ) -> List[str]:
        """
        Extract lines as separate strings.

        Parameters:
        -----------
        file_path : str
            Path to TXT file
        encoding : str, optional
            Force specific encoding
        strip_whitespace : bool
            Whether to strip whitespace

        Returns:
        --------
        List[str]:
            List of line texts

        Example:
        --------
        processor = TXTProcessor()
        lines = processor.extract_lines("document.txt")

        for i, line in enumerate(lines, 1):
            print(f"{i}. {line[:100]}...")
        """
        result = self.extract(file_path, encoding, strip_whitespace, compute_hash=False)
        return [line['text'] for line in result['pages']]

    def get_metadata(self, file_path: str, encoding: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract metadata from TXT file.

        Returns:
        --------
        dict:
            Metadata including line_count, encoding, file_size_bytes
        """
        try:
            result = self.extract(file_path, encoding, compute_hash=False)
            return result['metadata']
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return {
                'line_count': 0,
                'encoding': 'unknown',
                'file_size_bytes': 0
            }

    # =========================================================================
    # ENCODING DETECTION
    # =========================================================================

    def _detect_encoding(self, file_path: str) -> tuple[str, str]:
        """
        Detect file encoding using chardet or fallback strategies.

        Parameters:
        -----------
        file_path : str
            Path to file

        Returns:
        --------
        tuple[str, str]:
            (encoding, detection_method)
            Examples: ('utf-8', 'chardet'), ('latin-1', 'fallback')
        """
        # Strategy 1: Use chardet if available (best)
        if CHARDET_AVAILABLE:
            try:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()

                detection = chardet.detect(raw_data)
                encoding = detection.get('encoding', 'utf-8')
                confidence = detection.get('confidence', 0)

                if confidence > 0.7:  # High confidence
                    logger.debug(f"chardet detected: {encoding} (confidence: {confidence:.2f})")
                    return encoding.lower(), "chardet"
                else:
                    logger.debug(f"Low confidence ({confidence:.2f}), using fallback")

            except Exception as e:
                logger.debug(f"chardet detection failed: {e}, using fallback")

        # Strategy 2: Fallback - try common encodings
        fallback_encodings = ['utf-8', 'latin-1', 'windows-1252', 'ascii']

        for enc in fallback_encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    # Try to read first 1000 characters
                    f.read(1000)
                logger.debug(f"Fallback successful: {enc}")
                return enc, "fallback"

            except (UnicodeDecodeError, LookupError):
                continue

        # Strategy 3: Last resort - UTF-8 with error replacement
        logger.warning("All encodings failed, using UTF-8 with error replacement")
        return 'utf-8', "utf-8-fallback"


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of Phase 2 TXTProcessor.
    Run: python utils/file_parsers/txt_processor.py
    """
    import logging

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("Phase 2 TXTProcessor - Enhanced Plain Text Extraction")
    print("=" * 70)

    # Check available features
    print("\n1. Available Features")
    print("-" * 70)
    print(f"chardet: {'✅ Available' if CHARDET_AVAILABLE else '❌ Not installed (optional)'}")
    print("Note: chardet is optional but recommended for better encoding detection")
    print("Install with: pip install chardet")

    # Usage examples
    print("\n2. Phase 2 Usage Pattern")
    print("-" * 70)
    print("""
# Initialize processor
processor = TXTProcessor()

# Extract with Phase 2 format (auto-detect encoding)
result = processor.extract("document.txt")

if result['success']:
    print(f"Text: {len(result['text'])} characters")
    print(f"Lines: {result['metadata']['line_count']}")
    print(f"Encoding: {result['metadata']['encoding']}")
    print(f"Hash: {result['metadata']['file_hash']}")

    # Access per-line data
    for line in result['pages'][:5]:
        print(f"Line {line['page_num']}: {line['text'][:50]}...")
        print(f"  Position: {line['char_range']}")
else:
    print(f"Errors: {result['errors']}")

# Force specific encoding
result = processor.extract("document.txt", encoding="utf-8")

# Extract plain text (backward compatible)
text = processor.extract_text("document.txt")

# Extract lines separately
lines = processor.extract_lines("document.txt")

# Get metadata
metadata = processor.get_metadata("document.txt")
    """)

    # Encoding examples
    print("\n3. Encoding Detection Examples")
    print("-" * 70)
    print("""
# Auto-detect encoding (best for unknown files)
result = processor.extract("mystery.txt")
print(f"Detected: {result['metadata']['encoding']}")

# Force UTF-8 (if you know the encoding)
result = processor.extract("utf8_file.txt", encoding="utf-8")

# Force Latin-1 (for European languages)
result = processor.extract("european.txt", encoding="latin-1")

# Force Windows-1252 (for Windows files)
result = processor.extract("windows.txt", encoding="windows-1252")
    """)

    print("\n" + "=" * 70)
    print("✅ Phase 2 TXTProcessor ready to use!")
    print("=" * 70)
