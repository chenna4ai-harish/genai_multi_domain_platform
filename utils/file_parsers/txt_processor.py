"""
utils/file_parsers/txt_processor.py

This module provides utilities for extracting text from plain text files.

What Does This Do?
------------------
Handles plain text files (.txt) with different encodings.
Simplest processor - just reads and decodes text files.

Features:
---------
- Auto-detect encoding (UTF-8, Latin-1, Windows-1252, etc.)
- Handle different line endings (Windows, Unix, Mac)
- Strip or preserve whitespace

No External Dependencies Required!

Example Usage:
--------------
from utils.file_parsers.txt_processor import TXTProcessor

processor = TXTProcessor()
text = processor.extract_text("document.txt")
print(f"Extracted {len(text)} characters")
"""

from typing import List, Dict, Any
from pathlib import Path
import logging
import chardet  # For encoding detection (optional)

# Configure logging
logger = logging.getLogger(__name__)


class TXTProcessor:
    """
    Plain text file processor with encoding detection.

    Handles text files with various encodings and formats.

    Features:
    ---------
    - Auto-detect encoding (UTF-8, Latin-1, etc.)
    - Handle different line endings
    - Extract line by line
    - Get basic statistics

    Example:
    --------
    processor = TXTProcessor()

    # Extract all text
    text = processor.extract_text("notes.txt")

    # Extract lines
    lines = processor.extract_lines("notes.txt")
    for i, line in enumerate(lines, 1):
        print(f"{i}. {line}")
    """

    def __init__(self, default_encoding: str = "utf-8"):
        """
        Initialize TXT processor.

        Parameters:
        -----------
        default_encoding : str
            Default encoding to try first (default: "utf-8")
        """
        self.default_encoding = default_encoding
        logger.info(f"TXTProcessor initialized with encoding: {default_encoding}")

    def extract_text(self, file_path: str, encoding: str = None) -> str:
        """
        Extract text from a plain text file.

        Parameters:
        -----------
        file_path : str
            Path to text file
        encoding : str, optional
            File encoding (auto-detected if not provided)

        Returns:
        --------
        str:
            File content as string

        Example:
        --------
        text = processor.extract_text("document.txt")
        print(f"Length: {len(text)} characters")
        """
        try:
            # If encoding not specified, try to detect
            if encoding is None:
                encoding = self._detect_encoding(file_path)

            # Read file with detected/specified encoding
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()

            return text.strip()

        except UnicodeDecodeError:
            # Try fallback encodings
            fallback_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

            for enc in fallback_encodings:
                if enc == encoding:
                    continue  # Already tried this one

                try:
                    with open(file_path, 'r', encoding=enc) as f:
                        text = f.read()
                    logger.info(f"Successfully read {file_path} with encoding: {enc}")
                    return text.strip()
                except:
                    continue

            # If all encodings fail, raise error
            raise RuntimeError(
                f"Failed to decode {file_path} with any encoding. "
                f"Tried: {fallback_encodings}"
            )

        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            raise RuntimeError(f"TXT extraction failed: {e}")

    def extract_lines(self, file_path: str, encoding: str = None) -> List[str]:
        """
        Extract lines from text file.

        Parameters:
        -----------
        file_path : str
            Path to text file
        encoding : str, optional
            File encoding (auto-detected if not provided)

        Returns:
        --------
        List[str]:
            List of lines (stripped of whitespace)

        Example:
        --------
        lines = processor.extract_lines("document.txt")
        for i, line in enumerate(lines, 1):
            print(f"{i}. {line}")
        """
        text = self.extract_text(file_path, encoding)
        return [line.strip() for line in text.split('\n') if line.strip()]

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get basic statistics about text file.

        Returns:
        --------
        dict:
            Metadata including:
            - line_count: Number of lines
            - char_count: Number of characters
            - word_count: Number of words (approximate)
            - encoding: Detected encoding

        Example:
        --------
        metadata = processor.get_metadata("doc.txt")
        print(f"Lines: {metadata['line_count']}")
        print(f"Words: {metadata['word_count']}")
        """
        try:
            encoding = self._detect_encoding(file_path)
            text = self.extract_text(file_path, encoding)
            lines = text.split('\n')
            words = text.split()

            return {
                'line_count': len(lines),
                'char_count': len(text),
                'word_count': len(words),
                'encoding': encoding
            }

        except Exception as e:
            logger.error(f"Failed to get metadata from {file_path}: {e}")
            return {'line_count': 0, 'char_count': 0, 'word_count': 0, 'encoding': None}

    def _detect_encoding(self, file_path: str) -> str:
        """
        Detect file encoding.

        Uses chardet library if available, otherwise assumes UTF-8.

        Parameters:
        -----------
        file_path : str
            Path to file

        Returns:
        --------
        str:
            Detected encoding name
        """
        try:
            # Try using chardet for accurate detection
            import chardet

            with open(file_path, 'rb') as f:
                raw_data = f.read()

            result = chardet.detect(raw_data)
            encoding = result['encoding']

            logger.debug(f"Detected encoding for {file_path}: {encoding} (confidence: {result['confidence']})")

            return encoding if encoding else self.default_encoding

        except ImportError:
            # chardet not available, use default
            logger.debug(f"chardet not available, using default encoding: {self.default_encoding}")
            return self.default_encoding
        except Exception as e:
            logger.warning(f"Encoding detection failed for {file_path}: {e}")
            return self.default_encoding


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of TXTProcessor usage.
    Run: python utils/file_parsers/txt_processor.py
    """

    import logging
    import tempfile

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("TXTProcessor Usage Examples")
    print("=" * 70)

    # Example 1: Create test file
    print("\n1. Creating Test File")
    print("-" * 70)

    test_content = """This is a test document.
It has multiple lines.
And some special characters: café, naïve, résumé

This is paragraph two with more content.
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        test_file = f.name

    print(f"Created test file: {test_file}")

    # Example 2: Extract text
    print("\n2. Extracting Text")
    print("-" * 70)

    processor = TXTProcessor()
    text = processor.extract_text(test_file)

    print(f"Extracted text ({len(text)} characters):")
    print(text[:200] + "..." if len(text) > 200 else text)

    # Example 3: Extract lines
    print("\n3. Extracting Lines")
    print("-" * 70)

    lines = processor.extract_lines(test_file)
    print(f"Total lines: {len(lines)}")
    for i, line in enumerate(lines[:5], 1):
        print(f"{i}. {line}")

    # Example 4: Get metadata
    print("\n4. Getting Metadata")
    print("-" * 70)

    metadata = processor.get_metadata(test_file)
    print(f"Lines: {metadata['line_count']}")
    print(f"Characters: {metadata['char_count']}")
    print(f"Words: {metadata['word_count']}")
    print(f"Encoding: {metadata['encoding']}")

    # Cleanup
    import os

    os.unlink(test_file)

    print("\n" + "=" * 70)
    print("TXTProcessor examples completed!")
    print("=" * 70)
