"""

utils/file_parsers/pdf_processor.py

Phase 2 Enhanced PDF text extraction with multiple backend support and structured output.

What is This File?
-------------------
PDF processor with automatic backend selection that returns Phase 2 standardized
format with comprehensive metadata, page tracking, character ranges, and file hash.

Backends (in priority order):
1. PyMuPDF (fitz) - BEST quality, fastest, handles complex PDFs ⭐ RECOMMENDED
2. pdfplumber - Good quality, handles tables well
3. PyPDF2 - Basic fallback option

Phase 2 Features:
-----------------
- Standardized extract() method returning structured dict
- File hash computation for provenance tracking
- Character range tracking for citations
- Page-by-page extraction with position data
- Automatic backend selection with graceful fallbacks
- Comprehensive error handling

Installation:
-------------
pip install PyMuPDF        # Recommended (best quality)
pip install pdfplumber    # Alternative (good for tables)
pip install PyPDF2        # Fallback (basic)

Example Usage (Phase 2):
------------------------
processor = PDFProcessor()

# Extract with full Phase 2 metadata
result = processor.extract("document.pdf")

if result['success']:
    print(f"Text: {result['text'][:100]}...")
    print(f"Pages: {result['metadata']['page_count']}")
    print(f"Method: {result['metadata']['extraction_method']}")
    print(f"Hash: {result['metadata']['file_hash']}")
else:
    print(f"Errors: {result['errors']}")

References:
-----------
- Phase 2 Spec: Section 10.3 (Enhanced File Processors)
- PyMuPDF: https://pymupdf.readthedocs.io/
- pdfplumber: https://github.com/jsvine/pdfplumber
- PyPDF2: https://github.com/py-pdf/PyPDF2

"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import hashlib

# Try importing all available backends
try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import PyPDF2

    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

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
# PHASE 2 PDF PROCESSOR - MAIN CLASS
# =============================================================================

class PDFProcessor:
    """
    Phase 2 enhanced PDF processor with standardized output format.

    Automatically selects the best available backend for PDF extraction:
    1. PyMuPDF (fitz) - Best quality
    2. pdfplumber - Good quality for tables
    3. PyPDF2 - Basic fallback

    Returns structured Phase 2 format with metadata, page tracking, and file hash.

    Characteristics:
    ----------------
    - Multiple backend support with automatic selection
    - Graceful degradation on backend failures
    - Comprehensive page-level metadata
    - Character position tracking for citations
    - File hash computation for integrity
    - Complete error collection

    Example:
    --------
    # Initialize processor
    processor = PDFProcessor()

    # Extract with Phase 2 format
    result = processor.extract("document.pdf")

    # Check success
    if result['success']:
        print(f"Extracted: {len(result['text'])} characters")
        print(f"Pages: {len(result['pages'])}")
        print(f"Hash: {result['metadata']['file_hash']}")
    else:
        print(f"Errors: {result['errors']}")
    """

    def __init__(self, backend: str = "auto"):
        """
        Initialize PDF processor with backend selection.

        Parameters:
        -----------
        backend : str
            Backend to use: "auto", "pymupdf", "pdfplumber", "pypdf2"
            Default: "auto" (automatically selects best available)

        Raises:
        -------
        ImportError:
            If requested backend not available
        RuntimeError:
            If no PDF backends available at all
        """
        self.backend = backend.lower()

        # Auto-select best available backend
        if self.backend == "auto":
            if PYMUPDF_AVAILABLE:
                self.backend = "pymupdf"
                logger.info("✅ Auto-selected PyMuPDF backend (best quality)")
            elif PDFPLUMBER_AVAILABLE:
                self.backend = "pdfplumber"
                logger.info("✅ Auto-selected pdfplumber backend (good quality)")
            elif PYPDF2_AVAILABLE:
                self.backend = "pypdf2"
                logger.warning("⚠️  Auto-selected PyPDF2 backend (basic quality)")
                logger.warning("   For better quality, install: pip install PyMuPDF")
            else:
                raise RuntimeError(
                    "❌ No PDF processing library available!\n"
                    "Install one of:\n"
                    "  pip install PyMuPDF (recommended)\n"
                    "  pip install pdfplumber\n"
                    "  pip install PyPDF2"
                )

        # Validate requested backend is available
        if self.backend == "pymupdf" and not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF not installed. Install with: pip install PyMuPDF")
        elif self.backend == "pdfplumber" and not PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber not installed. Install with: pip install pdfplumber")
        elif self.backend == "pypdf2" and not PYPDF2_AVAILABLE:
            raise ImportError("PyPDF2 not installed. Install with: pip install PyPDF2")
        elif self.backend not in ["pymupdf", "pdfplumber", "pypdf2"]:
            raise ValueError(
                f"Unknown backend: {self.backend}\n"
                f"Valid options: 'auto', 'pymupdf', 'pdfplumber', 'pypdf2'"
            )

        logger.info(f"PDFProcessor initialized with backend: {self.backend}")

    def extract(
            self,
            file_path: str,
            password: Optional[str] = None,
            compute_hash: bool = True
    ) -> Dict[str, Any]:
        """
        Extract text and metadata from PDF (Phase 2 standardized format).

        This is the PRIMARY extraction method for Phase 2 service layer.
        Returns structured data with comprehensive metadata and error handling.

        Parameters:
        -----------
        file_path : str
            Path to PDF file
        password : str, optional
            Password for encrypted PDFs (PyPDF2 only)
        compute_hash : bool
            Whether to compute file hash (default: True)

        Returns:
        --------
        Dict[str, Any]:
            {
                'text': str,                    # Full extracted text
                'metadata': {
                    'page_count': int,
                    'file_hash': str,           # SHA-256 hash
                    'file_size_bytes': int,
                    'extraction_method': str,   # 'pymupdf', 'pdfplumber', 'pypdf2'
                    'title': Optional[str],
                    'author': Optional[str]
                },
                'pages': List[Dict],            # Per-page info
                'success': bool,                # Extraction successful
                'errors': List[str]             # Any warnings/errors
            }

        Page structure:
        ---------------
        'pages': [
            {
                'page_num': int,                # Page number (1-indexed)
                'text': str,                    # Page text
                'char_range': (start, end),     # Character positions
                'length': int                   # Page text length
            },
            ...
        ]

        Raises:
        -------
        FileNotFoundError:
            If PDF file doesn't exist
        RuntimeError:
            If extraction completely fails

        Example:
        --------
        processor = PDFProcessor()
        result = processor.extract("document.pdf")

        if result['success']:
            print(f"Pages: {result['metadata']['page_count']}")
            print(f"Method: {result['metadata']['extraction_method']}")

            for page in result['pages']:
                print(f"Page {page['page_num']}: {len(page['text'])} chars")
        else:
            print(f"Errors: {result['errors']}")
        """
        logger.info(f"Extracting PDF: {file_path} using {self.backend}")

        errors = []
        text = ""
        pages = []
        extraction_method = "unknown"
        doc_metadata = {}

        # Validate file exists
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        # Get file size
        try:
            file_size = file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Could not get file size: {e}")
            file_size = 0

        # Extract using selected backend
        try:
            if self.backend == "pymupdf":
                result = self._extract_pymupdf(str(file_path))
            elif self.backend == "pdfplumber":
                result = self._extract_pdfplumber(str(file_path))
            elif self.backend == "pypdf2":
                result = self._extract_pypdf2(str(file_path), password)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")

            text = result['text']
            pages = result['pages']
            doc_metadata = result['metadata']
            extraction_method = self.backend

            logger.info(f"✅ Extraction successful: {len(pages)} pages")

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            errors.append(f"Extraction failed: {str(e)}")
            raise RuntimeError(f"Failed to extract PDF: {e}")

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
            'text': text.strip(),
            'metadata': {
                'page_count': len(pages),
                'file_hash': file_hash,
                'file_size_bytes': file_size,
                'extraction_method': extraction_method,
                **doc_metadata  # Include title, author, etc. from backend
            },
            'pages': pages,
            'success': bool(text) and len(pages) > 0,
            'errors': errors
        }

        logger.info(
            f"✅ PDF extraction complete:\n"
            f"   Method: {extraction_method}\n"
            f"   Pages: {len(pages)}\n"
            f"   Text length: {len(text):,} chars\n"
            f"   Errors: {len(errors)}"
        )

        return result

    def extract_text(self, file_path: str, password: Optional[str] = None) -> str:
        """
        Extract plain text from PDF (backward compatibility).

        For Phase 2, use extract() method instead which returns structured data.

        Parameters:
        -----------
        file_path : str
            Path to PDF file
        password : str, optional
            Password for encrypted PDFs (PyPDF2 only)

        Returns:
        --------
        str:
            Extracted text from all pages
        """
        result = self.extract(file_path, password, compute_hash=False)
        return result['text']

    def extract_pages(self, file_path: str, password: Optional[str] = None) -> List[str]:
        """
        Extract text from each page separately.

        Parameters:
        -----------
        file_path : str
            Path to PDF file
        password : str, optional
            Password for encrypted PDFs

        Returns:
        --------
        List[str]:
            List of page texts

        Example:
        --------
        processor = PDFProcessor()
        pages = processor.extract_pages("document.pdf")

        for i, page_text in enumerate(pages, 1):
            print(f"Page {i}: {len(page_text)} characters")
        """
        result = self.extract(file_path, password, compute_hash=False)
        return [page['text'] for page in result['pages']]

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF.

        Parameters:
        -----------
        file_path : str
            Path to PDF file

        Returns:
        --------
        dict:
            Metadata including page_count, title, author
        """
        try:
            result = self.extract(file_path, compute_hash=False)
            return result['metadata']
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return {'page_count': 0, 'title': None, 'author': None}

    # =========================================================================
    # BACKEND IMPLEMENTATIONS
    # =========================================================================

    def _extract_pymupdf(self, file_path: str) -> Dict[str, Any]:
        """Extract using PyMuPDF (best quality) - BACKEND."""
        import fitz

        text_parts = []
        pages = []
        char_position = 0

        doc = fitz.open(file_path)

        try:
            for page_num, page in enumerate(doc, 1):
                try:
                    page_text = page.get_text()

                    if page_text.strip():
                        char_start = char_position
                        char_end = char_position + len(page_text)

                        pages.append({
                            'page_num': page_num,
                            'text': page_text.strip(),
                            'char_range': (char_start, char_end),
                            'length': len(page_text)
                        })

                        text_parts.append(page_text)
                        char_position = char_end + 2  # +2 for separator

                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num}: {e}")
                    continue

            metadata = doc.metadata or {}

            return {
                'text': "\n\n".join(text_parts),
                'pages': pages,
                'metadata': {
                    'title': metadata.get('title'),
                    'author': metadata.get('author')
                }
            }

        finally:
            doc.close()

    def _extract_pdfplumber(self, file_path: str) -> Dict[str, Any]:
        """Extract using pdfplumber (good quality) - BACKEND."""
        import pdfplumber

        text_parts = []
        pages = []
        char_position = 0

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()

                    if page_text and page_text.strip():
                        char_start = char_position
                        char_end = char_position + len(page_text)

                        pages.append({
                            'page_num': page_num,
                            'text': page_text.strip(),
                            'char_range': (char_start, char_end),
                            'length': len(page_text)
                        })

                        text_parts.append(page_text)
                        char_position = char_end + 2

                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num}: {e}")
                    continue

            metadata = pdf.metadata or {}

            return {
                'text': "\n\n".join(text_parts),
                'pages': pages,
                'metadata': {
                    'title': metadata.get('Title'),
                    'author': metadata.get('Author')
                }
            }

    def _extract_pypdf2(
            self,
            file_path: str,
            password: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract using PyPDF2 (basic quality) - BACKEND."""
        import PyPDF2

        text_parts = []
        pages = []
        char_position = 0

        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            # Handle encrypted PDFs
            if reader.is_encrypted:
                if password:
                    reader.decrypt(password)
                else:
                    logger.warning("PDF is encrypted but no password provided")
                    return {'text': '', 'pages': [], 'metadata': {}}

            # Extract each page
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()

                    if page_text and page_text.strip():
                        char_start = char_position
                        char_end = char_position + len(page_text)

                        pages.append({
                            'page_num': page_num,
                            'text': page_text.strip(),
                            'char_range': (char_start, char_end),
                            'length': len(page_text)
                        })

                        text_parts.append(page_text)
                        char_position = char_end + 2

                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num}: {e}")
                    continue

            # Extract metadata
            metadata = {}
            if reader.metadata:
                metadata['title'] = reader.metadata.get('/Title')
                metadata['author'] = reader.metadata.get('/Author')

            return {
                'text': "\n\n".join(text_parts),
                'pages': pages,
                'metadata': metadata
            }


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of Phase 2 PDFProcessor.
    Run: python utils/file_parsers/pdf_processor.py
    """
    import logging

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("Phase 2 PDFProcessor - Enhanced PDF Text Extraction")
    print("=" * 70)

    # Check available backends
    print("\n1. Available Backends")
    print("-" * 70)
    print(f"PyMuPDF (fitz):  {'✅ Available' if PYMUPDF_AVAILABLE else '❌ Not installed'}")
    print(f"pdfplumber:      {'✅ Available' if PDFPLUMBER_AVAILABLE else '❌ Not installed'}")
    print(f"PyPDF2:          {'✅ Available' if PYPDF2_AVAILABLE else '❌ Not installed'}")

    if not any([PYMUPDF_AVAILABLE, PDFPLUMBER_AVAILABLE, PYPDF2_AVAILABLE]):
        print("\n❌ No PDF libraries installed!")
        print("Install one of:")
        print("  pip install PyMuPDF (recommended)")
        print("  pip install pdfplumber")
        print("  pip install PyPDF2")
        exit(1)

    # Usage examples
    print("\n2. Phase 2 Usage Pattern")
    print("-" * 70)
    print("""
# Initialize processor
processor = PDFProcessor()  # Auto-selects best backend

# Extract with Phase 2 format
result = processor.extract("document.pdf")

if result['success']:
    print(f"Text: {len(result['text'])} characters")
    print(f"Pages: {result['metadata']['page_count']}")
    print(f"Method: {result['metadata']['extraction_method']}")
    print(f"Hash: {result['metadata']['file_hash']}")

    # Access per-page data
    for page in result['pages'][:3]:
        print(f"Page {page['page_num']}: {len(page['text'])} chars")
        print(f"  Position: {page['char_range']}")
else:
    print(f"Errors: {result['errors']}")

# Extract plain text (backward compatible)
text = processor.extract_text("document.pdf")

# Extract pages separately
pages = processor.extract_pages("document.pdf")

# Get metadata
metadata = processor.get_metadata("document.pdf")
    """)

    print("\n" + "=" * 70)
    print("✅ Phase 2 PDFProcessor ready to use!")
    print("=" * 70)
