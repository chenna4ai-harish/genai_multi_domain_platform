"""
utils/file_parsers/pdf_processor.py

PDF text extraction with multiple backend support.

Backends (in order of quality):
1. PyMuPDF (fitz) - BEST quality, fastest, handles complex PDFs ⭐ RECOMMENDED
2. pdfplumber - Good quality, handles tables well
3. PyPDF2 - Basic, fallback option

Installation:
-------------
pip install PyMuPDF              # Recommended (best quality)
pip install pdfplumber           # Alternative (good for tables)
pip install PyPDF2               # Fallback (basic)

Usage:
------
from utils.file_parsers.pdf_processor import PDFProcessor

# Use PyMuPDF (best)
processor = PDFProcessor(backend="pymupdf")
text = processor.extract_text("document.pdf")

# Or let it auto-select best available
processor = PDFProcessor()  # Defaults to pymupdf if available
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

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


class PDFProcessor:
    """
    PDF text extraction with automatic backend selection.

    Automatically uses the best available backend:
    1. PyMuPDF (if installed) - Best quality
    2. pdfplumber (if installed) - Good quality
    3. PyPDF2 (if installed) - Basic fallback

    Example:
    --------
    # Auto-select best backend
    processor = PDFProcessor()
    text = processor.extract_text("file.pdf")

    # Force specific backend
    processor = PDFProcessor(backend="pymupdf")
    text = processor.extract_text("file.pdf")

    # Extract page by page
    pages = processor.extract_pages("file.pdf")
    for i, page_text in enumerate(pages, 1):
        print(f"Page {i}: {page_text[:100]}...")

    # Get metadata
    metadata = processor.get_metadata("file.pdf")
    print(f"Pages: {metadata['page_count']}")
    """

    def __init__(self, backend: str = "auto"):
        """
        Initialize PDF processor.

        Parameters:
        -----------
        backend : str
            Backend to use: "auto", "pymupdf", "pdfplumber", "pypdf2"
            "auto" automatically selects best available (recommended)

        Raises:
        -------
        ImportError:
            If requested backend is not installed
        RuntimeError:
            If no PDF backends are available
        """
        self.backend = backend.lower()

        # Auto-select best available backend
        if self.backend == "auto":
            if PYMUPDF_AVAILABLE:
                self.backend = "pymupdf"
                logger.info("Auto-selected PyMuPDF backend (best quality)")
            elif PDFPLUMBER_AVAILABLE:
                self.backend = "pdfplumber"
                logger.info("Auto-selected pdfplumber backend (good quality)")
            elif PYPDF2_AVAILABLE:
                self.backend = "pypdf2"
                logger.warning("Auto-selected PyPDF2 backend (basic quality)")
                logger.warning("Install PyMuPDF for better quality: pip install PyMuPDF")
            else:
                raise RuntimeError(
                    "No PDF processing library available!\n"
                    "Install one of:\n"
                    "  pip install PyMuPDF (recommended)\n"
                    "  pip install pdfplumber\n"
                    "  pip install PyPDF2"
                )

        # Validate requested backend is available
        if self.backend == "pymupdf" and not PYMUPDF_AVAILABLE:
            raise ImportError(
                "PyMuPDF not installed.\n"
                "Install with: pip install PyMuPDF"
            )
        elif self.backend == "pdfplumber" and not PDFPLUMBER_AVAILABLE:
            raise ImportError(
                "pdfplumber not installed.\n"
                "Install with: pip install pdfplumber"
            )
        elif self.backend == "pypdf2" and not PYPDF2_AVAILABLE:
            raise ImportError(
                "PyPDF2 not installed.\n"
                "Install with: pip install PyPDF2"
            )
        elif self.backend not in ["pymupdf", "pdfplumber", "pypdf2"]:
            raise ValueError(
                f"Unknown backend: {self.backend}\n"
                f"Valid options: 'auto', 'pymupdf', 'pdfplumber', 'pypdf2'"
            )

        logger.info(f"PDFProcessor initialized with backend: {self.backend}")

    def extract_text(self, file_path: str, password: Optional[str] = None) -> str:
        """
        Extract all text from PDF.

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

        Raises:
        -------
        FileNotFoundError:
            If PDF file doesn't exist
        RuntimeError:
            If extraction fails

        Example:
        --------
        processor = PDFProcessor()
        text = processor.extract_text("document.pdf")
        print(f"Extracted {len(text)} characters")
        """
        # Validate file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.debug(f"Extracting text from {file_path} using {self.backend}")

        try:
            if self.backend == "pymupdf":
                return self._extract_text_pymupdf(file_path)
            elif self.backend == "pdfplumber":
                return self._extract_text_pdfplumber(file_path)
            elif self.backend == "pypdf2":
                return self._extract_text_pypdf2(file_path, password)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")

        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise RuntimeError(f"Failed to extract text from PDF: {e}")

    def extract_pages(self, file_path: str, password: Optional[str] = None) -> List[str]:
        """
        Extract text from each page separately.

        Parameters:
        -----------
        file_path : str
            Path to PDF file
        password : str, optional
            Password for encrypted PDFs (PyPDF2 only)

        Returns:
        --------
        List[str]:
            List of text strings, one per page

        Example:
        --------
        processor = PDFProcessor()
        pages = processor.extract_pages("doc.pdf")

        for i, page_text in enumerate(pages, 1):
            print(f"Page {i}: {len(page_text)} characters")
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.debug(f"Extracting pages from {file_path} using {self.backend}")

        try:
            if self.backend == "pymupdf":
                return self._extract_pages_pymupdf(file_path)
            elif self.backend == "pdfplumber":
                return self._extract_pages_pdfplumber(file_path)
            elif self.backend == "pypdf2":
                return self._extract_pages_pypdf2(file_path, password)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")

        except Exception as e:
            logger.error(f"Page extraction failed: {e}")
            raise RuntimeError(f"Failed to extract pages from PDF: {e}")

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
            Metadata with keys:
            - page_count: Number of pages
            - title: Document title (if available)
            - author: Document author (if available)
            - creation_date: Creation date (if available)

        Example:
        --------
        processor = PDFProcessor()
        metadata = processor.get_metadata("doc.pdf")
        print(f"Title: {metadata['title']}")
        print(f"Pages: {metadata['page_count']}")
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            if self.backend == "pymupdf":
                return self._get_metadata_pymupdf(file_path)
            elif self.backend == "pdfplumber":
                return self._get_metadata_pdfplumber(file_path)
            elif self.backend == "pypdf2":
                return self._get_metadata_pypdf2(file_path)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")

        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return {
                'page_count': 0,
                'title': None,
                'author': None,
                'creation_date': None
            }

    # =========================================================================
    # PyMuPDF (fitz) Backend - BEST QUALITY ⭐
    # =========================================================================

    def _extract_text_pymupdf(self, file_path: str) -> str:
        """Extract text using PyMuPDF (best quality)."""
        import fitz

        text = ""
        doc = fitz.open(file_path)

        for page_num, page in enumerate(doc, 1):
            try:
                page_text = page.get_text()
                if page_text.strip():
                    text += page_text + "\n\n"
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num}: {e}")
                continue

        doc.close()

        if not text.strip():
            logger.warning(f"No text extracted from {file_path}")

        return text.strip()

    def _extract_pages_pymupdf(self, file_path: str) -> List[str]:
        """Extract pages using PyMuPDF."""
        import fitz

        pages = []
        doc = fitz.open(file_path)

        for page in doc:
            try:
                page_text = page.get_text()
                pages.append(page_text.strip())
            except Exception as e:
                logger.warning(f"Failed to extract page: {e}")
                pages.append("")

        doc.close()
        return pages

    def _get_metadata_pymupdf(self, file_path: str) -> Dict[str, Any]:
        """Get metadata using PyMuPDF."""
        import fitz

        doc = fitz.open(file_path)
        metadata = doc.metadata

        result = {
            'page_count': len(doc),
            'title': metadata.get('title', None),
            'author': metadata.get('author', None),
            'creation_date': metadata.get('creationDate', None)
        }

        doc.close()
        return result

    # =========================================================================
    # pdfplumber Backend - GOOD QUALITY
    # =========================================================================

    def _extract_text_pdfplumber(self, file_path: str) -> str:
        """Extract text using pdfplumber."""
        import pdfplumber

        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num}: {e}")
                    continue

        return text.strip()

    def _extract_pages_pdfplumber(self, file_path: str) -> List[str]:
        """Extract pages using pdfplumber."""
        import pdfplumber

        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                try:
                    page_text = page.extract_text()
                    pages.append(page_text.strip() if page_text else "")
                except Exception as e:
                    logger.warning(f"Failed to extract page: {e}")
                    pages.append("")

        return pages

    def _get_metadata_pdfplumber(self, file_path: str) -> Dict[str, Any]:
        """Get metadata using pdfplumber."""
        import pdfplumber

        with pdfplumber.open(file_path) as pdf:
            return {
                'page_count': len(pdf.pages),
                'title': pdf.metadata.get('Title', None),
                'author': pdf.metadata.get('Author', None),
                'creation_date': pdf.metadata.get('CreationDate', None)
            }

    # =========================================================================
    # PyPDF2 Backend - BASIC FALLBACK
    # =========================================================================

    def _extract_text_pypdf2(self, file_path: str, password: Optional[str] = None) -> str:
        """Extract text using PyPDF2 (basic quality)."""
        import PyPDF2

        text = ""

        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            # Handle encrypted PDFs
            if reader.is_encrypted:
                if password:
                    reader.decrypt(password)
                else:
                    logger.warning("PDF is encrypted but no password provided")
                    return ""

            # Extract text from all pages
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num}: {e}")
                    continue

        return text.strip()

    def _extract_pages_pypdf2(self, file_path: str, password: Optional[str] = None) -> List[str]:
        """Extract pages using PyPDF2."""
        import PyPDF2

        pages = []

        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            # Handle encrypted PDFs
            if reader.is_encrypted:
                if password:
                    reader.decrypt(password)
                else:
                    logger.warning("PDF is encrypted but no password provided")
                    return []

            # Extract each page
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    pages.append(page_text.strip() if page_text else "")
                except Exception as e:
                    logger.warning(f"Failed to extract page: {e}")
                    pages.append("")

        return pages

    def _get_metadata_pypdf2(self, file_path: str) -> Dict[str, Any]:
        """Get metadata using PyPDF2."""
        import PyPDF2

        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            metadata = {
                'page_count': len(reader.pages),
                'title': None,
                'author': None,
                'creation_date': None
            }

            # Extract metadata if available
            if reader.metadata:
                metadata['title'] = reader.metadata.get('/Title', None)
                metadata['author'] = reader.metadata.get('/Author', None)
                metadata['creation_date'] = reader.metadata.get('/CreationDate', None)

            return metadata


# =============================================================================
# Convenience function for quick usage
# =============================================================================

def extract_pdf_text(file_path: str, backend: str = "auto") -> str:
    """
    Quick function to extract text from PDF.

    Parameters:
    -----------
    file_path : str
        Path to PDF file
    backend : str
        Backend to use (default: "auto" - selects best available)

    Returns:
    --------
    str:
        Extracted text

    Example:
    --------
    from utils.file_parsers.pdf_processor import extract_pdf_text

    text = extract_pdf_text("document.pdf")
    print(text)
    """
    processor = PDFProcessor(backend=backend)
    return processor.extract_text(file_path)


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of PDFProcessor.
    Run: python utils/file_parsers/pdf_processor.py
    """

    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("PDFProcessor - Multi-Backend PDF Text Extraction")
    print("=" * 70)

    # Check available backends
    print("\n1. Available Backends")
    print("-" * 70)
    print(f"PyMuPDF (fitz):  {'✅ Available' if PYMUPDF_AVAILABLE else '❌ Not installed'}")
    print(f"pdfplumber:      {'✅ Available' if PDFPLUMBER_AVAILABLE else '❌ Not installed'}")
    print(f"PyPDF2:          {'✅ Available' if PYPDF2_AVAILABLE else '❌ Not installed'}")

    if not any([PYMUPDF_AVAILABLE, PDFPLUMBER_AVAILABLE, PYPDF2_AVAILABLE]):
        print("\n❌ No PDF libraries installed!")
        print("\nInstall one of:")
        print("  pip install PyMuPDF (recommended)")
        print("  pip install pdfplumber")
        print("  pip install PyPDF2")
        exit(1)

    # Example usage
    print("\n2. Usage Examples")
    print("-" * 70)

    print("""
# Auto-select best backend
processor = PDFProcessor()  # or PDFProcessor(backend="auto")
text = processor.extract_text("document.pdf")

# Force specific backend
processor = PDFProcessor(backend="pymupdf")
text = processor.extract_text("document.pdf")

# Extract page by page
pages = processor.extract_pages("document.pdf")
for i, page_text in enumerate(pages, 1):
    print(f"Page {i}: {page_text[:100]}...")

# Get metadata
metadata = processor.get_metadata("document.pdf")
print(f"Pages: {metadata['page_count']}")
print(f"Title: {metadata['title']}")

# Quick extraction
from utils.file_parsers.pdf_processor import extract_pdf_text
text = extract_pdf_text("document.pdf")
    """)

    print("\n" + "=" * 70)
    print("PDFProcessor ready to use!")
    print("=" * 70)
