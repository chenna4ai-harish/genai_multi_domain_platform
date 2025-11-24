"""

utils/file_parsers/parser_factory.py

Factory for automatically selecting the correct file parser based on file extension.

What is This Factory?
----------------------
The FileParserFactory automatically selects the appropriate parser (PDF, DOCX, TXT)
based on file extension. This simplifies file processing throughout the system.

Why Use a Factory?
------------------
- **Automatic Selection**: No need to manually choose parser
- **Clean Code**: Hides parser implementation details
- **Extensible**: Easy to add new file types
- **Type Safety**: Returns consistent interface for all parsers

Phase 2 Integration:
--------------------
DocumentService uses this factory to process uploaded files:
1. User uploads "handbook.pdf"
2. Factory returns PDFProcessor
3. Service calls processor.extract() with standardized output
4. Works the same for DOCX, TXT, or future formats!

Supported File Types:
---------------------
- PDF: .pdf (PDFProcessor)
- DOCX: .docx (DOCXProcessor)
- TXT: .txt (TXTProcessor)

Future Extensions:
------------------
Easy to add support for:
- HTML: .html, .htm
- Markdown: .md
- CSV: .csv
- JSON: .json
- Images with OCR: .jpg, .png

Example Usage:
--------------
from utils.file_parsers.parser_factory import FileParserFactory

# Automatic parser selection
parser = FileParserFactory.create_parser("document.pdf")

# Extract with Phase 2 format (all parsers return same structure)
result = parser.extract("document.pdf")

print(f"Text: {len(result['text'])} characters")
print(f"Pages: {result['metadata']['page_count']}")
print(f"Hash: {result['metadata']['file_hash']}")

References:
-----------
- Phase 2 Spec: Section 10 (File Processing & Validation)
- Factory Pattern: https://refactoring.guru/design-patterns/factory-method

"""

from typing import Union
from pathlib import Path
import logging

# Import file processors
from utils.file_parsers.pdf_processor import PDFProcessor
from utils.file_parsers.docx_processor import DOCXProcessor
from utils.file_parsers.txt_processor import TXTProcessor

# Configure logging
logger = logging.getLogger(__name__)


class UnsupportedFileTypeError(Exception):
    """
    Raised when file type is not supported.

    Use this exception to distinguish unsupported file types
    from other processing errors.
    """
    pass


class FileParserFactory:
    """
    Factory for creating file parser instances based on file extension.

    Automatically selects the appropriate parser (PDF, DOCX, TXT) based on
    file extension, simplifying file processing throughout the application.

    All parsers return the same Phase 2 standardized format:
    {
        'text': str,
        'metadata': {...},
        'pages': [...],
        'success': bool,
        'errors': [...]
    }

    Supported File Types:
    ---------------------
    - .pdf → PDFProcessor
    - .docx → DOCXProcessor
    - .txt → TXTProcessor

    Methods:
    --------
    - create_parser(filename): Create parser for file
    - get_supported_extensions(): Get list of supported extensions
    - is_supported(filename): Check if file type supported

    Example:
    --------
    # Process any file type
    def process_file(filepath):
        # Factory automatically selects parser
        parser = FileParserFactory.create_parser(filepath)

        # All parsers have same extract() method
        result = parser.extract(filepath)

        return result

    # Works for PDF
    pdf_result = process_file("handbook.pdf")

    # Works for DOCX (same calling code!)
    docx_result = process_file("policy.docx")

    # Works for TXT (same calling code!)
    txt_result = process_file("notes.txt")
    """

    # Mapping of file extensions to parser classes
    _PARSERS = {
        '.pdf': PDFProcessor,
        '.docx': DOCXProcessor,
        '.txt': TXTProcessor,
    }

    @staticmethod
    def create_parser(
            filename: str
    ) -> Union[PDFProcessor, DOCXProcessor, TXTProcessor]:
        """
        Create appropriate parser based on file extension.

        This is the PRIMARY factory method.
        Automatically selects parser class based on file extension.

        Parameters:
        -----------
        filename : str
            Filename or filepath (extension used for detection)
            Examples: "document.pdf", "./uploads/file.docx", "notes.txt"

        Returns:
        --------
        Union[PDFProcessor, DOCXProcessor, TXTProcessor]:
            Parser instance for the file type
            All parsers have .extract() method returning Phase 2 format

        Raises:
        -------
        UnsupportedFileTypeError:
            If file extension not supported

        Example:
        --------
        # PDF file
        pdf_parser = FileParserFactory.create_parser("handbook.pdf")
        result = pdf_parser.extract("handbook.pdf")

        # DOCX file
        docx_parser = FileParserFactory.create_parser("policy.docx")
        result = docx_parser.extract("policy.docx")

        # TXT file
        txt_parser = FileParserFactory.create_parser("notes.txt")
        result = txt_parser.extract("notes.txt")

        # Unsupported file
        try:
            parser = FileParserFactory.create_parser("image.jpg")
        except UnsupportedFileTypeError as e:
            print(f"Error: {e}")
        """
        # Extract extension (convert to lowercase)
        ext = Path(filename).suffix.lower()

        logger.debug(f"Creating parser for file: {filename} (extension: {ext})")

        # Check if extension supported
        if ext not in FileParserFactory._PARSERS:
            supported = ', '.join(FileParserFactory._PARSERS.keys())
            raise UnsupportedFileTypeError(
                f"Unsupported file type: '{ext}'\n"
                f"Supported types: {supported}\n"
                f"File: {filename}"
            )

        # Get parser class
        parser_class = FileParserFactory._PARSERS[ext]

        # Create and return parser instance
        parser = parser_class()

        logger.info(
            f"✅ Created {parser_class.__name__} for file: {filename}"
        )

        return parser

    @staticmethod
    def get_supported_extensions() -> list[str]:
        """
        Get list of supported file extensions.

        Returns:
        --------
        list[str]:
            List of supported extensions (with dots)
            Example: ['.pdf', '.docx', '.txt']

        Example:
        --------
        supported = FileParserFactory.get_supported_extensions()
        print(f"Supported file types: {', '.join(supported)}")
        # Output: "Supported file types: .pdf, .docx, .txt"

        # Use for validation
        filename = "document.pdf"
        ext = Path(filename).suffix.lower()

        if ext in FileParserFactory.get_supported_extensions():
            print("File type supported!")
        else:
            print("File type not supported!")
        """
        return list(FileParserFactory._PARSERS.keys())

    @staticmethod
    def is_supported(filename: str) -> bool:
        """
        Check if file type is supported.

        Parameters:
        -----------
        filename : str
            Filename or filepath

        Returns:
        --------
        bool:
            True if file type supported, False otherwise

        Example:
        --------
        # Check before processing
        if FileParserFactory.is_supported("document.pdf"):
            parser = FileParserFactory.create_parser("document.pdf")
            result = parser.extract("document.pdf")
        else:
            print("File type not supported")

        # Batch validation
        files = ["doc1.pdf", "doc2.docx", "image.jpg", "notes.txt"]

        for file in files:
            if FileParserFactory.is_supported(file):
                print(f"✅ {file}")
            else:
                print(f"❌ {file}")
        """
        ext = Path(filename).suffix.lower()
        return ext in FileParserFactory._PARSERS

    @staticmethod
    def extract(filename: str, **kwargs) -> dict:
        """
        Convenience method: Create parser and extract in one call.

        Combines create_parser() and parser.extract() for simple use cases.

        Parameters:
        -----------
        filename : str
            Path to file
        **kwargs : dict
            Additional arguments passed to parser.extract()
            Examples: include_tables=True, compute_hash=True

        Returns:
        --------
        dict:
            Extraction result in Phase 2 format

        Example:
        --------
        # Single-line extraction
        result = FileParserFactory.extract("document.pdf")

        print(f"Text: {result['text'][:100]}...")
        print(f"Pages: {result['metadata']['page_count']}")

        # With options
        result = FileParserFactory.extract(
            "document.docx",
            include_tables=True,
            compute_hash=True
        )
        """
        logger.debug(f"Extracting file: {filename}")

        # Create parser
        parser = FileParserFactory.create_parser(filename)

        # Extract and return
        result = parser.extract(filename, **kwargs)

        logger.info(
            f"✅ Extracted {filename}:\n"
            f"   Text length: {len(result['text']):,} characters\n"
            f"   Method: {result['metadata']['extraction_method']}"
        )

        return result


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of FileParserFactory usage.
    Run: python utils/file_parsers/parser_factory.py
    """
    import logging

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("FileParserFactory - Automatic File Parser Selection")
    print("=" * 70)

    # Example 1: Supported file types
    print("\n1. Supported File Types")
    print("-" * 70)

    supported = FileParserFactory.get_supported_extensions()
    print(f"Supported extensions: {', '.join(supported)}")

    # Example 2: Check if file supported
    print("\n2. File Type Detection")
    print("-" * 70)

    test_files = [
        "document.pdf",
        "policy.docx",
        "notes.txt",
        "image.jpg",
        "data.csv",
        "no_extension"
    ]

    for filename in test_files:
        is_supported = FileParserFactory.is_supported(filename)
        status = "✅ Supported" if is_supported else "❌ Not supported"
        print(f"{status}: {filename}")

    # Example 3: Create parsers
    print("\n3. Parser Creation")
    print("-" * 70)

    # PDF parser
    try:
        pdf_parser = FileParserFactory.create_parser("document.pdf")
        print(f"✅ Created: {type(pdf_parser).__name__} for PDF files")
    except UnsupportedFileTypeError as e:
        print(f"❌ {e}")

    # DOCX parser
    try:
        docx_parser = FileParserFactory.create_parser("policy.docx")
        print(f"✅ Created: {type(docx_parser).__name__} for DOCX files")
    except UnsupportedFileTypeError as e:
        print(f"❌ {e}")

    # TXT parser
    try:
        txt_parser = FileParserFactory.create_parser("notes.txt")
        print(f"✅ Created: {type(txt_parser).__name__} for TXT files")
    except UnsupportedFileTypeError as e:
        print(f"❌ {e}")

    # Unsupported file
    try:
        unsupported_parser = FileParserFactory.create_parser("image.jpg")
        print(f"✅ Created parser for JPG")
    except UnsupportedFileTypeError as e:
        print(f"❌ Unsupported: {e}")

    # Example 4: Polymorphic usage
    print("\n4. Polymorphic File Processing")
    print("-" * 70)

    print("""
def process_any_file(filepath):
    '''Process any supported file type.'''
    # Factory automatically selects right parser
    parser = FileParserFactory.create_parser(filepath)

    # All parsers have same extract() interface
    result = parser.extract(filepath)

    return result

# Works for PDF
pdf_result = process_any_file("handbook.pdf")

# Works for DOCX (same code!)
docx_result = process_any_file("policy.docx")

# Works for TXT (same code!)
txt_result = process_any_file("notes.txt")

# All results have same structure (Phase 2 format)
for result in [pdf_result, docx_result, txt_result]:
    print(f"Text: {len(result['text'])} chars")
    print(f"Pages: {result['metadata']['page_count']}")
    print(f"Hash: {result['metadata']['file_hash']}")
    """)

    # Example 5: Integration with DocumentService
    print("\n5. DocumentService Integration Pattern")
    print("-" * 70)

    print("""
class DocumentService:
    def upload_document(self, file_path, metadata):
        '''Upload and process any file type.'''

        # Step 1: Validate file type
        if not FileParserFactory.is_supported(file_path):
            raise ValueError("File type not supported")

        # Step 2: Extract text (factory selects parser)
        result = FileParserFactory.extract(file_path)

        # Step 3: Validate extraction
        if not result['success']:
            raise RuntimeError(f"Extraction failed: {result['errors']}")

        # Step 4: Process (same for all file types!)
        text = result['text']
        file_hash = result['metadata']['file_hash']

        # Chunk, embed, store...
        chunks = self.chunk_text(text)
        embeddings = self.embed_chunks(chunks)
        self.store_chunks(chunks, embeddings)

        return {
            'doc_id': metadata['doc_id'],
            'file_hash': file_hash,
            'chunks': len(chunks)
        }

# Service works with ANY file type!
service.upload_document("handbook.pdf", metadata)  # PDF
service.upload_document("policy.docx", metadata)   # DOCX
service.upload_document("notes.txt", metadata)     # TXT
    """)

    # Example 6: Adding new file types
    print("\n6. Extending to New File Types")
    print("-" * 70)

    print("""
To add support for new file types (e.g., HTML, Markdown):

1. Create new processor:
   # utils/file_parsers/html_processor.py
   class HTMLProcessor:
       def extract(self, file_path):
           # Return Phase 2 format
           return {
               'text': extracted_text,
               'metadata': {...},
               'pages': [...],
               'success': True,
               'errors': []
           }

2. Register in factory:
   # utils/file_parsers/parser_factory.py
   _PARSERS = {
       '.pdf': PDFProcessor,
       '.docx': DOCXProcessor,
       '.txt': TXTProcessor,
       '.html': HTMLProcessor,  # NEW!
       '.htm': HTMLProcessor,   # NEW!
   }

3. Done! No changes to calling code needed.
   factory.create_parser("page.html")  # Works automatically!
    """)

    print("\n" + "=" * 70)
    print("✅ FileParserFactory ready to use!")
    print("=" * 70)
    print("\nNext Steps:")
    print("- Use in DocumentService for file processing")
    print("- Add validation before parser creation")
    print("- Extend to new file types as needed")
