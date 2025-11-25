"""
utils/file_parsers/parser_factory.py

Factory for automatically selecting the correct file parser based on file extension.
Supports: PDF (.pdf), DOCX (.docx), TXT (.txt)
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class UnsupportedFileTypeError(Exception):
    """Raised when unsupported file format is detected."""

class FileParserFactory:
    """
    Factory for creating file parser instances based on file extension.

    This lets you call:
        parser = FileParserFactory.create_parser(filename)
        text = parser.extract_text(filename)
    """

    @staticmethod
    def _get_parsers():
        """
        Lazily imports parser classes to avoid circular import issues.
        Returns a dictionary mapping supported file extensions to parser classes.
        """
        from .pdf_processor import PDFProcessor
        from .docx_processor import DOCXProcessor
        from .txt_processor import TXTProcessor

        return {
            '.pdf': PDFProcessor,
            '.docx': DOCXProcessor,
            '.txt': TXTProcessor,
        }

    @staticmethod
    def create_parser(filename: str):
        """
        Create and return an appropriate parser instance for the given filename.
        Raises UnsupportedFileTypeError if the extension is not supported.
        """
        ext = Path(filename).suffix.lower()
        logger.debug(f"File extension detected: {ext}")
        parsers = FileParserFactory._get_parsers()
        if ext not in parsers:
            supported = ', '.join(parsers.keys())
            raise UnsupportedFileTypeError(
                f"Unsupported file type: '{ext}'. Supported types: {supported}. File: {filename}"
            )
        parser_class = parsers[ext]
        return parser_class()

    @staticmethod
    def get_supported_extensions():
        """Returns a list of supported file extensions."""
        return list(FileParserFactory._get_parsers().keys())

    @staticmethod
    def is_supported(filename: str) -> bool:
        """Returns True if the file extension is supported."""
        ext = Path(filename).suffix.lower()
        return ext in FileParserFactory._get_parsers()

def extract_text_from_file(file_path: str, file_type: str = None) -> str:
    """
    Extracts text from the given file using the appropriate parser.
    """
    # Determine file type from extension if not provided
    ext = file_type.lower() if file_type else Path(file_path).suffix.lower()
    logger.info(f"Extracting text from: {file_path} (type: {ext})")
    try:
        parser = FileParserFactory.create_parser(file_path)
        return parser.extract_text(file_path)
    except UnsupportedFileTypeError as e:
        logger.error(str(e))
        raise
