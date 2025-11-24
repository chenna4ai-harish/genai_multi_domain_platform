"""
File parsers utilities for the multi-domain platform.

This package provides text extraction functions for:
- PDF files (using PyMuPDF/PyPDF2)
- DOCX files (using python-docx)
- TXT files (plain text)
"""

"""
File parsers utilities for the multi-domain platform.
"""

# Import the main extraction function from parser_factory
from .parser_factory import extract_text_from_file

__all__ = ['extract_text_from_file']

