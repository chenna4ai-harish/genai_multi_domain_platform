"""
test_pdf_processor.py

Comprehensive test suite for PDFProcessor.

Run:
----
python test_pdf_processor.py

Or with a specific PDF:
python test_pdf_processor.py path/to/your/document.pdf
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.file_parsers.pdf_processor import (
    PDFProcessor,
    extract_pdf_text,
    PYMUPDF_AVAILABLE,
    PDFPLUMBER_AVAILABLE,
    PYPDF2_AVAILABLE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_backend_availability():
    """Test 1: Check which backends are available."""
    print("\n" + "=" * 70)
    print("TEST 1: Backend Availability")
    print("=" * 70)

    backends = {
        'PyMuPDF (fitz)': PYMUPDF_AVAILABLE,
        'pdfplumber': PDFPLUMBER_AVAILABLE,
        'PyPDF2': PYPDF2_AVAILABLE
    }

    for backend, available in backends.items():
        status = "✅ AVAILABLE" if available else "❌ NOT INSTALLED"
        print(f"{backend:20} {status}")

        if not available:
            pkg_name = backend.split()[0]
            print(f"  Install with: pip install {pkg_name}")

    if not any(backends.values()):
        print("\n❌ CRITICAL: No PDF backends installed!")
        return False

    print("\n✅ Test 1 PASSED: At least one backend is available")
    return True


def test_auto_backend_selection():
    """Test 2: Test automatic backend selection."""
    print("\n" + "=" * 70)
    print("TEST 2: Auto Backend Selection")
    print("=" * 70)

    try:
        processor = PDFProcessor(backend="auto")
        print(f"✅ Auto-selected backend: {processor.backend}")

        expected_backends = []
        if PYMUPDF_AVAILABLE:
            expected_backends.append("pymupdf")
        elif PDFPLUMBER_AVAILABLE:
            expected_backends.append("pdfplumber")
        elif PYPDF2_AVAILABLE:
            expected_backends.append("pypdf2")

        if processor.backend in expected_backends:
            print(f"✅ Correct backend selected: {processor.backend}")
            return True
        else:
            print(f"❌ Unexpected backend: {processor.backend}")
            return False

    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        return False


def test_specific_backends():
    """Test 3: Test each available backend individually."""
    print("\n" + "=" * 70)
    print("TEST 3: Individual Backend Initialization")
    print("=" * 70)

    backends_to_test = []
    if PYMUPDF_AVAILABLE:
        backends_to_test.append("pymupdf")
    if PDFPLUMBER_AVAILABLE:
        backends_to_test.append("pdfplumber")
    if PYPDF2_AVAILABLE:
        backends_to_test.append("pypdf2")

    all_passed = True

    for backend in backends_to_test:
        try:
            processor = PDFProcessor(backend=backend)
            print(f"✅ {backend:15} initialized successfully")
        except Exception as e:
            print(f"❌ {backend:15} failed: {e}")
            all_passed = False

    if all_passed:
        print("\n✅ Test 3 PASSED: All available backends initialize correctly")
    else:
        print("\n❌ Test 3 FAILED: Some backends failed to initialize")

    return all_passed


def test_pdf_extraction(pdf_path: str):
    """Test 4: Extract text from actual PDF file."""
    print("\n" + "=" * 70)
    print("TEST 4: PDF Text Extraction")
    print("=" * 70)

    if not Path(pdf_path).exists():
        print(f"⚠️  PDF file not found: {pdf_path}")
        print("Skipping extraction test")
        return None

    print(f"Testing with file: {pdf_path}")
    print(f"File size: {Path(pdf_path).stat().st_size / 1024:.2f} KB")

    results = {}

    # Test each available backend
    backends = []
    if PYMUPDF_AVAILABLE:
        backends.append("pymupdf")
    if PDFPLUMBER_AVAILABLE:
        backends.append("pdfplumber")
    if PYPDF2_AVAILABLE:
        backends.append("pypdf2")

    for backend in backends:
        print(f"\n--- Testing {backend} ---")

        try:
            import time
            start_time = time.time()

            processor = PDFProcessor(backend=backend)
            text = processor.extract_text(pdf_path)

            extraction_time = time.time() - start_time

            results[backend] = {
                'success': True,
                'text_length': len(text),
                'time': extraction_time,
                'preview': text[:200] if text else ""
            }

            print(f"✅ Extracted {len(text)} characters in {extraction_time:.3f}s")
            print(f"Preview: {text[:100]}...")

        except Exception as e:
            print(f"❌ {backend} extraction failed: {e}")
            results[backend] = {
                'success': False,
                'error': str(e)
            }

    # Compare results
    if len(results) > 1:
        print("\n--- Comparison ---")
        for backend, result in results.items():
            if result['success']:
                print(f"{backend:15} {result['text_length']:8d} chars  {result['time']:.3f}s")

    return results


def test_page_extraction(pdf_path: str):
    """Test 5: Extract pages individually."""
    print("\n" + "=" * 70)
    print("TEST 5: Page-by-Page Extraction")
    print("=" * 70)

    if not Path(pdf_path).exists():
        print(f"⚠️  PDF file not found: {pdf_path}")
        return None

    try:
        processor = PDFProcessor()  # Auto-select backend
        pages = processor.extract_pages(pdf_path)

        print(f"✅ Extracted {len(pages)} pages")

        for i, page_text in enumerate(pages[:3], 1):  # Show first 3 pages
            print(f"\nPage {i}: {len(page_text)} characters")
            print(f"Preview: {page_text[:100]}...")

        if len(pages) > 3:
            print(f"\n... and {len(pages) - 3} more pages")

        return True

    except Exception as e:
        print(f"❌ Page extraction failed: {e}")
        return False


def test_metadata_extraction(pdf_path: str):
    """Test 6: Extract PDF metadata."""
    print("\n" + "=" * 70)
    print("TEST 6: Metadata Extraction")
    print("=" * 70)

    if not Path(pdf_path).exists():
        print(f"⚠️  PDF file not found: {pdf_path}")
        return None

    try:
        processor = PDFProcessor()
        metadata = processor.get_metadata(pdf_path)

        print("✅ Metadata extracted:")
        print(f"  Pages:         {metadata['page_count']}")
        print(f"  Title:         {metadata['title'] or 'N/A'}")
        print(f"  Author:        {metadata['author'] or 'N/A'}")
        print(f"  Creation Date: {metadata['creation_date'] or 'N/A'}")

        return True

    except Exception as e:
        print(f"❌ Metadata extraction failed: {e}")
        return False


def test_convenience_function(pdf_path: str):
    """Test 7: Test the convenience function."""
    print("\n" + "=" * 70)
    print("TEST 7: Convenience Function")
    print("=" * 70)

    if not Path(pdf_path).exists():
        print(f"⚠️  PDF file not found: {pdf_path}")
        return None

    try:
        # Test quick extraction function
        text = extract_pdf_text(pdf_path)

        print(f"✅ extract_pdf_text() works!")
        print(f"  Extracted: {len(text)} characters")
        print(f"  Preview: {text[:100]}...")

        return True

    except Exception as e:
        print(f"❌ Convenience function failed: {e}")
        return False


def run_all_tests(pdf_path: str = None):
    """Run all tests."""
    print("\n" + "=" * 70)
    print("PDF PROCESSOR - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    test_results = []

    # Tests that don't require a PDF file
    test_results.append(("Backend Availability", test_backend_availability()))
    test_results.append(("Auto Backend Selection", test_auto_backend_selection()))
    test_results.append(("Individual Backends", test_specific_backends()))

    # Tests that require a PDF file
    if pdf_path:
        test_results.append(("PDF Text Extraction", test_pdf_extraction(pdf_path)))
        test_results.append(("Page Extraction", test_page_extraction(pdf_path)))
        test_results.append(("Metadata Extraction", test_metadata_extraction(pdf_path)))
        test_results.append(("Convenience Function", test_convenience_function(pdf_path)))
    else:
        print("\n" + "=" * 70)
        print("⚠️  No PDF file provided - skipping extraction tests")
        print("Run with: python test_pdf_processor.py path/to/file.pdf")
        print("=" * 70)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in test_results if result is True)
    total = len([r for _, r in test_results if r is not None])

    for test_name, result in test_results:
        if result is None:
            status = "⏭️  SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"

        print(f"{test_name:30} {status}")

    print("-" * 70)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! PDF processor is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check output above for details.")

    print("=" * 70)


if __name__ == "__main__":
    # Get PDF path from command line or use None
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None

    if pdf_path and not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        print("\nUsage: python test_pdf_processor.py [path/to/file.pdf]")
        sys.exit(1)

    # Run all tests
    run_all_tests(pdf_path)
