"""
cli/ingest.py

Batch-ingest documents from a directory into a domain knowledge base.

Usage:
    python -m cli.ingest --domain hr --dir ./docs/hr_policies/ --uploader admin
    python -m cli.ingest --domain finance --dir ./docs/finance/ --uploader alice --replace
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.services.document_service import DocumentService, ValidationError, ProcessingError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cli.ingest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-ingest documents into a domain knowledge base."
    )
    parser.add_argument("--domain", required=True, help="Target domain (e.g. hr, finance)")
    parser.add_argument("--dir", required=True, help="Directory containing documents to ingest")
    parser.add_argument("--uploader", default="cli_ingest", help="Uploader ID for provenance")
    parser.add_argument(
        "--doc-type", default="document", help="Document type (policy, faq, manual, etc.)"
    )
    parser.add_argument(
        "--replace", action="store_true", help="Replace existing documents with same doc_id"
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=["pdf", "docx", "txt"],
        help="File extensions to process (default: pdf docx txt)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    doc_dir = Path(args.dir)
    if not doc_dir.is_dir():
        logger.error(f"Directory not found: {doc_dir}")
        sys.exit(1)

    # Collect matching files
    files = []
    for ext in args.extensions:
        files.extend(doc_dir.glob(f"*.{ext}"))
        files.extend(doc_dir.glob(f"*.{ext.upper()}"))

    if not files:
        logger.warning(f"No files found in {doc_dir} matching extensions: {args.extensions}")
        sys.exit(0)

    logger.info(f"Found {len(files)} file(s) to ingest into domain '{args.domain}'")

    # Initialize service once
    try:
        service = DocumentService(args.domain)
    except ValueError as e:
        logger.error(f"Failed to initialize service: {e}")
        sys.exit(1)

    success_count = 0
    failure_count = 0

    for file_path in sorted(files):
        doc_id = file_path.stem  # filename without extension as doc_id
        logger.info(f"Ingesting: {file_path.name}  (doc_id={doc_id})")

        try:
            with open(file_path, "rb") as f:
                # Attach the name attribute so the service can detect the file type
                f.name = str(file_path)
                result = service.upload_document(
                    file_obj=f,
                    metadata={
                        "doc_id": doc_id,
                        "title": file_path.stem.replace("_", " ").title(),
                        "doc_type": args.doc_type,
                        "uploader_id": args.uploader,
                    },
                    replace_existing=args.replace,
                )
            logger.info(
                f"  ✅ Ingested {result['chunks_ingested']} chunks  "
                f"(strategy={result.get('chunking_strategy')}, "
                f"model={result.get('embedding_model')})"
            )
            success_count += 1

        except ValidationError as e:
            logger.warning(f"  ⚠️  Skipped (validation): {e}")
            failure_count += 1

        except ProcessingError as e:
            logger.error(f"  ❌ Failed: {e}")
            failure_count += 1

        except Exception as e:
            logger.exception(f"  ❌ Unexpected error: {e}")
            failure_count += 1

    print()
    print(f"Ingestion complete: {success_count} succeeded, {failure_count} failed.")
    if failure_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
