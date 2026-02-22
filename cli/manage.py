"""
cli/manage.py

Domain and document management CLI: list, deprecate, delete, info.

Usage:
    python -m cli.manage list-docs   --domain hr
    python -m cli.manage list-chunks --domain hr --doc-id handbook_2025
    python -m cli.manage info        --domain hr --doc-id handbook_2025
    python -m cli.manage deprecate   --domain hr --doc-id handbook_2023 --reason "Superseded by 2025 version"
    python -m cli.manage delete      --domain hr --doc-id old_policy
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.services.document_service import (
    DocumentService,
    ValidationError,
    ProcessingError,
    DocumentNotFoundError,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cli.manage")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Domain and document management for the Multi-Domain RAG Platform."
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # list-docs
    p_list = subparsers.add_parser("list-docs", help="List all documents in a domain")
    p_list.add_argument("--domain", required=True)
    p_list.add_argument("--include-deprecated", action="store_true")
    p_list.add_argument("--json", action="store_true")

    # list-chunks
    p_chunks = subparsers.add_parser("list-chunks", help="List chunks for a document")
    p_chunks.add_argument("--domain", required=True)
    p_chunks.add_argument("--doc-id", required=True)
    p_chunks.add_argument("--limit", type=int, default=None)
    p_chunks.add_argument("--json", action="store_true")

    # info
    p_info = subparsers.add_parser("info", help="Get document info and statistics")
    p_info.add_argument("--domain", required=True)
    p_info.add_argument("--doc-id", required=True)
    p_info.add_argument("--json", action="store_true")

    # deprecate
    p_dep = subparsers.add_parser("deprecate", help="Mark a document as deprecated")
    p_dep.add_argument("--domain", required=True)
    p_dep.add_argument("--doc-id", required=True)
    p_dep.add_argument("--reason", required=True, help="Reason for deprecation")
    p_dep.add_argument("--superseded-by", default=None, help="doc_id of the replacement document")

    # delete
    p_del = subparsers.add_parser("delete", help="Permanently delete a document")
    p_del.add_argument("--domain", required=True)
    p_del.add_argument("--doc-id", required=True)
    p_del.add_argument("--confirm", action="store_true", help="Confirm destructive deletion")

    return parser.parse_args()


def get_service(domain: str) -> DocumentService:
    try:
        return DocumentService(domain)
    except ValueError as e:
        print(f"Error initializing service for domain '{domain}': {e}", file=sys.stderr)
        sys.exit(1)


def cmd_list_docs(args) -> None:
    service = get_service(args.domain)
    filters = {} if args.include_deprecated else {"deprecated": False}
    try:
        docs = service.list_documents(filters=filters if not args.include_deprecated else None)
    except ProcessingError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(docs, indent=2, default=str))
        return

    if not docs:
        print(f"No documents found in domain '{args.domain}'.")
        return

    print(f"\nDocuments in domain '{args.domain}' ({len(docs)} total):\n")
    print(f"{'doc_id':<30}  {'title':<35}  {'chunks':>6}  {'deprecated':>10}  {'uploaded'}")
    print("-" * 100)
    for d in docs:
        dep = "YES" if d.get("deprecated") else "no"
        ts = (d.get("first_seen") or "")[:19]
        print(
            f"{d.get('doc_id', ''):<30}  "
            f"{(d.get('title') or ''):<35}  "
            f"{d.get('chunk_count', 0):>6}  "
            f"{dep:>10}  "
            f"{ts}"
        )
    print()


def cmd_list_chunks(args) -> None:
    service = get_service(args.domain)
    try:
        chunks = service.list_chunks(doc_id=args.doc_id, limit=args.limit)
    except (DocumentNotFoundError, ProcessingError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(chunks, indent=2, default=str))
        return

    print(f"\nChunks for doc_id='{args.doc_id}' ({len(chunks)} total):\n")
    for i, c in enumerate(chunks, 1):
        text_preview = (c.get("text") or "")[:120].replace("\n", " ")
        print(f"[{i:3}]  id={c.get('id','?')[:16]}...  page={c.get('page_num','?')}  {text_preview}")
    print()


def cmd_info(args) -> None:
    service = get_service(args.domain)
    try:
        info = service.get_document_info(args.doc_id)
    except (ProcessingError, DocumentNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(info, indent=2, default=str))
        return

    print(f"\nDocument Info: {args.doc_id}\n")
    for k, v in info.items():
        print(f"  {k:<25} {v}")
    print()


def cmd_deprecate(args) -> None:
    service = get_service(args.domain)
    try:
        result = service.deprecate_document(
            doc_id=args.doc_id,
            reason=args.reason,
            superseded_by=args.superseded_by,
        )
    except (ValidationError, DocumentNotFoundError, ProcessingError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"✅ Deprecated '{args.doc_id}': "
        f"{result['chunks_deprecated']} chunks marked deprecated "
        f"(date={result['deprecated_date']})"
    )


def cmd_delete(args) -> None:
    if not args.confirm:
        print(
            f"⚠️  This will permanently delete ALL chunks for doc_id='{args.doc_id}' "
            f"from domain '{args.domain}'.\n"
            f"Re-run with --confirm to proceed."
        )
        sys.exit(0)

    service = get_service(args.domain)
    try:
        service.delete_document(args.doc_id)
        print(f"✅ Deleted all chunks for doc_id='{args.doc_id}' from domain '{args.domain}'.")
    except ProcessingError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    dispatch = {
        "list-docs": cmd_list_docs,
        "list-chunks": cmd_list_chunks,
        "info": cmd_info,
        "deprecate": cmd_deprecate,
        "delete": cmd_delete,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
