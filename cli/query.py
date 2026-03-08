"""
cli/query.py

Run natural-language queries against a domain knowledge base from the command line.

Usage:
    python -m cli.query --domain hr --query "What is the vacation policy?" --strategy hybrid
    python -m cli.query --domain finance --query "Capital expenditure limits" --top-k 3
    python -m cli.query --domain hr --query "Leave types" --strategy bm25 --show-metadata
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.services.document_service import DocumentService, ProcessingError

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cli.query")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query a domain knowledge base using natural language."
    )
    parser.add_argument("--domain", required=True, help="Domain to query (e.g. hr, finance)")
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["hybrid", "vector_similarity", "bm25"],
        help="Retrieval strategy (default: all configured strategies)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument(
        "--include-deprecated",
        action="store_true",
        help="Include deprecated documents in results",
    )
    parser.add_argument(
        "--filter",
        nargs="*",
        metavar="KEY=VALUE",
        help="Metadata filters (e.g. --filter doc_type=policy author=HR)",
    )
    parser.add_argument(
        "--show-metadata", action="store_true", help="Show full metadata for each result"
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def parse_filters(filter_args):
    if not filter_args:
        return {}
    filters = {}
    for item in filter_args:
        if "=" not in item:
            logger.warning(f"Ignoring malformed filter (expected KEY=VALUE): {item}")
            continue
        k, v = item.split("=", 1)
        # Coerce common boolean / int values
        if v.lower() == "true":
            v = True
        elif v.lower() == "false":
            v = False
        else:
            try:
                v = int(v)
            except ValueError:
                pass
        filters[k.strip()] = v
    return filters


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    metadata_filters = parse_filters(args.filter)

    try:
        service = DocumentService(args.domain)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        results = service.query(
            query_text=args.query,
            strategy=args.strategy,
            metadata_filters=metadata_filters if metadata_filters else None,
            top_k=args.top_k,
            include_deprecated=args.include_deprecated,
        )
    except ProcessingError as e:
        print(f"Query failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return

    if not results:
        print("No results found.")
        return

    print(f"\nQuery: {args.query}")
    print(f"Domain: {args.domain}  |  Strategy: {args.strategy or 'all'}  |  Top-K: {args.top_k}")
    print("=" * 70)

    for i, result in enumerate(results, 1):
        score = result.get("score", 0.0)
        text = result.get("document", "")
        meta = result.get("metadata", {}) or {}
        strategy_tag = result.get("strategy", args.strategy or "")

        title = meta.get("title") or meta.get("source_file") or meta.get("doc_id", "Unknown")
        page = meta.get("page_num")
        page_str = f"  p.{page}" if page else ""

        print(f"\n[{i}] Score: {score:.4f}  [{strategy_tag}]")
        print(f"    Source: {title}{page_str}")
        print(f"    {text[:300].replace(chr(10), ' ')}{'...' if len(text) > 300 else ''}")

        if args.show_metadata:
            print(f"    Metadata: {json.dumps(meta, indent=6, default=str)}")

    print()


if __name__ == "__main__":
    main()
