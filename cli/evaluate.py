"""
cli/evaluate.py

Run golden QA evaluation sets against a domain knowledge base.

Usage:
    python -m cli.evaluate --domain hr --golden-file tests/golden_qa/hr_qa.json
    python -m cli.evaluate --domain hr --golden-file tests/golden_qa/hr_qa.json --strategy hybrid --top-k 5
    python -m cli.evaluate --domain hr --golden-file tests/golden_qa/hr_qa.json --output results.json

Golden QA file format (JSON):
[
  {
    "id": "q1",
    "question": "How many vacation days do employees get?",
    "expected_doc_ids": ["handbook_2025", "leave_policy"],   // at least one must appear in top-K
    "expected_keywords": ["vacation", "annual leave", "days"] // at least one must appear in answer
  },
  ...
]

Metrics reported:
  - Recall@K  : fraction of questions where expected_doc_id appears in top-K results
  - Hit Rate  : fraction of questions with at least one expected keyword in retrieved text
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.services.document_service import DocumentService, ProcessingError

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cli.evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality using a golden QA set."
    )
    parser.add_argument("--domain", required=True, help="Domain to evaluate")
    parser.add_argument("--golden-file", required=True, help="Path to golden QA JSON file")
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["hybrid", "vector_similarity", "bm25"],
        help="Retrieval strategy (default: all configured)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to retrieve per question")
    parser.add_argument("--output", default=None, help="Write detailed results to this JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-question details")
    return parser.parse_args()


def load_golden_set(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        print(f"Golden file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p) as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("Golden file must be a JSON array of question objects.", file=sys.stderr)
        sys.exit(1)
    return data


def check_recall(results: List[Dict], expected_doc_ids: List[str]) -> bool:
    """Return True if any expected_doc_id appears in retrieved results."""
    if not expected_doc_ids:
        return True  # No expectation — skip
    retrieved_doc_ids = {
        r.get("metadata", {}).get("doc_id") for r in results if r.get("metadata")
    }
    return bool(retrieved_doc_ids & set(expected_doc_ids))


def check_keyword_hit(results: List[Dict], expected_keywords: List[str]) -> bool:
    """Return True if at least one keyword appears in any retrieved chunk text."""
    if not expected_keywords:
        return True
    combined_text = " ".join(r.get("document", "").lower() for r in results)
    return any(kw.lower() in combined_text for kw in expected_keywords)


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    golden_set = load_golden_set(args.golden_file)

    try:
        service = DocumentService(args.domain)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    total = len(golden_set)
    recall_hits = 0
    keyword_hits = 0
    detailed_results = []

    print(f"\nEvaluating {total} questions against domain '{args.domain}' "
          f"(strategy={args.strategy or 'default'}, top_k={args.top_k})\n")

    for item in golden_set:
        qid = item.get("id", "?")
        question = item.get("question", "")
        expected_doc_ids = item.get("expected_doc_ids", [])
        expected_keywords = item.get("expected_keywords", [])

        try:
            results = service.query(
                query_text=question,
                strategy=args.strategy,
                top_k=args.top_k,
                include_deprecated=False,
            )
        except ProcessingError as e:
            logger.warning(f"[{qid}] Query failed: {e}")
            results = []

        recall_ok = check_recall(results, expected_doc_ids)
        keyword_ok = check_keyword_hit(results, expected_keywords)

        if recall_ok:
            recall_hits += 1
        if keyword_ok:
            keyword_hits += 1

        status = "✅" if (recall_ok and keyword_ok) else "❌"

        if args.verbose:
            print(f"{status} [{qid}]  recall={'Y' if recall_ok else 'N'}  "
                  f"keyword={'Y' if keyword_ok else 'N'}  Q: {question[:80]}")

        detailed_results.append({
            "id": qid,
            "question": question,
            "recall_hit": recall_ok,
            "keyword_hit": keyword_ok,
            "retrieved_doc_ids": [
                r.get("metadata", {}).get("doc_id") for r in results if r.get("metadata")
            ],
            "top_scores": [round(r.get("score", 0), 4) for r in results[:3]],
        })

    # Summary
    recall_at_k = recall_hits / total if total else 0.0
    keyword_hit_rate = keyword_hits / total if total else 0.0

    print(f"\n{'=' * 50}")
    print(f"  Evaluation Summary — domain={args.domain}")
    print(f"{'=' * 50}")
    print(f"  Questions evaluated : {total}")
    print(f"  Strategy            : {args.strategy or 'default'}")
    print(f"  Top-K               : {args.top_k}")
    print(f"  Recall@{args.top_k:<3}           : {recall_at_k:.1%}  ({recall_hits}/{total})")
    print(f"  Keyword Hit Rate    : {keyword_hit_rate:.1%}  ({keyword_hits}/{total})")
    print(f"{'=' * 50}\n")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(
            {
                "domain": args.domain,
                "strategy": args.strategy,
                "top_k": args.top_k,
                "total": total,
                "recall_at_k": recall_at_k,
                "keyword_hit_rate": keyword_hit_rate,
                "results": detailed_results,
            },
            indent=2,
            default=str,
        ))
        print(f"Detailed results written to: {args.output}")


if __name__ == "__main__":
    main()
