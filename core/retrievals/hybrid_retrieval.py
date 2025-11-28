# core/retrievals/hybrid_retrieval.py

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class HybridRetrieval:
    """
    Hybrid retrieval combining dense (vector) + sparse (BM25) search.

    Formula: final_score = alpha * dense_score + (1 - alpha) * sparse_score

    - alpha = 1.0: Pure semantic (vector only)
    - alpha = 0.7: Balanced (recommended)
    - alpha = 0.5: Equal weight
    - alpha = 0.0: Pure keyword (BM25 only)
    """

    def __init__(
            self,
            vectorstore,
            embedding_model,  # ← REQUIRED!
            bm25_index,
            alpha: float = 0.7
    ):
        """
        Initialize hybrid retrieval.

        Parameters:
        -----------
        vectorstore : VectorStoreInterface
            Vector database
        embedding_model : EmbeddingInterface
            Model to embed queries (REQUIRED!)
        bm25_index : BM25Retrieval
            BM25 index for keyword search
        alpha : float
            Weight for dense vs sparse (0-1)
        """
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model  # ← STORE THIS!
        self.bm25_index = bm25_index
        self.alpha = alpha
        logger.info(
            f"HybridRetrieval initialized: "
            f"alpha={alpha}, model={embedding_model.get_model_name()}"
        )

    def retrieve(
            self,
            query_text: str,
            metadata_filters: Optional[Dict[str, Any]] = None,
            top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid retrieval.

        Steps:
        1. Dense search (vector similarity)
        2. Sparse search (BM25)
        3. Normalize scores to [0, 1]
        4. Combine with alpha weighting
        5. Sort and return top-k
        """
        logger.info(f"Hybrid search: '{query_text}' (alpha={self.alpha}, top_k={top_k})")

        # Step 1: Dense search
        query_embedding = self.embedding_model.embed_texts([query_text])[0]
        dense_results = self.vectorstore.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Over-fetch for better results
            filters=metadata_filters
        )
        dense_scores = {r['id']: r['score'] for r in dense_results}
        logger.debug(f"Dense search: {len(dense_results)} results")

        # Step 2: Sparse search (BM25)
        sparse_results = self.bm25_index.search(
            query_text=query_text,
            top_k=top_k * 2,
            filters=metadata_filters
        )
        sparse_scores = {r['id']: r['score'] for r in sparse_results}
        logger.debug(f"Sparse search: {len(sparse_results)} results")

        # Step 3: Normalize scores
        dense_norm = self._normalize_scores(dense_scores)
        sparse_norm = self._normalize_scores(sparse_scores)

        # Step 4: Combine scores
        all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        combined_scores = {}
        for chunk_id in all_ids:
            d_score = dense_norm.get(chunk_id, 0.0)
            s_score = sparse_norm.get(chunk_id, 0.0)
            combined_scores[chunk_id] = self.alpha * d_score + (1 - self.alpha) * s_score

        # Step 5: Sort and fetch documents
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        top_ids = sorted_ids[:top_k]

        # Fetch full documents
        results = []
        for chunk_id in top_ids:
            # Try to get from dense results first, then sparse
            doc = next((r for r in dense_results if r['id'] == chunk_id), None)
            if not doc:
                doc = next((r for r in sparse_results if r['id'] == chunk_id), None)

            if doc:
                doc['score'] = combined_scores[chunk_id]
                doc['dense_score'] = dense_norm.get(chunk_id, 0.0)
                doc['sparse_score'] = sparse_norm.get(chunk_id, 0.0)
                results.append(doc)

        logger.info(f"✅ Hybrid search returned {len(results)} results")
        return results

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Min-max normalize scores to [0, 1]."""
        if not scores:
            return {}

        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)

        if max_score == min_score:
            return {k: 1.0 for k in scores}

        return {
            k: (v - min_score) / (max_score - min_score)
            for k, v in scores.items()
        }
