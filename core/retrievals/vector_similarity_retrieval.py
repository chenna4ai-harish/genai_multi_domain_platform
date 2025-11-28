"""
core/retrievals/vector_similarity_retrieval.py

Vector similarity retrieval implementation using dense embeddings.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class VectorSimilarityRetrieval:
    """
    Dense vector similarity retrieval using semantic embeddings.

    This retriever:
    1. Embeds the query using the same model as ingestion
    2. Searches vector store for similar chunks
    3. Returns top-k results ranked by cosine similarity
    """

    def __init__(self, vectorstore, embedding_model):
        """
        Initialize vector similarity retrieval.

        Parameters:
        -----------
        vectorstore : VectorStoreInterface
            Vector database instance
        embedding_model : EmbeddingInterface
            Model to embed query text (REQUIRED!)
        """
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model  # ← STORE THIS!
        logger.info(
            f"VectorSimilarityRetrieval initialized with model: "
            f"{embedding_model.get_model_name()}"
        )

    def retrieve(
            self,
            query_text: str,
            metadata_filters: Optional[Dict[str, Any]] = None,
            top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents using vector search.

        Parameters:
        -----------
        query_text : str
            Natural language query
        metadata_filters : dict, optional
            Filters to apply (domain, doc_type, deprecated, etc.)
        top_k : int
            Number of results to return

        Returns:
        --------
        List[Dict[str, Any]]:
            Results with id, score, metadata, document fields
        """
        logger.info(f"Vector search: '{query_text}' (top_k={top_k})")

        # Step 1: Embed the query
        query_embedding = self.embedding_model.embed_texts([query_text])[0]
        logger.debug(f"Query embedded to {len(query_embedding)}-dim vector")

        # Step 2: Search vector store
        results = self.vectorstore.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=metadata_filters
        )

        logger.info(f"✅ Vector search returned {len(results)} results")
        return results
