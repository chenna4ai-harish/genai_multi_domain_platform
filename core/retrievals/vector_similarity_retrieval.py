"""
core/retrievals/vector_similarity_retrieval.py

Vector similarity retrieval implementation using dense embeddings.
"""

from typing import List, Dict, Any, Optional
import logging
from core.interfaces.retrieval_interface import RetrievalInterface

logger = logging.getLogger(__name__)


class VectorSimilarityRetrieval(RetrievalInterface):
    """
    Dense vector similarity retrieval strategy.

    Uses vector store (Chroma/Pinecone) for semantic similarity search.
    """

    def __init__(self, vector_store):
        """
        Initialize vector similarity retrieval.

        Parameters:
        -----------
        vector_store : VectorStoreInterface
            The vector store instance (ChromaDB or Pinecone)
        """
        self.vector_store = vector_store
        logger.info(f"Initialized VectorSimilarityRetrieval with {type(vector_store).__name__}")

    def retrieve(
            self,
            query_text: str,
            metadata_filters: Optional[Dict[str, Any]] = None,
            top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using vector similarity search.

        Parameters:
        -----------
        query_text : str
            Natural language query
        metadata_filters : Dict[str, Any], optional
            Metadata filters for the search
        top_k : int
            Number of results to return

        Returns:
        --------
        List[Dict[str, Any]]:
            List of retrieved documents with scores
        """
        # Validate input
        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty")
        if top_k < 1 or top_k > 1000:
            raise ValueError(f"top_k must be between 1 and 1000, got {top_k}")

        logger.debug(f"Vector search: '{query_text}' (top_k={top_k})")

        # Query vector store
        results = self.vector_store.query(
            query_texts=[query_text],
            n_results=top_k,
            where=metadata_filters
        )

        # Format results
        formatted_results = []
        if results and results.get('ids') and len(results['ids']) > 0:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'score': 1.0 - results['distances'][0][i],  # Convert distance to similarity
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results.get('metadatas') else {}
                })

        logger.debug(f"Vector retrieved {len(formatted_results)} results")
        return formatted_results
