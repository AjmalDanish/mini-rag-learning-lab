"""Retrieval and re-ranking for RAG."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Retriever:
    """Retrieval and re-ranking pipeline."""
    
    def __init__(self, cross_encoder_name: str = DEFAULT_CROSS_ENCODER):
        """Initialize the retriever.
        
        Args:
            cross_encoder_name: Name of the cross-encoder model for re-ranking.
        """
        self.cross_encoder = None
        if CROSS_ENCODER_AVAILABLE:
            try:
                self.cross_encoder = CrossEncoder(cross_encoder_name)
            except Exception:
                print(f"WARNING: Could not load cross-encoder {cross_encoder_name}")
    
    def retrieve(
        self,
        query_embedding: np.ndarray,
        chunk_embeddings: np.ndarray,
        chunks: list[str],
        metadatas: list[dict],
        top_k: int = 3,
    ) -> list[tuple[str, dict, float]]:
        """Retrieve top-K chunks by cosine similarity.
        
        Args:
            query_embedding: Query vector.
            chunk_embeddings: All chunk vectors.
            chunks: List of chunk texts.
            metadatas: List of chunk metadata dicts.
            top_k: Number of results.
            
        Returns:
            List of (text, metadata, score) tuples.
        """
        scores = cosine_similarity(
            query_embedding.reshape(1, -1),
            chunk_embeddings
        )[0]
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((chunks[idx], metadatas[idx], float(scores[idx])))
        
        return results
    
    def rerank(
        self,
        query: str,
        results: list[tuple[str, dict, float]],
    ) -> list[tuple[str, dict, float]]:
        """Re-rank results using cross-encoder.
        
        Args:
            query: Original query text.
            results: List of (text, metadata, score) tuples.
            
        Returns:
            Re-ranked list of (text, metadata, score) tuples.
        """
        if self.cross_encoder is None or not results:
            return results
        
        pairs = [(query, text) for text, _, _ in results]
        ce_scores = self.cross_encoder.predict(pairs)
        
        reranked = []
        for i, score in enumerate(ce_scores):
            text, meta, _ = results[i]
            reranked.append((text, meta, float(score)))
        
        reranked.sort(key=lambda x: x[2], reverse=True)
        return reranked
