"""Embedding model wrapper using sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"


class Embedder:
    """Wrapper around sentence-transformers embedding model."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """Initialize the embedder.
        
        Args:
            model_name: Name of the sentence-transformers model.
        """
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name)
        except Exception:
            print(f"WARNING: Could not load {model_name}, trying fallback")
            self.model_name = FALLBACK_MODEL
            self.model = SentenceTransformer(FALLBACK_MODEL)
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def encode(self, texts: str | list[str], normalize: bool = True) -> np.ndarray:
        """Encode text(s) into embedding vector(s).
        
        Args:
            texts: Single text or list of texts.
            normalize: Whether to L2-normalize vectors.
            
        Returns:
            numpy array of shape (n, dim) or (dim,) for single text.
        """
        if isinstance(texts, str):
            return self.model.encode(texts, normalize_embeddings=normalize)
        return self.model.encode(texts, normalize_embeddings=normalize)
