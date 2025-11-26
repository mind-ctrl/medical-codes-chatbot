"""
Embedding Generation Service
Uses sentence-transformers for local, cost-free embedding generation
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Manages embedding generation using sentence-transformers

    Uses all-MiniLM-L6-v2 model for fast, accurate embeddings:
    - 384 dimensions
    - ~15ms per embedding on CPU
    - Normalized vectors for cosine similarity
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model

        Args:
            model_name: HuggingFace model identifier
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text

        Args:
            text: Input text to embed

        Returns:
            Normalized embedding vector (numpy array)
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,  # Critical for cosine similarity
            show_progress_bar=False
        )
        return embedding

    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently

        Batching provides significant speedup:
        - Single: ~15ms each
        - Batch of 100: ~500ms total (5x faster)

        Args:
            texts: List of input texts
            batch_size: Batch size for processing (32 is optimal for CPU)

        Returns:
            Array of normalized embeddings, shape (len(texts), dimension)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings


# Global instance (singleton pattern)
# Load model once at startup, reuse for all requests
_embedding_service = None


def get_embedding_service(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> EmbeddingService:
    """
    Get or create embedding service singleton

    Using singleton ensures:
    - Model loaded only once (saves ~2s per request)
    - Consistent embeddings across requests
    - Memory efficient (one model in RAM)

    Args:
        model_name: Model to use (defaults to config)

    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(model_name)
    return _embedding_service
