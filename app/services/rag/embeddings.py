from typing import List, Optional
from openai import AsyncOpenAI
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings


class EmbeddingService:
    """Service for generating text embeddings."""

    SUPPORTED_MODELS = {
        "text-embedding-3-small": {"dimensions": 1536, "max_tokens": 8191},
        "text-embedding-3-large": {"dimensions": 3072, "max_tokens": 8191},
        "text-embedding-ada-002": {"dimensions": 1536, "max_tokens": 8191}
    }

    def __init__(
        self,
        model: str = None,
        api_key: str = None
    ):
        self.model = model or settings.EMBEDDING_MODEL
        self.client = AsyncOpenAI(api_key=api_key or settings.OPENAI_API_KEY)

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions for current model."""
        return self.SUPPORTED_MODELS.get(self.model, {}).get("dimensions", 1536)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        response = await self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            embeddings.extend([item.embedding for item in response.data])

        return embeddings

    async def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        a = np.array(embedding1)
        b = np.array(embedding2)

        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    async def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple]:
        """Find most similar embeddings to query.

        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embeddings
            top_k: Number of results to return

        Returns:
            List of (index, similarity_score) tuples
        """
        query = np.array(query_embedding)
        candidates = np.array(candidate_embeddings)

        # Compute all similarities at once
        similarities = np.dot(candidates, query) / (
            np.linalg.norm(candidates, axis=1) * np.linalg.norm(query)
        )

        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]


class SemanticCache:
    """Semantic caching for embedding queries."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        similarity_threshold: float = 0.95
    ):
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
        self.cache: dict = {}  # query -> (embedding, response)

    async def get(self, query: str) -> Optional[str]:
        """Get cached response for semantically similar query.

        Args:
            query: Query to look up

        Returns:
            Cached response if found, None otherwise
        """
        if not self.cache:
            return None

        query_embedding = await self.embedding_service.embed_text(query)

        for cached_query, (cached_embedding, response) in self.cache.items():
            similarity = await self.embedding_service.compute_similarity(
                query_embedding, cached_embedding
            )

            if similarity >= self.similarity_threshold:
                return response

        return None

    async def set(self, query: str, response: str) -> None:
        """Cache a query-response pair.

        Args:
            query: Query text
            response: Response to cache
        """
        embedding = await self.embedding_service.embed_text(query)
        self.cache[query] = (embedding, response)

    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
