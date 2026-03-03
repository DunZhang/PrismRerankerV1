"""Abstract base class for reranker models."""

from abc import ABC, abstractmethod


class BaseReranker(ABC):
    """Interface for all reranker implementations."""

    @abstractmethod
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Score each document for relevance to the query.

        Args:
            query: The search query.
            documents: List of candidate documents.

        Returns:
            List of relevance scores aligned with input documents (higher = more relevant).
        """
        ...

    def close(self) -> None:
        """Release any held resources (optional override)."""
