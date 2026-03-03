"""Voyage AI reranker scorer for teacher score generation.

Wraps the existing VoyageReranker with model name mapping so that
user-facing names (e.g. ``voyage-rerank-2``) are cleanly separated
from the Voyage API model identifiers (e.g. ``rerank-2``).
"""

import sys
from pathlib import Path

# Allow importing rank_evaluate as a sibling package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rank_evaluate.models.voyage import VoyageReranker  # noqa: E402

# User-facing model name -> Voyage API model identifier
MODEL_NAME_MAP: dict[str, str] = {
    "voyage-rerank-2": "rerank-2",
    "voyage-rerank-2-lite": "rerank-2-lite",
    "voyage-rerank-2.5": "rerank-2.5",
    "voyage-rerank-2.5-lite": "rerank-2.5-lite",
}

SUPPORTED_MODELS = ", ".join(sorted(MODEL_NAME_MAP))


class VoyageScorer:
    """Score query-document pairs using Voyage AI rerank API.

    Args:
        model_name: User-facing model identifier (e.g. ``voyage-rerank-2``).
    """

    def __init__(self, model_name: str) -> None:
        if model_name not in MODEL_NAME_MAP:
            raise ValueError(
                f"Unknown model_name: {model_name!r}\nSupported: {SUPPORTED_MODELS}"
            )
        api_model = MODEL_NAME_MAP[model_name]
        self._reranker = VoyageReranker(model=api_model)

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Return relevance scores aligned with *documents* order."""
        return self._reranker.rerank(query, documents)

    def close(self) -> None:
        """Release resources."""
        self._reranker.close()
