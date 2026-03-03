"""Ranking metric helpers for reranker evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence

DEFAULT_NDCG_K = 10


def ranked_relevance(relevance: Sequence[int], scores: Sequence[float]) -> list[int]:
    """Return relevance labels ordered by predicted score descending."""
    if len(relevance) != len(scores):
        raise ValueError("relevance and scores must have same length")

    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    return [relevance[idx] for idx in ranked_indices]


def dcg_at_k(relevance_ranked: Sequence[int], k: int) -> float:
    """Compute Discounted Cumulative Gain at rank k."""
    return sum(
        rel / math.log2(rank + 2)
        for rank, rel in enumerate(relevance_ranked[:k])
        if rel > 0
    )


def ndcg_at_k(
    relevance: Sequence[int],
    scores: Sequence[float],
    k: int = DEFAULT_NDCG_K,
) -> float:
    """Compute NDCG@k from binary or graded relevance labels."""
    predicted = ranked_relevance(relevance, scores)
    ideal = sorted(relevance, reverse=True)

    dcg = dcg_at_k(predicted, k)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mean_score(values: Sequence[float]) -> float:
    """Return the arithmetic mean, falling back to 0.0 for empty input."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def mean_ndcg(ndcg_scores: Sequence[float]) -> float:
    """Backward-compatible alias for mean_score()."""
    return mean_score(ndcg_scores)
