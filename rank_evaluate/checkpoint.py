"""Per-query checkpoint manager for resumable evaluation.

Cache layout:
  cache/{model_name}/{run_tag}/{dataset_stem}.jsonl

  run_tag encodes the parameters that affect individual query scores:
    neg{num_neg}_seed{seed}   e.g.  neg20_seed42

  max_queries is intentionally NOT part of the cache key: it only selects
  which queries to evaluate, not how each query is scored. This lets a
  max_queries=50 run and a full run share the same cache files.

Each line in the JSONL:
  {"idx": <int>, "ndcg": <float>, "scores": [<float>, ...], "relevance": [<int>, ...]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class CacheEntry:
    """Cached per-query evaluation result."""

    idx: int
    ndcg: float
    scores: list[float]
    relevance: list[int]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "idx": self.idx,
            "ndcg": self.ndcg,
            "scores": self.scores,
            "relevance": self.relevance,
        }


class CheckpointManager:
    """Manages per-query result caching so evaluation can be resumed."""

    def __init__(
        self,
        cache_dir: Path,
        model_name: str,
        dataset_name: str,
        run_tag: str = "",
    ) -> None:
        model_dir = cache_dir / _safe_name(model_name)
        subdir = model_dir / run_tag if run_tag else model_dir
        self.cache_file = subdir / f"{dataset_name}.jsonl"
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[int, CacheEntry] = self._load()

    def _load(self) -> dict[int, CacheEntry]:
        """Load existing cache from disk."""
        cache: dict[int, CacheEntry] = {}
        if not self.cache_file.exists():
            return cache
        for entry in load_cache_entries(self.cache_file):
            cache[entry.idx] = entry
        return cache

    def has(self, idx: int) -> bool:
        """Return True if this query index is already cached."""
        return idx in self._cache

    def get_ndcg(self, idx: int) -> float:
        """Return cached NDCG for a query index."""
        return self._cache[idx].ndcg

    def get_all_ndcg(self) -> list[float]:
        """Return all cached NDCG values in index order."""
        return [self._cache[i].ndcg for i in sorted(self._cache)]

    def save(
        self,
        idx: int,
        ndcg: float,
        scores: list[float],
        relevance: list[int],
    ) -> None:
        """Append a result for query index to the cache file."""
        entry = CacheEntry(idx=idx, ndcg=ndcg, scores=scores, relevance=relevance)
        self._cache[idx] = entry
        with open(self.cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_json_dict(), ensure_ascii=False) + "\n")

    @property
    def completed_count(self) -> int:
        return len(self._cache)


def load_cache_entries(cache_file: Path) -> list[CacheEntry]:
    """Read valid cache entries from disk, skipping malformed lines."""
    entries: list[CacheEntry] = []
    if not cache_file.exists():
        return entries

    with open(cache_file, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                entries.append(
                    CacheEntry(
                        idx=int(payload["idx"]),
                        ndcg=float(payload["ndcg"]),
                        scores=[float(score) for score in payload["scores"]],
                        relevance=[int(label) for label in payload["relevance"]],
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return entries


def _safe_name(name: str) -> str:
    """Convert model name to a filesystem-safe directory name."""
    return name.replace("/", "__").replace(":", "_").replace(" ", "_")
