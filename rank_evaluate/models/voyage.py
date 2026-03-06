"""Voyage AI reranker via official API."""

import os
import time
from collections import deque

import voyageai

from .base import BaseReranker

_DEFAULT_MODEL = "rerank-2-lite"
_MAX_DOCS_PER_CALL = 1000  # Voyage limit
_MAX_TOKENS_PER_BATCH = 500_000  # Voyage allows 600k, leave margin

# Voyage 2.5 series requires an instruction prefix on the query.
_INSTRUCTION_MODELS = {"rerank-2.5", "rerank-2.5-lite"}
_QUERY_INSTRUCTION = (
    "Rank documents by how well they answer the user's question. "
    "Prefer documents with specific, factual, and directly relevant information."
    "\nQuery: "
)
_RETRY_DELAYS = [2, 5, 15, 30]  # seconds between retries
_TPM_LIMIT = 2_000_000  # Voyage TPM cap
_TPM_MARGIN = 0.85  # start throttling at 85% of limit
_WINDOW_SECONDS = 60.0  # sliding window length


class _TokenRateLimiter:
    """Sliding-window token-per-minute rate limiter."""

    def __init__(
        self, tpm_limit: int = _TPM_LIMIT, margin: float = _TPM_MARGIN
    ) -> None:
        self._limit = int(tpm_limit * margin)
        self._window: deque[tuple[float, int]] = deque()
        self._total = 0

    def _evict(self) -> None:
        """Remove entries older than the sliding window."""
        cutoff = time.monotonic() - _WINDOW_SECONDS
        while self._window and self._window[0][0] < cutoff:
            _, tokens = self._window.popleft()
            self._total -= tokens

    def wait_if_needed(self, estimated_tokens: int) -> None:
        """Block until there is room in the window for *estimated_tokens*."""
        while True:
            self._evict()
            if self._total + estimated_tokens <= self._limit:
                return
            # Wait until the oldest entry expires
            oldest_time = self._window[0][0]
            sleep_for = oldest_time + _WINDOW_SECONDS - time.monotonic() + 0.1
            if sleep_for > 0:
                time.sleep(sleep_for)

    @property
    def used(self) -> int:
        """Tokens currently counted in the sliding window."""
        self._evict()
        return self._total

    def record(self, tokens: int) -> None:
        """Record actual token usage."""
        self._window.append((time.monotonic(), tokens))
        self._total += tokens


def _collect_api_keys() -> list[str]:
    """Discover Voyage API keys from environment variables.

    Scans VOYAGE_API_KEY_1, VOYAGE_API_KEY_2, ... in order.
    Falls back to VOYAGE_API_KEY if no numbered keys are found.
    """
    keys: list[str] = []
    idx = 1
    while True:
        key = os.environ.get(f"VOYAGE_API_KEY_{idx}")
        if not key:
            break
        keys.append(key)
        idx += 1

    if not keys:
        single = os.environ.get("VOYAGE_API_KEY")
        if single:
            keys.append(single)

    return keys


class VoyageReranker(BaseReranker):
    """Reranker using Voyage AI's reranking API.

    Supports multiple API keys (VOYAGE_API_KEY_1, VOYAGE_API_KEY_2, ...)
    for higher throughput via round-robin scheduling. Each key gets its
    own client and independent rate limiter.

    Falls back to a single VOYAGE_API_KEY if no numbered keys are found.
    """

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        keys = _collect_api_keys()
        if not keys:
            raise ValueError(
                "No Voyage API key found. "
                "Set VOYAGE_API_KEY_1, VOYAGE_API_KEY_2, ... "
                "or VOYAGE_API_KEY in environment."
            )

        self.model = model
        self._slots: list[tuple[voyageai.Client, _TokenRateLimiter]] = [
            (voyageai.Client(api_key=k), _TokenRateLimiter()) for k in keys
        ]
        self._slot_calls: list[int] = [0] * len(keys)
        self._slot_tokens: list[int] = [0] * len(keys)
        self._next_slot = 0
        self._total_calls = 0

        # Print key discovery summary
        n = len(keys)
        masked = [f"...{k[-4:]}" for k in keys]
        print(f"  [voyage] {n} API key(s) loaded: {', '.join(masked)}")
        tpm = _TPM_LIMIT * n
        print(f"  [voyage] Total TPM capacity: {tpm:,} ({_TPM_LIMIT:,} x {n})")

    def _pick_slot(self) -> tuple[int, voyageai.Client, _TokenRateLimiter]:
        """Round-robin select the next client/limiter pair."""
        idx = self._next_slot % len(self._slots)
        self._next_slot += 1
        client, limiter = self._slots[idx]
        return idx, client, limiter

    def _estimate_tokens(self, query: str, documents: list[str]) -> int:
        """Rough token estimate: ~1 token per 3 chars for CJK-heavy text."""
        total_chars = len(query) + sum(len(d) for d in documents)
        return max(total_chars // 3, 100)

    def _log_stats(self) -> None:
        """Print per-key usage stats periodically (every 50 calls)."""
        if self._total_calls % 50 != 0:
            return
        parts = []
        for i, (_, limiter) in enumerate(self._slots):
            calls = self._slot_calls[i]
            tokens = self._slot_tokens[i]
            window_tok = limiter.used
            parts.append(
                f"key{i + 1}: {calls} calls, "
                f"{tokens:,} tok total, "
                f"{window_tok:,} tok/min"
            )
        print(f"  [voyage] stats @ {self._total_calls} calls | {' | '.join(parts)}")

    def _split_batches(self, query: str, documents: list[str]) -> list[list[int]]:
        """Split document indices into batches respecting token limits.

        Each batch stays under both _MAX_DOCS_PER_CALL and
        _MAX_TOKENS_PER_BATCH (estimated).
        """
        query_tokens = max(len(query) // 3, 1)
        batches: list[list[int]] = []
        current_batch: list[int] = []
        current_tokens = query_tokens  # query counted once per batch

        for i, doc in enumerate(documents):
            doc_tokens = max(len(doc) // 3, 1)
            would_exceed_tokens = (current_tokens + doc_tokens) > _MAX_TOKENS_PER_BATCH
            would_exceed_docs = len(current_batch) >= _MAX_DOCS_PER_CALL

            if current_batch and (would_exceed_tokens or would_exceed_docs):
                batches.append(current_batch)
                current_batch = []
                current_tokens = query_tokens

            current_batch.append(i)
            current_tokens += doc_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    def _rerank_batch(
        self,
        query: str,
        documents: list[str],
        slot_idx: int,
        client: voyageai.Client,
        limiter: _TokenRateLimiter,
    ) -> list[tuple[int, float]]:
        """Call the Voyage API for a single batch with retries.

        Returns list of (index, score) pairs where index is relative
        to the provided documents list.
        """
        estimated = self._estimate_tokens(query, documents)
        limiter.wait_if_needed(estimated)

        last_err: Exception | None = None
        for delay in [0, *_RETRY_DELAYS]:
            if delay:
                time.sleep(delay)
            try:
                result = client.rerank(
                    query=query,
                    documents=documents,
                    model=self.model,
                    truncation=True,
                )
                actual_tokens = getattr(result, "total_tokens", None)
                used = actual_tokens or estimated
                limiter.record(used)
                self._slot_calls[slot_idx] += 1
                self._slot_tokens[slot_idx] += used
                self._total_calls += 1
                self._log_stats()

                return [
                    (item.index, float(item.relevance_score)) for item in result.results
                ]
            except Exception as e:
                last_err = e
                if "rate" in str(e).lower():
                    limiter.record(estimated)
                print(
                    f"  [voyage] key{slot_idx + 1} API error "
                    f"({type(e).__name__}: {e}), retrying..."
                )

        raise RuntimeError(
            f"Voyage API failed after {len(_RETRY_DELAYS) + 1} attempts"
        ) from last_err

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Score documents using the Voyage rerank API with rate limiting."""
        if self.model in _INSTRUCTION_MODELS:
            query = f"{_QUERY_INSTRUCTION}{query}"

        batches = self._split_batches(query, documents)
        scores = [0.0] * len(documents)

        for batch_indices in batches:
            slot_idx, client, limiter = self._pick_slot()
            batch_docs = [documents[i] for i in batch_indices]
            results = self._rerank_batch(query, batch_docs, slot_idx, client, limiter)
            for local_idx, score in results:
                scores[batch_indices[local_idx]] = score

        return scores
