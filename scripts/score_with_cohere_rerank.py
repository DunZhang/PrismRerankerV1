"""Score final_dev_data with Cohere rerank models via OpenRouter.

Groups documents by query to minimise API request count (451 requests
per model instead of 846). Checkpoint file enables resumable runs.

Run: uv run python scripts/score_with_cohere_rerank.py
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/rerank"
API_KEY = "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"

INPUT_PATH = Path("/mnt/g/PrismRerankerV1Data/final_dev_data.jsonl")
OUTPUT_PATH = Path("/mnt/g/PrismRerankerV1Data/final_dev_data_cohere.jsonl")
CHECKPOINT_PATH = Path(
    "/mnt/g/PrismRerankerV1Data/final_dev_data_cohere_checkpoint.jsonl"
)

MODELS: dict[str, str] = {
    "cohere_rerank-4-fast": "cohere/rerank-4-fast",
    "cohere_rerank-4-pro": "cohere/rerank-4-pro",
}

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0
REQUEST_DELAY = 0.1


def _load_data() -> list[dict]:
    rows: list[dict] = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    logger.info("loaded %d rows from %s", len(rows), INPUT_PATH)
    return rows


def _load_checkpoint() -> dict[str, dict[str, float]]:
    """Return {model_key: {query: {doc_idx: score}}} from checkpoint."""
    cp: dict[str, dict[str, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    if not CHECKPOINT_PATH.exists():
        return cp  # type: ignore[return-value]
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            model_key = obj["model_key"]
            query = obj["query"]
            for doc_idx_str, score in obj["scores"].items():
                cp[model_key][query][int(doc_idx_str)] = score
    logger.info(
        "checkpoint: %s",
        {k: len(v) for k, v in cp.items()},
    )
    return cp  # type: ignore[return-value]


def _rerank(
    model: str, query: str, documents: list[str], top_n: int
) -> list[dict]:
    """Call OpenRouter rerank API with retries."""
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_n,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                API_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=120,
            )
            if resp.status_code == 429:
                delay = RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "rate limited, retrying in %.1fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    MAX_RETRIES,
                )
                time.sleep(delay)
                continue
            resp.raise_for_status()
            return resp.json()["results"]
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY * (2**attempt)
            logger.warning(
                "request failed, retrying in %.1fs (attempt %d/%d)",
                delay,
                attempt + 1,
                MAX_RETRIES,
                exc_info=True,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")


def _build_query_groups(
    rows: list[dict],
) -> dict[str, list[tuple[int, str]]]:
    """Group row indices by query: {query: [(row_idx, document), ...]}."""
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[row["query"]].append((idx, row["document"]))
    logger.info(
        "grouped into %d unique queries (max docs/query=%d)",
        len(groups),
        max(len(v) for v in groups.values()),
    )
    return groups


def _score_model(
    model_key: str,
    model_id: str,
    rows: list[dict],
    query_groups: dict[str, list[tuple[int, str]]],
    checkpoint: dict,
) -> None:
    """Score all rows for one model, writing checkpoint incrementally."""
    done_queries: dict[str, dict[int, float]] = checkpoint.get(
        model_key, {}
    )
    total_queries = len(query_groups)
    skipped = 0

    for qi, (query, doc_entries) in enumerate(query_groups.items()):
        if query in done_queries and len(done_queries[query]) == len(
            doc_entries
        ):
            for local_idx, (row_idx, _doc) in enumerate(doc_entries):
                rows[row_idx][model_key] = done_queries[query][local_idx]
            skipped += 1
            continue

        documents = [doc for _row_idx, doc in doc_entries]
        results = _rerank(model_id, query, documents, top_n=len(documents))

        scores: dict[int, float] = {}
        for r in results:
            api_idx = r["index"]
            score = r["relevance_score"]
            row_idx = doc_entries[api_idx][0]
            rows[row_idx][model_key] = score
            scores[api_idx] = score

        cp_line = json.dumps(
            {"model_key": model_key, "query": query, "scores": scores},
            ensure_ascii=False,
        )
        with open(CHECKPOINT_PATH, "a", encoding="utf-8") as f:
            f.write(cp_line + "\n")

        done = qi + 1 - skipped
        remaining = total_queries - qi - 1
        if done % 50 == 0 or remaining == 0:
            logger.info(
                "[%s] progress: %d/%d queries scored (%d skipped from checkpoint)",
                model_key,
                qi + 1,
                total_queries,
                skipped,
            )

        time.sleep(REQUEST_DELAY)

    logger.info(
        "[%s] finished: %d queries total, %d from checkpoint",
        model_key,
        total_queries,
        skipped,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    rows = _load_data()
    query_groups = _build_query_groups(rows)
    checkpoint = _load_checkpoint()

    for model_key, model_id in MODELS.items():
        logger.info("=== scoring with %s (%s) ===", model_key, model_id)
        _score_model(model_key, model_id, rows, query_groups, checkpoint)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("wrote %d rows to %s", len(rows), OUTPUT_PATH)


if __name__ == "__main__":
    main()
