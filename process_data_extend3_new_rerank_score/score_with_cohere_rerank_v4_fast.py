"""Score every unique (query, document) pair across the step6~step9 data files.

Use Cohere rerank-v4.0-fast API. Documents are grouped by query so each API
call scores all documents for one query in a single request.

Multi-key concurrency: each ``CO_API_KEY_*`` key from ``.env`` runs in its own
worker process with an independent ``cohere.ClientV2`` and self-paced rate
limiter. Processes are fully isolated (separate connections, sessions, state).

Output is a single append-only JSONL with one row per unique pair, keyed by
an md5 hash so the run is resumable.

Run: ``uv run python -m process_data_extend3_new_rerank_score.score_with_cohere_rerank_v4_fast``
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL: str = "rerank-v4.0-fast"
SCORE_KEY: str = "cohere_rerank_4_fast"
OUTPUT_PATH: Path = Path(
    "/mnt/g/PrismRerankerV1Data/data_extend3_new_rerank_score/"
    "cohere_rerank_v4_fast_scores.jsonl"
)

INPUT_FILES: list[str] = [
    # "/mnt/g/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs.jsonl",
    "/mnt/g/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs_balanced.jsonl",
    # "/mnt/g/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs_no_medical.jsonl",
    "/mnt/g/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs_no_medical_length-score-balance.jsonl",
    "/mnt/g/PrismRerankerV1Data/step7_kalm_web-search_query_document_pairs_annotated.jsonl",
    "/mnt/g/PrismRerankerV1Data/step8_kalm_web-search_query_document_pairs_annotated_merged.jsonl",
    "/mnt/g/PrismRerankerV1Data/step9_kalm_web-search_query_document_pairs_contribution_evidence.jsonl",
    # "/mnt/g/PrismRerankerV1Data/data_extend2/step6_expanded2_web-search_query_document_pairs.jsonl",
    "/mnt/g/PrismRerankerV1Data/data_extend2/step6_expanded2_web-search_query_document_pairs_length-score-balance.jsonl",
    "/mnt/g/PrismRerankerV1Data/data_extend2/step7_expanded2_web-search_query_document_pairs_annotated.jsonl",
    "/mnt/g/PrismRerankerV1Data/data_extend2/step8_expanded2_web-search_query_document_pairs_annotated_merged.jsonl",
    "/mnt/g/PrismRerankerV1Data/data_extend2/step9_expanded2_web-search_query_document_contribution_evidence.jsonl",
]

REQUESTS_PER_MINUTE_PER_KEY: int = 10
MAX_RETRIES: int = 8
RETRY_BASE_DELAY: float = 5.0
WRITE_FLUSH_EVERY: int = 100

logger = logging.getLogger("cohere_rerank_v4_fast_score")


# ---------------------------------------------------------------------------
# Helpers (same pattern as other scorers)
# ---------------------------------------------------------------------------
def _pair_hash(query: str, document: str) -> str:
    """MD5 hash of (query, document) for deduplication / resume."""
    payload = (query + "\x1f" + document).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def _collect_unique_pairs() -> dict[str, tuple[str, str]]:
    """Read every input file; return {hash: (query, document)} deduped."""
    pairs: dict[str, tuple[str, str]] = {}
    total_rows = 0
    for fpath in INPUT_FILES:
        path = Path(fpath)
        if not path.exists():
            logger.warning("missing input file, skipping: %s", fpath)
            continue
        rows_in_file = 0
        added_from_file = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                query = obj["query"]
                document = obj["document"]
                h = _pair_hash(query, document)
                rows_in_file += 1
                if h not in pairs:
                    pairs[h] = (query, document)
                    added_from_file += 1
        total_rows += rows_in_file
        logger.info(
            "read %s: rows=%d new_unique=%d cumulative_unique=%d",
            path.name,
            rows_in_file,
            added_from_file,
            len(pairs),
        )
    logger.info(
        "collected total_rows=%d unique_pairs=%d dedup_ratio=%.2f%%",
        total_rows,
        len(pairs),
        100.0 * (1 - len(pairs) / max(total_rows, 1)),
    )
    return pairs


def _load_done_hashes() -> set[str]:
    """Stream existing output file to collect already-scored hashes."""
    if not OUTPUT_PATH.exists():
        return set()
    done: set[str] = set()
    bad_lines = 0
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(obj["hash"])
            except (json.JSONDecodeError, KeyError):
                bad_lines += 1
    if bad_lines:
        logger.warning("checkpoint had %d unparseable lines (ignored)", bad_lines)
    logger.info("checkpoint loaded: %d already-scored hashes", len(done))
    return done


def _load_api_keys() -> list[str]:
    """Load all CO_API_KEY_* from .env, sorted by suffix for determinism."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    env_vars = dotenv_values(env_path)
    keys: list[tuple[str, str]] = [
        (k, v) for k, v in env_vars.items() if k.startswith("CO_API_KEY_") and v
    ]
    keys.sort(key=lambda kv: kv[0])
    if not keys:
        raise RuntimeError(
            f"no CO_API_KEY_* found in {env_path}. "
            "Add at least one key (e.g. CO_API_KEY_1=xxx)."
        )
    effective_rpm = len(keys) * REQUESTS_PER_MINUTE_PER_KEY
    logger.info(
        "loaded %d Cohere API keys from %s  |  "
        "per-key rate: %d req/min  |  effective throughput: %d req/min",
        len(keys),
        env_path,
        REQUESTS_PER_MINUTE_PER_KEY,
        effective_rpm,
    )
    for name, _ in keys:
        logger.info("  key: %s (****%s)", name, _[-4:])
    return [v for _, v in keys]


# ---------------------------------------------------------------------------
# Query grouping
# ---------------------------------------------------------------------------
def _group_by_query(
    tasks: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    """Group tasks by query. Returns {query: [task_dict, ...]}."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for t in tasks:
        groups[t["query"]].append(t)
    return dict(groups)


# ---------------------------------------------------------------------------
# Worker process — one per API key
# ---------------------------------------------------------------------------
def _worker_loop(
    rank: int,
    api_key: str,
    rpm: int,
    in_q: Any,
    out_q: Any,
) -> None:
    """Independent worker process. Owns its own Cohere client and rate limiter."""
    import cohere

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    wlog = logging.getLogger(f"worker-{rank}")

    client = cohere.ClientV2(api_key=api_key)
    min_interval = 60.0 / rpm
    last_req = 0.0

    out_q.put(("ready", rank, None))
    wlog.info("ready (key=****%s, interval=%.1fs)", api_key[-4:], min_interval)

    while True:
        task = in_q.get()
        if task is None:
            break

        query: str = task["query"]
        doc_entries: list[dict[str, str]] = task["doc_entries"]
        documents = [e["document"] for e in doc_entries]

        # Self rate-limit
        now = time.monotonic()
        elapsed = now - last_req
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        # Retry loop
        response = None
        for attempt in range(MAX_RETRIES):
            try:
                last_req = time.monotonic()
                response = client.rerank(
                    model=MODEL,
                    query=query,
                    documents=documents,
                    top_n=len(documents),
                )
                break
            except Exception as exc:
                delay = RETRY_BASE_DELAY * (2**attempt)
                wlog.warning(
                    "rerank failed (attempt=%d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    MAX_RETRIES,
                    str(exc)[:200],
                    delay,
                )
                time.sleep(delay)

        if response is None:
            wlog.error(
                "giving up after %d retries, skipping query (%d docs): %s",
                MAX_RETRIES,
                len(doc_entries),
                query[:120],
            )
            out_q.put(("skip", rank, len(doc_entries)))
            continue

        # Map scores back via index
        score_map: dict[int, float] = {
            r.index: r.relevance_score for r in response.results
        }
        rows = [
            {
                "hash": e["hash"],
                "query": query,
                "document": e["document"],
                SCORE_KEY: float(score_map[i]),
            }
            for i, e in enumerate(doc_entries)
        ]
        out_q.put(("done", rank, rows))

    wlog.info("exiting")


# ---------------------------------------------------------------------------
# Main process orchestration
# ---------------------------------------------------------------------------
def _run(
    query_groups: dict[str, list[dict[str, str]]],
    api_keys: list[str],
) -> None:
    """Spawn one worker per key, distribute work, collect results."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    num_keys = len(api_keys)
    effective_rpm = num_keys * REQUESTS_PER_MINUTE_PER_KEY
    total_queries = len(query_groups)
    total_pairs = sum(len(docs) for docs in query_groups.values())
    est_minutes = total_queries / max(effective_rpm, 1)

    logger.info(
        "run config: workers=%d  interval=%.1fs/key  effective_rpm=%d",
        num_keys,
        60.0 / REQUESTS_PER_MINUTE_PER_KEY,
        effective_rpm,
    )
    logger.info(
        "workload: %d queries  %d pairs  est_time=%.1f min (%.1f hours)",
        total_queries,
        total_pairs,
        est_minutes,
        est_minutes / 60,
    )

    # --- spawn workers ---
    ctx = mp.get_context("spawn")
    in_qs: list[Any] = [ctx.Queue() for _ in range(num_keys)]
    out_q: Any = ctx.Queue()

    procs = []
    for rank in range(num_keys):
        p = ctx.Process(
            target=_worker_loop,
            args=(
                rank,
                api_keys[rank],
                REQUESTS_PER_MINUTE_PER_KEY,
                in_qs[rank],
                out_q,
            ),
        )
        p.start()
        procs.append(p)

    # Wait for all workers to be ready
    ready = 0
    while ready < num_keys:
        tag, _rank, _ = out_q.get()
        if tag == "ready":
            ready += 1
    logger.info("all %d workers ready", num_keys)

    # --- round-robin distribute query groups to workers ---
    query_list = list(query_groups.items())
    for i, (query, doc_entries) in enumerate(query_list):
        worker_rank = i % num_keys
        in_qs[worker_rank].put({"query": query, "doc_entries": doc_entries})

    # Send sentinel to each worker
    for rank in range(num_keys):
        in_qs[rank].put(None)

    logger.info("distributed %d query groups to %d workers", total_queries, num_keys)

    # --- collect results ---
    pbar = tqdm(total=total_queries, desc="scoring", unit="query", dynamic_ncols=True)
    pending_lines: list[str] = []
    ok_queries = 0
    ok_pairs = 0
    skipped_queries = 0
    skipped_pairs = 0

    def update_pbar() -> None:
        pbar.set_postfix_str(
            f"ok={ok_pairs} pairs  skip={skipped_queries}q/{skipped_pairs}p"
        )

    def flush() -> None:
        if not pending_lines:
            return
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            f.write("".join(pending_lines))
            f.flush()
            os.fsync(f.fileno())
        pending_lines.clear()

    t_start = time.monotonic()
    try:
        remaining = total_queries
        while remaining > 0:
            tag, rank, payload = out_q.get()
            if tag == "done":
                for row in payload:
                    pending_lines.append(json.dumps(row, ensure_ascii=False) + "\n")
                ok_queries += 1
                ok_pairs += len(payload)
            elif tag == "skip":
                skipped_queries += 1
                skipped_pairs += payload  # payload is num_pairs
            else:
                continue

            remaining -= 1
            pbar.update(1)
            update_pbar()

            if len(pending_lines) >= WRITE_FLUSH_EVERY:
                flush()

        flush()
    finally:
        pbar.close()
        for p in procs:
            p.join()

    elapsed = time.monotonic() - t_start
    actual_rpm = total_queries / max(elapsed / 60, 0.001)
    logger.info(
        "done. wrote %d pairs (%d queries) to %s  |  "
        "skipped %d queries (%d pairs)  |  "
        "elapsed=%.1f min  actual_rpm=%.1f",
        ok_pairs,
        ok_queries,
        OUTPUT_PATH,
        skipped_queries,
        skipped_pairs,
        elapsed / 60,
        actual_rpm,
    )
    if skipped_queries:
        logger.info(
            "hint: re-run to retry the %d skipped queries "
            "(already-scored pairs will be loaded from checkpoint)",
            skipped_queries,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logger.info("output: %s", OUTPUT_PATH)
    logger.info("model: %s  score_key: %s", MODEL, SCORE_KEY)

    pairs = _collect_unique_pairs()
    done = _load_done_hashes()

    tasks: list[dict[str, str]] = [
        {"hash": h, "query": q, "document": d}
        for h, (q, d) in pairs.items()
        if h not in done
    ]
    logger.info(
        "resume summary: unique=%d done=%d remaining=%d",
        len(pairs),
        len(done),
        len(tasks),
    )

    if not tasks:
        logger.info("nothing to score — exiting")
        return

    query_groups = _group_by_query(tasks)
    logger.info(
        "grouped into %d unique queries (total pairs=%d, max docs/query=%d)",
        len(query_groups),
        len(tasks),
        max(len(v) for v in query_groups.values()),
    )

    items = list(query_groups.items())
    random.shuffle(items)
    query_groups = dict(items)
    logger.info("shuffled query order")

    api_keys = _load_api_keys()
    _run(query_groups, api_keys)


if __name__ == "__main__":
    main()
