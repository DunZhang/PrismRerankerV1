"""Score every unique (query, document) pair across the step6~step9 data files.

Use Qwen3-Reranker-4B (prompt / scoring logic reused from
``evaluate_relevance_on_jina_bench.rerank_model_test``). Output is a single
append-only JSONL with one row per unique pair, keyed by an md5 hash so the
run is resumable: on restart, already-scored hashes are skipped.

Multi-GPU: one long-lived worker per CUDA device, each holding a full model
replica. Pairs are sorted by (query+document) char length so each batch groups
similar-length samples for minimal padding waste, then dealt round-robin to
workers so every shard keeps the length-locality and stays balanced.

Edit the constants below to reconfigure — no CLI flags.

Run: ``uv run python -m process_data_extend3_new_rerank_score.score_with_qwen3_reranker_4b``
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from evaluate_relevance_on_jina_bench.rerank_model_test import (
    _build_batch_ids,
    _compute_batch_scores,
    _left_pad,
    _load_model,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH: str = "/mnt/data/public_models/Qwen3-Reranker-4B"
OUTPUT_PATH: Path = Path(
    "/mnt/data/PrismRerankerV1Data/data_extend3_new_rerank_score/"
    "qwen3_reranker_4b_scores.jsonl"
)
BATCH_SIZE: int = 1
SCORE_KEY: str = "Qwen3-Reranker-4B_score"

INPUT_FILES: list[str] = [
    "/mnt/g/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs.jsonl",
    "/mnt/g/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs_balanced.jsonl",
    "/mnt/g/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs_no_medical.jsonl",
    "/mnt/g/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs_no_medical_length-score-balance.jsonl",
    "/mnt/g/PrismRerankerV1Data/step7_kalm_web-search_query_document_pairs_annotated.jsonl",
    "/mnt/g/PrismRerankerV1Data/step8_kalm_web-search_query_document_pairs_annotated_merged.jsonl",
    "/mnt/g/PrismRerankerV1Data/step9_kalm_web-search_query_document_pairs_contribution_evidence.jsonl",
    "/mnt/g/PrismRerankerV1Data/data_extend2/step6_expanded2_web-search_query_document_pairs.jsonl",
    "/mnt/g/PrismRerankerV1Data/data_extend2/step6_expanded2_web-search_query_document_pairs_length-score-balance.jsonl",
    "/mnt/g/PrismRerankerV1Data/data_extend2/step7_expanded2_web-search_query_document_pairs_annotated.jsonl",
    "/mnt/g/PrismRerankerV1Data/data_extend2/step8_expanded2_web-search_query_document_pairs_annotated_merged.jsonl",
    "/mnt/g/PrismRerankerV1Data/data_extend2/step9_expanded2_web-search_query_document_contribution_evidence.jsonl",
]
# INPUT_FILES: list[str] = [
#     "/mnt/data/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs.jsonl",
#     "/mnt/data/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs_balanced.jsonl",
#     "/mnt/data/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs_no_medical.jsonl",
#     "/mnt/data/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs_no_medical_length-score-balance.jsonl",
#     "/mnt/data/PrismRerankerV1Data/step7_kalm_web-search_query_document_pairs_annotated.jsonl",
#     "/mnt/data/PrismRerankerV1Data/step8_kalm_web-search_query_document_pairs_annotated_merged.jsonl",
#     "/mnt/data/PrismRerankerV1Data/step9_kalm_web-search_query_document_pairs_contribution_evidence.jsonl",
#     "/mnt/data/PrismRerankerV1Data/data_extend2/step6_expanded2_web-search_query_document_pairs.jsonl",
#     "/mnt/data/PrismRerankerV1Data/data_extend2/step6_expanded2_web-search_query_document_pairs_length-score-balance.jsonl",
#     "/mnt/data/PrismRerankerV1Data/data_extend2/step7_expanded2_web-search_query_document_pairs_annotated.jsonl",
#     "/mnt/data/PrismRerankerV1Data/data_extend2/step8_expanded2_web-search_query_document_pairs_annotated_merged.jsonl",
#     "/mnt/data/PrismRerankerV1Data/data_extend2/step9_expanded2_web-search_query_document_contribution_evidence.jsonl",
# ]
# Writer drains at least this many items from the result queue per flush, so
# the append-write + fsync amortizes the per-line IO cost.
WRITE_FLUSH_EVERY: int = 1024

logger = logging.getLogger("qwen3_rerank_score")


def _pair_hash(query: str, document: str) -> str:
    payload = (query + "\x1f" + document).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def _collect_unique_pairs() -> dict[str, tuple[str, str]]:
    """Read every input file; return {hash: (query, document)} deduped globally."""
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


def _worker_loop(
    rank: int,
    model_path: str,
    batch_size: int,
    in_q: Any,
    out_q: Any,
) -> None:
    """GPU worker: load model once, then score batches until sentinel arrives."""
    device = f"cuda:{rank}"
    bundle = _load_model(model_path, device)
    tokenizer = bundle["tokenizer"]
    out_q.put(("ready", rank, None))

    while True:
        task = in_q.get()
        if task is None:
            break
        # task: list[ dict(hash, query, document) ] — one full batch
        batch_items = [{"query": t["query"], "doc": t["document"]} for t in task]
        batch_ids = _build_batch_ids(bundle, batch_items)
        input_ids, attention_mask = _left_pad(batch_ids, tokenizer.pad_token_id, device)
        with torch.no_grad():
            logits = bundle["model"](
                input_ids=input_ids, attention_mask=attention_mask
            ).logits[:, -1, :]
        scores = _compute_batch_scores(bundle, logits)
        out_q.put(
            (
                "done",
                rank,
                [
                    {
                        "hash": t["hash"],
                        "query": t["query"],
                        "document": t["document"],
                        SCORE_KEY: float(s),
                    }
                    for t, s in zip(task, scores)
                ],
            )
        )


def _make_batches(
    tasks: list[dict[str, str]], world_size: int, batch_size: int
) -> list[list[list[dict[str, str]]]]:
    """Sort by length, shard round-robin across GPUs, chunk each shard into batches.

    Returns ``shards[rank]`` = list of batches (each batch is a list of task dicts).
    The round-robin shard keeps length locality (adjacent items have similar length)
    while balancing load across GPUs.
    """
    tasks.sort(key=lambda t: len(t["query"]) + len(t["document"]))
    shards: list[list[list[dict[str, str]]]] = []
    for rank in range(world_size):
        shard_items = tasks[rank::world_size]
        batches = [
            shard_items[i : i + batch_size]
            for i in range(0, len(shard_items), batch_size)
        ]
        shards.append(batches)
    return shards


def _run(tasks: list[dict[str, str]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    world_size = torch.cuda.device_count() or 1
    logger.info(
        "world_size=%d batch_size=%d model=%s", world_size, BATCH_SIZE, MODEL_PATH
    )

    shards = _make_batches(tasks, world_size, BATCH_SIZE)
    total_batches = sum(len(s) for s in shards)
    logger.info("prepared %d batches total", total_batches)

    ctx = mp.get_context("spawn")
    in_qs = [ctx.Queue() for _ in range(world_size)]
    out_q: Any = ctx.Queue()

    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker_loop,
            args=(rank, MODEL_PATH, BATCH_SIZE, in_qs[rank], out_q),
        )
        p.start()
        procs.append(p)

    ready = 0
    while ready < world_size:
        tag, rank, _ = out_q.get()
        if tag == "ready":
            ready += 1
            logger.info("worker rank=%d ready", rank)

    # Enqueue every batch up front — they're small (just dicts), so this is cheap
    # and lets each worker stream through its shard without waiting on the main
    # loop to feed it.
    for rank in range(world_size):
        for batch in shards[rank]:
            in_qs[rank].put(batch)
        in_qs[rank].put(None)  # sentinel

    pbar = tqdm(total=len(tasks), desc="scoring", unit="pair", dynamic_ncols=True)
    pending_lines: list[str] = []

    def flush() -> None:
        if not pending_lines:
            return
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            f.write("".join(pending_lines))
            f.flush()
            os.fsync(f.fileno())
        pending_lines.clear()

    try:
        batches_left = total_batches
        while batches_left > 0:
            tag, rank, payload = out_q.get()
            if tag != "done":
                continue
            for row in payload:
                pending_lines.append(json.dumps(row, ensure_ascii=False) + "\n")
            pbar.update(len(payload))
            batches_left -= 1
            if len(pending_lines) >= WRITE_FLUSH_EVERY:
                flush()
        flush()
    finally:
        pbar.close()
        for p in procs:
            p.join()

    logger.info("done. wrote %d new rows to %s", len(tasks), OUTPUT_PATH)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger.info("output: %s", OUTPUT_PATH)

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

    _run(tasks)


if __name__ == "__main__":
    main()
