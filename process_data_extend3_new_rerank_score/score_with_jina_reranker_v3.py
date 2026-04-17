"""Score every unique (query, document) pair across the step6~step9 data files.

Use jina-reranker-v3 (``model.rerank`` API). Output is a single append-only
JSONL with one row per unique pair, keyed by an md5 hash so the run is
resumable: on restart, already-scored hashes are skipped.

Multi-GPU: one long-lived worker per CUDA device, each holding a full model
replica. Pairs are sorted by (query+document) char length then dealt
round-robin to workers so every shard stays balanced.

IMPORTANT: each ``model.rerank`` call receives exactly ONE document to avoid
score fluctuation caused by listwise normalisation.

Run: ``uv run python -m process_data_extend3_new_rerank_score.score_with_jina_reranker_v3``
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
from transformers import AutoModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH: str = "/mnt/data/public_models/jina-reranker-v3"
OUTPUT_PATH: Path = Path(
    "/mnt/data/PrismRerankerV1Data/data_extend3_new_rerank_score/"
    "jina_reranker_v3_scores.jsonl"
)
SCORE_KEY: str = "jina-reranker-v3_score"

INPUT_FILES: list[str] = [
    "/mnt/data/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs.jsonl",
    "/mnt/data/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs_balanced.jsonl",
    "/mnt/data/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs_no_medical.jsonl",
    "/mnt/data/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs_no_medical_length-score-balance.jsonl",
    "/mnt/data/PrismRerankerV1Data/step7_kalm_web-search_query_document_pairs_annotated.jsonl",
    "/mnt/data/PrismRerankerV1Data/step8_kalm_web-search_query_document_pairs_annotated_merged.jsonl",
    "/mnt/data/PrismRerankerV1Data/step9_kalm_web-search_query_document_pairs_contribution_evidence.jsonl",
    "/mnt/data/PrismRerankerV1Data/data_extend2/step6_expanded2_web-search_query_document_pairs.jsonl",
    "/mnt/data/PrismRerankerV1Data/data_extend2/step6_expanded2_web-search_query_document_pairs_length-score-balance.jsonl",
    "/mnt/data/PrismRerankerV1Data/data_extend2/step7_expanded2_web-search_query_document_pairs_annotated.jsonl",
    "/mnt/data/PrismRerankerV1Data/data_extend2/step8_expanded2_web-search_query_document_pairs_annotated_merged.jsonl",
    "/mnt/data/PrismRerankerV1Data/data_extend2/step9_expanded2_web-search_query_document_contribution_evidence.jsonl",
]

WRITE_FLUSH_EVERY: int = 1024

logger = logging.getLogger("jina_reranker_v3_score")


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


def _load_jina_model(model_path: str, device: str) -> AutoModel:
    """Load jina-reranker-v3 onto the specified device."""
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)
    return model


def _worker_loop(
    rank: int,
    model_path: str,
    in_q: Any,
    out_q: Any,
) -> None:
    """GPU worker: load model once, then score one pair at a time until sentinel."""
    device = f"cuda:{rank}"
    model = _load_jina_model(model_path, device)
    out_q.put(("ready", rank, None))

    while True:
        task = in_q.get()
        if task is None:
            break
        # task: single dict(hash, query, document)
        query = task["query"]
        document = task["document"]
        with torch.no_grad():
            results = model.rerank(query, [document])
        score = results[0]["relevance_score"]
        out_q.put(
            (
                "done",
                rank,
                {
                    "hash": task["hash"],
                    "query": query,
                    "document": document,
                    SCORE_KEY: float(score),
                },
            )
        )


def _make_shards(
    tasks: list[dict[str, str]], world_size: int
) -> list[list[dict[str, str]]]:
    """Sort by length, shard round-robin across GPUs.

    Returns ``shards[rank]`` = list of single task dicts.
    """
    tasks.sort(key=lambda t: len(t["query"]) + len(t["document"]))
    shards: list[list[dict[str, str]]] = []
    for rank in range(world_size):
        shards.append(tasks[rank::world_size])
    return shards


def _run(tasks: list[dict[str, str]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    world_size = torch.cuda.device_count() or 1
    logger.info("world_size=%d model=%s", world_size, MODEL_PATH)

    shards = _make_shards(tasks, world_size)
    total_tasks = sum(len(s) for s in shards)
    logger.info("prepared %d tasks total", total_tasks)

    ctx = mp.get_context("spawn")
    in_qs = [ctx.Queue() for _ in range(world_size)]
    out_q: Any = ctx.Queue()

    procs = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker_loop,
            args=(rank, MODEL_PATH, in_qs[rank], out_q),
        )
        p.start()
        procs.append(p)

    ready = 0
    while ready < world_size:
        tag, rank, _ = out_q.get()
        if tag == "ready":
            ready += 1
            logger.info("worker rank=%d ready", rank)

    for rank in range(world_size):
        for task in shards[rank]:
            in_qs[rank].put(task)
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
        tasks_left = total_tasks
        while tasks_left > 0:
            tag, rank, payload = out_q.get()
            if tag != "done":
                continue
            pending_lines.append(json.dumps(payload, ensure_ascii=False) + "\n")
            pbar.update(1)
            tasks_left -= 1
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
