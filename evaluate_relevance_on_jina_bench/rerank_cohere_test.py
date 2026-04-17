"""Re-rank top-K candidate sets using OpenRouter's Cohere rerank endpoint.

Serial, single-key implementation (OpenRouter key has no strict rate limit).
Each query sends all 100 documents in one request.

Per file flow: load → score each query via one API call → reorder →
write reranked JSONL → compute NDCG@10 → print.

Robust retry: every API call retries on any failure with short backoff.
Per-query progress is saved so interrupted runs resume without re-scoring
completed queries.

Usage:
    uv run python -m evaluate_relevance_on_jina_bench.rerank_cohere_test
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import dotenv_values
from openpyxl import Workbook
from tqdm import tqdm

from .eval_topk import compute_ndcg10_from_jsonl

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL: str = "https://openrouter.ai/api/v1/rerank"
MODEL: str = "cohere/rerank-4-fast"
INPUT_DIR: str = "/mnt/g/PrismRerankerV1Data/jina_bench_result"
OUTPUT_DIR: str = "/mnt/g/PrismRerankerV1Data/jina_bench_result"
OUTPUT_SUBDIR: str = "cohere-rerank-v4-fast"

REQUEST_TIMEOUT: float = 120.0
RETRY_DELAY: float = 3.0
MAX_RETRY_DELAY: float = 30.0

logger = logging.getLogger("cohere_rerank_bench")


# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------
def _load_api_key() -> str:
    """Load OPENROUTER_API_KEY from .env."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    env_vars = dotenv_values(env_path)
    key = env_vars.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(f"OPENROUTER_API_KEY not found in {env_path}")
    logger.info("using OpenRouter key: ****%s", key[-4:])
    return key


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_file_items(
    fpath: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load one top-K JSONL file into (records, flat per-pair items)."""
    records: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    with open(fpath, encoding="utf-8") as f:
        for qid, line in enumerate(f):
            record = json.loads(line)
            records.append(record)
            for did, doc in enumerate(record["documents"]):
                items.append(
                    {
                        "qid": qid,
                        "did": did,
                        "query": record["query"],
                        "doc": doc["content"],
                    }
                )
    return records, items


def _group_items_by_query(
    items: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    """Group items by qid. Returns {qid: [item, ...]}."""
    groups: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(item["qid"], []).append(item)
    return groups


# ---------------------------------------------------------------------------
# Per-file progress (per-query resume)
# ---------------------------------------------------------------------------
def _progress_path(output_dir: Path, jsonl_name: str) -> Path:
    return output_dir / f".progress_{jsonl_name}.json"


def _load_progress(output_dir: Path, jsonl_name: str) -> dict[int, list[float]]:
    """Load already-scored qid -> scores mapping from progress file."""
    ppath = _progress_path(output_dir, jsonl_name)
    if not ppath.exists():
        return {}
    done: dict[int, list[float]] = {}
    with open(ppath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done[obj["qid"]] = obj["scores"]
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def _append_progress(
    output_dir: Path, jsonl_name: str, qid: int, scores: list[float]
) -> None:
    """Append one completed query's scores to the progress file."""
    ppath = _progress_path(output_dir, jsonl_name)
    with open(ppath, "a", encoding="utf-8") as f:
        f.write(json.dumps({"qid": qid, "scores": scores}, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _cleanup_progress(output_dir: Path, jsonl_name: str) -> None:
    """Remove progress file after successful completion."""
    ppath = _progress_path(output_dir, jsonl_name)
    if ppath.exists():
        ppath.unlink()


# ---------------------------------------------------------------------------
# OpenRouter rerank call with retry
# ---------------------------------------------------------------------------
def _rerank(
    session: requests.Session,
    api_key: str,
    query: str,
    documents: list[str],
) -> list[float]:
    """Call OpenRouter rerank; retry forever on any error with short backoff.

    Returns scores in the same order as ``documents``.
    """
    payload = {
        "model": MODEL,
        "query": query,
        "documents": documents,
        "top_n": len(documents),
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    attempt = 0
    while True:
        attempt += 1
        try:
            resp = session.post(
                API_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            if "results" not in data:
                raise ValueError(f"unexpected response: {str(data)[:300]}")
            score_map: dict[int, float] = {
                r["index"]: r["relevance_score"] for r in data["results"]
            }
            return [float(score_map[i]) for i in range(len(documents))]
        except Exception as exc:
            delay = min(RETRY_DELAY + random.uniform(0, 2), MAX_RETRY_DELAY)
            logger.warning(
                "rerank failed (attempt=%d): %s — retrying in %.1fs",
                attempt,
                str(exc)[:200],
                delay,
            )
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Reranked output
# ---------------------------------------------------------------------------
def _emit_reranked(
    records: list[dict[str, Any]],
    all_scores: dict[int, list[float]],
    out_path: Path,
) -> None:
    """Stamp scores back, sort docs per query, write reranked JSONL."""
    for qid, record in enumerate(records):
        scores = all_scores[qid]
        for did, doc in enumerate(record["documents"]):
            doc["_score"] = scores[did]

    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            record["documents"].sort(key=lambda d: d["_score"], reverse=True)
            for d in record["documents"]:
                d.pop("_score", None)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# NDCG evaluation
# ---------------------------------------------------------------------------
def _qrels_path(input_dir: Path, jsonl_name: str) -> Path:
    """``mteb__nfcorpus_top100.jsonl`` -> ``mteb__nfcorpus_qrels.json``."""
    stem = Path(jsonl_name).stem.rsplit("_top", 1)[0]
    return input_dir / f"{stem}_qrels.json"


def _evaluate_and_record(
    input_dir: Path,
    out_path: Path,
    results: list[tuple[str, float]],
) -> None:
    """Compute NDCG@10 for one reranked file and append to ``results``."""
    qrels_path = _qrels_path(input_dir, out_path.name)
    if not qrels_path.exists():
        print(f"Skipping {out_path.name} (qrels not found: {qrels_path.name})")
        return
    with open(qrels_path, encoding="utf-8") as f:
        qrels = json.load(f)
    score = compute_ndcg10_from_jsonl(out_path, qrels)
    results.append((out_path.name, score))
    print(f"{out_path.name}: NDCG@10 = {score:.4f}")


def _save_results_xlsx(results: list[tuple[str, float]], output_dir: Path) -> None:
    """Save NDCG@10 results to xlsx."""
    if not results:
        print("No NDCG results computed.")
        return
    wb = Workbook()
    ws = wb.active
    ws.title = "NDCG@10"
    ws.append(["File", "NDCG@10"])
    for name, score in results:
        ws.append([name, round(score, 4)])
    avg = sum(s for _, s in results) / len(results)
    ws.append(["Average", round(avg, 4)])

    xlsx_path = output_dir / "ndcg10_results.xlsx"
    wb.save(xlsx_path)
    print(f"\nResults saved to {xlsx_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point: rerank top-K JSONLs file-by-file via OpenRouter."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    input_dir = Path(INPUT_DIR)
    run_output_dir = Path(OUTPUT_DIR) / OUTPUT_SUBDIR
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {run_output_dir}")

    jsonl_paths = sorted(input_dir.glob("*_top*.jsonl"))
    if not jsonl_paths:
        print(f"No *_top*.jsonl found in {input_dir}")
        return

    def _line_count(p: Path) -> int:
        with open(p, encoding="utf-8") as f:
            return sum(1 for _ in f)

    jsonl_paths.sort(key=_line_count)
    print(f"Found {len(jsonl_paths)} files (sorted by query count, ascending):")
    for i, p in enumerate(jsonl_paths, 1):
        print(f"  {i}. {p.name} ({_line_count(p)} queries)")

    api_key = _load_api_key()
    session = requests.Session()
    print(f"Model: {MODEL}  |  endpoint: {API_URL}")

    results: list[tuple[str, float]] = []

    for file_idx, fpath in enumerate(jsonl_paths, start=1):
        out_path = run_output_dir / fpath.name

        if out_path.exists():
            print(
                f"\n=== [{file_idx}/{len(jsonl_paths)}] {fpath.name}: "
                f"cached, skipping scoring ==="
            )
            _evaluate_and_record(input_dir, out_path, results)
            continue

        records, items = _load_file_items(fpath)
        query_groups = _group_items_by_query(items)
        print(
            f"\n=== [{file_idx}/{len(jsonl_paths)}] {fpath.name}: "
            f"{len(records)} queries, {len(items)} pairs ==="
        )

        done_scores = _load_progress(run_output_dir, fpath.name)
        if done_scores:
            print(f"  Resuming: {len(done_scores)}/{len(query_groups)} queries done")

        pbar = tqdm(
            total=len(query_groups),
            initial=len(done_scores),
            desc=fpath.name,
            unit="query",
            dynamic_ncols=True,
        )

        for qid in sorted(query_groups.keys()):
            if qid in done_scores:
                continue

            group = query_groups[qid]
            query = group[0]["query"]
            documents = [item["doc"] for item in group]

            scores = _rerank(session, api_key, query, documents)
            done_scores[qid] = scores
            _append_progress(run_output_dir, fpath.name, qid, scores)
            pbar.update(1)

        pbar.close()

        _emit_reranked(records, done_scores, out_path)
        _cleanup_progress(run_output_dir, fpath.name)
        _evaluate_and_record(input_dir, out_path, results)

    _save_results_xlsx(results, run_output_dir)


if __name__ == "__main__":
    main()
