"""Re-rank top-K candidate sets file-by-file with a Qwen3.5 reranker and report NDCG@10.

Per file flow: load → score all (query, doc) pairs → reorder → write reranked JSONL →
compute NDCG@10 → print → move to next file. The model is loaded once and reused.

Multi-GPU is achieved by spawning long-lived worker processes (one per CUDA device) that
each hold a full model replica. The main process feeds them one file at a time via queues
and shards each file's pairs across workers for load balance.

Configuration is provided via global constants below — edit the file directly.

Usage:
    uv run python -m evaluate_relevance_on_jina_bench rerank_model_test
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp
from openpyxl import Workbook
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared.prompts import (
    TRAINING_INSTRUCTION,
    TRAINING_SYSTEM_PROMPT,
    render_raw_prompt,
)

from .eval_topk import compute_ndcg10_from_jsonl

# ---------------------------------------------------------------------------
# Global Config — edit these instead of passing CLI flags
# python -m evaluate_relevance_on_jina_bench rerank_model_test
# ---------------------------------------------------------------------------
# python -m evaluate_relevance_on_jina_bench rerank_model_test /root/prism_released_models/Prism-Qwen3-Reranker-4B-exp
# python -m evaluate_relevance_on_jina_bench rerank_model_test /root/prism_released_models/Prism-Qwen3.5-Reranker-0.8B
#
#
MODEL_PATH: str = os.environ.get(
    "MODEL_PATH", "/root/prism_released_models/Prism-Qwen3.5-Reranker-0.8B"
)
# MODEL_PATH: str = "/mnt/data/public_models/Qwen3-Reranker-8B"
INPUT_DIR: str = os.environ.get(
    "INPUT_DIR", "/mnt/data/PrismRerankerV1Data/jina_bench_result"
)
OUTPUT_DIR: str = os.environ.get(
    "OUTPUT_DIR", "/mnt/data/PrismRerankerV1Data/jina_bench_result"
)
BATCH_SIZE: int = 1
MAX_MODEL_LEN: int = 6000

# ---------------------------------------------------------------------------
# Qwen3-Reranker branch (activated when MODEL_PATH contains "Qwen3-Reranker")
# ---------------------------------------------------------------------------
QWEN3_RERANKER_MARKER: str = "Qwen3-Reranker"
QWEN3_RERANKER_MAX_LEN: int = 8192
QWEN3_SYSTEM: str = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)
QWEN3_INSTRUCTION: str = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
QWEN3_PREFIX: str = f"<|im_start|>system\n{QWEN3_SYSTEM}<|im_end|>\n<|im_start|>user\n"
QWEN3_SUFFIX: str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _is_qwen3_reranker(model_path: str) -> bool:
    return QWEN3_RERANKER_MARKER in model_path


def _format_qwen3_content(query: str, doc: str) -> str:
    return f"<Instruct>: {QWEN3_INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"


def _count_lines(fpath: Path) -> int:
    count = 0
    with open(fpath, "rb") as f:
        for _ in f:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Incremental checkpoint helpers
# ---------------------------------------------------------------------------


def _compute_pair_hash(query: str, doc: str) -> str:
    return hashlib.sha256((query + "\0" + doc).encode()).hexdigest()[:16]


def _checkpoint_dir(output_dir: Path, file_name: str) -> Path:
    return output_dir / ".checkpoints" / Path(file_name).stem


def _load_checkpoint(ckpt_dir: Path) -> dict[str, float]:
    """Read all checkpoint JSONL files, return hash -> score mapping."""
    result: dict[str, float] = {}
    if not ckpt_dir.exists():
        return result
    for ckpt_file in sorted(ckpt_dir.glob("*.jsonl")):
        with open(ckpt_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    result[entry["hash"]] = entry["score"]
                except (json.JSONDecodeError, KeyError):
                    continue
    return result


def _append_checkpoint(
    ckpt_path: Path, items: list[dict[str, Any]], scores: list[float]
) -> None:
    with open(ckpt_path, "a", encoding="utf-8") as f:
        for item, score in zip(items, scores):
            entry = {
                "hash": _compute_pair_hash(item["query"], item["doc"]),
                "score": score,
            }
            f.write(json.dumps(entry) + "\n")


def _clear_checkpoint(ckpt_dir: Path) -> None:
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)


def _split_cached_and_new(
    items: list[dict[str, Any]], cached_scores: dict[str, float]
) -> tuple[list[float], list[dict[str, Any]], list[int]]:
    """Separate items into cached (score filled) and new (need scoring).

    Returns (all_scores, new_items, new_indices). ``all_scores`` has cached
    values at the right positions and 0.0 for items that still need scoring.
    ``new_indices[i]`` is the position in the original ``items`` list for
    ``new_items[i]``.
    """
    all_scores: list[float] = [0.0] * len(items)
    new_items: list[dict[str, Any]] = []
    new_indices: list[int] = []
    for idx, item in enumerate(items):
        h = _compute_pair_hash(item["query"], item["doc"])
        if h in cached_scores:
            all_scores[idx] = cached_scores[h]
        else:
            new_items.append(item)
            new_indices.append(idx)
    return all_scores, new_items, new_indices


def _load_file_items(
    fpath: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load one top-K JSONL file into (records, flat per-pair items).

    Each item carries ``qid`` / ``did`` so scores can be stamped back onto the
    original record's ``documents`` list.
    """
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


def _left_pad(
    batch_ids: list[list[int]], pad_id: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(ids) for ids in batch_ids)
    padded: list[list[int]] = []
    masks: list[list[int]] = []
    for ids in batch_ids:
        pad = max_len - len(ids)
        padded.append([pad_id] * pad + ids)
        masks.append([0] * pad + [1] * len(ids))
    return (
        torch.tensor(padded, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


def _load_model(model_path: str, device: str) -> dict[str, Any]:
    """Load tokenizer + model and return a bundle with mode-specific state.

    Branches on ``_is_qwen3_reranker(model_path)``:

    * Qwen3-Reranker — follows the official README: ``flash_attention_2``,
      yes/no ids via ``convert_tokens_to_ids``, pre-tokenized prefix/suffix,
      fixed ``QWEN3_RERANKER_MAX_LEN``.
    * Prism (default) — preserves the existing behavior: ``sdpa``, yes/no
      ids via single-token ``encode``, prompt built from ``render_raw_prompt``.
    """
    is_qwen3 = _is_qwen3_reranker(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

    if is_qwen3:
        # README-exact: no pad-token fallback mutation, no encode() for yes/no.
        yes_id = tokenizer.convert_tokens_to_ids("yes")
        no_id = tokenizer.convert_tokens_to_ids("no")
        prefix_tokens = tokenizer.encode(QWEN3_PREFIX, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(QWEN3_SUFFIX, add_special_tokens=False)
        attn_impl = "flash_attention_2"
        max_len = QWEN3_RERANKER_MAX_LEN
        mode = "qwen3_reranker"
    else:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        yes_ids = tokenizer.encode("yes", add_special_tokens=False)
        no_ids = tokenizer.encode("no", add_special_tokens=False)
        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise ValueError(
                f"Expected single-token 'yes'/'no', got yes={yes_ids}, no={no_ids}"
            )
        yes_id, no_id = yes_ids[0], no_ids[0]
        prefix_tokens = []
        suffix_tokens = []
        attn_impl = "sdpa"
        max_len = MAX_MODEL_LEN
        mode = "prism"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation=attn_impl,
    )
    model.eval()

    return {
        "mode": mode,
        "model": model,
        "tokenizer": tokenizer,
        "yes_id": yes_id,
        "no_id": no_id,
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": suffix_tokens,
        "max_len": max_len,
    }


def _build_batch_ids(
    bundle: dict[str, Any], batch: list[dict[str, Any]]
) -> list[list[int]]:
    """Tokenize one batch of items into per-sample ``input_ids`` lists."""
    tokenizer = bundle["tokenizer"]
    max_len: int = bundle["max_len"]

    if bundle["mode"] == "qwen3_reranker":
        prefix_tokens: list[int] = bundle["prefix_tokens"]
        suffix_tokens: list[int] = bundle["suffix_tokens"]
        content_budget = max_len - len(prefix_tokens) - len(suffix_tokens)
        if content_budget <= 0:
            raise ValueError(
                f"max_len={max_len} too small for prefix/suffix "
                f"({len(prefix_tokens)} + {len(suffix_tokens)})"
            )
        # README-exact: tokenizer(__call__) with longest_first truncation,
        # then prepend prefix and append suffix per sample.
        pairs = [_format_qwen3_content(item["query"], item["doc"]) for item in batch]
        encoded = tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=content_budget,
        )
        return [prefix_tokens + ids + suffix_tokens for ids in encoded["input_ids"]]

    return [
        tokenizer.encode(
            render_raw_prompt(
                item["query"],
                item["doc"],
                instruction=TRAINING_INSTRUCTION,
                system_prompt=TRAINING_SYSTEM_PROMPT,
            ),
            add_special_tokens=False,
        )[:max_len]
        for item in batch
    ]


def _compute_batch_scores(bundle: dict[str, Any], logits: torch.Tensor) -> list[float]:
    """Convert last-token logits to per-sample yes-probabilities."""
    yes_id: int = bundle["yes_id"]
    no_id: int = bundle["no_id"]

    if bundle["mode"] == "qwen3_reranker":
        # README-exact: no dtype cast, 2-way log_softmax over [no, yes], exp of yes.
        true_vector = logits[:, yes_id]
        false_vector = logits[:, no_id]
        stacked = torch.stack([false_vector, true_vector], dim=1)
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
        return log_probs[:, 1].exp().tolist()

    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    yes_p = torch.exp(log_probs[:, yes_id])
    no_p = torch.exp(log_probs[:, no_id])
    return (yes_p / (yes_p + no_p)).tolist()


def _score_items(
    bundle: dict[str, Any],
    items: list[dict[str, Any]],
    batch_size: int,
    device: str,
    desc: str,
    pbar_position: int = 0,
    ckpt_path: Path | None = None,
) -> list[float]:
    """Score every item; branches on ``bundle['mode']`` for Qwen3-Reranker vs Prism."""
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]

    total = len(items)
    scores: list[float] = [0.0] * total
    pbar = tqdm(
        total=total,
        desc=desc,
        unit="pair",
        position=pbar_position,
        dynamic_ncols=True,
        smoothing=0.1,
    )
    for start in range(0, total, batch_size):
        batch = items[start : start + batch_size]
        batch_ids = _build_batch_ids(bundle, batch)
        input_ids, attention_mask = _left_pad(batch_ids, tokenizer.pad_token_id, device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[
                :, -1, :
            ]

        batch_scores = _compute_batch_scores(bundle, logits)
        for i, s in enumerate(batch_scores):
            scores[start + i] = s

        if ckpt_path is not None:
            _append_checkpoint(ckpt_path, batch, batch_scores)

        pbar.update(len(batch))

    pbar.close()
    return scores


def _worker_loop(
    rank: int,
    world_size: int,
    model_path: str,
    batch_size: int,
    in_q: Any,
    out_q: Any,
) -> None:
    """Long-lived GPU worker: load model once, then process per-file tasks."""
    device = f"cuda:{rank}"
    bundle = _load_model(model_path, device)
    out_q.put(("ready", rank, None))

    while True:
        task = in_q.get()
        if task is None:
            break
        file_name, items, ckpt_dir_str = task
        shard = items[rank::world_size]
        ckpt_path: Path | None = None
        if ckpt_dir_str:
            ckpt_path = Path(ckpt_dir_str) / f"scores_rank{rank}.jsonl"
        scores = _score_items(
            bundle,
            shard,
            batch_size,
            device,
            desc=f"GPU{rank} {file_name}",
            pbar_position=rank,
            ckpt_path=ckpt_path,
        )
        out_q.put(("done", rank, scores))


def _emit_reranked(
    records: list[dict[str, Any]],
    items: list[dict[str, Any]],
    scores: list[float],
    out_path: Path,
) -> None:
    """Stamp scores back, sort docs per query, write reranked JSONL."""
    for item, score in zip(items, scores):
        records[item["qid"]]["documents"][item["did"]]["_score"] = score

    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            record["documents"].sort(key=lambda d: d["_score"], reverse=True)
            for d in record["documents"]:
                d.pop("_score", None)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def _run_single_gpu(
    jsonl_paths: list[Path], input_dir: Path, output_dir: Path
) -> list[tuple[str, float]]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    bundle: dict[str, Any] | None = None

    results: list[tuple[str, float]] = []
    for file_idx, fpath in enumerate(jsonl_paths, start=1):
        out_path = output_dir / fpath.name
        if out_path.exists():
            print(
                f"\n=== [{file_idx}/{len(jsonl_paths)}] {fpath.name}: "
                f"cached, skipping scoring ==="
            )
            _evaluate_and_record(input_dir, out_path, results)
            continue

        records, items = _load_file_items(fpath)
        ckpt_dir = _checkpoint_dir(output_dir, fpath.name)
        cached_scores = _load_checkpoint(ckpt_dir)
        all_scores, new_items, new_indices = _split_cached_and_new(
            items, cached_scores
        )
        print(
            f"\n=== [{file_idx}/{len(jsonl_paths)}] {fpath.name}: "
            f"{len(records)} queries, {len(items)} pairs "
            f"({len(items) - len(new_items)} cached, {len(new_items)} new) ==="
        )

        if new_items:
            if bundle is None:
                print(f"Loading model on {device} ...")
                bundle = _load_model(MODEL_PATH, device)
                print(
                    f"Model loaded (mode={bundle['mode']}, "
                    f"max_len={bundle['max_len']})."
                )
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "scores.jsonl"
            new_scores = _score_items(
                bundle,
                new_items,
                BATCH_SIZE,
                device,
                desc=fpath.name,
                ckpt_path=ckpt_path,
            )
            for i, idx in enumerate(new_indices):
                all_scores[idx] = new_scores[i]

        _emit_reranked(records, items, all_scores, out_path)
        _clear_checkpoint(ckpt_dir)
        _evaluate_and_record(input_dir, out_path, results)
    return results


def _run_multi_gpu(
    jsonl_paths: list[Path],
    input_dir: Path,
    output_dir: Path,
    world_size: int,
) -> list[tuple[str, float]]:
    pending = [p for p in jsonl_paths if not (output_dir / p.name).exists()]
    cached = [p for p in jsonl_paths if (output_dir / p.name).exists()]

    results: list[tuple[str, float]] = []
    for fpath in cached:
        out_path = output_dir / fpath.name
        print(f"\n=== {fpath.name}: cached, skipping scoring ===")
        _evaluate_and_record(input_dir, out_path, results)

    if not pending:
        return results

    ctx = mp.get_context("spawn")
    in_qs = [ctx.Queue() for _ in range(world_size)]
    out_q: Any = ctx.Queue()
    procs: list[Any] = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_worker_loop,
            args=(
                rank,
                world_size,
                MODEL_PATH,
                BATCH_SIZE,
                in_qs[rank],
                out_q,
            ),
        )
        p.start()
        procs.append(p)

    ready = 0
    while ready < world_size:
        tag, _, _ = out_q.get()
        if tag == "ready":
            ready += 1
    print(f"All {world_size} workers ready.")

    try:
        for file_idx, fpath in enumerate(pending, start=1):
            records, items = _load_file_items(fpath)
            ckpt_dir = _checkpoint_dir(output_dir, fpath.name)
            cached_scores = _load_checkpoint(ckpt_dir)
            all_scores, new_items, new_indices = _split_cached_and_new(
                items, cached_scores
            )
            print(
                f"\n=== [{file_idx}/{len(pending)}] {fpath.name}: "
                f"{len(records)} queries, {len(items)} pairs "
                f"({len(items) - len(new_items)} cached, "
                f"{len(new_items)} new) ==="
            )

            if new_items:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_dir_str = str(ckpt_dir)
                for rank in range(world_size):
                    in_qs[rank].put((fpath.name, new_items, ckpt_dir_str))

                shard_scores: dict[int, list[float]] = {}
                received = 0
                while received < world_size:
                    tag, rank, payload = out_q.get()
                    if tag == "done":
                        shard_scores[rank] = payload
                        received += 1

                new_scores = [0.0] * len(new_items)
                for rank, s in shard_scores.items():
                    for i, sc in zip(
                        range(rank, len(new_items), world_size), s
                    ):
                        new_scores[i] = sc

                for i, idx in enumerate(new_indices):
                    all_scores[idx] = new_scores[i]

            out_path = output_dir / fpath.name
            _emit_reranked(records, items, all_scores, out_path)
            _clear_checkpoint(ckpt_dir)
            _evaluate_and_record(input_dir, out_path, results)
    finally:
        for rank in range(world_size):
            in_qs[rank].put(None)
        for p in procs:
            p.join()

    return results


def main() -> None:
    """Entry point: rerank top-K JSONLs file-by-file and report NDCG@10."""
    global MODEL_PATH
    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]

    input_dir = Path(INPUT_DIR)
    # Per-model subdirectory so swapping MODEL_PATH never collides with another
    # model's file-level cache.
    run_output_dir = Path(OUTPUT_DIR) / Path(MODEL_PATH).name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {run_output_dir}")

    jsonl_paths = sorted(input_dir.glob("*_top*.jsonl"))
    if not jsonl_paths:
        print(f"No *_top*.jsonl found in {input_dir}")
        return

    jsonl_paths.sort(key=_count_lines)
    print(f"Found {len(jsonl_paths)} files (sorted by query count asc). batch_size={BATCH_SIZE}")

    num_gpus = torch.cuda.device_count() or 1
    world_size = num_gpus
    print(f"Using {world_size} GPU process(es)")

    if world_size <= 1:
        results = _run_single_gpu(jsonl_paths, input_dir, run_output_dir)
    else:
        results = _run_multi_gpu(jsonl_paths, input_dir, run_output_dir, world_size)

    _save_results_xlsx(results, run_output_dir)


if __name__ == "__main__":
    main()
