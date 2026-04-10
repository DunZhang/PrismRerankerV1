"""Batch inference script for Qwen3 / Qwen3.5 Reranker on JSONL test data.

Uses SGLang offline engine for fast batched inference. For each (query, document) pair:
- Score = softmax(yes_logit, no_logit)[yes] at the first generated token
- Generated text = greedy decoded reasoning output

Usage:
    uv run python infer_on_test_data_sglang.py
"""

from __future__ import annotations

import json
import math
from typing import Any

import sglang as sgl

from shared.prompts import (
    TRAINING_INSTRUCTION,
    TRAINING_SYSTEM_PROMPT,
    render_raw_prompt,
)

# ---------------------------------------------------------------------------
# Global Config
# ---------------------------------------------------------------------------
MODEL_PATH: str = "/root/Qwen3.5-2B-epoch-1"
INPUT_PATH: str = "/mnt/data/PrismRerankerV1Data/final_dev_data.jsonl"
OUTPUT_PATH: str = "/mnt/data/PrismRerankerV1Data/valid_3_5_pred_res.jsonl"

MAX_SAMPLES: int = 4000000
MAX_MODEL_LEN: int = 10240
MAX_NEW_TOKENS: int = 2048
GPU_MEMORY_UTILIZATION: float = 0.6


def build_prompts(
    rows: list[dict[str, Any]], tokenizer: Any
) -> list[dict[str, list[int]]]:
    """Render and tokenize prompts for all rows."""
    prompts: list[dict[str, list[int]]] = []
    for row in rows:
        raw = render_raw_prompt(
            row["query"],
            row["document"],
            instruction=TRAINING_INSTRUCTION,
            system_prompt=TRAINING_SYSTEM_PROMPT,
        )
        ids = tokenizer.encode(raw, add_special_tokens=False)[:MAX_MODEL_LEN]
        prompts.append({"input_ids": ids})
    return prompts


def main() -> None:
    import torch

    tp_size = max(1, torch.cuda.device_count())
    print(f"Loading SGLang engine from {MODEL_PATH} (tp={tp_size}) ...")

    llm = sgl.Engine(
        model_path=MODEL_PATH,
        tp_size=tp_size,
        mem_fraction_static=GPU_MEMORY_UTILIZATION,
        context_length=MAX_MODEL_LEN + MAX_NEW_TOKENS + 64,
    )
    tokenizer = llm.tokenizer_manager.tokenizer

    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError(
            f"Expected single-token 'yes'/'no', got yes={yes_ids}, no={no_ids}"
        )
    yes_token_id, no_token_id = yes_ids[0], no_ids[0]
    print(f"yes_id={yes_token_id}, no_id={no_token_id}")

    sampling_params: dict[str, Any] = {
        "temperature": 0,
        "max_new_tokens": MAX_NEW_TOKENS,
    }

    with open(INPUT_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f.readlines()[:MAX_SAMPLES]]
    print(f"Total samples: {len(rows)}")

    prompts = build_prompts(rows, tokenizer)
    outputs = llm.generate(
        input_ids=[p["input_ids"] for p in prompts],
        sampling_params=sampling_params,
        return_logprob=True,
        top_logprobs_num=20,
    )

    results: list[dict[str, Any]] = []
    for i, (row, output) in enumerate(zip(rows, outputs)):
        meta = output["meta_info"]
        first_top_logprobs = meta["output_top_logprobs"][0]

        # SGLang format: list of (logprob, token_id, token_str) tuples
        logprob_map: dict[int, float] = {
            tok_id: lp for lp, tok_id, _ in first_top_logprobs
        }
        yes_lp = logprob_map.get(yes_token_id, -10.0)
        no_lp = logprob_map.get(no_token_id, -10.0)
        yes_p = math.exp(yes_lp)
        no_p = math.exp(no_lp)
        score = yes_p / (yes_p + no_p)

        row["pred_score"] = score
        row["pred_text"] = output["text"]
        results.append(row)

        print(f"[{i + 1}/{len(rows)}] score={score:.6f} | {row['query'][:60]}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    llm.shutdown()
    print(f"\nDone. Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
