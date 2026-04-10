"""Batch inference script for Qwen3 / Qwen3.5 Reranker on JSONL test data.

Uses HuggingFace Transformers (no vLLM). For each (query, document) pair:
- Score = softmax(yes_logit, no_logit)[yes] at the first generated token
- Generated text = greedy decoded reasoning output

Supports multi-GPU data parallelism via torch.multiprocessing.

Usage:
    uv run python infer_on_test_data_hf.py
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared.prompts import (
    TRAINING_INSTRUCTION,
    TRAINING_SYSTEM_PROMPT,
    render_raw_prompt,
)

# ---------------------------------------------------------------------------
# Global Config
# ---------------------------------------------------------------------------
MODEL_PATH: str = "/root/qwen3_5_2B_v2-epoch-1"
INPUT_PATH: str = "/mnt/data/PrismRerankerV1Data/final_dev_data.jsonl"
OUTPUT_PATH: str = "/mnt/data/PrismRerankerV1Data/qwen3_5_2B_v2_pred_res.jsonl"

MAX_SAMPLES: int = 400000
MAX_MODEL_LEN: int = 10240
MAX_NEW_TOKENS: int = 2048
BATCH_SIZE: int = 1
NUM_GPUS: int = torch.cuda.device_count() or 1


def build_prompt_ids(
    row: dict[str, Any], tokenizer: AutoTokenizer
) -> list[int]:
    """Render and tokenize a single prompt."""
    raw = render_raw_prompt(
        row["query"],
        row["document"],
        instruction=TRAINING_INSTRUCTION,
        system_prompt=TRAINING_SYSTEM_PROMPT,
    )
    return tokenizer.encode(raw, add_special_tokens=False)[:MAX_MODEL_LEN]


def left_pad_batch(
    batch_ids: list[list[int]], pad_token_id: int, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Left-pad a batch of token lists and return (input_ids, attention_mask, prompt_lens)."""
    prompt_lens = [len(ids) for ids in batch_ids]
    max_len = max(prompt_lens)
    padded_ids: list[list[int]] = []
    masks: list[list[int]] = []
    for ids in batch_ids:
        pad_len = max_len - len(ids)
        padded_ids.append([pad_token_id] * pad_len + ids)
        masks.append([0] * pad_len + [1] * len(ids))
    input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
    return input_ids, attention_mask, prompt_lens


def infer_worker(rank: int, world_size: int, rows: list[dict[str, Any]]) -> None:
    """Run inference on a shard of data on a single GPU."""
    device = f"cuda:{rank}"
    shard = rows[rank::world_size]
    print(f"[GPU {rank}] Processing {len(shard)} samples ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    no_ids = tokenizer.encode("no", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError(
            f"Expected single-token 'yes'/'no', got yes={yes_ids}, no={no_ids}"
        )
    yes_token_id, no_token_id = yes_ids[0], no_ids[0]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"[GPU {rank}] Model loaded on {device}.")

    results: list[dict[str, Any]] = []
    total = len(shard)
    for batch_start in range(0, total, BATCH_SIZE):
        batch_rows = shard[batch_start : batch_start + BATCH_SIZE]
        batch_ids = [build_prompt_ids(row, tokenizer) for row in batch_rows]

        input_ids, attention_mask, _ = left_pad_batch(
            batch_ids, tokenizer.pad_token_id, device
        )

        with torch.no_grad():
            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        first_logits = gen_out.scores[0].float()
        first_logprobs = torch.log_softmax(first_logits, dim=-1)
        yes_lps = first_logprobs[:, yes_token_id]
        no_lps = first_logprobs[:, no_token_id]
        yes_ps = torch.exp(yes_lps)
        no_ps = torch.exp(no_lps)
        scores = (yes_ps / (yes_ps + no_ps)).tolist()

        padded_prompt_len = input_ids.shape[1]
        for j, row in enumerate(batch_rows):
            gen_ids = gen_out.sequences[j, padded_prompt_len:].tolist()
            pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            row["pred_score"] = scores[j]
            row["pred_text"] = pred_text
            results.append(row)

        done = min(batch_start + BATCH_SIZE, total)
        print(f"[GPU {rank}] [{done}/{total}] last_score={scores[-1]:.6f}")

    # Write shard results to a temp file
    tmp_path = f"{OUTPUT_PATH}.{rank}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[GPU {rank}] Done. Shard saved to {tmp_path}")


def main() -> None:
    with open(INPUT_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f.readlines()[:MAX_SAMPLES]]

    # Tag each row with its original index for ordered merge
    for i, row in enumerate(rows):
        row["_original_idx"] = i

    world_size = min(NUM_GPUS, len(rows))
    print(f"Total samples: {len(rows)}, using {world_size} GPU(s), batch_size={BATCH_SIZE}")

    if world_size <= 1:
        # Single GPU: run directly without spawning
        infer_worker(0, 1, rows)
    else:
        mp.spawn(infer_worker, args=(world_size, rows), nprocs=world_size, join=True)

    # Merge shard results in original order
    all_results: list[dict[str, Any]] = []
    for rank in range(world_size):
        tmp_path = f"{OUTPUT_PATH}.{rank}"
        with open(tmp_path, encoding="utf-8") as f:
            for line in f:
                all_results.append(json.loads(line))
        os.remove(tmp_path)

    all_results.sort(key=lambda r: r["_original_idx"])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in all_results:
            del row["_original_idx"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(all_results)} results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
