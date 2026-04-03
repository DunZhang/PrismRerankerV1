"""Batch inference script for Qwen3-Reranker on JSONL test data.

Usage:
    uv run python infer_on_test_data.py
"""

from __future__ import annotations

import json
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared.prompts import (
    TRAINING_INSTRUCTION,
    TRAINING_SYSTEM_PROMPT,
    render_raw_prompt,
)

# ---------------------------------------------------------------------------
# Global Config
# ---------------------------------------------------------------------------
MODEL_PATH: str = "/root/epoch-3-merged/"
INPUT_PATH: str = "/mnt/data/PrismRerankerV1Data/test_data_for_only_sft_model.jsonl"
OUTPUT_PATH: str = "/mnt/data/PrismRerankerV1Data/test_data_for_only_sft_model_result.jsonl"

MAX_NEW_TOKENS: int = 1024
YES_TOKEN_ID: int = 9693
NO_TOKEN_ID: int = 2152


def run_single_inference(
    query: str,
    doc: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> tuple[float, str]:
    """Run reranker inference on a single query-document pair.

    Returns:
        A tuple of (score, generated_text).
    """
    prompt = render_raw_prompt(
        query,
        doc,
        instruction=TRAINING_INSTRUCTION,
        system_prompt=TRAINING_SYSTEM_PROMPT,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    yes_no = logits[:, [YES_TOKEN_ID, NO_TOKEN_ID]]
    probs = torch.softmax(yes_no, dim=1)
    score = probs[0, 0].item()

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    generated_text = tokenizer.decode(
        gen_ids[0][input_len:], skip_special_tokens=True,
    )

    return score, generated_text


def main() -> None:
    print(f"Loading model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    print("Model loaded.\n")

    with open(INPUT_PATH, encoding="utf-8") as f:
        lines = f.readlines()[:400]

    total = len(lines)
    print(f"Total samples: {total}")

    results: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        row = json.loads(line)
        query = row["query"]
        doc = row["document"]

        score, generated_text = run_single_inference(query, doc, model, tokenizer)
        row["pred_score"] = score
        row["pred_text"] = generated_text
        results.append(row)

        print(f"[{i + 1}/{total}] score={score:.6f} | {query[:60]}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDone. Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
