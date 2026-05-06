"""Verify that Prism reranker scores match between HF direct usage and CrossEncoder.

Both paths must produce the same final score:
    score = sigmoid(logit_yes - logit_no)

Run with:  uv run python verify_cross_encoder.py [MODEL_PATH]
"""

from __future__ import annotations

import sys

import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = (
    sys.argv[1] if len(sys.argv) > 1
    else "/mnt/g/prism_released_models/Prism-Qwen3.5-Reranker-0.8B"
)

SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on "
    "the Query and the Instruct provided. "
)
# Hardcoded training instruction. The chat_template.jinja embeds this verbatim,
# so the HF baseline must use the same string to produce matching scores.
INSTRUCTION = (
    'Judge if the document is relevant to the query. Reply "yes" or "no".\n'
    'On "yes", also emit:\n'
    "<contribution>One sentence covering every core point the document "
    "contributes to the query, without elaboration.</contribution>\n"
    "<evidence>Self-contained rewrite of the query-relevant content. Rules:\n"
    "- Faithful: rephrase only; add or infer nothing.\n"
    "- Self-contained: evidence alone must fully answer the query.\n"
    "- Concise: drop query-irrelevant background.\n"
    "- Verbatim (no translation): proper nouns, terms, abbreviations, "
    "numbers, dates, code, URLs.\n"
    "- Output language: multilingual doc → query's language; else doc's language."
    "</evidence>"
)
PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n"
    "<Instruct>: {instruction}\n"
    "<Query>: {query}\n"
    "<Document>: {doc}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

PAIRS: list[tuple[str, str]] = [
    (
        "What is the capital of China?",
        "The capital of China is Beijing.",
    ),
    (
        "What is the capital of China?",
        "Gravity is a force that attracts two bodies towards each other.",
    ),
    (
        "What is the boiling point of water at sea level?",
        (
            "Water boils at 100 C (212 F) at standard atmospheric pressure (1 atm), "
            "which corresponds to sea-level conditions."
        ),
    ),
    (
        "Explain photosynthesis briefly.",
        "Photosynthesis converts light energy into chemical energy stored in glucose.",
    ),
    (
        "Explain photosynthesis briefly.",
        "Apples are a popular fruit grown in temperate climates.",
    ),
]


@torch.no_grad()
def hf_scores(pairs: list[tuple[str, str]], dtype: torch.dtype) -> list[float]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map="cuda",
        attn_implementation="sdpa",
    ).eval()

    yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("no", add_special_tokens=False)[0]

    scores: list[float] = []
    for query, doc in pairs:
        prompt = PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            instruction=INSTRUCTION,
            query=query,
            doc=doc,
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        # Stay in the model's native dtype for the diff so we match CrossEncoder's
        # LogitScore module, which subtracts in the model dtype before activation.
        logits = model(input_ids=input_ids).logits[0, -1]
        diff = logits[yes_id] - logits[no_id]
        scores.append(torch.sigmoid(diff).float().item())

    del model
    torch.cuda.empty_cache()
    return scores


def ce_scores(pairs: list[tuple[str, str]], dtype: torch.dtype) -> list[float]:
    ce = CrossEncoder(MODEL_PATH, model_kwargs={"torch_dtype": dtype})
    return ce.predict(pairs, batch_size=8, convert_to_numpy=True).tolist()


def _compare(hf: list[float], ce: list[float], tol: float, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"{'idx':<4} {'hf_score':<12} {'ce_score':<12} {'diff':<12}")
    print("-" * 46)
    max_diff = 0.0
    for i, (h, c) in enumerate(zip(hf, ce)):
        diff = abs(h - c)
        max_diff = max(max_diff, diff)
        print(f"{i:<4} {h:<12.6f} {c:<12.6f} {diff:<12.2e}")
    print(f"max |hf - ce| = {max_diff:.2e} (tol = {tol:.0e})")
    assert max_diff < tol, f"Scores diverge: max diff {max_diff:.2e} > {tol:.0e}"
    print(f"PASS: HF scores match CrossEncoder scores ({label}).")


def main() -> None:
    print(f"Model: {MODEL_PATH}")

    # 1. fp32 path: arithmetic is exact, scores must match to ~1e-6
    hf32 = hf_scores(PAIRS, torch.float32)
    ce32 = ce_scores(PAIRS, torch.float32)
    _compare(hf32, ce32, tol=1e-5, label="float32")

    # 2. bf16 path: real-world dtype. Scores match within bf16 noise. The
    # CrossEncoder path also batches+left-pads while the HF baseline runs one
    # sample at a time, which costs a few extra bf16 ULPs around the activation.
    hf16 = hf_scores(PAIRS, torch.bfloat16)
    ce16 = ce_scores(PAIRS, torch.bfloat16)
    _compare(hf16, ce16, tol=5e-2, label="bfloat16")


if __name__ == "__main__":
    main()
