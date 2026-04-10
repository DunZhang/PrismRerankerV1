from __future__ import annotations

from typing import Any

from shared.prompts import TRAINING_INSTRUCTION, render_raw_prompt

# Re-export for backward compatibility (train/data.py uses build_prompt)
build_prompt = render_raw_prompt

DEFAULT_LORA_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Legacy hardcoded IDs for Qwen3-Reranker-0.6B / 4B / 8B.
YES_TOKEN_ID: int = 9693
NO_TOKEN_ID: int = 2152


def resolve_yes_no_token_ids(tokenizer: Any) -> tuple[int, int]:
    """Resolve YES/NO token IDs dynamically from the tokenizer.

    Training targets are lowercase ``yes``/``no`` (matches the original
    Qwen3-Reranker hardcoded IDs 9693/2152). For Qwen3.5 this resolves to
    9405/2083.
    """
    yes_ids: list[int] = tokenizer.encode("yes", add_special_tokens=False)
    no_ids: list[int] = tokenizer.encode("no", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise ValueError(
            f"Expected single-token encoding for 'Yes'/'No', "
            f"got yes={yes_ids}, no={no_ids}"
        )
    return yes_ids[0], no_ids[0]


__all__ = [
    "TRAINING_INSTRUCTION",
    "build_prompt",
    "DEFAULT_LORA_TARGET_MODULES",
    "YES_TOKEN_ID",
    "NO_TOKEN_ID",
    "resolve_yes_no_token_ids",
]
