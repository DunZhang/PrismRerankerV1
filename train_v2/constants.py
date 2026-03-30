from __future__ import annotations

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

# Token IDs are shared by Qwen3-Reranker-0.6B, 4B and 8B.
YES_TOKEN_ID: int = 9693
NO_TOKEN_ID: int = 2152

__all__ = [
    "TRAINING_INSTRUCTION",
    "build_prompt",
    "DEFAULT_LORA_TARGET_MODULES",
    "YES_TOKEN_ID",
    "NO_TOKEN_ID",
]
