from __future__ import annotations

from typing import Any

import torch

from train_v2.config import TrainConfig
from train_v2.constants import NO_TOKEN_ID, YES_TOKEN_ID

DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    dtype = DTYPE_MAP.get(name)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {name}")
    return dtype


def load_model_and_tokenizer(cfg: TrainConfig) -> tuple[Any, Any]:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = resolve_dtype(cfg.model.dtype)
    model_kwargs: dict[str, Any] = {}

    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if cfg.model.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.model.attn_implementation

    if cfg.model.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype or torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path)
    model = AutoModelForCausalLM.from_pretrained(cfg.model.path, **model_kwargs)
    model.config.use_cache = not cfg.model.gradient_checkpointing

    if cfg.model.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=cfg.model.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    elif cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if cfg.lora.enabled and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if cfg.lora.enabled:
        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
            use_rslora=cfg.lora.use_rslora,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def extract_yes_no_logits(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    last_logits = outputs.logits[:, -1, :]
    return last_logits[:, YES_TOKEN_ID] - last_logits[:, NO_TOKEN_ID]
