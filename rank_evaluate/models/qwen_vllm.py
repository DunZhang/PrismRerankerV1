"""Qwen3-Reranker via vLLM (GPU inference).

Scores (query, document) pairs by generating a single token and comparing
the logprobs of "yes" vs "no" to produce a relevance probability.

Uses vLLM's prefix caching for efficient batch inference — the shared
system prompt and instruction prefix are cached across all documents
within the same query.

Supported models (auto-downloaded from HuggingFace):
  - Qwen/Qwen3-Reranker-0.6B
  - Qwen/Qwen3-Reranker-4B
  - Qwen/Qwen3-Reranker-8B

Or pass ``--model_path`` to load from a local directory.
"""

from __future__ import annotations

import math

from shared.prompts import DEFAULT_EVAL_INSTRUCTION, render_raw_prompt

from .base import BaseReranker

_DEFAULT_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"


class QwenVLLMReranker(BaseReranker):
    """Qwen3-Reranker loaded via vLLM for fast GPU inference.

    Args:
        model_id: HuggingFace model ID or local path.
        instruction: Task instruction prefix.
        max_length: Max token length per prompt (truncates if exceeded).
        gpu_memory_utilization: Fraction of GPU memory for vLLM.
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        instruction: str = DEFAULT_EVAL_INSTRUCTION,
        max_length: int = 8192,
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        print(f"[qwen-vllm] Loading {model_id} ...")

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._tokenizer.padding_side = "left"
        self._tokenizer.pad_token = self._tokenizer.eos_token

        tp_size = torch.cuda.device_count()
        self._model = LLM(
            model=model_id,
            tensor_parallel_size=tp_size,
            max_model_len=max_length + 64,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        self._true_token = self._tokenizer("yes", add_special_tokens=False).input_ids[0]
        self._false_token = self._tokenizer("no", add_special_tokens=False).input_ids[0]

        self._sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self._true_token, self._false_token],
        )

        self._instruction = instruction
        self._max_length = max_length
        print(
            f"[qwen-vllm] Model loaded. tp={tp_size}, "
            f"yes_id={self._true_token}, no_id={self._false_token}"
        )

    def _build_token_prompts(self, query: str, documents: list[str]) -> list[list[int]]:
        """Build tokenised prompts for all (query, doc) pairs."""
        from vllm.inputs.data import TokensPrompt

        prompts = []
        for doc in documents:
            raw = render_raw_prompt(query, doc, instruction=self._instruction)
            ids = self._tokenizer.encode(raw, add_special_tokens=False)
            ids = ids[: self._max_length]
            prompts.append(TokensPrompt(prompt_token_ids=ids))
        return prompts  # type: ignore[return-value]

    def rerank(
        self, query: str, documents: list[str], batch_size: int = 4
    ) -> list[float]:
        """Score all documents for a query via vLLM batch generation.

        Args:
            query: The search query.
            documents: Documents to score against the query.
            batch_size: Max documents per vLLM call. 0 means all at once.
        """
        prompts = self._build_token_prompts(query, documents)

        if batch_size > 0:
            outputs = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]
                outputs.extend(
                    self._model.generate(batch, self._sampling_params, use_tqdm=False)
                )
        else:
            outputs = self._model.generate(
                prompts, self._sampling_params, use_tqdm=False
            )

        scores: list[float] = []
        for output in outputs:
            final_logits = output.outputs[0].logprobs[-1]

            true_lp = (
                final_logits[self._true_token].logprob
                if self._true_token in final_logits
                else -10.0
            )
            false_lp = (
                final_logits[self._false_token].logprob
                if self._false_token in final_logits
                else -10.0
            )

            true_score = math.exp(true_lp)
            false_score = math.exp(false_lp)
            scores.append(true_score / (true_score + false_score))

        return scores

    def close(self) -> None:
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
        )

        destroy_model_parallel()
        del self._model
        del self._tokenizer

        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
