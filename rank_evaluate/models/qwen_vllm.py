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

from .base import BaseReranker

_DEFAULT_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


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
        instruction: str = _DEFAULT_INSTRUCTION,
        max_length: int = 8192,
        gpu_memory_utilization: float = 0.8,
        prompt_template: str | None = None,
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

        self._suffix_tokens = self._tokenizer.encode(_SUFFIX, add_special_tokens=False)
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
        self._prompt_template = prompt_template
        print(
            f"[qwen-vllm] Model loaded. tp={tp_size}, "
            f"yes_id={self._true_token}, no_id={self._false_token}"
        )

    def _build_token_prompts(self, query: str, documents: list[str]) -> list[list[int]]:
        """Build tokenised prompts for all (query, doc) pairs."""
        from vllm.inputs.data import TokensPrompt

        # Raw template mode: tokenize the pre-formatted prompt directly.
        if self._prompt_template is not None:
            prompts = []
            for doc in documents:
                raw = self._prompt_template.format(query=query, doc=doc)
                ids = self._tokenizer.encode(raw, add_special_tokens=False)
                ids = ids[: self._max_length]
                prompts.append(TokensPrompt(prompt_token_ids=ids))
            return prompts  # type: ignore[return-value]

        # Default: build via chat template.
        messages_list = [
            [
                {
                    "role": "system",
                    "content": (
                        "Judge whether the Document meets the requirements "
                        "based on the Query and the Instruct provided. "
                        'Note that the answer can only be "yes" or "no".'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"<Instruct>: {self._instruction}\n\n"
                        f"<Query>: {query}\n\n"
                        f"<Document>: {doc}"
                    ),
                },
            ]
            for doc in documents
        ]

        body_limit = self._max_length - len(self._suffix_tokens)
        token_ids_list: list[list[int]] = self._tokenizer.apply_chat_template(
            messages_list,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prompts = [
            TokensPrompt(prompt_token_ids=ids[:body_limit] + self._suffix_tokens)
            for ids in token_ids_list
        ]
        return prompts  # type: ignore[return-value]

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Score all documents for a query via vLLM batch generation."""
        prompts = self._build_token_prompts(query, documents)
        outputs = self._model.generate(prompts, self._sampling_params, use_tqdm=False)

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
