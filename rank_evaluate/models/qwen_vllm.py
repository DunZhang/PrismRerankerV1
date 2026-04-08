"""Qwen3-Reranker / Prism-Reranker via vLLM (GPU inference).

Scores (query, document) pairs by generating a single token and comparing
the logprobs of "yes" vs "no" to produce a relevance probability.

Uses vLLM's prefix caching for efficient batch inference — the shared
system prompt and instruction prefix are cached across all documents
within the same query.

For Qwen3 models, uses ``apply_chat_template`` with the official prompt
format.  For Prism models, uses ``render_raw_prompt`` from shared templates.
"""

from __future__ import annotations

import math

from .base import BaseReranker

# ---------------------------------------------------------------------------
# Qwen3-Reranker official constants (hardcoded to match upstream exactly)
# ---------------------------------------------------------------------------
_QWEN3_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on "
    "the Query and the Instruct provided. Note that the answer "
    'can only be "yes" or "no".'
)
_QWEN3_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
_QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class QwenVLLMReranker(BaseReranker):
    """Qwen3-Reranker / Prism-Reranker loaded via vLLM.

    Args:
        model_id: HuggingFace model ID or local path.
        instruction: Task instruction prefix (only used when
            ``use_chat_template=False``).
        use_chat_template: If ``True`` (default), use
            ``tokenizer.apply_chat_template`` with the official Qwen3
            prompt format.  If ``False``, use ``render_raw_prompt`` from
            the shared Jinja2 template (for Prism models).
        max_length: Max token length per prompt (truncates if exceeded).
        gpu_memory_utilization: Fraction of GPU memory for vLLM.
    """

    def __init__(
        self,
        model_id: str,
        instruction: str | None = None,
        use_chat_template: bool = True,
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

        self._true_token = self._tokenizer(
            "yes", add_special_tokens=False
        ).input_ids[0]
        self._false_token = self._tokenizer(
            "no", add_special_tokens=False
        ).input_ids[0]

        self._sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self._true_token, self._false_token],
        )

        self._use_chat_template = use_chat_template
        self._instruction = instruction or _QWEN3_INSTRUCTION
        self._max_length = max_length

        # Pre-compute suffix tokens for the chat-template path
        if use_chat_template:
            self._suffix_tokens: list[int] = self._tokenizer.encode(
                _QWEN3_SUFFIX, add_special_tokens=False
            )

        print(
            f"[qwen-vllm] Model loaded. tp={tp_size}, "
            f"yes_id={self._true_token}, no_id={self._false_token}"
        )

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_token_prompts_chat_template(
        self, query: str, documents: list[str]
    ) -> list:
        """Qwen3 official path: apply_chat_template + suffix."""
        from vllm.inputs.data import TokensPrompt

        messages_batch = [
            [
                {"role": "system", "content": _QWEN3_SYSTEM_PROMPT},
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

        tokenised_batch: list[list[int]] = self._tokenizer.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        body_max = self._max_length - len(self._suffix_tokens)
        return [
            TokensPrompt(prompt_token_ids=ids[:body_max] + self._suffix_tokens)
            for ids in tokenised_batch
        ]

    def _build_token_prompts_raw(self, query: str, documents: list[str]) -> list:
        """Prism path: render_raw_prompt from shared Jinja2 template."""
        from shared.prompts import TRAINING_SYSTEM_PROMPT, render_raw_prompt
        from vllm.inputs.data import TokensPrompt

        prompts = []
        for doc in documents:
            raw = render_raw_prompt(
                query,
                doc,
                instruction=self._instruction,
                system_prompt=TRAINING_SYSTEM_PROMPT,
            )
            ids = self._tokenizer.encode(raw, add_special_tokens=False)
            ids = ids[: self._max_length]
            prompts.append(TokensPrompt(prompt_token_ids=ids))
        return prompts

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def rerank(
        self, query: str, documents: list[str], batch_size: int = 4
    ) -> list[float]:
        """Score all documents for a query via vLLM batch generation."""
        if self._use_chat_template:
            prompts = self._build_token_prompts_chat_template(query, documents)
        else:
            prompts = self._build_token_prompts_raw(query, documents)

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
