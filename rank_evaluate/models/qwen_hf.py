"""Qwen3-Reranker-0.6B via HuggingFace transformers (GPU inference).

Scores (query, document) pairs by extracting the softmax probability of the
"yes" token at the last position of the pre-filled assistant turn.

Batching strategy:
  - Documents are sorted by length (ascending) before batching so that
    sequences within a batch have similar lengths, minimizing padding waste.
  - With left-padding, the last real token is always at position -1, so
    logits[:, -1, :] correctly extracts the scoring position for the whole batch.

Template format (from official model card):
  <|im_start|>system
  Judge whether the Document meets the requirements...<|im_end|>
  <|im_start|>user
  <Instruct>: {instruction}
  <Query>: {query}
  <Document>: {doc}<|im_end|>
  <|im_start|>assistant
  <think>

  </think>

"""

import torch

from shared.prompts import DEFAULT_EVAL_INSTRUCTION, render_raw_prompt

from .base import BaseReranker

_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"


class QwenHFReranker(BaseReranker):
    """Qwen3-Reranker-0.6B loaded via HuggingFace transformers with CUDA.

    Args:
        model_id: HuggingFace model ID or local path.
        device: Torch device string (default: "cuda").
        torch_dtype: Dtype for model weights (default: bfloat16).
        instruction: Task instruction prefix.
        max_length: Max token length per prompt (truncates document if exceeded).
        batch_size: Number of (query, doc) pairs to score in one forward pass.
    """

    def __init__(
        self,
        model_id: str = _MODEL_ID,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        instruction: str = DEFAULT_EVAL_INSTRUCTION,
        max_length: int = 8192,
        batch_size: int = 8,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[qwen-hf] Loading {model_id} ...")
        # left-padding so sequences always end at position -1 regardless of length
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self._model.eval()
        self._device = device
        self._instruction = instruction
        self._max_length = max_length
        self._batch_size = batch_size

        self._yes_id = self._tokenizer.convert_tokens_to_ids("yes")
        self._no_id = self._tokenizer.convert_tokens_to_ids("no")
        print(
            f"[qwen-hf] Model loaded. "
            f"yes_id={self._yes_id}, no_id={self._no_id}, "
            f"batch_size={batch_size}"
        )

    def _score_batch(self, prompts: list[str]) -> list[float]:
        """Run one batched forward pass and return yes-probabilities."""
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Left-padding → last real token always at position -1 for every sequence
        logits = outputs.logits[:, -1, :]  # (batch, vocab)
        yes_no = logits[:, [self._yes_id, self._no_id]]  # (batch, 2)
        probs = torch.softmax(yes_no, dim=1)
        return probs[:, 0].tolist()  # yes probability

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Score all documents, batching by similar length to minimise padding."""
        prompts = [
            render_raw_prompt(query, doc, instruction=self._instruction)
            for doc in documents
        ]

        # Sort by document length (good proxy for token count; query prefix is constant)
        order = sorted(range(len(documents)), key=lambda i: len(documents[i]))
        sorted_prompts = [prompts[i] for i in order]

        # Batched inference
        sorted_scores: list[float] = []
        for start in range(0, len(sorted_prompts), self._batch_size):
            batch = sorted_prompts[start : start + self._batch_size]
            sorted_scores.extend(self._score_batch(batch))

        # Restore original document order
        scores = [0.0] * len(documents)
        for rank, orig_idx in enumerate(order):
            scores[orig_idx] = sorted_scores[rank]
        return scores

    def close(self) -> None:
        del self._model
        del self._tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
