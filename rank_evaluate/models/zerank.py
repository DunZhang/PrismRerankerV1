"""ZeRank-2 reranker via sentence-transformers CrossEncoder (GPU, BF16).

Uses ``sentence_transformers.CrossEncoder`` with ``trust_remote_code=True``.
Scores are Elo-style relevance scores returned directly by ``.predict()``.

Model card: https://huggingface.co/zeroentropy/zerank-2
  - Parameters:  4B (based on Qwen3-4B)
  - Context:     32 768 tokens
  - Tensor type: BF16

Batching strategy:
  - Documents are sorted by length before batching to minimise padding.
  - ``CrossEncoder.predict()`` handles batching internally; we pass
    ``batch_size`` directly and sort externally for efficiency.
"""

from __future__ import annotations

from .base import BaseReranker

_DEFAULT_MODEL_ID = "zeroentropy/zerank-2"


class ZeRankReranker(BaseReranker):
    """ZeRank-2 CrossEncoder loaded via sentence-transformers with BF16.

    Args:
        model_id: HuggingFace model ID or local path.
        device: Torch device string (default: "cuda").
        batch_size: Number of (query, doc) pairs per forward pass.
        max_length: Max token length passed to the CrossEncoder.
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        device: str = "cuda",
        batch_size: int = 8,
        max_length: int = 32768,
    ) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        print(f"[zerank] Loading {model_id} ...")
        self._model = CrossEncoder(
            model_id,
            trust_remote_code=True,
            device=device,
            automodel_args={"torch_dtype": torch.bfloat16},
            max_length=max_length,
        )
        # Qwen3 tokenizer and model config have no pad_token; set to eos_token
        # so batched inference works. transformers checks model.config.pad_token_id,
        # so both the tokenizer and the model config must be updated.
        tokenizer = self._model.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model_config = self._model.model.config
        if model_config.pad_token_id is None:
            model_config.pad_token_id = tokenizer.eos_token_id
        self._batch_size = batch_size
        print(f"[zerank] Model loaded. batch_size={batch_size}, device={device}")

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Score all documents for relevance to query using batched prediction.

        Documents are sorted by length before batching to reduce padding waste,
        then scores are restored to the original document order.
        """
        # Sort by document length to minimise padding within each batch
        order = sorted(range(len(documents)), key=lambda i: len(documents[i]))
        sorted_pairs = [(query, documents[i]) for i in order]

        sorted_scores: list[float] = self._model.predict(
            sorted_pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        ).tolist()

        # Restore original order
        scores = [0.0] * len(documents)
        for rank, orig_idx in enumerate(order):
            scores[orig_idx] = sorted_scores[rank]
        return scores

    def close(self) -> None:
        import gc

        import torch

        del self._model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
