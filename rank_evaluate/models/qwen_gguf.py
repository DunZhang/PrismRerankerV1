"""Qwen3-Reranker via llama-cpp-python (GPU inference).

Supports any Qwen3-Reranker GGUF model (0.6B, 4B, 8B, etc.).

The model scores (query, document) pairs by extracting the logprob of the
"yes" token at the position where the model would generate its final answer,
after a pre-filled empty <think> block.

Template format (from official model card):
  <|im_start|>system
  Judge whether the Document meets the requirements based on the Query and the
  Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
  <|im_start|>user
  <Instruct>: {instruction}
  <Query>: {query}
  <Document>: {doc}<|im_end|>
  <|im_start|>assistant
  <think>

  </think>

"""

import math

from .base import BaseReranker

_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
_DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

# Suffix pre-fills the empty think block so model jumps straight to yes/no
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# Variants to try when looking for yes/no in logprobs dict
_YES_VARIANTS = ("yes", "Yes", " yes", " Yes", "YES")
_NO_VARIANTS = ("no", "No", " no", " No", "NO")


class QwenGGUFReranker(BaseReranker):
    """Qwen3-Reranker loaded via llama-cpp-python with CUDA.

    Works with any Qwen3-Reranker GGUF file (0.6B, 4B, etc.).

    Args:
        model_path: Local path to the GGUF file (required).
        n_gpu_layers: GPU layers to offload (-1 = all).
        n_ctx: Context window size.
        instruction: Task instruction prefix (optional).
        logprobs_k: Number of top-k logprobs to retrieve per position.
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        instruction: str = _DEFAULT_INSTRUCTION,
        logprobs_k: int = 200,
    ) -> None:
        try:
            from ..cuda_libs import preload_cuda_libs

            preload_cuda_libs()
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python not installed. Install with CUDA support:\n"
                "  uv add llama-cpp-python "
                "--extra-index-url https://abetlen.github.io/"
                "llama-cpp-python/whl/cu124"
            ) from e

        print(f"[qwen-gguf] Loading model from {model_path}")
        self._llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            logits_all=True,
            verbose=False,
        )
        self._instruction = instruction
        self._logprobs_k = logprobs_k
        print("[qwen-gguf] Model loaded.")

    def _build_prompt(self, query: str, document: str) -> str:
        prefix = (
            f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"<Instruct>: {self._instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )
        return prefix + _SUFFIX

    def _score_pair(self, query: str, document: str) -> float:
        """Return yes-probability for a single (query, doc) pair."""
        prompt = self._build_prompt(query, document)
        result = self._llm.create_completion(
            prompt,
            max_tokens=1,
            temperature=0.0,
            logprobs=self._logprobs_k,
        )
        top_lp: dict[str, float] = result["choices"][0]["logprobs"]["top_logprobs"][0]

        yes_lp = _find_lp(top_lp, _YES_VARIANTS)
        no_lp = _find_lp(top_lp, _NO_VARIANTS)

        if yes_lp is None and no_lp is None:
            # Fall back: use the actual generated token
            token = result["choices"][0]["text"].strip().lower()
            return 1.0 if token.startswith("y") else 0.0

        yes_p = math.exp(yes_lp) if yes_lp is not None else 1e-9
        no_p = math.exp(no_lp) if no_lp is not None else 1e-9
        return yes_p / (yes_p + no_p)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Score all documents for the query sequentially."""
        return [self._score_pair(query, doc) for doc in documents]

    def close(self) -> None:
        del self._llm


def _find_lp(
    top_lp: dict[str, float],
    variants: tuple[str, ...],
) -> float | None:
    """Find the logprob for any of the given token string variants."""
    for v in variants:
        if v in top_lp:
            return top_lp[v]
    return None
