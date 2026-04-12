"""Prism rerank evaluation module."""

from __future__ import annotations


def patch_transformers_compat() -> None:
    """Patch transformers v5 compatibility for jina-embeddings-v3.

    The model's custom XLMRobertaLoRA class lacks ``all_tied_weights_keys``
    which was added in transformers v5. We patch ``PreTrainedModel`` so that
    the property falls back to ``_tied_weights_keys`` if missing.
    """
    from transformers import PreTrainedModel

    if hasattr(PreTrainedModel, "_original_mark_tied"):
        return  # already patched

    original = PreTrainedModel.mark_tied_weights_as_initialized

    def _patched_mark_tied(
        self: PreTrainedModel, *args: object, **kwargs: object
    ) -> None:
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = getattr(self, "_tied_weights_keys", None) or {}
        return original(self, *args, **kwargs)

    PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark_tied  # type: ignore[assignment]
    PreTrainedModel._original_mark_tied = original  # type: ignore[attr-defined]
