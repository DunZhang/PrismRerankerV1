"""Model selection helpers for teacher score generation."""

from rank_evaluate.model_registry import (
    build_model,
    get_model_definition,
    list_supported_models,
)
from rank_evaluate.models.base import BaseReranker

SUPPORTED_MODELS = ", ".join(
    definition.name for definition in list_supported_models(backend="voyage-api")
)


def create_voyage_scorer(model_name: str) -> BaseReranker:
    """Create a teacher-score reranker from the shared model registry."""
    try:
        definition = get_model_definition(model_name)
    except ValueError as exc:
        raise ValueError(
            f"Unknown model_name: {model_name!r}\nSupported: {SUPPORTED_MODELS}"
        ) from exc

    if definition.backend != "voyage-api":
        raise ValueError(
            f"Teacher-score generation only supports Voyage models: {model_name!r}\n"
            f"Supported: {SUPPORTED_MODELS}"
        )

    return build_model(definition.name)
