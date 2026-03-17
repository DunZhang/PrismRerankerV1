"""Model registry and factory helpers for reranker evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models.base import BaseReranker


@dataclass(slots=True, frozen=True)
class ModelDefinition:
    """Metadata and factory for a supported reranker model."""

    name: str
    description: str
    backend: str
    requires_model_path: bool = False
    aliases: tuple[str, ...] = ()


def list_supported_models(backend: str | None = None) -> list[ModelDefinition]:
    """Return supported models in CLI display order."""
    definitions = [
        ModelDefinition(
            name="voyage-rerank-2-lite",
            description="Voyage AI rerank-2-lite API",
            backend="voyage-api",
        ),
        ModelDefinition(
            name="voyage-rerank-2",
            description="Voyage AI rerank-2 API",
            backend="voyage-api",
        ),
        ModelDefinition(
            name="voyage-rerank-2.5",
            description="Voyage AI rerank-2.5 API",
            backend="voyage-api",
        ),
        ModelDefinition(
            name="voyage-rerank-2.5-lite",
            description="Voyage AI rerank-2.5-lite API",
            backend="voyage-api",
        ),
        ModelDefinition(
            name="qwen3-reranker-0.6b-gguf",
            description="Qwen3-Reranker-0.6B GGUF via llama-cpp-python",
            backend="llama-cpp",
            requires_model_path=True,
            aliases=("qwen3-reranker-gguf",),
        ),
        ModelDefinition(
            name="qwen3-reranker-4b-gguf",
            description="Qwen3-Reranker-4B GGUF via llama-cpp-python",
            backend="llama-cpp",
            requires_model_path=True,
        ),
        ModelDefinition(
            name="qwen3-reranker-0.6b",
            description="Qwen3-Reranker-0.6B via HuggingFace transformers",
            backend="transformers",
            aliases=("qwen3-reranker-hf",),
        ),
        ModelDefinition(
            name="qwen3-reranker-0.6b-vllm",
            description="Qwen3-Reranker-0.6B via vLLM",
            backend="vllm",
        ),
        ModelDefinition(
            name="qwen3-reranker-4b-vllm",
            description="Qwen3-Reranker-4B via vLLM",
            backend="vllm",
        ),
        ModelDefinition(
            name="qwen3-reranker-8b-vllm",
            description="Qwen3-Reranker-8B via vLLM",
            backend="vllm",
        ),
        ModelDefinition(
            name="prism-reranker-0.6b-vllm",
            description="Fine-tuned Prism Reranker via vLLM",
            backend="vllm",
            requires_model_path=True,
        ),
        ModelDefinition(
            name="zerank-2",
            description="ZeRank-2 CrossEncoder via sentence-transformers (BF16)",
            backend="sentence-transformers",
        ),
    ]
    if backend is None:
        return definitions
    return [definition for definition in definitions if definition.backend == backend]


def supported_model_names(include_aliases: bool = False) -> list[str]:
    """Return canonical model names, optionally including aliases."""
    names: list[str] = []
    for definition in list_supported_models():
        names.append(definition.name)
        if include_aliases:
            names.extend(definition.aliases)
    return names


def supported_models_help() -> str:
    """Render a compact help string for the CLI."""
    lines = ["Supported models:"]
    for definition in list_supported_models():
        model_path_note = (
            " (requires --model_path)" if definition.requires_model_path else ""
        )
        alias_note = (
            f" [aliases: {', '.join(definition.aliases)}]" if definition.aliases else ""
        )
        lines.append(
            f"  {definition.name:<24} {definition.description}{model_path_note}{alias_note}"
        )
    return "\n".join(lines)


def get_model_definition(model_name: str) -> ModelDefinition:
    """Resolve a canonical model definition from a name or alias."""
    normalized = model_name.lower().strip()
    for definition in list_supported_models():
        if normalized == definition.name or normalized in definition.aliases:
            return definition
    raise ValueError(f"Unknown model: {model_name!r}\n{supported_models_help()}")


def build_model(model_name: str, model_path: Path | None = None) -> BaseReranker:
    """Instantiate the requested reranker model."""
    definition = get_model_definition(model_name)

    if definition.requires_model_path and model_path is None:
        raise ValueError(f"{definition.name} requires --model_path.")

    if definition.name.startswith("voyage-rerank-"):
        from .models.voyage import VoyageReranker

        voyage_model = definition.name.removeprefix("voyage-")
        return VoyageReranker(model=voyage_model)

    if definition.name.endswith("-gguf"):
        from .models.qwen_gguf import QwenGGUFReranker

        if model_path is None:
            raise ValueError(f"{definition.name} requires --model_path.")
        return QwenGGUFReranker(model_path=str(model_path))

    if definition.name == "qwen3-reranker-0.6b":
        from .models.qwen_hf import QwenHFReranker

        if model_path is None:
            return QwenHFReranker()
        return QwenHFReranker(model_id=str(model_path))

    if definition.name in {
        "qwen3-reranker-0.6b-vllm",
        "qwen3-reranker-4b-vllm",
        "qwen3-reranker-8b-vllm",
    }:
        from .models.qwen_vllm import QwenVLLMReranker

        default_model_ids = {
            "qwen3-reranker-0.6b-vllm": "Qwen/Qwen3-Reranker-0.6B",
            "qwen3-reranker-4b-vllm": "Qwen/Qwen3-Reranker-4B",
            "qwen3-reranker-8b-vllm": "Qwen/Qwen3-Reranker-8B",
        }
        model_id = str(model_path) if model_path else default_model_ids[definition.name]
        return QwenVLLMReranker(model_id=model_id)

    if definition.name == "prism-reranker-0.6b-vllm":
        from shared.prompts import TRAINING_INSTRUCTION

        from .models.qwen_vllm import QwenVLLMReranker

        if model_path is None:
            raise ValueError(f"{definition.name} requires --model_path.")
        return QwenVLLMReranker(
            model_id=str(model_path),
            instruction=TRAINING_INSTRUCTION,
        )

    if definition.name == "zerank-2":
        from .models.zerank import ZeRankReranker

        if model_path is not None:
            return ZeRankReranker(model_id=str(model_path))
        return ZeRankReranker()

    raise ValueError(f"Unsupported model configuration for {definition.name!r}.")
