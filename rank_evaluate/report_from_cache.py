"""Generate an xlsx report from existing cache data.

Only includes datasets that ALL cached models have completed.

Usage:
  uv run python -m rank_evaluate.report_from_cache
  uv run python -m rank_evaluate.report_from_cache --run_tag neg100_seed42
  uv run python -m rank_evaluate.report_from_cache --output comparison.xlsx
  uv run python -m rank_evaluate.report_from_cache --models voyage-rerank-2-lite qwen3-reranker-0.6b
"""

import argparse
from pathlib import Path

from .checkpoint import load_cache_entries
from .config import DEFAULT_CACHE_DIR, DEFAULT_OUTPUT
from .metrics import mean_ndcg
from .report import save_results


def _load_ndcg_from_cache(cache_file: Path) -> float:
    """Read a cache JSONL and return the mean NDCG over all entries."""
    return mean_ndcg([entry.ndcg for entry in load_cache_entries(cache_file)])


def collect_results(
    cache_dir: Path,
    run_tag: str,
    models: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Scan cache and collect per-model results for shared datasets.

    Args:
        cache_dir: Root cache directory.
        run_tag: Run tag subdirectory (e.g. "neg100_seed42").
        models: If given, only include these model names. Otherwise all.

    Returns:
        Dict mapping model_name -> {dataset_stem: mean_ndcg}.
    """
    # Discover models
    model_dirs = sorted(
        d for d in cache_dir.iterdir() if d.is_dir() and (d / run_tag).is_dir()
    )
    if models:
        model_dirs = [d for d in model_dirs if d.name in models]

    if not model_dirs:
        raise FileNotFoundError(
            f"No models found in {cache_dir} with run_tag={run_tag!r}"
        )

    # Collect dataset sets per model
    model_datasets: dict[str, set[str]] = {}
    for model_dir in model_dirs:
        tag_dir = model_dir / run_tag
        datasets = {f.stem for f in tag_dir.glob("*.jsonl")}
        model_datasets[model_dir.name] = datasets

    # Find intersection (datasets all models share)
    common = set.intersection(*model_datasets.values())
    if not common:
        model_info = ", ".join(
            f"{name}({len(ds)})" for name, ds in model_datasets.items()
        )
        raise ValueError(
            f"No common datasets across models. Counts: {model_info}"
        )

    print(f"Models: {len(model_dirs)}  |  Common datasets: {len(common)}")
    for name, ds in model_datasets.items():
        print(f"  {name}: {len(ds)} total, {len(common)} shared")

    # Load NDCG values
    results: dict[str, dict[str, float]] = {}
    for model_dir in model_dirs:
        model_name = model_dir.name
        model_results: dict[str, float] = {}
        tag_dir = model_dir / run_tag
        for dataset_stem in sorted(common):
            cache_file = tag_dir / f"{dataset_stem}.jsonl"
            model_results[dataset_stem] = _load_ndcg_from_cache(cache_file)
        results[model_name] = model_results

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate xlsx report from cached evaluation results."
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--run_tag",
        default="neg100_seed42",
        help="Run tag subdirectory (default: neg100_seed42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output xlsx path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Only include these models (default: all in cache).",
    )
    args = parser.parse_args()

    all_results = collect_results(args.cache_dir, args.run_tag, args.models)

    # Write each model as a column, starting fresh
    output = args.output
    if output.exists():
        output.unlink()

    for model_name, results in all_results.items():
        save_results(results, model_name, output)
        avg = mean_ndcg(list(results.values()))
        print(f"  Written: {model_name}  (avg={avg:.4f})")


if __name__ == "__main__":
    main()
