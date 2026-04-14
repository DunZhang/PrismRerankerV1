import argparse
import sys
from pathlib import Path

from .config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_OUTPUT,
    EvaluationConfig,
    load_env,
    resolve_data_dir,
)
from .evaluator import run_evaluation
from .model_registry import build_model, supported_models_help
from .report import save_results


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    load_env(args.env_file)

    try:
        config = EvaluationConfig(
            model_name=args.model,
            model_path=args.model_path,
            num_neg=args.num_neg,
            data_dir=resolve_data_dir(args.data_dir),
            output_path=args.output,
            cache_dir=args.cache_dir or DEFAULT_CACHE_DIR,
            seed=args.seed,
            max_queries=args.max_queries,
        )
        config.validate()
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Building model: {config.model_name}")
    model = build_model(config.model_name, model_path=config.model_path)

    try:
        summary = run_evaluation(model=model, config=config)
    finally:
        model.close()

    save_results(summary.dataset_scores, config.model_name, config.output_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate reranker models on JSONL benchmark datasets.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help=supported_models_help(),
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="Path to a local model directory or file when the backend needs one.",
    )
    parser.add_argument(
        "--num_neg",
        type=int,
        default=10,
        help="Number of negatives to sample per query (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output xlsx path (default: evaluation_results.xlsx).",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help=(
            "Benchmark JSONL directory. "
            "Falls back to POSIR_DATA_DIR env var if not set."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=None,
        help=(
            "Cache directory for per-query checkpoints "
            "(default: cache/ inside the module directory)."
        ),
    )
    parser.add_argument(
        "--env_file",
        type=Path,
        default=None,
        help="Path to .env file for API keys (default: project-root .env).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for negative sampling and query subsampling (default: 42).",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help=(
            "If set, randomly subsample this many queries per dataset. "
            "Sampling is fixed by --seed, so results are reproducible. "
            "Checkpoint keys are original indices, so partial runs can "
            "be extended to the full dataset later."
        ),
    )
    return parser


if __name__ == "__main__":
    main()
