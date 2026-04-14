"""Entry point: uv run python -m evaluate_relevance_on_jina_bench <task> [options].

Usage:
    uv run python -m evaluate_relevance_on_jina_bench beir --batch-size 4
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    """Load shared config, then dispatch to the appropriate evaluation task."""
    # Load HF_TOKEN from .env for faster HuggingFace downloads
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
    if os.environ.get("HF_TOKEN"):
        print(f"HF_TOKEN loaded from {env_path}")

    if len(sys.argv) < 2:
        print("Usage: python -m evaluate_relevance_on_jina_bench <task> [options]")
        print("Available tasks: beir, eval_topk, rerank_model_test")
        sys.exit(1)

    task = sys.argv.pop(1)

    if task == "beir":
        from .beir import main as beir_main

        beir_main()
    elif task == "eval_topk":
        from .eval_topk import main as eval_topk_main

        eval_topk_main()
    elif task == "rerank_model_test":
        from .rerank_model_test import main as rerank_model_test_main

        rerank_model_test_main()
    else:
        print(f"Unknown task: {task}")
        print("Available tasks: beir, eval_topk, rerank_model_test")
        sys.exit(1)


main()
