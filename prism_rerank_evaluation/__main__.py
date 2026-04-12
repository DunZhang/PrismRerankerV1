"""Entry point: uv run python -m prism_rerank_evaluation <task> [options].

Usage:
    uv run python -m prism_rerank_evaluation beir --batch-size 4
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
        print("Usage: python -m prism_rerank_evaluation <task> [options]")
        print("Available tasks: beir")
        sys.exit(1)

    task = sys.argv.pop(1)

    if task == "beir":
        from .beir import main as beir_main

        beir_main()
    else:
        print(f"Unknown task: {task}")
        print("Available tasks: beir")
        sys.exit(1)


main()
