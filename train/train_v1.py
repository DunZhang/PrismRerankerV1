"""Entry point for reranker distillation training.

Usage:
    uv run python train/train_v1.py --config train/train_config.yaml
    uv run accelerate launch --num_processes 2 train/train_v1.py --config train/train_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.config import TrainConfig
from train.trainer import RerankerTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reranker distillation training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    RerankerTrainer(cfg, config_path=args.config).train()


if __name__ == "__main__":
    main()
