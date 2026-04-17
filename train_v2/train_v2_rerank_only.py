"""Entry point for rerank-only training.

与 ``train_v2.py`` 的区别：SFT loss 只作用在 prompt 之后的第一个 token
（即 ``yes`` / ``no``），不再监督 ``contribution_evidence``。
所有配置字段与 ``train_v2.py`` 共用，直接复用同一份 YAML 即可。

Usage:
    uv run python train_v2/train_v2_rerank_only.py --config train_v2/train_config_local.yaml
    uv run accelerate launch --num_processes 2 \\
        train_v2/train_v2_rerank_only.py --config train_v2/train_config_local.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_v2.config import TrainConfig
from train_v2.trainer_rerank_only import RerankerOnlyTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reranker training (SFT restricted to yes/no token)",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    RerankerOnlyTrainer(cfg, config_path=args.config).train()


if __name__ == "__main__":
    main()
