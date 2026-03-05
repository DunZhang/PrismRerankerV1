"""Analyze a training dataset directory and recommend initial hyperparameters.

Usage:
    uv run python train/analyze_data.py --data-dir /path/to/data
    uv run python train/analyze_data.py --data-dir /path/to/data --max-samples 5000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, stdev
from typing import NamedTuple


class _Sample(NamedTuple):
    pos_score: float
    neg_scores: list[float]
    query_len: int
    doc_lens: list[int]


def _percentile(sorted_data: list[float], p: float) -> float:
    """Return the p-th percentile (0-100) of sorted_data."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (k - lo) * (sorted_data[hi] - sorted_data[lo])


def _load_samples(data_dir: str, max_samples: int | None) -> list[_Sample]:
    """Load samples from a .jsonl file or all .jsonl files in a directory."""
    p = Path(data_dir)
    if p.is_file():
        paths = [p]
    else:
        paths = sorted(p.rglob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")

    reservoir: list[_Sample] = []
    rng = random.Random(42)
    total_seen = 0

    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)

                pos_scores = data["teacher_pos_scores"]
                neg_scores = data["teacher_neg_scores"]
                if len(pos_scores) != 1 or len(neg_scores) != 7:
                    continue

                query = data.get("query", "")
                docs = data.get("pos_list", []) + data.get("neg_list", [])

                sample = _Sample(
                    pos_score=pos_scores[0],
                    neg_scores=neg_scores,
                    query_len=len(query),
                    doc_lens=[len(d) for d in docs],
                )

                total_seen += 1
                if max_samples is None:
                    reservoir.append(sample)
                elif total_seen <= max_samples:
                    reservoir.append(sample)
                else:
                    replace_at = rng.randint(0, total_seen - 1)
                    if replace_at < max_samples:
                        reservoir[replace_at] = sample

    print(f"找到文件: {[p.name for p in paths]}")
    print(f"总行数: {total_seen:,}，已加载: {len(reservoir):,} 条样本\n")
    return reservoir


def _analyze(samples: list[_Sample]) -> dict:
    """Compute all statistics from loaded samples."""
    n = len(samples)

    pos_scores = [s.pos_score for s in samples]
    all_neg_scores = [score for s in samples for score in s.neg_scores]
    all_scores = pos_scores + all_neg_scores

    # hardest margin = pos - max(negs)
    hardest_margins = sorted(s.pos_score - max(s.neg_scores) for s in samples)

    # listwise std = std of 7 neg scores per sample
    listwise_stds = []
    for s in samples:
        if len(s.neg_scores) > 1:
            listwise_stds.append(stdev(s.neg_scores))

    # doc character lengths: use query + each doc, estimate tokens
    all_doc_chars: list[int] = []
    for s in samples:
        for doc_len in s.doc_lens:
            all_doc_chars.append(s.query_len + doc_len)
    all_doc_chars.sort()

    all_scores_sorted = sorted(all_scores)
    scores_std = stdev(all_scores) if len(all_scores) > 1 else 0.0
    scores_near_half = sum(1 for s in all_scores if 0.4 <= s <= 0.6) / len(all_scores)

    return {
        "n": n,
        "pos_scores_mean": mean(pos_scores),
        "pos_scores_std": stdev(pos_scores) if len(pos_scores) > 1 else 0.0,
        "neg_scores_mean": mean(all_neg_scores),
        "neg_scores_std": stdev(all_neg_scores) if len(all_neg_scores) > 1 else 0.0,
        "all_scores_std": scores_std,
        "all_scores_p25": _percentile(all_scores_sorted, 25),
        "all_scores_p75": _percentile(all_scores_sorted, 75),
        "scores_near_half_ratio": scores_near_half,
        "hardest_margin_p25": _percentile(hardest_margins, 25),
        "hardest_margin_p50": _percentile(hardest_margins, 50),
        "hardest_margin_p75": _percentile(hardest_margins, 75),
        "listwise_std_mean": mean(listwise_stds) if listwise_stds else 0.0,
        "doc_chars_p50": _percentile(all_doc_chars, 50),
        "doc_chars_p90": _percentile(all_doc_chars, 90),
        "doc_chars_p99": _percentile(all_doc_chars, 99),
    }


def _estimate_tokens(chars: float, lang_hint: str = "auto") -> int:
    """Rough character-to-token estimate."""
    # heuristic: average 2.5 chars per token (mixed CJK + Latin)
    return int(chars / 2.5)


def _recommend(stats: dict) -> dict[str, object]:
    """Apply mapping rules to produce recommended hyperparameter values."""
    n = stats["n"]
    all_scores_std = stats["all_scores_std"]
    scores_near_half = stats["scores_near_half_ratio"]
    margin_p50 = stats["hardest_margin_p50"]
    listwise_std = stats["listwise_std_mean"]

    # LoRA
    if n < 5000:
        r, alpha, dropout, use_rslora = 8, 16, 0.05, False
    elif n <= 50000:
        r, alpha, dropout, use_rslora = 16, 16, 0.0, False
    else:
        r, alpha, dropout, use_rslora = 16, 16, 0.0, False

    # temperature
    if all_scores_std < 0.15:
        temperature = 1.5
    elif all_scores_std < 0.25:
        temperature = 2.0
    else:
        temperature = min(3.0, round(2.0 + (all_scores_std - 0.25) * 4, 1))

    # gamma_point
    if scores_near_half > 0.30:
        gamma_point = 0.8
    elif all_scores_std < 0.15:
        gamma_point = 0.3
    elif all_scores_std < 0.25:
        gamma_point = 0.5
    else:
        gamma_point = 0.7

    # alpha_rank, hard_neg_scale
    if margin_p50 < 0.05:
        alpha_rank, hard_neg_scale = 1.5, 8.0
    elif margin_p50 < 0.15:
        alpha_rank, hard_neg_scale = 1.2, 6.0
    elif margin_p50 < 0.30:
        alpha_rank, hard_neg_scale = 1.0, 5.0
    else:
        alpha_rank, hard_neg_scale = 0.8, 3.0

    # beta_list
    if listwise_std < 0.05:
        beta_list = 0.5
    elif listwise_std < 0.15:
        beta_list = 1.0
    else:
        beta_list = 1.5

    # max_seq_length suggestion
    tokens_p90 = _estimate_tokens(stats["doc_chars_p90"])
    tokens_p99 = _estimate_tokens(stats["doc_chars_p99"])
    if tokens_p99 < 1024:
        seq_suggestion = 1024
    elif tokens_p99 < 2048:
        seq_suggestion = 2048
    else:
        seq_suggestion = 4096

    return {
        "lora": {
            "r": r,
            "alpha": alpha,
            "dropout": dropout,
            "use_rslora": use_rslora,
        },
        "loss": {
            "alpha_rank": alpha_rank,
            "beta_list": beta_list,
            "gamma_point": gamma_point,
            "temperature": temperature,
            "hard_neg_scale": hard_neg_scale,
        },
        "_seq": {
            "tokens_p90": tokens_p90,
            "tokens_p99": tokens_p99,
            "suggestion": seq_suggestion,
        },
    }


def _print_report(stats: dict, rec: dict) -> None:
    SEP = "=" * 60
    lora = rec["lora"]
    loss = rec["loss"]
    seq = rec["_seq"]

    n = stats["n"]
    margin_p50 = stats["hardest_margin_p50"]
    listwise_std = stats["listwise_std_mean"]
    all_scores_std = stats["all_scores_std"]
    near_half = stats["scores_near_half_ratio"]

    print(SEP)
    print("数据集分析报告")
    print(SEP)
    print(f"总样本数:           {n:,}")
    print()
    print(
        f"正样本分数: mean={stats['pos_scores_mean']:.4f}  std={stats['pos_scores_std']:.4f}"
    )
    print(
        f"负样本分数: mean={stats['neg_scores_mean']:.4f}  std={stats['neg_scores_std']:.4f}"
    )
    print(
        f"全体分数 std:        {all_scores_std:.4f}  (0.4-0.6 区间占比: {near_half:.1%})"
    )
    print()
    print(
        f"最难负样本 Margin:   p25={stats['hardest_margin_p25']:.4f}  p50={margin_p50:.4f}  p75={stats['hardest_margin_p75']:.4f}"
    )
    print(f"负样本内部排序 std:  均值={listwise_std:.4f}")
    print()
    print(
        f"文档长度(字符):      p50={stats['doc_chars_p50']:.0f}  p90={stats['doc_chars_p90']:.0f}  p99={stats['doc_chars_p99']:.0f}"
    )
    print(
        f"估算 Token 数:       p90≈{seq['tokens_p90']}  p99≈{seq['tokens_p99']}  → 建议 max_seq_length={seq['suggestion']}"
    )
    print()
    print(SEP)
    print("推荐超参数配置")
    print(SEP)
    print("lora:")
    print(f"  r: {lora['r']}")
    print(f"  alpha: {lora['alpha']}")
    print(f"  dropout: {lora['dropout']}")
    print(f"  use_rslora: {str(lora['use_rslora']).lower()}")
    print()
    print("loss:")
    print(
        f"  alpha_rank: {loss['alpha_rank']}    # hardest_margin p50={margin_p50:.4f}"
    )
    print(
        f"  beta_list: {loss['beta_list']}     # neg内部排序 std均值={listwise_std:.4f}"
    )
    print(f"  gamma_point: {loss['gamma_point']}   # 0.4-0.6分数占比={near_half:.1%}")
    print(f"  temperature: {loss['temperature']}   # 全体分数 std={all_scores_std:.4f}")
    print(
        f"  hard_neg_scale: {loss['hard_neg_scale']}  # hardest_margin p50={margin_p50:.4f}"
    )
    print(SEP)
    print()
    print("注意：以上为数据驱动的初始值建议，仍需根据实际训练曲线调整。")
    print("建议调参顺序：先跑1个epoch观察 loss_rank/loss_list/loss_point 各分量。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze training data and suggest hyperparameters."
    )
    parser.add_argument(
        "--data-dir", required=True, help="Directory containing .jsonl files"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to load (reservoir sampling). Default: all.",
    )
    args = parser.parse_args()

    samples = _load_samples(args.data_dir, args.max_samples)
    if not samples:
        print("错误：没有加载到任何有效样本，请检查数据格式。")
        return

    stats = _analyze(samples)
    rec = _recommend(stats)
    _print_report(stats, rec)


if __name__ == "__main__":
    main()
