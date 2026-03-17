"""分析 reranker 正例/负例得分分布，输出阈值建议和统计摘要。"""

import argparse
import json
import statistics
from pathlib import Path


MODEL_NAMES = ["voyage-rerank-2", "voyage-rerank-2.5"]


def load_scores(input_path: str) -> tuple[list[float], list[float]]:
    """读取 JSONL，将两个模型的分数取平均，返回 (all_pos_scores, all_neg_scores)。"""
    all_pos: list[float] = []
    all_neg: list[float] = []

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            # 正例：每个模型有 list[1]，取平均
            pos_scores_per_model = [
                record[f"{m}_pos_scores"] for m in MODEL_NAMES
            ]
            # 对每个 pos doc，跨模型取平均
            for i in range(len(pos_scores_per_model[0])):
                avg = statistics.mean(
                    model_scores[i] for model_scores in pos_scores_per_model
                )
                all_pos.append(avg)

            # 负例：每个模型有 list[N]，取平均
            neg_scores_per_model = [
                record[f"{m}_neg_scores"] for m in MODEL_NAMES
            ]
            for i in range(len(neg_scores_per_model[0])):
                avg = statistics.mean(
                    model_scores[i] for model_scores in neg_scores_per_model
                )
                all_neg.append(avg)

    return all_pos, all_neg


def compute_stats(scores: list[float]) -> dict:
    """计算分数列表的统计信息。"""
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    percentile_keys = [5, 10, 25, 50, 75, 90, 95]
    percentiles = {}
    for p in percentile_keys:
        idx = int(n * p / 100)
        idx = min(idx, n - 1)
        percentiles[f"p{p}"] = round(sorted_scores[idx], 6)

    return {
        "count": n,
        "mean": round(statistics.mean(scores), 6),
        "std": round(statistics.stdev(scores), 6) if n > 1 else 0.0,
        "min": round(min(scores), 6),
        "max": round(max(scores), 6),
        "percentiles": percentiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="分析 reranker 分数分布，输出阈值建议"
    )
    parser.add_argument("--input", required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output", required=True, help="输出 JSON 文件路径")
    args = parser.parse_args()

    print(f"读取数据: {args.input}")
    all_pos, all_neg = load_scores(args.input)
    print(f"正例数: {len(all_pos)}, 负例数: {len(all_neg)}")

    pos_stats = compute_stats(all_pos)
    neg_stats = compute_stats(all_neg)

    # 阈值建议（经人工抽样验证调整）
    # - 负例 P90 (~0.5) 作为不相关上界：验证发现 <0.5 的文档基本无法回答 query
    # - 0.5 同时作为相关下界：验证发现 >0.5 的文档多数已能提供有用信息
    irrelevant_below = neg_stats["percentiles"]["p90"]
    relevant_above = 0.5

    # 按阈值统计正负例的覆盖情况
    neg_below_threshold = sum(1 for s in all_neg if s < relevant_above)
    neg_total = len(all_neg)
    pos_above_threshold = sum(1 for s in all_pos if s >= relevant_above)
    pos_total = len(all_pos)

    result = {
        "data_file": str(Path(args.input).name),
        "num_queries": len(all_pos),
        "models": MODEL_NAMES,
        "pos_scores_stats": pos_stats,
        "neg_scores_stats": neg_stats,
        "thresholds": {
            "irrelevant_below": irrelevant_below,
            "relevant_above": relevant_above,
            "description": {
                "irrelevant_below": "低于此分数的文档与 query 基本无关，可直接丢弃",
                "relevant_above": "高于此分数的文档大概率与 query 相关，可作为正例",
            },
        },
        "validation": {
            "neg_below_threshold_ratio": round(neg_below_threshold / neg_total, 4),
            "pos_above_threshold_ratio": round(pos_above_threshold / pos_total, 4),
            "description": {
                "neg_below_threshold_ratio": "负例中低于 relevant_above 的比例（越高越好）",
                "pos_above_threshold_ratio": "正例中高于 relevant_above 的比例（越高越好）",
            },
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n=== 统计摘要 ===")
    print(f"正例 mean={pos_stats['mean']:.4f}, std={pos_stats['std']:.4f}")
    print(f"负例 mean={neg_stats['mean']:.4f}, std={neg_stats['std']:.4f}")
    print(f"\n=== 建议阈值 ===")
    print(f"不相关: < {irrelevant_below:.4f}")
    print(f"相关:   >= {relevant_above:.4f}")
    print(f"\n=== 覆盖率验证 ===")
    print(
        f"负例中 < {relevant_above}: "
        f"{neg_below_threshold}/{neg_total} "
        f"({neg_below_threshold / neg_total:.1%})"
    )
    print(
        f"正例中 >= {relevant_above}: "
        f"{pos_above_threshold}/{pos_total} "
        f"({pos_above_threshold / pos_total:.1%})"
    )
    print(f"\n结果已写入: {args.output}")


if __name__ == "__main__":
    main()
