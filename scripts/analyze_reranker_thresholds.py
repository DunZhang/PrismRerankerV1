"""Reranker 分数阈值分析脚本。

基于贝叶斯后验概率估计，从正负例得分分布中提取多级阈值。
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde

# ── 配置 ──────────────────────────────────────────────────────────────
DATA_PATH = Path(
    "/mnt/g/PrismRerankerV1Data/"
    "KaLM__all_retrieval_voyage-rerank2_voyage-rerank2.5_web-search-processed.jsonl"
)
MODEL_NAMES = ["voyage-rerank-2", "voyage-rerank-2.5"]
PRIOR_POS = 1 / 8  # P(正例)，基于 1:7 正负比
PRIOR_NEG = 7 / 8  # P(负例)
TRAIN_RATIO = 0.7
SEED = 42
# 后验概率档位：以 0.01 为步长，正例从 0.50~0.99，负例从 0.01~0.50
POS_LEVELS = [round(p / 100, 2) for p in range(99, 49, -1)]  # 0.99, 0.98, ..., 0.50
NEG_LEVELS = [round(p / 100, 2) for p in range(1, 51)]  # 0.01, 0.02, ..., 0.50


def load_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """加载数据，将两个模型的分数取平均，返回 (pos_scores, neg_scores)。"""
    pos_scores: list[float] = []
    neg_scores: list[float] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # 每个模型的 pos_scores 是 list[float]，长度 1
            # neg_scores 是 list[float]，长度 7
            model_pos = [row[f"{m}_pos_scores"] for m in MODEL_NAMES]
            model_neg = [row[f"{m}_neg_scores"] for m in MODEL_NAMES]

            # 对两个模型取平均
            avg_pos = [
                sum(scores[i] for scores in model_pos) / len(MODEL_NAMES)
                for i in range(len(model_pos[0]))
            ]
            avg_neg = [
                sum(scores[i] for scores in model_neg) / len(MODEL_NAMES)
                for i in range(len(model_neg[0]))
            ]

            pos_scores.extend(avg_pos)
            neg_scores.extend(avg_neg)

    return np.array(pos_scores), np.array(neg_scores)


def find_threshold(
    grid: np.ndarray, posterior: np.ndarray, target_prob: float
) -> float | None:
    """在后验概率曲线上找到目标概率对应的分数值。"""
    # 找到后验概率穿越 target_prob 的位置
    diff = posterior - target_prob
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        return None
    # 取最后一个交叉点（后验是单调递增的，理论上只有一个）
    idx = sign_changes[-1]
    # 线性插值
    x0, x1 = grid[idx], grid[idx + 1]
    y0, y1 = posterior[idx], posterior[idx + 1]
    if abs(y1 - y0) < 1e-12:
        return float(x0)
    t = (target_prob - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0))


def compute_precision_recall(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    threshold: float,
    direction: str,
) -> tuple[float, float]:
    """计算给定阈值的 Precision 和 Recall。

    direction="above": score > threshold 预测为正例
    direction="below": score < threshold 预测为负例
    """
    if direction == "above":
        tp = np.sum(pos_scores >= threshold)
        fp = np.sum(neg_scores >= threshold)
        fn = np.sum(pos_scores < threshold)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:  # below → 预测为负例
        tn = np.sum(neg_scores <= threshold)
        fn_neg = np.sum(pos_scores <= threshold)  # 正例被误判为负
        tp_neg = tn  # 负例被正确判为负
        fp_neg = fn_neg  # 正例被误判为负
        precision = tp_neg / (tp_neg + fp_neg) if (tp_neg + fp_neg) > 0 else 0.0
        recall = tp_neg / len(neg_scores) if len(neg_scores) > 0 else 0.0

    return float(precision), float(recall)


def main() -> None:
    print("加载数据...")
    pos_scores, neg_scores = load_scores(DATA_PATH)
    print(f"  正例得分: {len(pos_scores)} 个, 范围 [{pos_scores.min():.4f}, {pos_scores.max():.4f}]")
    print(f"  负例得分: {len(neg_scores)} 个, 范围 [{neg_scores.min():.4f}, {neg_scores.max():.4f}]")

    # ── 按 query 级别做 70/30 split ──
    rng = np.random.default_rng(SEED)
    n_queries = len(pos_scores)  # 每个 query 有 1 个正例
    n_neg_per_query = len(neg_scores) // n_queries
    indices = np.arange(n_queries)
    rng.shuffle(indices)
    split = int(n_queries * TRAIN_RATIO)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_pos = pos_scores[train_idx]
    val_pos = pos_scores[val_idx]

    # 负例按 query 分组后取对应 query 的
    neg_reshaped = neg_scores.reshape(n_queries, n_neg_per_query)
    train_neg = neg_reshaped[train_idx].ravel()
    val_neg = neg_reshaped[val_idx].ravel()

    print(f"  训练集: {len(train_pos)} 正例, {len(train_neg)} 负例")
    print(f"  验证集: {len(val_pos)} 正例, {len(val_neg)} 负例")

    # ── KDE 拟合 ──
    print("\n拟合 KDE...")
    kde_pos = gaussian_kde(train_pos, bw_method="scott")
    kde_neg = gaussian_kde(train_neg, bw_method="scott")
    print(f"  KDE bandwidth (正例): {kde_pos.factor:.6f}")
    print(f"  KDE bandwidth (负例): {kde_neg.factor:.6f}")

    # ── 计算后验概率 ──
    score_min = min(pos_scores.min(), neg_scores.min()) - 0.05
    score_max = max(pos_scores.max(), neg_scores.max()) + 0.05
    grid = np.linspace(score_min, score_max, 10000)

    density_pos = kde_pos(grid)
    density_neg = kde_neg(grid)

    posterior = (density_pos * PRIOR_POS) / (
        density_pos * PRIOR_POS + density_neg * PRIOR_NEG
    )

    # ── 收集正例阈值 ──
    pos_rows: list[tuple[float, float, float, float, int]] = []
    for prob in POS_LEVELS:
        threshold = find_threshold(grid, posterior, prob)
        if threshold is None:
            continue
        prec, rec = compute_precision_recall(val_pos, val_neg, threshold, "above")
        n_hit = int(np.sum(val_pos >= threshold) + np.sum(val_neg >= threshold))
        pos_rows.append((prob, threshold, prec, rec, n_hit))

    # ── 收集负例阈值 ──
    neg_rows: list[tuple[float, float, float, float, int]] = []
    for prob in NEG_LEVELS:
        threshold = find_threshold(grid, posterior, prob)
        if threshold is None:
            continue
        prec, rec = compute_precision_recall(val_pos, val_neg, threshold, "below")
        n_hit = int(np.sum(val_pos <= threshold) + np.sum(val_neg <= threshold))
        neg_rows.append((prob, threshold, prec, rec, n_hit))

    # ── 打印到终端 ──
    print("\n" + "=" * 78)
    print("  正例阈值表：score >= 阈值 → 判为正例")
    print("=" * 78)
    print(f"  {'P(正例)':>8}  {'阈值':>8}  {'Precision':>10}  {'Recall':>10}  {'命中数':>10}")
    print("-" * 78)
    for prob, threshold, prec, rec, n_hit in pos_rows:
        print(f"  > {prob:.2f}    {threshold:8.4f}  {prec:10.4f}  {rec:10.4f}  {n_hit:10d}")

    print("\n" + "=" * 78)
    print("  负例阈值表：score <= 阈值 → 判为负例")
    print("=" * 78)
    print(f"  {'P(正例)':>8}  {'阈值':>8}  {'Precision':>10}  {'Recall':>10}  {'命中数':>10}")
    print("-" * 78)
    for prob, threshold, prec, rec, n_hit in neg_rows:
        print(f"  < {prob:.2f}    {threshold:8.4f}  {prec:10.4f}  {rec:10.4f}  {n_hit:10d}")

    # ── 写入 Markdown ──
    md_path = Path("/mnt/d/Codes/PrismRerankerV1/docs/Reranker threshold results.md")
    lines: list[str] = []
    lines.append("# Reranker 阈值分析结果\n")
    lines.append("## 方法\n")
    lines.append("- 模型：voyage-rerank-2 与 voyage-rerank-2.5 得分取平均\n")
    lines.append(f"- 数据量：{len(pos_scores)} 正例，{len(neg_scores)} 负例（1:7）\n")
    lines.append(f"- 划分：70% 训练（KDE 拟合），30% 验证（Precision/Recall 计算）\n")
    lines.append(f"- 先验：P(正例) = {PRIOR_POS:.4f}，P(负例) = {PRIOR_NEG:.4f}\n")
    lines.append("- 核密度估计 bandwidth：scott 方法\n")
    lines.append("- 贝叶斯后验：P(正例|score) = p(score|正例) × P(正例) / [p(score|正例) × P(正例) + p(score|负例) × P(负例)]\n")

    lines.append("\n## 正例阈值表\n")
    lines.append("score >= 阈值 → 判为正例\n\n")
    lines.append("| P(正例) | 阈值 | Precision | Recall | 验证集命中数 |\n")
    lines.append("|--------:|-----:|----------:|-------:|------------:|\n")
    for prob, threshold, prec, rec, n_hit in pos_rows:
        lines.append(f"| > {prob:.2f} | {threshold:.4f} | {prec:.4f} | {rec:.4f} | {n_hit} |\n")

    lines.append("\n## 负例阈值表\n")
    lines.append("score <= 阈值 → 判为负例\n\n")
    lines.append("| P(正例) | 阈值 | Precision | Recall | 验证集命中数 |\n")
    lines.append("|--------:|-----:|----------:|-------:|------------:|\n")
    for prob, threshold, prec, rec, n_hit in neg_rows:
        lines.append(f"| < {prob:.2f} | {threshold:.4f} | {prec:.4f} | {rec:.4f} | {n_hit} |\n")

    md_path.write_text("".join(lines), encoding="utf-8")
    print(f"\n结果已写入: {md_path}")

if __name__ == "__main__":
    main()
