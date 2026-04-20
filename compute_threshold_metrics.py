"""Compute accuracy and positive recall at varying score thresholds."""

from __future__ import annotations

import json
from pathlib import Path

from openpyxl import Workbook

DATA_PATHS = [
    Path(
        "/mnt/g/PrismRerankerV1Data/step8_kalm_web-search_query_document_pairs_annotated_merged.jsonl"
    ),
    Path(
        "/mnt/g/PrismRerankerV1Data/data_extend2/step8_expanded2_web-search_query_document_pairs_annotated_merged.jsonl"
    ),
]
OUTPUT_PATH = Path("/mnt/d/Codes/PrismRerankerV1/threshold_metrics.xlsx")
SCORE_KEY = "voyage-rerank-2.5_score"
STEP = 0.01


def load_pairs(paths: list[Path], score_key: str) -> list[tuple[float, int]]:
    """Load (score, label_int) pairs from all paths; label 1 for 'yes', 0 for 'no'."""
    pairs: list[tuple[float, int]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                label = 1 if d["annotated_label"] == "yes" else 0
                pairs.append((float(d[score_key]), label))
    return pairs


def compute_metrics(
    pairs: list[tuple[float, int]], thresholds: list[float]
) -> list[dict[str, float]]:
    """For each threshold t, score >= t predicts positive."""
    scores_sorted = sorted(pairs, key=lambda p: p[0])
    scores = [p[0] for p in scores_sorted]
    labels = [p[1] for p in scores_sorted]
    n = len(pairs)
    total_pos = sum(labels)
    total_neg = n - total_pos

    # prefix_pos[i] = number of positives with score < scores[i] index (among sorted)
    # we want, for threshold t: count where score >= t (predicted positive)
    # use bisect: idx = first i where scores[i] >= t; everything from idx..n-1 is predicted positive.
    import bisect

    results: list[dict[str, float]] = []
    cum_pos = [0] * (n + 1)
    for i, lab in enumerate(labels):
        cum_pos[i + 1] = cum_pos[i] + lab

    for t in thresholds:
        idx = bisect.bisect_left(scores, t)
        pred_pos_count = n - idx
        tp = total_pos - cum_pos[idx]
        fp = pred_pos_count - tp
        fn = total_pos - tp
        tn = total_neg - fp
        accuracy = (tp + tn) / n if n else 0.0
        recall_pos = tp / total_pos if total_pos else 0.0
        precision_pos = tp / pred_pos_count if pred_pos_count else 0.0
        results.append(
            {
                "threshold": round(t, 4),
                "accuracy": accuracy,
                "positive_recall": recall_pos,
                "positive_precision": precision_pos,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "pred_positive": pred_pos_count,
            }
        )
    return results


def write_xlsx(results: list[dict[str, float]], out: Path) -> None:
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "threshold_metrics"
    headers = [
        "threshold",
        "accuracy",
        "positive_recall",
        "positive_precision",
        "TP",
        "FP",
        "FN",
        "TN",
        "pred_positive",
    ]
    ws.append(headers)
    for row in results:
        ws.append([row[h] for h in headers])
    for col_idx, h in enumerate(headers, start=1):
        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = max(
            12, len(h) + 2
        )
    wb.save(out)


def main() -> None:
    pairs = load_pairs(DATA_PATHS, SCORE_KEY)
    n_pos = sum(1 for _, lab in pairs if lab == 1)
    print(f"Loaded {len(pairs)} rows; positives={n_pos}, negatives={len(pairs) - n_pos}")
    thresholds = [round(i * STEP, 4) for i in range(int(1.0 / STEP) + 1)]
    results = compute_metrics(pairs, thresholds)
    write_xlsx(results, OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH} with {len(results)} rows")


if __name__ == "__main__":
    main()
