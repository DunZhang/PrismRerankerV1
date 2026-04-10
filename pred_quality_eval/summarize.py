"""Summarize the enriched JSONL produced by ``evaluate.py``.

Outputs:
- ``metrics.json``: all numerical metrics in one file.
- ``summary.md``: human-friendly markdown report with ASCII histograms.

Usage:
    uv run python -m pred_quality_eval.summarize \\
        --input  /mnt/g/PrismRerankerV1Data/Qwen3.5-2B-samples-4000-result.eval.jsonl \\
        --outdir /mnt/g/PrismRerankerV1Data/eval_summary
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pred_quality_eval.evaluate import SCORE_FIELDS

log = logging.getLogger("pred_quality_summarize")

DEFAULT_INPUT_PATH = Path(
    "/mnt/g/PrismRerankerV1Data/sft_test_data/qwen3_5_2B_v2_eval.jsonl"
)
LOW_SCORE_EXAMPLE_LIMIT = 20


def _default_outdir(input_path: Path) -> Path:
    """Derive outdir as a sibling folder named ``<stem>_summary`` next to input."""
    return input_path.parent / f"{input_path.stem}_summary"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _load_rows(input_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _ascii_bar(count: int, max_count: int, width: int = 30) -> str:
    if max_count <= 0:
        return ""
    filled = round(count / max_count * width)
    return "█" * filled + "░" * (width - filled)


def _compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)

    # --- Task 1: label accuracy & confusion matrix ---
    cm = {
        "true_yes_pred_yes": 0,
        "true_yes_pred_no": 0,
        "true_no_pred_yes": 0,
        "true_no_pred_no": 0,
        "pred_unparseable": 0,
    }
    for row in rows:
        annotated = row.get("annotated_label")
        pred = row.get("pred_label")
        if pred not in ("yes", "no"):
            cm["pred_unparseable"] += 1
            continue
        key = f"true_{annotated}_pred_{pred}"
        if key in cm:
            cm[key] += 1

    parseable = total - cm["pred_unparseable"]
    correct = cm["true_yes_pred_yes"] + cm["true_no_pred_no"]
    label_accuracy = correct / parseable if parseable else 0.0

    tp = cm["true_yes_pred_yes"]
    fp = cm["true_no_pred_yes"]
    fn = cm["true_yes_pred_no"]
    precision_yes = tp / (tp + fp) if (tp + fp) else 0.0
    recall_yes = tp / (tp + fn) if (tp + fn) else 0.0
    f1_yes = (
        2 * precision_yes * recall_yes / (precision_yes + recall_yes)
        if (precision_yes + recall_yes)
        else 0.0
    )

    # --- Task 2: score aggregation ---
    status_counts: Counter[str] = Counter()
    for row in rows:
        status_counts[str(row.get("eval_status"))] += 1

    scored_rows = [r for r in rows if r.get("eval_status") == "scored"]
    scored_count = len(scored_rows)

    avg_scores: dict[str, float] = {}
    distributions: dict[str, dict[str, int]] = {}
    for field in SCORE_FIELDS:
        vals = [
            int(r["eval_scores"][field])
            for r in scored_rows
            if isinstance(r.get("eval_scores"), dict) and field in r["eval_scores"]
        ]
        avg_scores[field] = sum(vals) / len(vals) if vals else 0.0
        dist = {str(i): 0 for i in range(1, 6)}
        for v in vals:
            if 1 <= v <= 5:
                dist[str(v)] += 1
        distributions[field] = dist

    # --- Low-score examples (overall <= 2) ---
    low_score_examples: list[dict[str, Any]] = []
    for row in scored_rows:
        scores = row.get("eval_scores") or {}
        if scores.get("overall", 5) <= 2:
            low_score_examples.append(
                {
                    "query": row.get("query"),
                    "contribution": row.get("parsed_contribution"),
                    "evidence": row.get("parsed_evidence"),
                    "scores": scores,
                    "reason": row.get("eval_reason"),
                }
            )
        if len(low_score_examples) >= LOW_SCORE_EXAMPLE_LIMIT:
            break

    return {
        "total_rows": total,
        "label_accuracy": round(label_accuracy, 4),
        "confusion_matrix": cm,
        "precision_yes": round(precision_yes, 4),
        "recall_yes": round(recall_yes, 4),
        "f1_yes": round(f1_yes, 4),
        "eval_status_counts": dict(status_counts),
        "eval_scored_count": scored_count,
        "avg_scores": {k: round(v, 3) for k, v in avg_scores.items()},
        "score_distributions": distributions,
        "low_score_examples": low_score_examples,
    }


def _render_markdown(metrics: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Pred Quality Evaluation Summary\n")
    lines.append(f"- Total rows: **{metrics['total_rows']}**")
    lines.append(f"- Label accuracy: **{metrics['label_accuracy']:.4f}**")
    lines.append(
        f"- Precision (yes): **{metrics['precision_yes']:.4f}** | "
        f"Recall (yes): **{metrics['recall_yes']:.4f}** | "
        f"F1 (yes): **{metrics['f1_yes']:.4f}**"
    )

    lines.append("\n## Confusion Matrix\n")
    cm = metrics["confusion_matrix"]
    lines.append("| | pred yes | pred no | unparseable |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| **true yes** | {cm['true_yes_pred_yes']} | {cm['true_yes_pred_no']} | — |"
    )
    lines.append(
        f"| **true no**  | {cm['true_no_pred_yes']} | {cm['true_no_pred_no']} | — |"
    )
    lines.append(f"| **—**        | — | — | {cm['pred_unparseable']} |")

    lines.append("\n## Eval Status Counts\n")
    for status, count in sorted(metrics["eval_status_counts"].items()):
        lines.append(f"- `{status}`: {count}")

    lines.append(f"\n## Quality Scores ({metrics['eval_scored_count']} rows scored)\n")
    lines.append("| Dimension | Avg |")
    lines.append("|---|---:|")
    for field in SCORE_FIELDS:
        lines.append(f"| {field} | {metrics['avg_scores'][field]:.3f} |")

    lines.append("\n## Score Distributions\n")
    for field in SCORE_FIELDS:
        dist = metrics["score_distributions"][field]
        max_c = max(dist.values()) if dist else 0
        lines.append(f"\n**{field}**")
        for score in ("1", "2", "3", "4", "5"):
            c = dist.get(score, 0)
            bar = _ascii_bar(c, max_c)
            lines.append(f"- `{score}`: {bar} {c}")

    lines.append("\n## Low-score Examples (overall ≤ 2)\n")
    examples = metrics["low_score_examples"]
    if not examples:
        lines.append("_(none)_")
    else:
        for i, ex in enumerate(examples, 1):
            lines.append(f"\n### Example {i}")
            lines.append(f"- **query**: {ex['query']}")
            lines.append(f"- **contribution**: {ex['contribution']}")
            lines.append(f"- **evidence**: {ex['evidence']}")
            lines.append(f"- **scores**: `{ex['scores']}`")
            lines.append(f"- **reason**: {ex['reason']}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize pred-quality eval JSONL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output dir. Defaults to <input_parent>/<input_stem>_summary.",
    )
    args = parser.parse_args()

    _setup_logging()

    if not args.input.exists():
        log.error("input not found: %s", args.input)
        sys.exit(1)

    outdir: Path = args.outdir or _default_outdir(args.input)

    rows = _load_rows(args.input)
    log.info("Loaded %d rows from %s", len(rows), args.input)

    metrics = _compute_metrics(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    log.info("Output dir: %s", outdir)

    metrics_path = outdir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    log.info("Wrote %s", metrics_path)

    summary_path = outdir / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(_render_markdown(metrics))
    log.info("Wrote %s", summary_path)

    log.info("-" * 60)
    log.info("Label accuracy: %.4f", metrics["label_accuracy"])
    log.info(
        "Avg overall score: %.3f (%d scored rows)",
        metrics["avg_scores"]["overall"],
        metrics["eval_scored_count"],
    )


if __name__ == "__main__":
    main()
