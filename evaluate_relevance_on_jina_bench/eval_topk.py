"""Compute NDCG@10 from Top-K retrieval JSONL files and write to xlsx.

Reads JSONL files produced by the BEIR retrieval pipeline, loads qrels
from local JSON files, and saves results to an Excel spreadsheet.

Usage:
    uv run python -m evaluate_relevance_on_jina_bench eval_topk [--results-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from math import log2

from openpyxl import Workbook


def dcg_at_k(relevances: list[float], k: int) -> float:
    """Discounted cumulative gain at rank k."""
    return sum(rel / log2(i + 2) for i, rel in enumerate(relevances[:k]))


def _qrels_path_for_jsonl(jsonl_path: Path) -> Path:
    """Derive qrels JSON path from JSONL path.

    'mteb__nfcorpus_top100.jsonl' -> 'mteb__nfcorpus_qrels.json'
    """
    stem = jsonl_path.stem
    parts = stem.rsplit("_top", 1)
    qrels_name = f"{parts[0]}_qrels.json"
    return jsonl_path.parent / qrels_name


def compute_ndcg10_from_jsonl(
    path: Path,
    qrels: dict[str, dict[str, int]],
) -> float:
    """Compute mean NDCG@10 using full qrels for correct IDCG.

    Args:
        path: JSONL file path.
        qrels: Full qrels keyed by query_text: {query_text: {corpus_id: relevance}}.
    """
    ndcg_scores: list[float] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            query_text = record["query"]
            if query_text not in qrels:
                continue

            # DCG from retrieved rank order
            relevance = [
                float(doc["relevance"]) for doc in record["documents"]
            ]
            dcg = dcg_at_k(relevance, 10)

            # IDCG from ALL relevant docs in qrels
            all_relevant = sorted(qrels[query_text].values(), reverse=True)
            idcg = dcg_at_k([float(v) for v in all_relevant], 10)
            if idcg == 0.0:
                continue

            ndcg_scores.append(dcg / idcg)

    if not ndcg_scores:
        return 0.0
    return sum(ndcg_scores) / len(ndcg_scores)


def main() -> None:
    """CLI entry point for eval_topk."""
    parser = argparse.ArgumentParser(
        description="Compute NDCG@10 from Top-K JSONL files"
    )
    default_dir = str(Path(__file__).resolve().parent / "results")
    parser.add_argument(
        "--results-dir",
        default=default_dir,
        help=f"Directory containing JSONL files (default: {default_dir})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output xlsx path (default: <results-dir>/ndcg10_results.xlsx)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    jsonl_files = sorted(results_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in {results_dir}")
        return

    output_path = (
        Path(args.output) if args.output else results_dir / "ndcg10_results.xlsx"
    )

    results: list[tuple[str, float]] = []
    for fpath in jsonl_files:
        qrels_path = _qrels_path_for_jsonl(fpath)
        if not qrels_path.exists():
            print(f"Skipping {fpath.name} (qrels not found: {qrels_path.name})")
            continue

        short_name = fpath.stem.rsplit("_top", 1)[0]
        print(f"Processing {short_name} ...", end=" ")

        with open(qrels_path, encoding="utf-8") as f:
            qrels = json.load(f)

        score = compute_ndcg10_from_jsonl(fpath, qrels)
        results.append((fpath.name, score))
        print(f"NDCG@10 = {score:.4f}")

    if not results:
        print("No valid results computed.")
        return

    # Write xlsx
    wb = Workbook()
    ws = wb.active
    ws.title = "NDCG@10"
    ws.append(["File", "NDCG@10"])
    for name, score in results:
        ws.append([name, round(score, 4)])

    avg = sum(s for _, s in results) / len(results)
    ws.append(["Average", round(avg, 4)])

    wb.save(output_path)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
