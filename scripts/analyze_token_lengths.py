"""
Analyze token length statistics for queries and documents in a JSONL file.

Usage:
    uv run python scripts/analyze_token_lengths.py \
        --path /mnt/g/PrismRerankerV1Data/KaLM__all_retrieval_voyage-rerank2_voyage-rerank2.5_web-search-processed-T2.5.jsonl \
        --n 3000
"""

import argparse
import json
import statistics
from pathlib import Path

import tiktoken


def percentile(sorted_data: list[int], p: float) -> float:
    idx = (len(sorted_data) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


def print_stats(name: str, lengths: list[int]) -> None:
    if not lengths:
        print(f"  {name}: no data")
        return
    s = sorted(lengths)
    print(f"  {name} (n={len(s)}):")
    print(f"    mean={statistics.mean(s):.1f}  median={statistics.median(s):.1f}  "
          f"stdev={statistics.stdev(s):.1f}")
    print(f"    min={s[0]}  p25={percentile(s,25):.0f}  p75={percentile(s,75):.0f}")
    print(f"    p90={percentile(s,90):.0f}  p91={percentile(s,91):.0f}  p92={percentile(s,92):.0f}  "
          f"p93={percentile(s,93):.0f}  p94={percentile(s,94):.0f}  p95={percentile(s,95):.0f}  "
          f"p96={percentile(s,96):.0f}  p97={percentile(s,97):.0f}  p98={percentile(s,98):.0f}  "
          f"p99={percentile(s,99):.0f}  max={s[-1]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--n", type=int, default=3000, help="Number of rows to sample")
    parser.add_argument("--encoding", default="cl100k_base", help="tiktoken encoding name")
    args = parser.parse_args()

    enc = tiktoken.get_encoding(args.encoding)
    print(f"Encoding: {args.encoding}")
    print(f"Reading first {args.n} rows from {args.path.name}\n")

    query_lens: list[int] = []
    pos_lens: list[int] = []
    neg_lens: list[int] = []
    web_lens: list[int] = []

    with open(args.path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            def tok(text: str) -> int:
                return len(enc.encode(text, disallowed_special=()))

            query_lens.append(tok(row["query"]))

            for doc in row.get("pos_list", []):
                pos_lens.append(tok(doc))

            for doc in row.get("neg_list", []):
                neg_lens.append(tok(doc))

            for doc in row.get("web_search_topk_docs", []):
                web_lens.append(tok(doc))

    print("=== Token Length Statistics ===\n")
    print_stats("query", query_lens)
    print()
    print_stats("pos_list docs", pos_lens)
    print()
    print_stats("neg_list docs", neg_lens)
    print()
    print_stats("web_search_topk_docs", web_lens)


if __name__ == "__main__":
    main()
