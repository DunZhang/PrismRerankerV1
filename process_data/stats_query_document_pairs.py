"""统计 query-document pairs JSONL 数据的各种特征。

用法:
    uv run python process_data/stats_query_document_pairs.py [jsonl_path]

默认读取:
    /mnt/g/PrismRerankerV1Data/kalm_web-search_query_document_pairs.jsonl
"""

import json
import sys
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import tiktoken

# ── 配置 ──────────────────────────────────────────────────────────────
DEFAULT_FILE = Path(
    "/mnt/g/PrismRerankerV1Data/kalm_web-search_query_document_pairs.jsonl"
)
SCORE_KEY = "voyage-rerank-2_and_2.5_score"
# 多进程 batch 大小
_BATCH_SIZE = 2000

# 分数桶宽
SCORE_BUCKET_WIDTH = 0.05
# 长度桶宽（token 数）
LENGTH_BUCKET_WIDTH = 50


# ── 工具函数 ──────────────────────────────────────────────────────────
def _batch_token_len(texts: list[str]) -> list[int]:
    """在子进程中批量计算 token 长度（每个子进程初始化自己的 encoder）。"""
    enc = tiktoken.get_encoding("cl100k_base")
    return [len(enc.encode(t)) for t in texts]


def parallel_token_len(texts: list[str], max_workers: int | None = None) -> list[int]:
    """多进程并行计算 token 长度，返回与输入顺序一致的结果。"""
    if len(texts) <= _BATCH_SIZE:
        return _batch_token_len(texts)

    batches = [
        texts[i : i + _BATCH_SIZE] for i in range(0, len(texts), _BATCH_SIZE)
    ]
    results: list[int] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for batch_result in pool.map(_batch_token_len, batches):
            results.extend(batch_result)
    return results


def char_len(text: str) -> int:
    """返回文本的字符数。"""
    return len(text)


def percentiles(data: list[float | int], ps: list[int]) -> dict[str, float]:
    """计算百分位数。"""
    sorted_data = sorted(data)
    n = len(sorted_data)
    result: dict[str, float] = {}
    for p in ps:
        idx = int(n * p / 100)
        idx = min(idx, n - 1)
        result[f"p{p}"] = sorted_data[idx]
    return result


def distribution_summary(
    data: list[float | int],
    label: str,
) -> None:
    """打印数值分布的完整摘要。"""
    print(f"\n  {label}:")
    print(f"    count:  {len(data)}")
    print(f"    min:    {min(data):.4g}")
    print(f"    max:    {max(data):.4g}")
    print(f"    mean:   {statistics.mean(data):.4f}")
    print(f"    median: {statistics.median(data):.4f}")
    print(f"    stdev:  {statistics.stdev(data):.4f}" if len(data) > 1 else "")
    pcts = percentiles(data, [5, 10, 25, 75, 90, 95, 99])
    print(f"    p5={pcts['p5']:.4g}  p10={pcts['p10']:.4g}  "
          f"p25={pcts['p25']:.4g}  p75={pcts['p75']:.4g}  "
          f"p90={pcts['p90']:.4g}  p95={pcts['p95']:.4g}  "
          f"p99={pcts['p99']:.4g}")


def print_histogram(
    data: list[float | int],
    bucket_width: float,
    label: str,
    *,
    max_bar_width: int = 50,
    hide_zero: bool = True,
) -> None:
    """打印带柱状图的分桶分布。跳过 count=0 的桶以避免超长输出。"""
    buckets: Counter[int] = Counter()
    for v in data:
        buckets[int(v / bucket_width)] += 1

    print(f"\n  {label} (bucket width={bucket_width}):")
    max_count = max(buckets.values())
    for b in range(min(buckets), max(buckets) + 1):
        count = buckets.get(b, 0)
        if hide_zero and count == 0:
            continue
        lo = b * bucket_width
        hi = lo + bucket_width
        bar_len = int(count / max_count * max_bar_width) if max_count > 0 else 0
        bar = "█" * bar_len
        if bucket_width == int(bucket_width):
            print(f"    [{lo:6.0f}, {hi:6.0f}): {count:>6}  {bar}")
        else:
            print(f"    [{lo:.2f}, {hi:.2f}): {count:>6}  {bar}")


def print_top_n(
    counter: Counter[int | str],
    label: str,
    n: int = 15,
) -> None:
    """打印出现次数最多的 top-N。"""
    print(f"\n  {label} (top {n}):")
    for val, cnt in counter.most_common(n):
        print(f"    {val}: {cnt}")


# ── 主逻辑 ────────────────────────────────────────────────────────────
def main() -> None:
    input_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE
    print(f"Loading: {input_file}")

    # 加载全部数据
    records: list[dict[str, str | float]] = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Total records: {len(records)}")

    # 提取字段
    queries = [r["query"] for r in records]
    documents = [r["document"] for r in records]
    scores = [float(r[SCORE_KEY]) for r in records]

    # ── 1. 基本统计 ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("1. BASIC OVERVIEW")
    print("=" * 70)
    unique_queries = set(queries)
    unique_docs = set(documents)
    print(f"  Total query-document pairs: {len(records)}")
    print(f"  Unique queries:             {len(unique_queries)}")
    print(f"  Unique documents:           {len(unique_docs)}")
    print(f"  Query reuse rate:           {len(records) / len(unique_queries):.2f}x")
    print(f"  Document reuse rate:        {len(records) / len(unique_docs):.2f}x")

    # ── 2. 每个 query 的 document 数量分布 ───────────────────────────
    print("\n" + "=" * 70)
    print("2. DOCUMENTS PER QUERY")
    print("=" * 70)
    docs_per_query: Counter[str] = Counter(queries)
    doc_counts = list(docs_per_query.values())
    distribution_summary(doc_counts, "Docs per query")

    doc_count_freq: Counter[int] = Counter(doc_counts)
    print("\n  Frequency of doc-count values:")
    for k in sorted(doc_count_freq):
        pct = doc_count_freq[k] / len(doc_counts) * 100
        print(f"    {k} docs: {doc_count_freq[k]} queries ({pct:.1f}%)")

    # ── 3. 每个 document 被多少 query 引用 ──────────────────────────
    print("\n" + "=" * 70)
    print("3. QUERIES PER DOCUMENT")
    print("=" * 70)
    queries_per_doc: Counter[str] = Counter(documents)
    qpd_counts = list(queries_per_doc.values())
    distribution_summary(qpd_counts, "Queries per document")

    qpd_freq: Counter[int] = Counter(qpd_counts)
    print("\n  Frequency of query-count values (top 10):")
    for k in sorted(qpd_freq)[:10]:
        pct = qpd_freq[k] / len(qpd_counts) * 100
        print(f"    {k} queries: {qpd_freq[k]} docs ({pct:.1f}%)")

    # ── 4. 分数分布 ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("4. SCORE DISTRIBUTION")
    print("=" * 70)
    distribution_summary(scores, "Score")
    print_histogram(scores, SCORE_BUCKET_WIDTH, "Score histogram")

    # 更细的分桶 (0.01)
    print_histogram(scores, 0.01, "Score fine-grained histogram (0.01)")

    # ── 5. Query 长度分布（token） ───────────────────────────────────
    print("\n" + "=" * 70)
    print("5. QUERY LENGTH (tokens, cl100k_base)")
    print("=" * 70)
    # 对 unique query 只计算一次（多进程加速）
    unique_query_list = list(unique_queries)
    query_tl_values = parallel_token_len(unique_query_list)
    query_token_lens = dict(zip(unique_query_list, query_tl_values))
    query_tl_list = query_tl_values
    distribution_summary(query_tl_list, "Query token length (unique)")
    print_histogram(query_tl_list, LENGTH_BUCKET_WIDTH, "Query token length histogram")

    # 字符长度
    query_char_lens = [char_len(q) for q in unique_queries]
    distribution_summary(query_char_lens, "Query char length (unique)")

    # ── 6. Document 长度分布（token） ────────────────────────────────
    print("\n" + "=" * 70)
    print("6. DOCUMENT LENGTH (tokens, cl100k_base)")
    print("=" * 70)
    unique_doc_list = list(unique_docs)
    doc_tl_values = parallel_token_len(unique_doc_list)
    doc_token_lens = dict(zip(unique_doc_list, doc_tl_values))
    doc_tl_list = doc_tl_values
    distribution_summary(doc_tl_list, "Document token length (unique)")
    print_histogram(doc_tl_list, 500, "Document token length histogram")

    # 字符长度
    doc_char_lens = [char_len(d) for d in unique_docs]
    distribution_summary(doc_char_lens, "Document char length (unique)")

    # ── 7. Query+Document 组合长度 ───────────────────────────────────
    print("\n" + "=" * 70)
    print("7. QUERY + DOCUMENT COMBINED LENGTH (tokens)")
    print("=" * 70)
    combined_lens = [
        query_token_lens[r["query"]] + doc_token_lens[r["document"]]
        for r in records
    ]
    distribution_summary(combined_lens, "Combined token length")
    print_histogram(combined_lens, 500, "Combined token length histogram")

    # ── 8. 分数 vs 长度 关系 ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("8. SCORE vs DOCUMENT LENGTH")
    print("=" * 70)
    score_buckets_doc_len: defaultdict[int, list[int]] = defaultdict(list)
    for r in records:
        b = int(float(r[SCORE_KEY]) / SCORE_BUCKET_WIDTH)
        score_buckets_doc_len[b].append(doc_token_lens[r["document"]])
    print("\n  Avg document token length per score bucket:")
    for b in sorted(score_buckets_doc_len):
        lo = b * SCORE_BUCKET_WIDTH
        hi = lo + SCORE_BUCKET_WIDTH
        lens = score_buckets_doc_len[b]
        avg = statistics.mean(lens)
        med = statistics.median(lens)
        print(f"    [{lo:.2f}, {hi:.2f}): n={len(lens):>5}  "
              f"avg_len={avg:>7.1f}  median_len={med:>7.1f}")

    # ── 9. 分数 vs Query 长度 关系 ───────────────────────────────────
    print("\n" + "=" * 70)
    print("9. SCORE vs QUERY LENGTH")
    print("=" * 70)
    score_buckets_q_len: defaultdict[int, list[int]] = defaultdict(list)
    for r in records:
        b = int(float(r[SCORE_KEY]) / SCORE_BUCKET_WIDTH)
        score_buckets_q_len[b].append(query_token_lens[r["query"]])
    print("\n  Avg query token length per score bucket:")
    for b in sorted(score_buckets_q_len):
        lo = b * SCORE_BUCKET_WIDTH
        hi = lo + SCORE_BUCKET_WIDTH
        lens = score_buckets_q_len[b]
        avg = statistics.mean(lens)
        med = statistics.median(lens)
        print(f"    [{lo:.2f}, {hi:.2f}): n={len(lens):>5}  "
              f"avg_len={avg:>6.1f}  median_len={med:>6.1f}")

    # ── 10. 重复检测 ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("10. DUPLICATE DETECTION")
    print("=" * 70)
    pair_counter: Counter[tuple[str, str]] = Counter()
    for r in records:
        pair_counter[(r["query"], r["document"])] += 1
    dup_pairs = {k: v for k, v in pair_counter.items() if v > 1}
    print(f"  Total unique (query, document) pairs: {len(pair_counter)}")
    print(f"  Duplicate pairs (appearing > 1 time): {len(dup_pairs)}")
    if dup_pairs:
        print(f"  Max duplication count: {max(dup_pairs.values())}")
        total_dup_records = sum(v - 1 for v in dup_pairs.values())
        print(f"  Total redundant records: {total_dup_records}")

    # ── 11. Query 语言 / 字符集估计 ──────────────────────────────────
    print("\n" + "=" * 70)
    print("11. LANGUAGE / CHARACTER SET ESTIMATION")
    print("=" * 70)
    cjk_queries = sum(
        1 for q in unique_queries if any("\u4e00" <= c <= "\u9fff" for c in q)
    )
    cjk_docs = sum(
        1 for d in unique_docs if any("\u4e00" <= c <= "\u9fff" for c in d)
    )
    print(f"  Queries containing CJK chars: {cjk_queries} / {len(unique_queries)} "
          f"({cjk_queries / len(unique_queries) * 100:.1f}%)")
    print(f"  Docs containing CJK chars:    {cjk_docs} / {len(unique_docs)} "
          f"({cjk_docs / len(unique_docs) * 100:.1f}%)")

    # ASCII-only
    ascii_queries = sum(1 for q in unique_queries if q.isascii())
    ascii_docs = sum(1 for d in unique_docs if d.isascii())
    print(f"  ASCII-only queries: {ascii_queries} / {len(unique_queries)} "
          f"({ascii_queries / len(unique_queries) * 100:.1f}%)")
    print(f"  ASCII-only docs:    {ascii_docs} / {len(unique_docs)} "
          f"({ascii_docs / len(unique_docs) * 100:.1f}%)")

    # ── 12. 空 / 超短文本检测 ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("12. SHORT / EMPTY TEXT DETECTION")
    print("=" * 70)
    empty_q = sum(1 for q in unique_queries if len(q.strip()) == 0)
    empty_d = sum(1 for d in unique_docs if len(d.strip()) == 0)
    short_q = sum(1 for q in unique_queries if query_token_lens[q] <= 3)
    short_d = sum(1 for d in unique_docs if doc_token_lens[d] <= 5)
    long_d = sum(1 for d in unique_docs if doc_token_lens[d] > 2000)
    print(f"  Empty queries:           {empty_q}")
    print(f"  Empty documents:         {empty_d}")
    print(f"  Very short queries (≤3 tokens):  {short_q}")
    print(f"  Very short docs (≤5 tokens):     {short_d}")
    print(f"  Very long docs (>2000 tokens):   {long_d}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
