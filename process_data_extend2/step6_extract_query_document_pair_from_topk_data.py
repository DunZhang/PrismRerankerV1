"""从 KaLM topk 数据中抽取分数均衡的 query-document 对。

来源：web_search_topk_docs（及对应 reranker 分数）

对每个来源按 avg_score 分桶（桶宽 0.05）均衡采样，
最后按 query 分组（query 间随机排列）输出 JSONL。
"""

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import tiktoken

# ── 配置 ──────────────────────────────────────────────────────────────
INPUT_FILE = Path(
    "/mnt/g/PrismRerankerV1Data/data_extend2/"
    "step5_expanded2_web_search-processed-Rerank2.5-Rerank2_keywords.jsonl"
)
OUTPUT_FILE = INPUT_FILE.parent / "step6_expanded2_web-search_query_document_pairs.jsonl"
BUCKET_WIDTH = 0.05
DISCARD_IF_EXCEEDS_TOKENS = 10240  # query+doc 超过此 token 数则丢弃整条
TRUNCATE_DOC_TO_TOKENS = 7168  # document 超过此 token 数则截断到该长度
BUCKET_MIN_RATIO = 0.7  # 中间桶的最小/最大数量比 ≥ 此值
EXTREME_BUCKET_LO = 0.20  # 低于此分数的桶视为极端桶（有多少用多少）
EXTREME_BUCKET_HI = 0.95  # 高于此分数的桶视为极端桶（有多少用多少）
# 桶数量倍率：分数 >= 阈值的中间桶，采样数量乘以对应倍率
# 格式 (score_threshold, multiplier)，按阈值从高到低匹配第一个命中的
BUCKET_MULTIPLIERS: list[tuple[float, float]] = [
    (0.65, 1.7),
]
SEED = 42

ENCODING = tiktoken.get_encoding("cl100k_base")

# 分数 key
S2_WEB = "voyage-rerank-2_web_search_topk_docs_scores"
S25_WEB = "voyage-rerank-2.5_web_search_topk_docs_scores"

QueryDocScore = tuple[str, str, float]


# ── 工具函数 ──────────────────────────────────────────────────────────
def score_to_bucket(score: float) -> int:
    """将分数映射到桶索引（桶宽 BUCKET_WIDTH）。"""
    return int(score / BUCKET_WIDTH)


def _is_extreme_bucket(bucket_id: int) -> bool:
    """判断桶是否为极端桶（有多少用多少，不参与均衡）。"""
    lo = bucket_id * BUCKET_WIDTH
    return lo < EXTREME_BUCKET_LO or lo >= EXTREME_BUCKET_HI


def _bucket_multiplier(bucket_id: int) -> float:
    """返回桶的数量倍率（按 BUCKET_MULTIPLIERS 从高到低匹配）。"""
    lo = bucket_id * BUCKET_WIDTH
    for threshold, multiplier in sorted(BUCKET_MULTIPLIERS, reverse=True):
        if lo >= threshold:
            return multiplier
    return 1.0


def balanced_sample(
    pairs: list[QueryDocScore],
    rng: random.Random,
) -> list[QueryDocScore]:
    """按分数分桶，极端桶全部保留，中间桶用 BUCKET_MIN_RATIO 约束最大化采样。

    每个中间桶的采样上限 = base_cap * multiplier，其中 multiplier
    由 BUCKET_MULTIPLIERS 配置决定。base_cap 的计算保证所有中间桶
    的实际数量/目标上限 >= BUCKET_MIN_RATIO。
    """
    buckets: dict[int, list[QueryDocScore]] = defaultdict(list)
    for item in pairs:
        buckets[score_to_bucket(item[2])].append(item)

    # 分离极端桶和中间桶
    middle: dict[int, list[QueryDocScore]] = {}
    for bid, items in buckets.items():
        if not _is_extreme_bucket(bid):
            middle[bid] = items

    # 中间桶：base_cap = min(size_i / mult_i) / BUCKET_MIN_RATIO
    # 每个桶的 cap_i = base_cap * mult_i
    if middle:
        base_cap = int(
            min(
                len(items) / _bucket_multiplier(bid)
                for bid, items in middle.items()
            )
            / BUCKET_MIN_RATIO
        )
    else:
        base_cap = 0

    sampled: list[QueryDocScore] = []
    for bucket_id in sorted(buckets):
        items = buckets[bucket_id]
        if _is_extreme_bucket(bucket_id):
            sampled.extend(items)  # 极端桶全部保留
        else:
            cap = int(base_cap * _bucket_multiplier(bucket_id))
            if len(items) > cap:
                sampled.extend(rng.sample(items, cap))
            else:
                sampled.extend(items)
    return sampled


def print_bucket_distribution(
    pairs: list[QueryDocScore],
    label: str,
) -> None:
    """打印分桶分布。"""
    buckets: dict[int, int] = defaultdict(int)
    for _, _, s in pairs:
        buckets[score_to_bucket(s)] += 1
    print(f"\n  {label} bucket distribution:")
    for b in sorted(buckets):
        lo = b * BUCKET_WIDTH
        hi = lo + BUCKET_WIDTH
        print(f"    [{lo:.2f}, {hi:.2f}): {buckets[b]}")


def print_score_stats(pairs: list[QueryDocScore], label: str) -> None:
    """打印分数统计。"""
    scores = [s for _, _, s in pairs]
    print(f"\n  {label} score stats:")
    print(f"    count: {len(scores)}")
    print(f"    min:   {min(scores):.4f}")
    print(f"    max:   {max(scores):.4f}")
    print(f"    mean:  {statistics.mean(scores):.4f}")
    print(f"    median:{statistics.median(scores):.4f}")
    print(f"    stdev: {statistics.stdev(scores):.4f}")


# ── 主逻辑 ────────────────────────────────────────────────────────────
def main() -> None:
    rng = random.Random(SEED)

    # Step 1: 加载数据，构建 (query, document, avg_score)
    print("Loading data...")
    pairs: list[QueryDocScore] = []
    query_to_keywords: dict[str, list[str]] = {}

    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            query: str = rec["query"]

            if "Dialogue History:" in query:
                continue
            if "根据以下商品信息检索对应的商品描述" in query:
                continue

            if "keywords" in rec:
                query_to_keywords[query] = rec["keywords"]

            # web_search_topk_docs
            for doc, s2, s25 in zip(
                rec["web_search_topk_docs"], rec[S2_WEB], rec[S25_WEB], strict=True
            ):
                pairs.append((query, doc, (s2 + s25) / 2))

    print(f"Web search pairs: {len(pairs)}")

    # Step 1.5: 截断过长 document，然后丢弃总 token 数仍超限的
    print(f"\nTruncating docs > {TRUNCATE_DOC_TO_TOKENS} tokens, "
          f"discarding pairs > {DISCARD_IF_EXCEEDS_TOKENS} total tokens...")

    def _truncate_and_filter(
        pair: QueryDocScore,
    ) -> QueryDocScore | None:
        query, doc, score = pair
        q_tokens = ENCODING.encode(query, disallowed_special=())
        d_tokens = ENCODING.encode(doc, disallowed_special=())
        # 截断 document
        if len(d_tokens) > TRUNCATE_DOC_TO_TOKENS:
            d_tokens = d_tokens[:TRUNCATE_DOC_TO_TOKENS]
            doc = ENCODING.decode(d_tokens)
        # 丢弃总长超限
        if len(q_tokens) + len(d_tokens) > DISCARD_IF_EXCEEDS_TOKENS:
            return None
        return (query, doc, score)

    truncated, discarded = 0, 0
    new_pairs: list[QueryDocScore] = []
    for p in pairs:
        result = _truncate_and_filter(p)
        if result is None:
            discarded += 1
        else:
            if result[1] != p[1]:
                truncated += 1
            new_pairs.append(result)

    total = len(pairs)
    trunc_pct = truncated / total * 100 if total else 0
    disc_pct = discarded / total * 100 if total else 0
    kept = len(new_pairs)
    kept_pct = kept / total * 100 if total else 0
    print(f"  Total:     {total}")
    print(f"  Truncated: {truncated} ({trunc_pct:.1f}%)  "
          f"[doc > {TRUNCATE_DOC_TO_TOKENS} tokens, cut to {TRUNCATE_DOC_TO_TOKENS}]")
    print(f"  Discarded: {discarded} ({disc_pct:.1f}%)  "
          f"[query+doc > {DISCARD_IF_EXCEEDS_TOKENS} tokens after truncation]")
    print(f"  Kept:      {kept} ({kept_pct:.1f}%)")
    pairs = new_pairs

    # Step 2: 分桶均衡采样（极端桶保留，中间桶 min/max >= 0.8）
    print("\n" + "=" * 60)
    print(f"Score-balanced sampling (min_ratio={BUCKET_MIN_RATIO}, "
          f"extreme < {EXTREME_BUCKET_LO} or >= {EXTREME_BUCKET_HI})")
    print("=" * 60)

    print("\n--- Web search ---")
    print_bucket_distribution(pairs, "Before sampling")
    sampled = balanced_sample(pairs, rng)
    print_bucket_distribution(sampled, "After sampling")
    print(f"  Sampled: {len(pairs)} -> {len(sampled)}")

    # Step 3: 按 query 分组（query 间随机排列）
    query_groups: dict[str, list[QueryDocScore]] = defaultdict(list)
    for item in sampled:
        query_groups[item[0]].append(item)

    query_keys = list(query_groups.keys())
    rng.shuffle(query_keys)

    ordered_pairs: list[QueryDocScore] = []
    for q in query_keys:
        ordered_pairs.extend(query_groups[q])

    # Step 4: 写出 JSONL
    print(f"\nWriting {len(ordered_pairs)} pairs to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for query, doc, score in ordered_pairs:
            obj: dict[str, object] = {
                "query": query,
                "document": doc,
                "voyage-rerank-2_and_2.5_score": round(score, 6),
            }
            if query in query_to_keywords:
                obj["keywords"] = query_to_keywords[query]
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Step 5: 统计
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)

    print(f"\n  Total pairs: {len(ordered_pairs)}")
    print(f"  Unique queries: {len(query_keys)}")

    # 每个 query 的 doc 数量分布
    docs_per_query = [len(query_groups[q]) for q in query_keys]
    print("\n  Docs per query:")
    print(f"    min:    {min(docs_per_query)}")
    print(f"    max:    {max(docs_per_query)}")
    print(f"    mean:   {statistics.mean(docs_per_query):.2f}")
    print(f"    median: {statistics.median(docs_per_query):.1f}")

    # 最终分数分布
    print_score_stats(ordered_pairs, "Final")
    print_bucket_distribution(ordered_pairs, "Final")

    print(f"\nDone! Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
