"""Classify queries into predefined categories using DeepSeek LLM.

Usage:
    uv run python scripts/classify_queries.py \
        --input /mnt/g/PrismRerankerV1Data/step10_sft_queries.json \
        --output /mnt/g/PrismRerankerV1Data/step10_sft_queries_classified.json \
        --batch_size 50 --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from shared.env import DEFAULT_PROJECT_ENV_FILE, load_optional_dotenv

log = logging.getLogger("classify_queries")

CATEGORIES = [
    "交通运输",
    "医学_健康_心理_中医",
    "数学_统计学",
    "时政_政务_行政",
    "消防安全_食品安全",
    "石油化工",
    "计算机_通信",
    "人工智能_机器学习",
    "其他信息服务_信息安全",
    "学科教育_教育",
    "文学_情感",
    "水利_海洋",
    "游戏",
    "科技_科学研究",
    "采矿",
    "住宿_餐饮_酒店",
    "其他制造",
    "影视_娱乐",
    "新闻传媒",
    "汽车",
    "生物医药",
    "航空航天",
    "金融_经济",
    "体育",
    "农林牧渔",
    "房地产_建筑",
    "旅游_地理",
    "法律_司法",
    "电力能源",
    "计算机编程_代码",
    "其他",
]

CATEGORY_LIST_STR = "\n".join(f"{i+1}. {c}" for i, c in enumerate(CATEGORIES))

SYSTEM_PROMPT = f"""你是一个query分类器。给定一个用户query，你需要将其归类到以下类别之一。
只输出类别名称，不要输出任何其他内容。

类别列表：
{CATEGORY_LIST_STR}"""

MAX_RETRIES = 2


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-5s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)


def _classify_batch(queries: list[str], model: str) -> list[str]:
    """Classify a batch of queries in a single LLM call."""
    import litellm

    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(queries))
    user_prompt = (
        f"请对以下{len(queries)}个query逐一分类，每行输出一个类别名称，"
        f"顺序与输入一致，不要输出序号和其他内容。\n\n{numbered}"
    )

    for attempt in range(1 + MAX_RETRIES):
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=len(queries) * 30,
            )
            text = response.choices[0].message.content.strip()
            lines = [ln.strip().lstrip("0123456789.、) ").strip() for ln in text.splitlines() if ln.strip()]

            # Validate and fix labels
            valid = set(CATEGORIES)
            results = []
            for ln in lines:
                results.append(ln if ln in valid else "其他")

            # If line count mismatch, pad or truncate
            if len(results) < len(queries):
                results.extend(["其他"] * (len(queries) - len(results)))
            elif len(results) > len(queries):
                results = results[: len(queries)]

            return results
        except Exception as exc:
            log.warning("Batch call failed (attempt %d): %s: %s", attempt + 1, type(exc).__name__, exc)
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            return ["其他"] * len(queries)

    return ["其他"] * len(queries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify queries into categories.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/mnt/g/PrismRerankerV1Data/step10_sft_queries.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/mnt/g/PrismRerankerV1Data/step10_sft_queries_classified.json"),
    )
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    load_optional_dotenv()

    queries: list[str] = json.loads(args.input.read_text(encoding="utf-8"))
    log.info("Loaded %d queries from %s", len(queries), args.input)

    # Split into batches
    batches: list[list[str]] = []
    for i in range(0, len(queries), args.batch_size):
        batches.append(queries[i : i + args.batch_size])

    # Classify in parallel
    results: dict[int, list[str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_classify_batch, batch, args.model): idx
            for idx, batch in enumerate(batches)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Classifying"):
            idx = futures[fut]
            results[idx] = fut.result()

    # Flatten results in order
    all_labels: list[str] = []
    for idx in range(len(batches)):
        all_labels.extend(results[idx])

    # Build output
    classified = [{"query": q, "category": label} for q, label in zip(queries, all_labels)]

    args.output.write_text(
        json.dumps(classified, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Saved classified queries to %s", args.output)

    # Print statistics
    from collections import Counter

    counter = Counter(all_labels)
    print("\n===== 分类统计 =====")
    print(f"{'类别':<30s} {'数量':>6s} {'占比':>8s}")
    print("-" * 48)
    total = len(all_labels)
    for cat, cnt in counter.most_common():
        print(f"{cat:<30s} {cnt:>6d} {cnt/total*100:>7.1f}%")
    print("-" * 48)
    print(f"{'总计':<30s} {total:>6d} {100.0:>7.1f}%")


if __name__ == "__main__":
    main()
