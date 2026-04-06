"""Filter out medical-related queries from a JSONL dataset using DeepSeek Chat.

Step 1: Extract unique queries, call DeepSeek to judge if medical-related.
Step 2: Remove all rows with medical queries and save a new JSONL file.

Usage:
    uv run python scripts/filter_medical_queries.py \
        --input /mnt/g/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs.jsonl \
        --workers 16 --batch_size 50
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

from shared.env import load_optional_dotenv

log = logging.getLogger("filter_medical")

SYSTEM_PROMPT = (
    "你是一个query分类助手。给定一批query，判断每个query是否与医学、医疗、"
    "健康、疾病、药物、临床、中医、心理健康等医学相关领域有关。\n"
    "对每个query只输出 yes 或 no，每行一个，顺序与输入一致。\n"
    "yes 表示医学相关，no 表示不相关。不要输出任何其他内容。\n\n"
    "示例输入：\n"
    "1. 高血压的治疗方法有哪些\n"
    "2. Python如何读取CSV文件\n"
    "3. 糖尿病患者的饮食注意事项\n\n"
    "示例输出：\n"
    "yes\n"
    "no\n"
    "yes"
)

MAX_RETRIES = 3


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-5s | %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S", stream=sys.stderr)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)


def _judge_batch(queries: list[str], model: str) -> list[bool]:
    """Judge a batch of queries. Returns list of bools (True = medical)."""
    import litellm

    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(queries))
    user_prompt = (
        f"请判断以下{len(queries)}个query是否与医学相关，"
        f"每行输出yes或no，顺序与输入一致。\n\n{numbered}"
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=len(queries) * 5,
            )
            text = response.choices[0].message.content.strip()
            lines = [
                ln.strip().lstrip("0123456789.、) ").strip().lower()
                for ln in text.splitlines()
                if ln.strip()
            ]

            results: list[bool] = []
            for ln in lines:
                results.append(ln.startswith("yes"))

            # Pad or truncate
            if len(results) < len(queries):
                results.extend([False] * (len(queries) - len(results)))
            elif len(results) > len(queries):
                results = results[: len(queries)]

            return results
        except Exception as exc:
            log.warning(
                "Batch failed (attempt %d/%d): %s: %s",
                attempt + 1,
                MAX_RETRIES,
                type(exc).__name__,
                exc,
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(2**attempt)
                continue
            # Default to not medical on failure
            return [False] * len(queries)

    return [False] * len(queries)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter medical queries from JSONL dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/mnt/g/PrismRerankerV1Data/"
            "step6_kalm_web-search_query_document_pairs.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Default: input with _no_medical suffix.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Cache file for medical labels. Default: input dir / medical_labels.json",
    )
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    load_optional_dotenv()

    if args.output is None:
        args.output = args.input.with_name(
            args.input.stem + "_no_medical" + args.input.suffix
        )
    if args.cache is None:
        args.cache = args.input.parent / "medical_labels_cache.json"

    # --- Step 1: Extract unique queries ---
    log.info("Reading queries from %s ...", args.input)
    unique_queries: list[str] = []
    seen: set[str] = set()
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)["query"]
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
    log.info("Found %d unique queries", len(unique_queries))

    # --- Load cache (already judged queries) ---
    medical_map: dict[str, bool] = {}
    if args.cache.exists():
        medical_map = json.loads(args.cache.read_text(encoding="utf-8"))
        log.info("Loaded %d cached labels from %s", len(medical_map), args.cache)

    # Filter out already cached queries
    todo_queries = [q for q in unique_queries if q not in medical_map]
    log.info("%d queries need classification", len(todo_queries))

    # --- Step 2: Batch classify with multithreading ---
    if todo_queries:
        batches: list[list[str]] = []
        for i in range(0, len(todo_queries), args.batch_size):
            batches.append(todo_queries[i : i + args.batch_size])

        results: dict[int, list[bool]] = {}
        save_interval = 50  # Save cache every N batches
        completed = 0

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_judge_batch, batch, args.model): idx
                for idx, batch in enumerate(batches)
            }
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="Judging"
            ):
                idx = futures[fut]
                batch_results = fut.result()
                results[idx] = batch_results

                # Update medical_map incrementally
                batch_queries = batches[idx]
                for q, is_med in zip(batch_queries, batch_results):
                    medical_map[q] = is_med

                completed += 1
                if completed % save_interval == 0:
                    args.cache.write_text(
                        json.dumps(medical_map, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    log.info("Cache saved (%d/%d batches done)", completed, len(batches))

        # Final cache save
        args.cache.write_text(
            json.dumps(medical_map, ensure_ascii=False), encoding="utf-8"
        )
        log.info("Cache saved to %s", args.cache)

    # --- Statistics ---
    medical_count = sum(1 for q in unique_queries if medical_map.get(q, False))
    non_medical_count = len(unique_queries) - medical_count
    log.info(
        "Medical: %d (%.1f%%), Non-medical: %d (%.1f%%)",
        medical_count,
        medical_count / len(unique_queries) * 100,
        non_medical_count,
        non_medical_count / len(unique_queries) * 100,
    )

    # --- Step 3: Filter and save ---
    log.info("Filtering and writing to %s ...", args.output)
    kept = 0
    removed = 0
    with (
        open(args.input, encoding="utf-8") as fin,
        open(args.output, "w", encoding="utf-8") as fout,
    ):
        for line in tqdm(fin, desc="Filtering", total=149957):
            row = json.loads(line)
            if medical_map.get(row["query"], False):
                removed += 1
            else:
                fout.write(line)
                kept += 1

    log.info("Done. Kept %d rows, removed %d rows.", kept, removed)
    log.info("Output: %s", args.output)


if __name__ == "__main__":
    main()
