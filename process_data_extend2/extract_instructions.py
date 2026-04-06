"""从 Infinity-Instruct 7M 提取 instruction 数据并均衡采样。

用法:
    uv run python process_data_extend2/extract_instructions.py --analyze   # 只看原始分布
    uv run python process_data_extend2/extract_instructions.py --extract   # 均衡采样并保存
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 配置
# ============================================================
DATA_DIR = Path("/mnt/g/Infinity-Instruct/7M")
OUTPUT_PATH = Path(
    "/mnt/g/PrismRerankerV1Data/data_extend2/instructions_balanced.jsonl"
)
NUM_PARQUETS = 75
ZH_EN_RATIO = (7, 3)  # zh : en
SEED = 42
FLOAT_RATIO = 0.1  # 类别间允许 ±10% 浮动
EXTREME_PERCENTILE = 10  # 低于 P10 的类别视为极端，不参与均衡目标计算

# 黑名单类别，直接丢弃
CATE_BLACKLIST: set[str] = {
    "数学能力",
}


# ============================================================
# 数据加载
# ============================================================
def load_all_data() -> list[dict]:
    """从所有 parquet 文件提取 instruction / lang / cate。

    cate 取原始 label["cate_ability_zh"] 的最后一个元素，
    黑名单类别直接跳过。
    """
    records: list[dict] = []
    for i in range(NUM_PARQUETS):
        path = DATA_DIR / f"train-{i:05d}-of-{NUM_PARQUETS:05d}.parquet"
        print(f"Loading {path.name} ...")
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            conversations = row["conversations"]
            if conversations is None or len(conversations) == 0:
                continue
            instruction = conversations[0].get("value", "")
            if not instruction:
                continue

            lang = row.get("langdetect", "")
            if lang not in ("en", "zh-cn"):
                continue

            label = row.get("label", {})
            raw_cates = label.get("cate_ability_zh", [])
            if raw_cates is None or len(raw_cates) == 0:
                continue

            cate = str(list(raw_cates)[-1])
            if not cate or cate in CATE_BLACKLIST:
                continue

            records.append({
                "instruction": instruction,
                "lang": lang,
                "cate": cate,
            })
    print(f"Total valid records: {len(records)}")
    return records


# ============================================================
# 统计分析
# ============================================================
def analyze(records: list[dict]) -> None:
    """打印语言分布和每个语言下的类别分布。"""
    lang_counter: Counter[str] = Counter()
    cate_counter: Counter[str] = Counter()
    cate_per_lang: dict[str, Counter[str]] = defaultdict(Counter)

    for r in records:
        lang_counter[r["lang"]] += 1
        cate_counter[r["cate"]] += 1
        cate_per_lang[r["lang"]][r["cate"]] += 1

    print("\n=== 语言分布 ===")
    for lang, cnt in lang_counter.most_common():
        print(f"  {lang}: {cnt}")

    print(f"\n=== 类别分布 ({len(cate_counter)} 类) ===")
    for cate, cnt in cate_counter.most_common():
        print(f"  {cate}: {cnt}")

    print("\n=== 各语言的类别分布 ===")
    for lang in sorted(cate_per_lang):
        print(f"\n  [{lang}]")
        for cate, cnt in cate_per_lang[lang].most_common():
            print(f"    {cate}: {cnt}")


# ============================================================
# 均衡采样
# ============================================================
def balanced_sample(records: list[dict]) -> list[dict]:
    """按 zh:en 比例和类别均衡采样。

    步骤:
    1. 按语言分组，语言内按 cate 分桶
    2. 极端稀有类别（低于 P10）不参与目标计算，全取
    3. 正常类别：以中位数为目标，每类取 [目标*0.9, 目标*1.1] 条
    4. 两个语言各自均衡后，按 zh:en = 7:3 等比例截断
    5. 全局 shuffle 后返回
    """
    rng = random.Random(SEED)

    # 按语言分组
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_lang[r["lang"]].append(r)

    # 每个语言组内按 cate 分桶做类别均衡
    balanced_by_lang: dict[str, list[dict]] = {}
    for lang in ("zh-cn", "en"):
        lang_records = by_lang.get(lang, [])
        if not lang_records:
            continue

        # 分桶并打乱
        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in lang_records:
            buckets[r["cate"]].append(r)
        for v in buckets.values():
            rng.shuffle(v)

        # 区分极端类别和正常类别
        sizes = sorted(len(v) for v in buckets.values())
        p_threshold = float(np.percentile(sizes, EXTREME_PERCENTILE))

        extreme_cates: set[str] = set()
        normal_cates: set[str] = set()
        for cate, items in buckets.items():
            if len(items) <= p_threshold:
                extreme_cates.add(cate)
            else:
                normal_cates.add(cate)

        # 正常类别的中位数作为目标，±FLOAT_RATIO 浮动
        normal_sizes = [len(buckets[c]) for c in normal_cates]
        target_per_cate = int(np.median(normal_sizes))
        lower = int(target_per_cate * (1 - FLOAT_RATIO))
        upper = int(target_per_cate * (1 + FLOAT_RATIO))

        print(f"\n  [{lang}] 桶数: {len(buckets)}, "
              f"极端: {len(extreme_cates)}, 正常: {len(normal_cates)}")
        print(f"  目标/类: {target_per_cate} (范围 {lower}~{upper})")
        if extreme_cates:
            print(f"  极端类别(全取): "
                  f"{', '.join(f'{c}({len(buckets[c])})' for c in extreme_cates)}")

        # 采样
        sampled: list[dict] = []
        for cate, items in buckets.items():
            if cate in extreme_cates:
                sampled.extend(items)
            else:
                take = min(len(items), upper)
                take = max(take, min(lower, len(items)))
                sampled.extend(items[:take])

        balanced_by_lang[lang] = sampled
        print(f"  均衡后总量: {len(sampled)}")

    # 按 zh:en = 7:3 调整数量
    n_zh_bal = len(balanced_by_lang.get("zh-cn", []))
    n_en_bal = len(balanced_by_lang.get("en", []))

    target_zh = n_zh_bal
    target_en = int(target_zh * ZH_EN_RATIO[1] / ZH_EN_RATIO[0])
    if target_en > n_en_bal:
        target_en = n_en_bal
        target_zh = int(target_en * ZH_EN_RATIO[0] / ZH_EN_RATIO[1])

    print(f"\n=== 采样目标 (7:3) ===")
    print(f"  zh-cn: {target_zh} (均衡后 {n_zh_bal})")
    print(f"  en: {target_en} (均衡后 {n_en_bal})")
    print(f"  总计: {target_zh + target_en}")
    print(f"  实际比例: {target_zh / (target_zh + target_en) * 100:.1f}%"
          f" : {target_en / (target_zh + target_en) * 100:.1f}%")

    # 截取时按桶等比例缩减，保持类别均衡
    result: list[dict] = []
    for lang, target_n in [("zh-cn", target_zh), ("en", target_en)]:
        pool = balanced_by_lang.get(lang, [])
        if len(pool) <= target_n:
            sampled = pool
        else:
            sub_buckets: dict[str, list[dict]] = defaultdict(list)
            for r in pool:
                sub_buckets[r["cate"]].append(r)
            ratio = target_n / len(pool)
            sampled = []
            remainder = 0.0
            for items in sub_buckets.values():
                exact = len(items) * ratio + remainder
                take = int(exact)
                remainder = exact - take
                sampled.extend(items[:take])

        result.extend(sampled)

        # 打印每个类别的最终数量
        sampled_buckets: Counter[str] = Counter()
        for r in sampled:
            sampled_buckets[r["cate"]] += 1
        print(f"\n  [{lang}] 最终采样 ({len(sampled)} 条):")
        for cate, cnt in sampled_buckets.most_common():
            print(f"    {cate}: {cnt}")

    rng.shuffle(result)
    return result


# ============================================================
# 保存
# ============================================================
def save_jsonl(records: list[dict], path: Path) -> None:
    """保存为 jsonl 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(records)} records to {path}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="提取并均衡采样 instruction 数据"
    )
    parser.add_argument(
        "--analyze", action="store_true", help="只做统计分析"
    )
    parser.add_argument(
        "--extract", action="store_true", help="提取并均衡采样"
    )
    args = parser.parse_args()

    if not args.analyze and not args.extract:
        parser.print_help()
        return

    records = load_all_data()

    if args.analyze:
        analyze(records)

    if args.extract:
        sampled = balanced_sample(records)
        analyze(sampled)
        save_jsonl(sampled, OUTPUT_PATH)


if __name__ == "__main__":
    main()
