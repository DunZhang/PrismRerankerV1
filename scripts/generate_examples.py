# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow>=15.0", "pandas>=2.0", "numpy"]
# ///
"""
从 KaLM-embedding-finetuning-data 的每个子文件夹中读取 parquet 文件，
随机抽取最多 100 条记录，生成 example.jsonl 文件方便观察数据。

用法: uv run scripts/generate_examples.py [--data-dir PATH] [--num-samples N]
"""

import argparse
import json
from pathlib import Path

import random

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _make_serializable(obj):
    """将 ndarray 等不可序列化的类型转为 Python 原生类型。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


def _read_parquet_fallback(parquet_files: list[Path]) -> list[dict]:
    """当 pd.read_parquet 失败时，用 pyarrow iter_batches 逐批读取。"""
    all_rows = []
    for f in parquet_files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=1024):
            batch_dict = batch.to_pydict()
            cols = list(batch_dict.keys())
            for i in range(len(batch_dict[cols[0]])):
                all_rows.append({c: batch_dict[c][i] for c in cols})
    return all_rows


def generate_example(folder: Path, num_samples: int) -> None:
    parquet_files = sorted(folder.glob("*.parquet"))
    if not parquet_files:
        return

    output_path = folder / "example.jsonl"

    try:
        # 常规方式：用 pandas 读取并合并
        dfs = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(dfs, ignore_index=True)
        total = len(df)
        n = min(num_samples, total)
        sampled = df.sample(n=n, random_state=42)
        with open(output_path, "w", encoding="utf-8") as f:
            for _, row in sampled.iterrows():
                record = _make_serializable(row.to_dict())
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Fallback：用 pyarrow iter_batches 处理嵌套类型
        all_rows = _read_parquet_fallback(parquet_files)
        total = len(all_rows)
        n = min(num_samples, total)
        rng = random.Random(42)
        sampled = rng.sample(all_rows, n)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in sampled:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] {folder.name}: {n}/{total} samples -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="为每个子文件夹的 parquet 生成 example.jsonl")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/mnt/g/KaLM-embedding-finetuning-data"),
        help="数据根目录",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="每个文件夹抽取的样本数 (默认 100)",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        print(f"错误: 目录不存在 {data_dir}")
        return

    folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"共发现 {len(folders)} 个子文件夹\n")

    for folder in folders:
        try:
            generate_example(folder, args.num_samples)
        except Exception as e:
            print(f"[FAIL] {folder.name}: {e}")

    print("\n完成!")


if __name__ == "__main__":
    main()
