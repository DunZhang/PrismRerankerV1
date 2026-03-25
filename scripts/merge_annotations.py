"""Merge long-format annotations into one row per (query, document).

Input (long): each line has ``model_name`` and ``annotated_label``, so the same
query-document pair may appear multiple times (once per model).

Output (wide): one row per unique (query, document), with each model's label as
``{model_name}_annotated_label`` column.  Base fields (query, document, scores…)
are preserved as-is.

Usage:
uv run python scripts/merge_annotations.py \
    --input_path /mnt/g/.../annotated.jsonl \
    --save_path /mnt/g/.../annotated_merged.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


def _pair_hash(query: str, document: str) -> str:
    return hashlib.sha256(f"{query}\n{document}".encode()).hexdigest()


def merge(input_path: Path, save_path: Path) -> None:
    # hash -> (base_fields, {model_name: label})
    rows: OrderedDict[str, tuple[dict[str, Any], dict[str, Any]]] = OrderedDict()
    model_names: list[str] = []  # preserve discovery order
    seen_models: set[str] = set()

    total_in = 0
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_in += 1

            model_name = row.pop("model_name", None)
            label = row.pop("annotated_label", None)

            h = _pair_hash(row["query"], row["document"])

            if h not in rows:
                rows[h] = (row, {})
            else:
                # Merge any new base fields (shouldn't differ, but be safe)
                base, _ = rows[h]
                for k, v in row.items():
                    if k not in base:
                        base[k] = v

            if model_name is not None:
                rows[h][1][model_name] = label
                if model_name not in seen_models:
                    seen_models.add(model_name)
                    model_names.append(model_name)

    # Write merged output
    total_out = 0
    with open(save_path, "w", encoding="utf-8") as f:
        for base, labels in rows.values():
            out = dict(base)
            for mn in model_names:
                out[f"{mn}_annotated_label"] = labels.get(mn)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            total_out += 1

    print(f"Models found: {model_names}", file=sys.stderr)
    print(
        f"Done: {total_in} input rows -> {total_out} merged rows",
        file=sys.stderr,
    )
    print(f"Saved to: {save_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge long-format annotations into one row per (query, document)."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"Error: {args.input_path} not found", file=sys.stderr)
        sys.exit(1)

    merge(args.input_path, args.save_path)


if __name__ == "__main__":
    main()
