"""One-time migration: convert wide-format annotated JSONL to long-format.

Wide format (old): one row per query-document, multiple ``*_annotated_label`` columns.
Long format (new): one row per (query, document, model_name), with ``annotated_label``.

Usage:
uv run python process_data/migrate_annotated_to_long.py \
    --input_path /mnt/g/.../annotated.jsonl \
    --save_path /mnt/g/.../annotated_long.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LABEL_SUFFIX = "_annotated_label"


def migrate(input_path: Path, save_path: Path) -> None:
    total_in = 0
    total_out = 0

    with (
        open(input_path, encoding="utf-8") as fin,
        open(save_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total_in += 1

            # Separate base fields from label fields
            base: dict[str, object] = {}
            labels: dict[str, object] = {}
            for k, v in row.items():
                if k.endswith(LABEL_SUFFIX):
                    model_name = k.removesuffix(LABEL_SUFFIX)
                    labels[model_name] = v
                else:
                    base[k] = v

            # Write one row per model
            for model_name, label in labels.items():
                out_row = {**base, "model_name": model_name, "annotated_label": label}
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                total_out += 1

    print(f"Done: {total_in} input rows -> {total_out} output rows", file=sys.stderr)
    print(f"Saved to: {save_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate wide-format annotated JSONL to long-format."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--save_path", type=Path, required=True)
    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"Error: {args.input_path} not found", file=sys.stderr)
        sys.exit(1)

    if args.save_path.exists():
        print(
            f"Error: {args.save_path} already exists, refusing to overwrite",
            file=sys.stderr,
        )
        sys.exit(1)

    migrate(args.input_path, args.save_path)


if __name__ == "__main__":
    main()
