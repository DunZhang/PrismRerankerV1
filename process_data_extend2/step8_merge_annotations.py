"""Merge long-format annotations into one row per (query, document).

Input (long): each line has ``model_name`` and ``annotated_label``, so the same
query-document pair may appear multiple times (once per model).

Output (wide): one row per unique (query, document), with each model's label as
``{model_name}_annotated_label`` column.  Base fields (query, document, scores…)
are preserved as-is.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

# ============ 修改这里 ============
INPUT_PATH = Path("/mnt/g/PrismRerankerV1Data/data_extend2/step7_expanded2_web-search_query_document_pairs_annotated.jsonl")
SAVE_PATH = Path("/mnt/g/PrismRerankerV1Data/data_extend2/step8_expanded2_web-search_query_document_pairs_annotated_merged.jsonl")
# =================================

MODELS = [
    "deepseek-chat_annotated_label",
    "google/gemini-3-flash-preview_annotated_label",
    "openai/gpt-5.4-mini_annotated_label",
    "qwen3.5-397b-a17b_annotated_label",
    "anthropic/claude-haiku-4.5_annotated_label",
]
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

    # Write merged output – keep rows where yes or no votes > 2
    total_out = 0
    total_yes = 0
    total_no = 0
    skipped = 0
    required_models = {col.removesuffix("_annotated_label") for col in MODELS}
    with open(save_path, "w", encoding="utf-8") as f:
        for base, labels in rows.values():
            # Count yes/no votes from available models
            voted = {
                mn: labels[mn]
                for mn in required_models
                if labels.get(mn) is not None
            }
            yes_count = sum(1 for v in voted.values() if v == "yes")
            no_count = sum(1 for v in voted.values() if v == "no")
            # Skip if neither yes nor no has majority (> 2)
            if yes_count <= 2 and no_count <= 2:
                skipped += 1
                continue
            out = dict(base)
            out.pop("annotated_score", None)
            score = out.get("voyage-rerank-2_and_2.5_score")
            if score is not None:
                out["revised_score"] = score ** 1.609
            for mn in required_models:
                out[f"{mn}_annotated_label"] = labels.get(mn)
            final_label = "yes" if yes_count > no_count else "no"
            out["annotated_label"] = final_label
            if final_label == "yes":
                total_yes += 1
            else:
                total_no += 1
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            total_out += 1

    print(f"Models found: {model_names}", file=sys.stderr)
    print(f"Required models: {sorted(required_models)}", file=sys.stderr)
    print(f"Skipped (no majority): {skipped}", file=sys.stderr)
    print(f"Label distribution: yes={total_yes}, no={total_no}", file=sys.stderr)
    print(
        f"Done: {total_in} input rows -> {total_out} merged rows",
        file=sys.stderr,
    )
    print(f"Saved to: {save_path}", file=sys.stderr)


if __name__ == "__main__":
    if not INPUT_PATH.exists():
        print(f"Error: {INPUT_PATH} not found", file=sys.stderr)
        sys.exit(1)

    merge(INPUT_PATH, SAVE_PATH)
