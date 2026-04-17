"""Write reranker scores back into the original JSONL files.

Supported score sources:
    - cohere rerank-v4.0-fast

For every input file listed in ``score_with_qwen3_reranker_4b.INPUT_FILES``:
    - read each row, hash (query, document) the same way the scoring script did
    - look the hash up in the cohere score table
    - if found: add the cohere score field to the row
    - if not found: keep the row as-is (no field added)
    - write to a sibling ``*.tmp`` then atomically replace the original
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from process_data_extend3_new_rerank_score.score_with_qwen3_reranker_4b import (
    INPUT_FILES,
    _pair_hash,
)

SCORE_SOURCES: list[dict[str, str]] = [
    {
        "file": (
            "/mnt/g/PrismRerankerV1Data/data_extend3_new_rerank_score/"
            "cohere_rerank_v4_fast_scores.jsonl"
        ),
        "key": "cohere_rerank_4_fast",
    },
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("write_back_scores")


def _load_score_table(path: str, key: str) -> dict[str, float]:
    """Stream the score JSONL and return ``{hash: score}``."""
    table: dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            table[obj["hash"]] = obj[key]
    logger.info("loaded score table %s: %d entries", key, len(table))
    return table


def _process_file(
    path: Path, tables: list[tuple[str, dict[str, float]]]
) -> tuple[int, int, int]:
    """Rewrite one file in place. Returns (total, enriched, missed)."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    total = enriched = missed = 0

    with (
        open(path, encoding="utf-8") as fin,
        open(tmp_path, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            h = _pair_hash(obj["query"], obj["document"])
            row_enriched = False
            for key, table in tables:
                score = table.get(h)
                if score is not None:
                    obj[key] = score
                    row_enriched = True
            if row_enriched:
                enriched += 1
            else:
                missed += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    os.replace(tmp_path, path)
    return total, enriched, missed


def main() -> None:
    tables: list[tuple[str, dict[str, float]]] = []
    for src in SCORE_SOURCES:
        table = _load_score_table(src["file"], src["key"])
        tables.append((src["key"], table))

    grand_total = grand_enriched = grand_missed = 0
    for fpath in INPUT_FILES:
        path = Path(fpath)
        if not path.exists():
            logger.warning("missing input file, skipping: %s", fpath)
            continue
        total, enriched, missed = _process_file(path, tables)
        grand_total += total
        grand_enriched += enriched
        grand_missed += missed
        logger.info(
            "%s: total=%d enriched=%d missed=%d",
            path.name,
            total,
            enriched,
            missed,
        )

    logger.info("=" * 60)
    logger.info(
        "GRAND TOTAL: total=%d enriched=%d missed=%d",
        grand_total,
        grand_enriched,
        grand_missed,
    )


if __name__ == "__main__":
    main()
