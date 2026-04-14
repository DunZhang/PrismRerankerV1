"""Write Qwen3-Reranker-4B scores back into the original JSONL files.

For every input file listed in ``score_with_qwen3_reranker_4b.INPUT_FILES``:
    - read each row, hash (query, document) the same way the scoring script did
    - look the hash up in the global score table
    - if found: add ``Qwen3-Reranker-4B_score`` and keep the row
    - if missing: drop the row and bump the ``dropped`` counter
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

SCORE_FILE: str = (
    "/mnt/g/PrismRerankerV1Data/data_extend3_new_rerank_score/"
    "qwen3_reranker_4b_scores.jsonl"
)
SCORE_KEY: str = "Qwen3-Reranker-4B_score"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("write_back_scores")


def _load_score_table(path: str) -> dict[str, float]:
    """Stream the score JSONL and return ``{hash: score}``."""
    table: dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            table[obj["hash"]] = obj[SCORE_KEY]
    logger.info("loaded score table: %d entries", len(table))
    return table


def _process_file(path: Path, table: dict[str, float]) -> tuple[int, int, int]:
    """Rewrite one file in place. Returns (total, kept, dropped)."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    total = kept = dropped = 0

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
            score = table.get(h)
            if score is None:
                dropped += 1
                continue
            obj[SCORE_KEY] = score
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    os.replace(tmp_path, path)
    return total, kept, dropped


def main() -> None:
    table = _load_score_table(SCORE_FILE)

    grand_total = grand_kept = grand_dropped = 0
    for fpath in INPUT_FILES:
        path = Path(fpath)
        if not path.exists():
            logger.warning("missing input file, skipping: %s", fpath)
            continue
        total, kept, dropped = _process_file(path, table)
        grand_total += total
        grand_kept += kept
        grand_dropped += dropped
        logger.info(
            "%s: total=%d kept=%d dropped=%d",
            path.name,
            total,
            kept,
            dropped,
        )

    logger.info("=" * 60)
    logger.info(
        "GRAND TOTAL: total=%d kept=%d dropped=%d",
        grand_total,
        grand_kept,
        grand_dropped,
    )


if __name__ == "__main__":
    main()
