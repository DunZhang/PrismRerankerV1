"""回写 voyage-rerank-2 / voyage-rerank-2.5 单独得分到 data_extend2 的 step6-9 文件。

source: step5_expanded2_web_search-processed-Rerank2.5-Rerank2_keywords.jsonl
只有 web_search_topk_docs 一组，截断阈值 7168 tokens（与 extend2 step6 一致）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import tiktoken
from tqdm import tqdm

DATA_DIR = Path("/mnt/g/PrismRerankerV1Data/data_extend2")

SOURCE_FILE = (
    DATA_DIR / "step5_expanded2_web_search-processed-Rerank2.5-Rerank2_keywords.jsonl"
)

TARGET_FILES: list[Path] = [
    DATA_DIR / "step6_expanded2_web-search_query_document_pairs.jsonl",
    DATA_DIR
    / "step6_expanded2_web-search_query_document_pairs_length-score-balance.jsonl",
    DATA_DIR / "step7_expanded2_web-search_query_document_pairs_annotated.jsonl",
    DATA_DIR / "step8_expanded2_web-search_query_document_pairs_annotated_merged.jsonl",
    DATA_DIR / "step9_expanded2_web-search_query_document_contribution_evidence.jsonl",
]

TRUNCATE_DOC_TO_TOKENS = 7168
ENCODING = tiktoken.get_encoding("cl100k_base")

DOCS_KEY = "web_search_topk_docs"
V2_KEY = "voyage-rerank-2_web_search_topk_docs_scores"
V25_KEY = "voyage-rerank-2.5_web_search_topk_docs_scores"


def truncate_doc(doc: str) -> str:
    toks = ENCODING.encode(doc, disallowed_special=())
    if len(toks) > TRUNCATE_DOC_TO_TOKENS:
        return ENCODING.decode(toks[:TRUNCATE_DOC_TO_TOKENS])
    return doc


def build_lookup(source: Path) -> dict[tuple[str, str], tuple[float, float]]:
    lookup: dict[tuple[str, str], tuple[float, float]] = {}
    with open(source, encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading {source.name}"):
            rec = json.loads(line)
            query: str = rec["query"]
            docs = rec.get(DOCS_KEY, [])
            v2s = rec.get(V2_KEY, [])
            v25s = rec.get(V25_KEY, [])
            for doc, s2, s25 in zip(docs, v2s, v25s, strict=True):
                key = (query, truncate_doc(doc))
                if key not in lookup:
                    lookup[key] = (float(s2), float(s25))
    return lookup


def backfill_file(
    path: Path,
    lookup: dict[tuple[str, str], tuple[float, float]],
) -> tuple[int, int]:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    kept = 0
    dropped = 0
    with (
        open(path, encoding="utf-8") as fin,
        open(tmp_path, "w", encoding="utf-8") as fout,
    ):
        for line in tqdm(fin, desc=f"Backfill {path.name}"):
            rec = json.loads(line)
            key = (rec["query"], rec["document"])
            scores = lookup.get(key)
            if scores is None:
                dropped += 1
                continue
            rec["voyage-rerank-2_score"] = scores[0]
            rec["voyage-rerank-2.5_score"] = scores[1]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
    os.replace(tmp_path, path)
    return kept, dropped


def main() -> None:
    print(f"Building lookup from: {SOURCE_FILE}")
    lookup = build_lookup(SOURCE_FILE)
    print(f"Lookup size: {len(lookup):,} (query, document) pairs\n")

    for target in TARGET_FILES:
        if not target.exists():
            print(f"[SKIP] not found: {target}")
            continue
        kept, dropped = backfill_file(target, lookup)
        print(f"  {target.name}: kept={kept:,}, dropped={dropped:,}")


if __name__ == "__main__":
    main()
