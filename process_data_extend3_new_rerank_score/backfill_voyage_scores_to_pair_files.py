"""回写 voyage-rerank-2 / voyage-rerank-2.5 单独得分到 step6-9 文件。

从 step5 源文件构建 (query, document) -> (v2_score, v25_score) 查找表，
然后为 step6-9 各目标文件每一行补上两个分数字段。查不到的行直接删除。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import tiktoken
from tqdm import tqdm

SOURCE_FILE = Path(
    "/mnt/g/PrismRerankerV1Data/"
    "step5_KaLM__all_retrieval_voyage-rerank2_voyage-rerank2.5_"
    "web-search-processed_keywords.jsonl"
)

TARGET_FILES: list[Path] = [
    Path("/mnt/g/PrismRerankerV1Data/step6_kalm_web-search_query_document_pairs.jsonl"),
    Path(
        "/mnt/g/PrismRerankerV1Data/"
        "step6_kalm_web-search_query_document_pairs_balanced.jsonl"
    ),
    Path(
        "/mnt/g/PrismRerankerV1Data/"
        "step6_kalm_web-search_query_document_pairs_no_medical.jsonl"
    ),
    Path(
        "/mnt/g/PrismRerankerV1Data/"
        "step6_kalm_web-search_query_document_pairs_no_medical_length-score-balance.jsonl"
    ),
    Path(
        "/mnt/g/PrismRerankerV1Data/"
        "step7_kalm_web-search_query_document_pairs_annotated.jsonl"
    ),
    Path(
        "/mnt/g/PrismRerankerV1Data/"
        "step8_kalm_web-search_query_document_pairs_annotated_merged.jsonl"
    ),
    Path(
        "/mnt/g/PrismRerankerV1Data/"
        "step9_kalm_web-search_query_document_pairs_contribution_evidence.jsonl"
    ),
]

TRUNCATE_DOC_TO_TOKENS = 4096
ENCODING = tiktoken.get_encoding("cl100k_base")

V2_POS = "voyage-rerank-2_pos_scores"
V2_NEG = "voyage-rerank-2_neg_scores"
V2_WEB = "voyage-rerank-2_web_search_topk_docs_scores"
V25_POS = "voyage-rerank-2.5_pos_scores"
V25_NEG = "voyage-rerank-2.5_neg_scores"
V25_WEB = "voyage-rerank-2.5_web_search_topk_docs_scores"


def truncate_doc(doc: str) -> str:
    toks = ENCODING.encode(doc, disallowed_special=())
    if len(toks) > TRUNCATE_DOC_TO_TOKENS:
        return ENCODING.decode(toks[:TRUNCATE_DOC_TO_TOKENS])
    return doc


def build_lookup(source: Path) -> dict[tuple[str, str], tuple[float, float]]:
    """扫描 step5，构建 (query, truncated_doc) -> (v2, v25) 查找表。"""
    lookup: dict[tuple[str, str], tuple[float, float]] = {}
    with open(source, encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading {source.name}"):
            rec = json.loads(line)
            query: str = rec["query"]
            groups = [
                (rec.get("pos_list", []), rec.get(V2_POS, []), rec.get(V25_POS, [])),
                (rec.get("neg_list", []), rec.get(V2_NEG, []), rec.get(V25_NEG, [])),
                (
                    rec.get("web_search_topk_docs", []),
                    rec.get(V2_WEB, []),
                    rec.get(V25_WEB, []),
                ),
            ]
            for docs, v2_scores, v25_scores in groups:
                for doc, s2, s25 in zip(docs, v2_scores, v25_scores, strict=True):
                    key = (query, truncate_doc(doc))
                    if key not in lookup:
                        lookup[key] = (float(s2), float(s25))
    return lookup


def backfill_file(
    path: Path,
    lookup: dict[tuple[str, str], tuple[float, float]],
) -> tuple[int, int]:
    """为一个目标文件补写分数字段。返回 (kept, dropped)。"""
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
