"""
Convert PosIR-Benchmark-v1 datasets into hard-negative JSONL files.

For each (language, dataset) pair, reads corpus/queries/qrels from parquet,
encodes all texts with static-similarity-mrl-multilingual-v1, and writes:
  {query: str, pos_list: list[str], neg_list: list[str]}
where neg_list contains top-K corpus docs (by cosine similarity) after removing positives,
in descending similarity order.

Output filename: {output_dir}/{lang}__{dataset}_top{k}.jsonl

Usage:
    export HF_ENDPOINT=https://hf-mirror.com
    uv run python scripts/mine_topk_negatives.py \
        --benchmark-dir /mnt/g/PosIR-Benchmark-v1 \
        --output-dir /mnt/g/PrismRerankerV1Data/posir_benchmark
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    """Encode texts, returning L2-normalized embeddings."""
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def process_dataset(
    dataset_dir: Path,
    lang: str,
    dataset_name: str,
    model: SentenceTransformer,
    batch_size: int,
    top_k: int,
    output_path: Path,
) -> int:
    """Process one dataset directory and write results to output_path. Returns record count."""
    corpus_df = pl.read_parquet(dataset_dir / "corpus.parquet")
    queries_df = pl.read_parquet(dataset_dir / "queries.parquet")
    qrels_df = pl.read_parquet(dataset_dir / "qrels" / "test.parquet")

    corpus_ids: list[str] = corpus_df["_id"].to_list()
    corpus_texts: list[str] = corpus_df["text"].to_list()
    corpus_id_to_idx: dict[str, int] = {cid: i for i, cid in enumerate(corpus_ids)}

    # Build {query_id -> [positive corpus_ids]} from qrels
    qrels_pos = (
        qrels_df
        .filter(pl.col("score") > 0)
        .group_by("query-id")
        .agg(pl.col("corpus-id").alias("pos_corpus_ids"))
    )
    qrels_dict: dict[str, list[str]] = {
        row["query-id"]: row["pos_corpus_ids"]
        for row in qrels_pos.iter_rows(named=True)
    }

    query_ids: list[str] = queries_df["_id"].to_list()
    query_texts: list[str] = queries_df["text"].to_list()

    # Filter to queries that have at least one positive
    valid_indices = [i for i, qid in enumerate(query_ids) if qid in qrels_dict]
    if not valid_indices:
        print(f"  [{lang}] {dataset_name}: no queries with positives, skipping")
        return 0

    print(f"  [{lang}] {dataset_name}: encoding {len(corpus_texts)} corpus docs...")
    corpus_embeddings = encode_texts(model, corpus_texts, batch_size)

    filtered_query_texts = [query_texts[i] for i in valid_indices]
    filtered_query_ids = [query_ids[i] for i in valid_indices]

    print(f"  [{lang}] {dataset_name}: encoding {len(filtered_query_texts)} queries...")
    query_embeddings = encode_texts(model, filtered_query_texts, batch_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as fw:
        for qi, (qid, query_text) in enumerate(zip(filtered_query_ids, filtered_query_texts)):
            pos_ids = qrels_dict[qid]
            pos_set = set(pos_ids)

            # Collect positive texts (only those present in corpus)
            pos_texts = []
            for pid in pos_ids:
                idx = corpus_id_to_idx.get(pid)
                if idx is not None:
                    pos_texts.append(corpus_texts[idx])
            if not pos_texts:
                continue

            # Compute cosine similarities (both sides are normalized → dot product)
            q_emb = query_embeddings[qi]  # (D,)
            sims = corpus_embeddings @ q_emb  # (N,)
            sorted_idxs = np.argsort(-sims)  # descending

            # Collect top_k negatives in ranked order
            neg_texts = []
            for idx in sorted_idxs:
                if corpus_ids[idx] not in pos_set:
                    neg_texts.append(corpus_texts[idx])
                if len(neg_texts) >= top_k:
                    break

            fw.write(json.dumps({"query": query_text, "pos_list": pos_texts, "neg_list": neg_texts}, ensure_ascii=False) + "\n")
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare PosIR-Benchmark hard-negative JSONL files")
    parser.add_argument("--benchmark-dir", default="/mnt/g/PosIR-Benchmark-v1", help="Root benchmark directory")
    parser.add_argument("--output-dir", default="/mnt/g/PrismRerankerV1Data/posir_benchmark", help="Output directory")
    parser.add_argument("--languages", nargs="+", default=["cmn-Hans", "eng-Latn"], help="Language codes to process")
    parser.add_argument("--batch-size", type=int, default=512, help="Embedding batch size")
    parser.add_argument("--model-name", default="sentence-transformers/static-similarity-mrl-multilingual-v1", help="SentenceTransformer model")
    parser.add_argument("--top-k", type=int, default=100, help="Number of hard negatives to retrieve")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    total_records = 0
    for lang in args.languages:
        lang_dir = benchmark_dir / lang
        if not lang_dir.exists():
            print(f"Language directory not found: {lang_dir}, skipping")
            continue

        datasets = sorted([d.name for d in lang_dir.iterdir() if d.is_dir()])
        print(f"\n[{lang}] Found {len(datasets)} datasets")

        for di, dataset_name in enumerate(datasets, 1):
            dataset_dir = lang_dir / dataset_name
            output_path = output_dir / f"{lang}__{dataset_name}_top{args.top_k}.jsonl"

            if output_path.exists():
                print(f"  [{lang}] ({di}/{len(datasets)}) {dataset_name}: already exists, skipping")
                continue

            print(f"  [{lang}] ({di}/{len(datasets)}) {dataset_name}...")
            n = process_dataset(
                dataset_dir=dataset_dir,
                lang=lang,
                dataset_name=dataset_name,
                model=model,
                batch_size=args.batch_size,
                top_k=args.top_k,
                output_path=output_path,
            )
            print(f"  [{lang}] {dataset_name}: wrote {n} records → {output_path.name}")
            total_records += n

    print(f"\nDone. Total records written: {total_records}")


if __name__ == "__main__":
    main()
