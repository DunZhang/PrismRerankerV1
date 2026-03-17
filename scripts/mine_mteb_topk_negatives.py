"""
Mine hard negatives from MTEB retrieval datasets.

For each dataset, loads corpus/queries/qrels from HuggingFace, encodes all
texts with static-similarity-mrl-multilingual-v1, retrieves top-101 docs per
query, removes positives, and writes:
  {query: str, pos_list: list[str], neg_list: list[str]}

One output JSONL file per dataset.

Usage:
    export HF_ENDPOINT=https://hf-mirror.com
    uv run python scripts/mine_mteb_topk_negatives.py \
        --output-dir /mnt/g/PrismRerankerV1Data/mteb_benchmark
"""

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from sentence_transformers import SentenceTransformer

MTEB_DATASETS: list[str] = [
    "mteb/arguana",
    "mteb/cqadupstack-gaming",
    "mteb/cqadupstack-unix",
    "mteb/ClimateFEVER_test_top_250_only_w_correct-v2",
    "mteb/FEVER_test_top_250_only_w_correct-v2",
    "mteb/fiqa",
    "mteb/HotpotQA_test_top_250_only_w_correct-v2",
    "mteb/scidocs",
    "mteb/trec-covid",
    "mteb/webis-touche2020-v3",
    "mteb/T2Retrieval",
    "mteb/MMarcoRetrieval",
    "mteb/DuRetrieval",
    "mteb/CovidRetrieval",
    "mteb/CmedqaRetrieval",
    "mteb/EcomRetrieval",
    "mteb/MedicalRetrieval",
    "mteb/VideoRetrieval",
]


def encode_texts(
    model: SentenceTransformer, texts: list[str], batch_size: int
) -> np.ndarray:
    """Encode texts, returning L2-normalized embeddings."""
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def get_split(ds: DatasetDict) -> Dataset:
    """Get 'test' split if available, otherwise the first split."""
    if "test" in ds:
        return ds["test"]
    return next(iter(ds.values()))


def build_corpus_texts(
    corpus_ds: Dataset,
) -> tuple[list[str], list[str]]:
    """Extract corpus IDs and texts, prepending title if present."""
    corpus_ids = [str(x) for x in corpus_ds["_id"]]
    has_title = "title" in corpus_ds.column_names
    corpus_texts: list[str] = []
    if has_title:
        titles = corpus_ds["title"]
        texts = corpus_ds["text"]
        for title, text in zip(titles, texts):
            title = (title or "").strip()
            text = text or ""
            if title:
                corpus_texts.append(f"{title} {text}")
            else:
                corpus_texts.append(text)
    else:
        corpus_texts = [t or "" for t in corpus_ds["text"]]
    return corpus_ids, corpus_texts


def build_qrels_dict(qrels_ds: Dataset) -> dict[str, list[str]]:
    """Build {query_id -> [positive corpus_ids]} from qrels (score > 0)."""
    qrels_dict: dict[str, list[str]] = {}
    for row in qrels_ds:
        if row["score"] > 0:
            qid = str(row["query-id"])
            cid = str(row["corpus-id"])
            qrels_dict.setdefault(qid, []).append(cid)
    return qrels_dict


def process_dataset(
    dataset_name: str,
    model: SentenceTransformer,
    batch_size: int,
    top_k: int,
    output_path: Path,
) -> int:
    """Process one MTEB dataset and write JSONL. Returns record count."""
    print(f"  Loading corpus for {dataset_name}...")
    corpus_ds = get_split(load_dataset(dataset_name, "corpus"))
    corpus_ids, corpus_texts = build_corpus_texts(corpus_ds)
    corpus_id_to_idx: dict[str, int] = {cid: i for i, cid in enumerate(corpus_ids)}

    print(f"  Loading queries for {dataset_name}...")
    queries_ds = get_split(load_dataset(dataset_name, "queries"))
    query_ids = [str(x) for x in queries_ds["_id"]]
    query_texts: list[str] = queries_ds["text"]

    print(f"  Loading qrels for {dataset_name}...")
    qrels_ds = get_split(load_dataset(dataset_name, "default"))
    qrels_dict = build_qrels_dict(qrels_ds)

    # Filter to queries with at least one positive
    valid_indices = [i for i, qid in enumerate(query_ids) if qid in qrels_dict]
    if not valid_indices:
        print(f"  {dataset_name}: no queries with positives, skipping")
        return 0

    filtered_query_ids = [query_ids[i] for i in valid_indices]
    filtered_query_texts = [query_texts[i] for i in valid_indices]

    print(f"  {dataset_name}: encoding {len(corpus_texts)} corpus docs...")
    corpus_embeddings = encode_texts(model, corpus_texts, batch_size)

    print(f"  {dataset_name}: encoding {len(filtered_query_texts)} queries...")
    query_embeddings = encode_texts(model, filtered_query_texts, batch_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as fw:
        for qi, (qid, query_text) in enumerate(
            zip(filtered_query_ids, filtered_query_texts)
        ):
            pos_ids = qrels_dict[qid]
            pos_set = set(pos_ids)

            # Collect positive texts (only those in corpus)
            pos_texts = []
            for pid in pos_ids:
                idx = corpus_id_to_idx.get(pid)
                if idx is not None:
                    pos_texts.append(corpus_texts[idx])
            if not pos_texts:
                continue

            # Cosine sim via dot product (normalized embeddings)
            q_emb = query_embeddings[qi]
            sims = corpus_embeddings @ q_emb
            sorted_idxs = np.argsort(-sims)[:top_k]

            # Remove positives from top-K results
            neg_texts = [
                corpus_texts[idx]
                for idx in sorted_idxs
                if corpus_ids[idx] not in pos_set
            ]

            fw.write(
                json.dumps(
                    {
                        "query": query_text,
                        "pos_list": pos_texts,
                        "neg_list": neg_texts,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1

    return count


def make_output_filename(dataset_name: str, top_k: int) -> str:
    """Convert 'mteb/arguana' to 'mteb__arguana_top101.jsonl'."""
    safe_name = dataset_name.replace("/", "__")
    return f"{safe_name}_top{top_k}.jsonl"


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Mine MTEB hard-negative JSONL files")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Embedding batch size",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/static-similarity-mrl-multilingual-v1",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=101,
        help="Number of top docs to retrieve per query",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=MTEB_DATASETS,
        help="MTEB dataset names to process",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)

    total_records = 0
    for di, dataset_name in enumerate(args.datasets, 1):
        filename = make_output_filename(dataset_name, args.top_k)
        output_path = output_dir / filename

        if output_path.exists():
            print(
                f"  ({di}/{len(args.datasets)}) {dataset_name}: "
                "already exists, skipping"
            )
            continue

        print(f"\n  ({di}/{len(args.datasets)}) Processing {dataset_name}...")
        n = process_dataset(
            dataset_name=dataset_name,
            model=model,
            batch_size=args.batch_size,
            top_k=args.top_k,
            output_path=output_path,
        )
        print(f"  {dataset_name}: wrote {n} records -> {output_path.name}")
        total_records += n

    print(f"\nDone. Total records written: {total_records}")


if __name__ == "__main__":
    main()
