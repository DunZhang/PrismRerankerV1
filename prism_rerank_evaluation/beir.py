"""Evaluate embedding models on BEIR retrieval benchmarks.

Loads BEIR datasets from HuggingFace, encodes corpus & queries with a
SentenceTransformer model, computes NDCG@10, and saves Top-K results
as JSONL for downstream reranker evaluation.

Usage:
    uv run python -m prism_rerank_evaluation beir --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rank_evaluate.metrics import dcg_at_k, mean_score

# Sorted by corpus size ascending for faster iteration on small datasets first
BEIR_DATASETS: list[str] = [
    "mteb/nfcorpus",  # ~3.6K
    "mteb/scifact",  # ~5K
    "mteb/scidocs",  # ~25K
    "mteb/fiqa",  # ~57K
    "mteb/trec-covid",  # ~171K
    "mteb/touche2020",  # ~382K
    "mteb/dbpedia",  # ~4.6M
    "mteb/nq",  # ~2.7M
    "mteb/hotpotqa",  # ~5.2M
]

DEFAULT_MODEL = "/mnt/d/PublicModels/jina-embeddings-v3"
DEFAULT_BATCH_SIZE = 128
DEFAULT_TOP_K = 100
DEFAULT_QUERY_CHUNK = 1000
DEFAULT_CORPUS_CHUNK = 100_000


# ---------------------------------------------------------------------------
# Data loading helpers (adapted from scripts/mine_mteb_topk_negatives.py)
# ---------------------------------------------------------------------------


def get_split(ds: DatasetDict | Dataset) -> Dataset:
    """Get 'test' split if available, otherwise the first split."""
    if isinstance(ds, Dataset):
        return ds
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


def build_qrels_dict(
    qrels_ds: Dataset,
) -> dict[str, dict[str, int]]:
    """Build {query_id -> {corpus_id: relevance_score}} from qrels.

    Preserves graded relevance (e.g. TREC-COVID uses 0/1/2) for
    accurate NDCG computation. Only entries with score > 0 are kept.
    """
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        score = int(row["score"])
        if score > 0:
            qid = str(row["query-id"])
            cid = str(row["corpus-id"])
            qrels.setdefault(qid, {})[cid] = score
    return qrels


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _encode_length_sorted(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    task: str,
) -> np.ndarray:
    """Encode texts sorted by length for better GPU utilization.

    Sorts texts by length so that batches contain similarly-sized texts,
    reducing padding waste. Passes task/prompt_name for LoRA adapter selection.
    """
    n = len(texts)
    sorted_indices = sorted(range(n), key=lambda i: len(texts[i]))
    sorted_texts = [texts[i] for i in sorted_indices]

    # Pass both task (LoRA adapter) and prompt_name per official usage
    sorted_embeddings = model.encode(
        sorted_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        task=task,
        prompt_name=task,
    )

    # Restore original order
    embeddings = np.empty_like(sorted_embeddings)
    for new_pos, orig_pos in enumerate(sorted_indices):
        embeddings[orig_pos] = sorted_embeddings[new_pos]
    return embeddings


def encode_corpus(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    """Encode corpus passages with retrieval.passage LoRA adapter."""
    return _encode_length_sorted(model, texts, batch_size, "retrieval.passage")


def encode_queries(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    """Encode queries with retrieval.query LoRA adapter."""
    return _encode_length_sorted(model, texts, batch_size, "retrieval.query")


# ---------------------------------------------------------------------------
# FAISS-based encoding & retrieval (memory-efficient for large corpora)
# ---------------------------------------------------------------------------


def build_faiss_index(
    model: SentenceTransformer,
    corpus_texts: list[str],
    batch_size: int,
    corpus_chunk_size: int = DEFAULT_CORPUS_CHUNK,
) -> faiss.IndexFlatIP:
    """Encode corpus in chunks and build a FAISS exact inner-product index.

    Encodes ``corpus_chunk_size`` documents at a time, adds them to the
    index, then frees the chunk embeddings.  This avoids holding 2x full
    corpus embeddings in memory simultaneously.
    """
    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)

    num_docs = len(corpus_texts)
    for start in tqdm(
        range(0, num_docs, corpus_chunk_size),
        desc="  Encoding corpus chunks",
        unit="chunk",
    ):
        end = min(start + corpus_chunk_size, num_docs)
        chunk_embeddings = _encode_length_sorted(
            model, corpus_texts[start:end], batch_size, "retrieval.passage"
        )
        index.add(np.ascontiguousarray(chunk_embeddings, dtype=np.float32))
        del chunk_embeddings

    return index


def faiss_retrieve_top_k(
    query_embeddings: np.ndarray,
    index: faiss.IndexFlatIP,
    top_k: int,
    query_chunk_size: int = DEFAULT_QUERY_CHUNK,
) -> np.ndarray:
    """Retrieve top-k corpus indices using FAISS exact inner-product search.

    Processes queries in chunks.  FAISS internally tiles the computation
    so the full similarity matrix is never materialised.

    Returns:
        ``(num_queries, top_k)`` int64 array of corpus indices sorted by
        descending similarity per query.
    """
    num_queries = query_embeddings.shape[0]
    actual_k = min(top_k, index.ntotal)
    all_top_indices = np.empty((num_queries, actual_k), dtype=np.int64)

    for start in tqdm(
        range(0, num_queries, query_chunk_size),
        desc="  Retrieving",
        unit="chunk",
    ):
        end = min(start + query_chunk_size, num_queries)
        q_chunk = np.ascontiguousarray(query_embeddings[start:end], dtype=np.float32)
        _scores, indices = index.search(q_chunk, actual_k)
        all_top_indices[start:end] = indices

    return all_top_indices


# ---------------------------------------------------------------------------
# Retrieval (legacy numpy — kept for reference)
# ---------------------------------------------------------------------------


def retrieve_top_k(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    top_k: int,
    query_chunk_size: int = DEFAULT_QUERY_CHUNK,
) -> np.ndarray:
    """Return (num_queries, top_k) array of corpus indices by descending similarity.

    Processes queries in chunks to avoid OOM on large corpora.
    Uses argpartition (O(n)) for initial selection, then sorts the top-k.
    """
    num_queries = query_embeddings.shape[0]
    num_corpus = corpus_embeddings.shape[0]
    actual_k = min(top_k, num_corpus)
    all_top_indices = np.empty((num_queries, actual_k), dtype=np.int64)

    for start in tqdm(
        range(0, num_queries, query_chunk_size),
        desc="  Retrieving",
        unit="chunk",
    ):
        end = min(start + query_chunk_size, num_queries)
        sims = query_embeddings[start:end] @ corpus_embeddings.T

        if actual_k >= num_corpus:
            sorted_indices = np.argsort(-sims, axis=1)[:, :actual_k]
        else:
            top_k_unsorted = np.argpartition(-sims, actual_k, axis=1)[:, :actual_k]
            rows = np.arange(end - start)[:, None]
            top_k_scores = sims[rows, top_k_unsorted]
            sorted_within = np.argsort(-top_k_scores, axis=1)
            sorted_indices = top_k_unsorted[rows, sorted_within]

        all_top_indices[start:end] = sorted_indices

    return all_top_indices


# ---------------------------------------------------------------------------
# NDCG computation
# ---------------------------------------------------------------------------


def compute_ndcg_scores(
    query_ids: list[str],
    corpus_ids: list[str],
    top_k_indices: np.ndarray,
    qrels: dict[str, dict[str, int]],
    k: int = 10,
) -> list[float]:
    """Compute NDCG@k for each query using graded relevance from qrels.

    Documents in top_k_indices are already sorted by descending similarity,
    so relevance is directly in rank order. IDCG uses all relevant docs
    from qrels (not just those retrieved).
    """
    ndcg_list: list[float] = []
    for qi, qid in enumerate(query_ids):
        if qid not in qrels:
            continue
        retrieved_ids = [corpus_ids[idx] for idx in top_k_indices[qi]]
        relevance = [float(qrels[qid].get(did, 0)) for did in retrieved_ids]

        # IDCG: best possible DCG@k using all relevant docs from qrels
        ideal = sorted(qrels[qid].values(), reverse=True)
        idcg = dcg_at_k(ideal, k)
        if idcg == 0.0:
            continue

        dcg = dcg_at_k(relevance, k)
        ndcg_list.append(dcg / idcg)
    return ndcg_list


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------


def write_top_k_jsonl(
    query_ids: list[str],
    query_texts: list[str],
    corpus_ids: list[str],
    corpus_texts: list[str],
    top_k_indices: np.ndarray,
    qrels: dict[str, dict[str, int]],
    output_path: Path,
) -> int:
    """Write Top-K retrieval results as JSONL.

    Format: {"query": str, "documents": [{"content": str, "relevance": int}]}
    Documents are in retrieval rank order with graded relevance from qrels.
    Returns number of records written.
    """
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fw:
        for qi, (qid, query_text) in enumerate(zip(query_ids, query_texts)):
            query_qrels = qrels.get(qid, {})
            documents = [
                {
                    "content": corpus_texts[idx],
                    "relevance": query_qrels.get(corpus_ids[idx], 0),
                }
                for idx in top_k_indices[qi]
            ]
            fw.write(
                json.dumps(
                    {"query": query_text, "documents": documents},
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


# ---------------------------------------------------------------------------
# Per-dataset orchestration
# ---------------------------------------------------------------------------


def process_dataset(
    dataset_name: str,
    model: SentenceTransformer,
    batch_size: int,
    top_k: int,
    query_chunk_size: int,
    corpus_chunk_size: int,
    output_path: Path,
) -> tuple[float, int]:
    """Evaluate one BEIR dataset. Returns (mean_ndcg10, num_queries)."""
    short_name = dataset_name.split("/")[-1]

    # --- Load data ---
    print(f"  Loading corpus for {short_name}...")
    corpus_ds = get_split(load_dataset(dataset_name, "corpus"))
    corpus_ids, corpus_texts = build_corpus_texts(corpus_ds)
    print(f"  Corpus: {len(corpus_texts):,} documents")

    print(f"  Loading queries for {short_name}...")
    queries_ds = get_split(load_dataset(dataset_name, "queries"))
    query_ids = [str(x) for x in queries_ds["_id"]]
    query_texts: list[str] = queries_ds["text"]

    print(f"  Loading qrels for {short_name}...")
    qrels_ds = get_split(load_dataset(dataset_name, "default"))
    qrels = build_qrels_dict(qrels_ds)

    # Filter to queries with at least one positive
    valid_indices = [i for i, qid in enumerate(query_ids) if qid in qrels]
    if not valid_indices:
        print(f"  {short_name}: no queries with positives, skipping")
        return 0.0, 0

    filtered_query_ids = [query_ids[i] for i in valid_indices]
    filtered_query_texts = [query_texts[i] for i in valid_indices]
    print(f"  Queries: {len(filtered_query_texts):,} (with positives)")

    # --- Encode corpus into FAISS index ---
    print(f"  Encoding {len(corpus_texts):,} corpus documents into FAISS index...")
    index = build_faiss_index(model, corpus_texts, batch_size, corpus_chunk_size)
    print(f"  FAISS index built: {index.ntotal:,} vectors")

    # --- Encode queries ---
    print(f"  Encoding {len(filtered_query_texts):,} queries...")
    query_embeddings = encode_queries(model, filtered_query_texts, batch_size)

    # --- Retrieve ---
    print(f"  Computing top-{top_k} retrieval via FAISS...")
    top_k_indices = faiss_retrieve_top_k(
        query_embeddings,
        index,
        top_k,
        query_chunk_size,
    )

    # --- NDCG@10 ---
    ndcg_scores = compute_ndcg_scores(
        filtered_query_ids, corpus_ids, top_k_indices, qrels, k=10
    )
    avg_ndcg = mean_score(ndcg_scores)

    # --- Save JSONL ---
    count = write_top_k_jsonl(
        query_ids=filtered_query_ids,
        query_texts=filtered_query_texts,
        corpus_ids=corpus_ids,
        corpus_texts=corpus_texts,
        top_k_indices=top_k_indices,
        qrels=qrels,
        output_path=output_path,
    )
    print(f"  Saved {count:,} records -> {output_path.name}")

    # Free memory
    del index, query_embeddings, top_k_indices
    return avg_ndcg, len(ndcg_scores)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def make_output_filename(dataset_name: str, top_k: int) -> str:
    """Convert 'mteb/trec-covid' to 'mteb__trec-covid_top100.jsonl'."""
    safe_name = dataset_name.replace("/", "__")
    return f"{safe_name}_top{top_k}.jsonl"


def print_summary(results: dict[str, float]) -> None:
    """Print a formatted summary table of all dataset NDCG@10 scores."""
    if not results:
        print("\nNo results to summarize.")
        return

    print(f"\n{'=' * 50}")
    print(f"  {'Dataset':<30} {'NDCG@10':>10}")
    print(f"  {'-' * 44}")
    for name, score in results.items():
        short_name = name.split("/")[-1]
        print(f"  {short_name:<30} {score:>10.4f}")
    avg = sum(results.values()) / len(results)
    print(f"  {'-' * 44}")
    print(f"  {'Average':<30} {avg:>10.4f}")
    print(f"{'=' * 50}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate embedding models on BEIR retrieval benchmarks"
    )
    default_output = str(Path(__file__).resolve().parent / "results")
    parser.add_argument(
        "--output-dir",
        default=default_output,
        help=f"Output directory for Top-K JSONL files (default: {default_output})",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Encoding batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top docs to retrieve per query (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--query-chunk-size",
        type=int,
        default=DEFAULT_QUERY_CHUNK,
        help="Query chunk size for similarity computation "
        f"(default: {DEFAULT_QUERY_CHUNK})",
    )
    parser.add_argument(
        "--corpus-chunk-size",
        type=int,
        default=DEFAULT_CORPUS_CHUNK,
        help="Corpus encoding chunk size for incremental FAISS indexing "
        f"(default: {DEFAULT_CORPUS_CHUNK})",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=BEIR_DATASETS,
        help="BEIR dataset names to evaluate",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_name}")
    from . import patch_transformers_compat

    patch_transformers_compat()
    model = SentenceTransformer(
        args.model_name,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": "bfloat16"},
        truncate_dim=None,
    )
    model.max_seq_length = 4096
    print(
        f"Model loaded. "
        f"dtype: {next(model[0].auto_model.parameters()).dtype}, "
        f"dim: {model.get_sentence_embedding_dimension()}"
    )

    results: dict[str, float] = {}
    total = len(args.datasets)

    for di, dataset_name in enumerate(args.datasets, 1):
        filename = make_output_filename(dataset_name, args.top_k)
        output_path = output_dir / filename

        print(f"\n{'=' * 50}")
        print(f"[{di}/{total}] {dataset_name}")
        print(f"{'=' * 50}")

        if output_path.exists():
            print(f"  Already exists, skipping: {filename}")
            continue

        try:
            avg_ndcg, num_queries = process_dataset(
                dataset_name=dataset_name,
                model=model,
                batch_size=args.batch_size,
                top_k=args.top_k,
                query_chunk_size=args.query_chunk_size,
                corpus_chunk_size=args.corpus_chunk_size,
                output_path=output_path,
            )
            results[dataset_name] = avg_ndcg
            print(f"  NDCG@10 = {avg_ndcg:.4f}  ({num_queries:,} queries)")
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            raise

    print_summary(results)


if __name__ == "__main__":
    main()
