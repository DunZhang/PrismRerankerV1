"""Core evaluation loop for reranker benchmarks."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from utils.segmenter import segment_document

from .checkpoint import CheckpointManager
from .config import DEFAULT_CACHE_DIR, EvaluationConfig, make_run_tag
from .data_loader import QuerySample, list_datasets, load_dataset
from .metrics import DEFAULT_NDCG_K, mean_score, ndcg_at_k
from .models.base import BaseReranker


@dataclass(slots=True, frozen=True)
class DatasetEvaluationResult:
    """Aggregated result for one dataset."""

    dataset_name: str
    mean_ndcg: float
    total_queries: int
    cached_queries: int


@dataclass(slots=True, frozen=True)
class EvaluationSummary:
    """Aggregated result for an entire evaluation run."""

    model_name: str
    dataset_scores: dict[str, float]

    @property
    def average_score(self) -> float:
        return mean_score(list(self.dataset_scores.values()))


def evaluate_dataset(
    model: BaseReranker,
    model_name: str,
    dataset_path: Path,
    num_neg: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    seed: int = 42,
    max_queries: int | None = None,
    segment_docs: bool = False,
) -> DatasetEvaluationResult:
    """Evaluate a single dataset, using cached per-query results when available.

    Args:
        model: Reranker model instance.
        model_name: Display name (used for cache directory naming).
        dataset_path: Path to the JSONL benchmark file.
        num_neg: Number of negatives to mix per query.
        cache_dir: Directory for per-query checkpoint files.
        seed: Random seed for negative sampling and query subsampling.
        max_queries: If set, randomly subsample this many queries (fixed by seed).
            Checkpoint keys are always original dataset indices, so a run with
            max_queries=100 and a subsequent full run share the same cache.
        segment_docs: If True, apply ``segment_document()`` to each document
            before passing it to the model (used for prism-reranker models).

    Returns:
        Mean NDCG@10 over the evaluated queries in this dataset.
    """
    dataset_name = dataset_path.stem
    run_tag = make_run_tag(num_neg, seed)
    ckpt = CheckpointManager(cache_dir, model_name, dataset_name, run_tag)

    all_samples: list[QuerySample] = load_dataset(dataset_path, num_neg, seed)
    selected_indices = _select_query_indices(len(all_samples), max_queries, seed)

    total = len(selected_indices)
    already_done = sum(1 for idx in selected_indices if ckpt.has(idx))
    remaining = total - already_done

    if remaining == 0:
        ndcg_list = [ckpt.get_ndcg(idx) for idx in selected_indices]
        return DatasetEvaluationResult(
            dataset_name=dataset_name,
            mean_ndcg=mean_score(ndcg_list),
            total_queries=total,
            cached_queries=already_done,
        )

    suffix = f" (max_queries={max_queries})" if max_queries is not None else ""
    print(
        f"  {dataset_name}: {already_done}/{total} cached, "
        f"running {remaining} queries{suffix}..."
    )

    for idx in tqdm(selected_indices, desc=f"  {dataset_name[:40]}", leave=False):
        if ckpt.has(idx):
            continue

        sample = all_samples[idx]
        documents = (
            [segment_document(doc) for doc in sample.documents]
            if segment_docs
            else list(sample.documents)
        )
        relevance = list(sample.relevance)

        # Filter out empty/whitespace-only documents (some APIs reject them)
        non_empty = [
            (doc, rel)
            for doc, rel in zip(documents, relevance)
            if doc.strip()
        ]
        if len(non_empty) < len(documents):
            documents = [doc for doc, _ in non_empty]
            relevance = [rel for _, rel in non_empty]

        scores = model.rerank(sample.query, documents)
        score = ndcg_at_k(relevance, scores, k=DEFAULT_NDCG_K)
        ckpt.save(idx, score, scores, relevance)

    ndcg_list = [ckpt.get_ndcg(idx) for idx in selected_indices]
    return DatasetEvaluationResult(
        dataset_name=dataset_name,
        mean_ndcg=mean_score(ndcg_list),
        total_queries=total,
        cached_queries=already_done,
    )


def run_evaluation(
    model: BaseReranker,
    config: EvaluationConfig,
) -> EvaluationSummary:
    """Run evaluation over all benchmark datasets."""
    datasets = list_datasets(config.data_dir)
    if not datasets:
        raise FileNotFoundError(f"No JSONL files found in {config.data_dir}")

    # Shuffle dataset evaluation order with a fixed seed for reproducibility.
    rng = random.Random(42)
    rng.shuffle(datasets)

    max_q_str = (
        f"max_queries={config.max_queries}"
        if config.max_queries is not None
        else "all queries"
    )
    print(f"\n{'=' * 60}")
    print(
        f"Model: {config.model_name}  |  num_neg={config.num_neg}  |  NDCG@{DEFAULT_NDCG_K}"
    )
    print(f"Datasets: {len(datasets)}  |  {max_q_str}")
    print(f"{'=' * 60}\n")

    segment_docs = "prism-reranker" in config.model_name.lower()
    if segment_docs:
        print("Document segmentation: ENABLED (prism-reranker)")

    results: dict[str, float] = {}
    for dataset_path in datasets:
        dataset_result = evaluate_dataset(
            model,
            config.model_name,
            dataset_path,
            config.num_neg,
            cache_dir=config.cache_dir,
            seed=config.seed,
            max_queries=config.max_queries,
            segment_docs=segment_docs,
        )
        results[dataset_result.dataset_name] = dataset_result.mean_ndcg
        print(
            f"  ✓ {dataset_path.name}: {dataset_result.mean_ndcg:.4f} "
            f"({dataset_result.total_queries} queries, {dataset_result.cached_queries} cached)"
        )

    summary = EvaluationSummary(model_name=config.model_name, dataset_scores=results)
    print(f"\nOverall mean NDCG@{DEFAULT_NDCG_K}: {summary.average_score:.4f}")
    return summary


def evaluate_all(
    model: BaseReranker,
    model_name: str,
    num_neg: int,
    data_dir: Path,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    seed: int = 42,
    max_queries: int | None = None,
) -> dict[str, float]:
    """Backward-compatible wrapper returning dataset scores only."""
    config = EvaluationConfig(
        model_name=model_name,
        model_path=None,
        num_neg=num_neg,
        data_dir=data_dir,
        output_path=Path("unused.xlsx"),
        cache_dir=cache_dir,
        seed=seed,
        max_queries=max_queries,
    )
    return run_evaluation(model=model, config=config).dataset_scores


def _select_query_indices(
    total_queries: int,
    max_queries: int | None,
    seed: int,
) -> list[int]:
    """Choose which dataset rows should be evaluated for this run."""
    if max_queries is None or max_queries >= total_queries:
        return list(range(total_queries))
    rng = random.Random(seed)
    return sorted(rng.sample(range(total_queries), max_queries))
