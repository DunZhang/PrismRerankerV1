"""Load and validate benchmark datasets for reranker evaluation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class QueryRecord:
    """Raw dataset record loaded from one JSONL line."""

    query: str
    positives: list[str]
    negatives: list[str]


@dataclass(slots=True, frozen=True)
class QuerySample:
    """A single query with mixed positive and sampled negative documents."""

    query: str
    documents: list[str]
    relevance: list[int]


def iter_records(jsonl_path: Path) -> list[QueryRecord]:
    """Read and validate all records from a benchmark JSONL file."""
    records: list[QueryRecord] = []
    with open(jsonl_path, encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {jsonl_path} at line {line_number}: {exc}"
                ) from exc
            records.append(_parse_record(payload, jsonl_path, line_number))
    return records


def build_sample(
    record: QueryRecord,
    num_neg: int,
    rng: random.Random,
) -> QuerySample:
    """Sample negatives and shuffle the final candidate list for one query."""
    sampled_negs = rng.sample(record.negatives, min(num_neg, len(record.negatives)))
    documents_and_relevance = [(doc, 1) for doc in record.positives] + [
        (doc, 0) for doc in sampled_negs
    ]
    rng.shuffle(documents_and_relevance)
    documents = [document for document, _ in documents_and_relevance]
    relevance = [label for _, label in documents_and_relevance]
    return QuerySample(query=record.query, documents=documents, relevance=relevance)


def load_dataset(
    jsonl_path: Path,
    num_neg: int,
    seed: int = 42,
) -> list[QuerySample]:
    """Load a dataset and prepare reranking inputs deterministically."""
    rng = random.Random(seed)
    return [build_sample(record, num_neg=num_neg, rng=rng) for record in iter_records(jsonl_path)]


def list_datasets(data_dir: Path) -> list[Path]:
    """Return sorted list of benchmark JSONL files."""
    return sorted(path for path in data_dir.glob("*.jsonl") if path.is_file())


def _parse_record(payload: Any, jsonl_path: Path, line_number: int) -> QueryRecord:
    """Validate one JSONL payload and convert it to a typed record."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected an object in {jsonl_path} at line {line_number}, got {type(payload).__name__}."
        )

    query = _require_string(payload, "query", jsonl_path, line_number)
    positives = _require_string_list(payload, "pos_list", jsonl_path, line_number)
    negatives = _require_string_list(payload, "neg_list", jsonl_path, line_number)

    if not positives:
        raise ValueError(
            f"{jsonl_path} line {line_number} has empty pos_list; each query needs at least one positive example."
        )

    return QueryRecord(query=query, positives=positives, negatives=negatives)


def _require_string(
    payload: dict[str, Any],
    key: str,
    jsonl_path: Path,
    line_number: int,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(
            f"{jsonl_path} line {line_number} field {key!r} must be a string."
        )
    return value


def _require_string_list(
    payload: dict[str, Any],
    key: str,
    jsonl_path: Path,
    line_number: int,
) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(
            f"{jsonl_path} line {line_number} field {key!r} must be a list[str]."
        )
    return value
