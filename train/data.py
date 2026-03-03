from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from train.constants import build_prompt

EXPECTED_POSITIVES = 1
EXPECTED_NEGATIVES = 7
EXPECTED_DOCS = EXPECTED_POSITIVES + EXPECTED_NEGATIVES


@dataclass(frozen=True)
class RerankSample:
    query: str
    docs: list[str]
    teacher_scores: list[float]


def _validate_sample_shape(data: dict[str, Any], line_number: int, source: str) -> None:
    pos_list = data["pos_list"]
    neg_list = data["neg_list"]
    teacher_pos_scores = data["teacher_pos_scores"]
    teacher_neg_scores = data["teacher_neg_scores"]

    if len(pos_list) != EXPECTED_POSITIVES:
        raise ValueError(
            f"{source}:{line_number} expects {EXPECTED_POSITIVES} positive doc, "
            f"got {len(pos_list)}."
        )
    if len(neg_list) != EXPECTED_NEGATIVES:
        raise ValueError(
            f"{source}:{line_number} expects {EXPECTED_NEGATIVES} negative docs, "
            f"got {len(neg_list)}."
        )
    if len(teacher_pos_scores) != EXPECTED_POSITIVES:
        raise ValueError(
            f"{source}:{line_number} expects {EXPECTED_POSITIVES} positive score, "
            f"got {len(teacher_pos_scores)}."
        )
    if len(teacher_neg_scores) != EXPECTED_NEGATIVES:
        raise ValueError(
            f"{source}:{line_number} expects {EXPECTED_NEGATIVES} negative scores, "
            f"got {len(teacher_neg_scores)}."
        )

    all_scores = teacher_pos_scores + teacher_neg_scores
    if any(score < 0.0 or score > 1.0 for score in all_scores):
        raise ValueError(f"{source}:{line_number} teacher scores must be in [0, 1].")


def _parse_sample(data: dict[str, Any], line_number: int, source: str) -> RerankSample:
    _validate_sample_shape(data, line_number, source)
    docs = data["pos_list"] + data["neg_list"]
    teacher_scores = data["teacher_pos_scores"] + data["teacher_neg_scores"]

    if len(docs) != EXPECTED_DOCS or len(teacher_scores) != EXPECTED_DOCS:
        raise ValueError(f"{source}:{line_number} does not contain {EXPECTED_DOCS} docs.")

    return RerankSample(
        query=data["query"],
        docs=docs,
        teacher_scores=teacher_scores,
    )


class RerankerDataset(Dataset[RerankSample]):
    """JSONL dataset with optional reservoir sampling."""

    def __init__(
        self,
        path: str,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        self.samples: list[RerankSample] = []
        rng = random.Random(seed)

        with open(path, encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                data = json.loads(line)
                sample = _parse_sample(data, index, path)

                if max_samples is None:
                    self.samples.append(sample)
                    continue
                if index <= max_samples:
                    self.samples.append(sample)
                    continue

                replace_at = rng.randint(0, index - 1)
                if replace_at < max_samples:
                    self.samples[replace_at] = sample

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> RerankSample:
        return self.samples[index]


def make_collate_fn(tokenizer: Any, max_length: int) -> Any:
    def collate_fn(batch: list[RerankSample]) -> dict[str, torch.Tensor]:
        if len(batch) != 1:
            raise ValueError("This trainer expects DataLoader(batch_size=1).")

        sample = batch[0]
        prompts = [build_prompt(sample.query, doc) for doc in sample.docs]
        encoded = tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "teacher_scores": torch.tensor(sample.teacher_scores, dtype=torch.float32),
        }

    return collate_fn
