from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from train.constants import build_prompt

@dataclass(frozen=True)
class RerankSample:
    query: str
    docs: list[str]
    teacher_scores: list[float]
    num_positives: int


def _validate_sample_shape(data: dict[str, Any], line_number: int, source: str) -> None:
    pos_list = data["pos_list"]
    neg_list = data["neg_list"]
    teacher_pos_scores = data["teacher_pos_scores"]
    teacher_neg_scores = data["teacher_neg_scores"]

    if len(pos_list) < 1:
        raise ValueError(f"{source}:{line_number} requires at least 1 positive doc.")
    if len(neg_list) < 1:
        raise ValueError(f"{source}:{line_number} requires at least 1 negative doc.")
    if len(teacher_pos_scores) != len(pos_list):
        raise ValueError(
            f"{source}:{line_number} teacher_pos_scores length ({len(teacher_pos_scores)}) "
            f"!= pos_list length ({len(pos_list)})."
        )
    if len(teacher_neg_scores) != len(neg_list):
        raise ValueError(
            f"{source}:{line_number} teacher_neg_scores length ({len(teacher_neg_scores)}) "
            f"!= neg_list length ({len(neg_list)})."
        )

    all_scores = teacher_pos_scores + teacher_neg_scores
    if any(score < 0.0 or score > 1.0 for score in all_scores):
        raise ValueError(f"{source}:{line_number} teacher scores must be in [0, 1].")


def _parse_sample(data: dict[str, Any], line_number: int, source: str) -> RerankSample:
    _validate_sample_shape(data, line_number, source)
    docs = data["pos_list"] + data["neg_list"]
    teacher_scores = data["teacher_pos_scores"] + data["teacher_neg_scores"]

    return RerankSample(
        query=data["query"],
        docs=docs,
        teacher_scores=teacher_scores,
        num_positives=len(data["pos_list"]),
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
            "num_positives": sample.num_positives,
        }

    return collate_fn
