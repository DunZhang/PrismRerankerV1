from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from train.constants import build_prompt


@dataclass(frozen=True)
class RerankSample:
    query: str
    docs: list[str]
    teacher_scores: list[float] | None
    num_positives: int


def _validate_sample_shape(
    data: dict[str, Any],
    line_number: int,
    source: str,
    require_teacher: bool = True,
) -> None:
    pos_list = data["pos_list"]
    neg_list = data["neg_list"]

    if len(pos_list) < 1:
        raise ValueError(f"{source}:{line_number} requires at least 1 positive doc.")
    if len(neg_list) < 1:
        raise ValueError(f"{source}:{line_number} requires at least 1 negative doc.")

    if not require_teacher and "teacher_pos_scores" not in data:
        return

    teacher_pos_scores = data["teacher_pos_scores"]
    teacher_neg_scores = data["teacher_neg_scores"]

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


def _parse_sample(
    data: dict[str, Any],
    line_number: int,
    source: str,
    require_teacher: bool = True,
    num_neg: int | None = None,
) -> RerankSample:
    _validate_sample_shape(data, line_number, source, require_teacher)

    pos_list = data["pos_list"]
    neg_list = data["neg_list"]
    if num_neg is not None:
        neg_list = neg_list[:num_neg]

    docs = pos_list + neg_list

    has_teacher = "teacher_pos_scores" in data
    teacher_scores: list[float] | None = None
    if has_teacher:
        teacher_neg_scores = data["teacher_neg_scores"]
        if num_neg is not None:
            teacher_neg_scores = teacher_neg_scores[:num_neg]
        teacher_scores = data["teacher_pos_scores"] + teacher_neg_scores

    return RerankSample(
        query=data["query"],
        docs=docs,
        teacher_scores=teacher_scores,
        num_positives=len(pos_list),
    )


class RerankerDataset(Dataset[RerankSample]):
    """JSONL dataset that reads the first ``max_samples`` lines then stops."""

    def __init__(
        self,
        path: str,
        max_samples: int | None = None,
        seed: int = 42,
        require_teacher: bool = True,
        num_neg: int | None = None,
    ) -> None:
        self.samples: list[RerankSample] = []

        with open(path, encoding="utf-8") as handle:
            for index, line in tqdm(
                enumerate(handle, start=1),
                desc=f"Loading {Path(path).name}",
                total=max_samples,
                unit=" lines",
            ):
                data = json.loads(line)
                sample = _parse_sample(data, index, path, require_teacher, num_neg)
                self.samples.append(sample)

                if max_samples is not None and index >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> RerankSample:
        return self.samples[index]


def make_collate_fn(tokenizer: Any, max_length: int) -> Any:
    def collate_fn(batch: list[RerankSample]) -> dict[str, Any]:
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

        result: dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "num_positives": sample.num_positives,
        }
        if sample.teacher_scores is not None:
            result["teacher_scores"] = torch.tensor(
                sample.teacher_scores, dtype=torch.float32
            )
        return result

    return collate_fn


def load_eval_datasets(
    dev_dir: str,
    max_samples: int | None = None,
    max_files: int | None = None,
    seed: int = 42,
    num_neg: int | None = None,
) -> dict[str, RerankerDataset]:
    """Load JSONL files from a directory as evaluation datasets.

    When ``max_files`` is set and fewer than the total number of files,
    a deterministic random subset is selected using ``seed``.
    """
    dir_path = Path(dev_dir)
    if not dir_path.is_dir():
        raise ValueError(f"dev_dir is not a directory: {dev_dir}")

    all_files = sorted(dir_path.glob("*.jsonl"))
    if max_files is not None and len(all_files) > max_files:
        rng = random.Random(seed)
        all_files = sorted(rng.sample(all_files, max_files))

    datasets: dict[str, RerankerDataset] = {}
    for jsonl_file in all_files:
        ds = RerankerDataset(
            str(jsonl_file),
            max_samples=max_samples,
            seed=seed,
            require_teacher=False,
            num_neg=num_neg,
        )
        if len(ds) > 0:
            datasets[jsonl_file.stem] = ds

    if not datasets:
        raise ValueError(f"No non-empty JSONL files found in {dev_dir}")

    return datasets
