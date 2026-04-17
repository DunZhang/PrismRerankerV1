from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from train_v2.constants import build_prompt

VALID_LOSS_TYPES = {"point-wise", "sft", "point-wise;sft"}


@dataclass(frozen=True)
class FlatSample:
    """One query-document pair with associated loss metadata."""

    query: str
    document: str
    loss_type: str  # "point-wise" / "sft" / "point-wise;sft"
    teacher_score: float | None = None  # for point-wise
    target_text: str | None = None  # for sft


def _parse_flat_sample(
    data: dict[str, Any],
    line_number: int,
    source: str,
) -> FlatSample:
    """Parse one JSONL line into a FlatSample."""
    loss_type = data.get("loss_type", "")
    if loss_type not in VALID_LOSS_TYPES:
        raise ValueError(
            f"{source}:{line_number} invalid loss_type={loss_type!r}, "
            f"expected one of {VALID_LOSS_TYPES}"
        )

    teacher_score: float | None = None
    if "point-wise" in loss_type:
        teacher_score = float(data["revised_score"])
        if not 0.0 <= teacher_score <= 1.0:
            raise ValueError(f"{source}:{line_number} revised_score must be in [0, 1].")

    target_text: str | None = None
    if "sft" in loss_type:
        label = data["annotated_label"]
        evidence = data.get("contribution_evidence", "")
        target_text = f"{label}\n{evidence}".strip()

    keywords = (data.get("keywords", "") or "").strip()
    query = keywords if keywords else data["query"]
    return FlatSample(
        query=query,
        document=data["document"],
        loss_type=loss_type,
        teacher_score=teacher_score,
        target_text=target_text,
    )


class FlatDataset(Dataset[FlatSample]):
    """JSONL dataset where each line is a single query-document pair."""

    def __init__(
        self,
        path: str,
        max_samples: int | None = None,
    ) -> None:
        self.samples: list[FlatSample] = []
        with open(path, encoding="utf-8") as handle:
            for index, line in tqdm(
                enumerate(handle, start=1),
                desc=f"Loading {Path(path).name}",
                total=max_samples,
                unit=" lines",
            ):
                data = json.loads(line)
                self.samples.append(_parse_flat_sample(data, index, path))
                if max_samples is not None and index >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> FlatSample:
        return self.samples[index]


class InterleavedDataset(Dataset[FlatSample]):
    """按 ``sft_fraction`` 混合两个数据源。

    行为分三种：

    - 只提供一个数据源：忽略 ``sft_fraction``，完整走一轮该数据源。
    - ``sft_fraction is None`` 且两源都在：**不做采样**，两份并集走一轮，
      每条样本恰好出现一次（长度 = n_sft + n_pw）。
    - 两源都在且 ``sft_fraction`` 是 ``(0, 1)`` 内的小数：按该比例混合，
      较大的源完整看一次，较小的源循环过采样到凑够比例。
    """

    def __init__(
        self,
        sft_dataset: FlatDataset | None,
        point_wise_dataset: FlatDataset | None,
        sft_fraction: float | None = None,
        seed: int = 42,
    ) -> None:
        self.sft_dataset = sft_dataset
        self.pw_dataset = point_wise_dataset
        self.sft_fraction = sft_fraction
        self.seed = seed

        n_sft = len(sft_dataset) if sft_dataset else 0
        n_pw = len(point_wise_dataset) if point_wise_dataset else 0

        if n_sft == 0 and n_pw == 0:
            raise ValueError("Both datasets are empty.")

        # 单源：无采样，直接一轮
        if n_sft == 0:
            self._length = n_pw
            self._source = [False] * n_pw
            self._local_idx = list(range(n_pw))
            return
        if n_pw == 0:
            self._length = n_sft
            self._source = [True] * n_sft
            self._local_idx = list(range(n_sft))
            return

        # 两源都在且 sft_fraction 为 None：不采样，并集一轮
        if sft_fraction is None:
            self._length = n_sft + n_pw
            self._source = [True] * n_sft + [False] * n_pw
            self._local_idx = list(range(n_sft)) + list(range(n_pw))
            return

        # 两源都在且按比例混合（小源循环过采样）
        self._length = max(
            math.ceil(n_pw / (1.0 - sft_fraction)),
            math.ceil(n_sft / sft_fraction),
        )
        rng = random.Random(seed)
        self._source = []
        self._local_idx = []
        sft_cursor = 0
        pw_cursor = 0
        for _ in range(self._length):
            use_sft = rng.random() < sft_fraction
            self._source.append(use_sft)
            if use_sft:
                self._local_idx.append(sft_cursor % n_sft)
                sft_cursor += 1
            else:
                self._local_idx.append(pw_cursor % n_pw)
                pw_cursor += 1

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> FlatSample:
        is_sft = self._source[index]
        local_idx = self._local_idx[index]
        if is_sft:
            assert self.sft_dataset is not None
            return self.sft_dataset[local_idx]
        else:
            assert self.pw_dataset is not None
            return self.pw_dataset[local_idx]


def make_train_collate_fn(tokenizer: Any, max_length: int) -> Any:
    """Build a collate function for training (batch_size=1)."""
    eos_token = tokenizer.eos_token or "<|im_end|>"

    def collate_fn(batch: list[FlatSample]) -> dict[str, Any]:
        if len(batch) != 1:
            raise ValueError("Training requires DataLoader(batch_size=1).")

        sample = batch[0]
        prompt_str = build_prompt(sample.query, sample.document)
        has_sft = "sft" in sample.loss_type
        has_pw = "point-wise" in sample.loss_type

        result: dict[str, Any] = {"loss_type": sample.loss_type}

        if has_sft:
            # Build full sequence: prompt + target + <|im_end|>
            full_str = prompt_str + sample.target_text + eos_token
            full_enc = tokenizer(
                full_str,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            # Get prompt length by tokenizing prompt alone
            prompt_enc = tokenizer(
                prompt_str,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            prompt_length = len(prompt_enc["input_ids"])

            result["input_ids"] = full_enc["input_ids"]
            result["attention_mask"] = full_enc["attention_mask"]

            # Labels: mask prompt positions with -100
            labels = full_enc["input_ids"].clone()
            labels[:, :prompt_length] = -100
            result["labels"] = labels
            result["prompt_length"] = prompt_length

            if has_pw:
                result["teacher_score"] = torch.tensor(
                    [sample.teacher_score], dtype=torch.float32
                )
        else:
            # Point-wise only: tokenize prompt only
            prompt_enc = tokenizer(
                prompt_str,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            result["input_ids"] = prompt_enc["input_ids"]
            result["attention_mask"] = prompt_enc["attention_mask"]
            result["teacher_score"] = torch.tensor(
                [sample.teacher_score], dtype=torch.float32
            )

        return result

    return collate_fn
