"""通用 dev-set 评估，适用于 ``train_v2`` 所有 trainer 变体。

评估只看 rerank 分数：模型对每条 (query, document) 计算
``sigmoid(yes_logit - no_logit)``，然后和 dev 里的监督信号做对齐。

支持多卡：数据通过 ``accelerator.prepare(DataLoader)`` 分片，分数用
``gather_for_metrics`` 汇聚到所有 rank。指标只在 main process 计算。

dev 数据里期望至少有：
  - ``query``
  - ``document``
  - ``annotated_label``（``"yes"`` / ``"no"``；作为二分类标签）
  - ``revised_score``（可选；作为连续教师分，用来算 Pearson / Spearman）
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train_v2.constants import build_prompt


@dataclass(frozen=True)
class DevSample:
    query: str
    document: str
    label: float  # 1.0 for "yes", 0.0 for "no", NaN otherwise
    teacher_score: float  # revised_score or NaN if missing
    target_text: str | None  # "{annotated_label}\n{contribution_evidence}" (strip)


class DevDataset(Dataset[DevSample]):
    def __init__(self, path: str) -> None:
        self.samples: list[DevSample] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                keywords = (data.get("keywords", "") or "").strip()
                query = keywords if keywords else data["query"]
                label_raw = data.get("annotated_label")
                if label_raw == "yes":
                    label_val = 1.0
                elif label_raw == "no":
                    label_val = 0.0
                else:
                    label_val = math.nan
                teacher = data.get("revised_score")
                teacher_val = float(teacher) if teacher is not None else math.nan

                # SFT target text — 与训练端 data._parse_flat_sample 保持一致
                if label_raw in ("yes", "no"):
                    evidence = data.get("contribution_evidence", "") or ""
                    target_text: str | None = f"{label_raw}\n{evidence}".strip()
                else:
                    target_text = None

                self.samples.append(
                    DevSample(
                        query=query,
                        document=data["document"],
                        label=label_val,
                        teacher_score=teacher_val,
                        target_text=target_text,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DevSample:
        return self.samples[index]


def make_eval_collate_fn(tokenizer: Any, max_length: int) -> Any:
    def collate(batch: list[DevSample]) -> dict[str, Any]:
        prompts = [build_prompt(s.query, s.document) for s in batch]
        enc = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        labels = torch.tensor([s.label for s in batch], dtype=torch.float32)
        teacher = torch.tensor([s.teacher_score for s in batch], dtype=torch.float32)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "teacher_scores": teacher,
        }

    return collate


def make_eval_sft_collate_fn(tokenizer: Any, max_length: int) -> Any:
    """batch_size=1 的 SFT collate，与训练端 make_train_collate_fn 的 sft 分支一致。"""
    eos_token = tokenizer.eos_token or "<|im_end|>"

    def collate(batch: list[DevSample]) -> dict[str, Any]:
        if len(batch) != 1:
            raise ValueError("Eval SFT collate requires batch_size=1.")
        sample = batch[0]
        if sample.target_text is None:
            raise ValueError("Sample missing target_text; filter before collating.")

        prompt_str = build_prompt(sample.query, sample.document)
        full_str = prompt_str + sample.target_text + eos_token
        full_enc = tokenizer(
            full_str,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        prompt_enc = tokenizer(
            prompt_str,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        prompt_length = len(prompt_enc["input_ids"])

        labels = full_enc["input_ids"].clone()
        labels[:, :prompt_length] = -100
        return {
            "input_ids": full_enc["input_ids"],
            "attention_mask": full_enc["attention_mask"],
            "labels": labels,
        }

    return collate


class _SftSubset(Dataset[DevSample]):
    """仅保留具有 target_text 的 dev 样本，用于 SFT 交叉熵评估。"""

    def __init__(self, parent: DevDataset) -> None:
        self.samples: list[DevSample] = [
            s for s in parent.samples if s.target_text is not None
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DevSample:
        return self.samples[index]


# ---------------------------------------------------------------------------
# Metric helpers (numpy-only, no scipy / sklearn dependency)
# ---------------------------------------------------------------------------
def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Ranks with average tie-breaking, 1-indexed."""
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg_rank = (i + j + 2) / 2.0  # average of (i+1) .. (j+1)
            ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return math.nan
    xm = x - x.mean()
    ym = y - y.mean()
    denom = math.sqrt(float((xm**2).sum()) * float((ym**2).sum()))
    if denom == 0:
        return math.nan
    return float((xm * ym).sum() / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return math.nan
    return pearson_corr(_rankdata_average(x), _rankdata_average(y))


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via Mann-Whitney U; returns NaN if only one class present."""
    mask = ~(np.isnan(scores) | np.isnan(labels))
    scores = scores[mask]
    labels = labels[mask]
    if len(scores) < 2:
        return math.nan
    pos = labels > 0.5
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = _rankdata_average(scores)
    sum_pos_ranks = float(ranks[pos].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def accuracy_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5
) -> float:
    mask = ~(np.isnan(scores) | np.isnan(labels))
    scores = scores[mask]
    labels = labels[mask]
    if len(scores) == 0:
        return math.nan
    preds = (scores >= threshold).astype(np.float32)
    return float((preds == (labels > 0.5).astype(np.float32)).mean())


# ---------------------------------------------------------------------------
# SFT cross-entropy eval loss
# ---------------------------------------------------------------------------
@torch.no_grad()
def _compute_sft_eval_loss(
    *,
    dataset: DevDataset,
    transformer: Any,
    lm_head: torch.nn.Module,
    tokenizer: Any,
    accelerator: Accelerator,
    max_length: int,
) -> float:
    """在 dev 集上计算 SFT 交叉熵损失。

    target text 按训练端相同方式拼接：``f"{annotated_label}\\n{contribution_evidence}".strip()``；
    loss 只在 target + eos 位置计算（prompt 部分的 labels 置为 -100）。
    返回 token 级加权平均 loss。
    """
    subset = _SftSubset(dataset)
    if len(subset) == 0:
        return math.nan

    collate = make_eval_sft_collate_fn(tokenizer, max_length)
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    loader = accelerator.prepare_data_loader(loader, device_placement=True)

    loss_chunks: list[torch.Tensor] = []
    valid_chunks: list[torch.Tensor] = []

    progress = tqdm(
        loader,
        desc="Eval-SFT",
        disable=not accelerator.is_main_process,
        total=len(loader),
    )
    with accelerator.autocast():
        for batch in progress:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = transformer(
                input_ids=input_ids, attention_mask=attention_mask
            )
            hidden = outputs[0]

            shift_hidden = hidden[:, :-1, :]
            shift_labels = labels[:, 1:].contiguous()
            token_mask = shift_labels.view(-1) != -100

            if int(token_mask.sum().item()) == 0:
                loss_val = torch.zeros(1, device=hidden.device, dtype=torch.float32)
                valid_val = torch.zeros(1, device=hidden.device, dtype=torch.float32)
            else:
                valid_hidden = shift_hidden.reshape(-1, shift_hidden.size(-1))[token_mask]
                valid_labels = shift_labels.view(-1)[token_mask]
                logits = lm_head(valid_hidden)
                loss_mean = F.cross_entropy(
                    logits.float(), valid_labels, reduction="mean"
                )
                loss_val = loss_mean.detach().unsqueeze(0).to(torch.float32)
                valid_val = torch.ones(1, device=hidden.device, dtype=torch.float32)

            loss_chunks.append(
                accelerator.gather_for_metrics(loss_val).detach().cpu()
            )
            valid_chunks.append(
                accelerator.gather_for_metrics(valid_val).detach().cpu()
            )

    losses = torch.cat(loss_chunks)
    valid_flags = torch.cat(valid_chunks)
    denom = float(valid_flags.sum().item())
    if denom <= 0:
        return math.nan
    return float((losses * valid_flags).sum().item() / denom)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_evaluation(
    *,
    model: Any,
    transformer: Any,
    lm_head: torch.nn.Module,
    tokenizer: Any,
    accelerator: Accelerator,
    yes_token_id: int,
    no_token_id: int,
    dev_path: str,
    max_length: int,
    batch_size: int,
) -> dict[str, float]:
    """Run dev evaluation across all ranks; return metrics (main rank only has
    meaningful values, other ranks receive the same numbers because
    ``gather_for_metrics`` syncs tensors to every process).
    """
    was_training = model.training
    model.eval()

    dataset = DevDataset(dev_path)
    collate = make_eval_collate_fn(tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    loader = accelerator.prepare_data_loader(loader, device_placement=True)

    score_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    teacher_chunks: list[torch.Tensor] = []

    progress = tqdm(
        loader,
        desc="Eval",
        disable=not accelerator.is_main_process,
        total=len(loader),
    )
    with accelerator.autocast():
        for batch in progress:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            outputs = transformer(input_ids=input_ids, attention_mask=attention_mask)
            # left-padding + 取最后一位 = 每条序列最后一个真实 token
            last_hidden = outputs[0][:, -1, :]
            logits = lm_head(last_hidden)
            student_z = logits[:, yes_token_id] - logits[:, no_token_id]
            student_score = torch.sigmoid(student_z.float())

            gathered_score = accelerator.gather_for_metrics(student_score)
            gathered_label = accelerator.gather_for_metrics(
                batch["labels"].to(student_score.device)
            )
            gathered_teacher = accelerator.gather_for_metrics(
                batch["teacher_scores"].to(student_score.device)
            )
            score_chunks.append(gathered_score.detach().cpu())
            label_chunks.append(gathered_label.detach().cpu())
            teacher_chunks.append(gathered_teacher.detach().cpu())

    scores = torch.cat(score_chunks).numpy().astype(np.float64)
    labels = torch.cat(label_chunks).numpy().astype(np.float64)
    teacher = torch.cat(teacher_chunks).numpy().astype(np.float64)

    n_valid_label = int((~np.isnan(labels)).sum())
    n_valid_teacher = int((~np.isnan(teacher)).sum())

    eval_loss = _compute_sft_eval_loss(
        dataset=dataset,
        transformer=transformer,
        lm_head=lm_head,
        tokenizer=tokenizer,
        accelerator=accelerator,
        max_length=max_length,
    )

    if was_training:
        model.train()

    return {
        "n_dev": float(len(scores)),
        "n_valid_label": float(n_valid_label),
        "n_valid_teacher": float(n_valid_teacher),
        "eval_loss": eval_loss,
        "pearson_teacher": pearson_corr(scores, teacher),
        "spearman_teacher": spearman_corr(scores, teacher),
        "pearson_label": pearson_corr(scores, labels),
        "auc": auc_score(scores, labels),
        "accuracy@0.5": accuracy_at_threshold(scores, labels, 0.5),
    }
