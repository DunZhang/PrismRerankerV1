from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO
from zoneinfo import ZoneInfo

import torch
import torch.nn.functional as F
import wandb
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from train_v2.config import TrainConfig
from train_v2.constants import NO_TOKEN_ID, YES_TOKEN_ID
from train_v2.data import (
    FlatDataset,
    InterleavedDataset,
    make_train_collate_fn,
)
from train_v2.modeling import load_model_and_tokenizer

BEIJING_TZ = ZoneInfo("Asia/Shanghai")


# ---------------------------------------------------------------------------
# Loss breakdown
# ---------------------------------------------------------------------------
@dataclass
class LossBreakdown:
    total: torch.Tensor
    pointwise: torch.Tensor
    sft: torch.Tensor

    def as_float_dict(self) -> dict[str, float]:
        return {
            "loss_total": self.total.detach().item(),
            "loss_pointwise": self.pointwise.detach().item(),
            "loss_sft": self.sft.detach().item(),
        }


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------
@dataclass
class TrainingState:
    global_samples_seen: int = 0
    epoch_samples_seen: int = 0
    optimizer_steps: int = 0
    next_save_at: int = 0


# ---------------------------------------------------------------------------
# Metric tracker
# ---------------------------------------------------------------------------
class MetricTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._totals = {
            "loss_total": 0.0,
            "loss_pointwise": 0.0,
            "loss_sft": 0.0,
        }
        self.count_total = 0
        self.count_pointwise = 0
        self.count_sft = 0

    def update(self, losses: LossBreakdown, loss_type: str) -> None:
        for key, value in losses.as_float_dict().items():
            self._totals[key] += value
        self.count_total += 1
        if "point-wise" in loss_type:
            self.count_pointwise += 1
        if "sft" in loss_type:
            self.count_sft += 1

    def distributed_averages(self, accelerator: Accelerator) -> dict[str, float]:
        if self.count_total == 0:
            return {key: 0.0 for key in self._totals}

        stats = torch.tensor(
            [
                self._totals["loss_total"],
                self._totals["loss_pointwise"],
                self._totals["loss_sft"],
                float(self.count_total),
                float(self.count_pointwise),
                float(self.count_sft),
            ],
            dtype=torch.float32,
            device=accelerator.device,
        )
        gathered = accelerator.gather(stats).view(-1, stats.numel()).sum(dim=0)
        total_count = max(gathered[3].item(), 1.0)
        pw_count = max(gathered[4].item(), 1.0)
        sft_count = max(gathered[5].item(), 1.0)
        return {
            "loss_total": gathered[0].item() / total_count,
            "loss_pointwise": gathered[1].item() / pw_count,
            "loss_sft": gathered[2].item() / sft_count,
        }


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
class TrainLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO = open(path, "a", encoding="utf-8")  # noqa: SIM115

    def _timestamp(self) -> str:
        return datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S UTC+08:00")

    def info(self, message: str) -> None:
        self._handle.write(f"[{self._timestamp()}] {message}\n")
        self._handle.flush()

    def log_config(self, cfg: TrainConfig) -> None:
        self.info("=" * 60)
        self.info("Resolved training configuration")
        self.info("=" * 60)
        for key, value in cfg.to_flat_dict().items():
            self.info(f"  {key}: {value}")
        self.info("=" * 60)

    def log_step(
        self,
        *,
        step: int,
        epoch: int,
        samples_seen: int,
        learning_rate: float,
        losses: dict[str, float],
        grad_norm: float | None,
    ) -> None:
        parts = [
            f"step={step}",
            f"epoch={epoch}",
            f"samples={samples_seen}",
            f"lr={learning_rate:.2e}",
            f"loss_total={losses['loss_total']:.6f}",
            f"loss_point={losses['loss_pointwise']:.6f}",
            f"loss_sft={losses['loss_sft']:.6f}",
        ]
        if grad_norm is not None:
            parts.append(f"grad_norm={grad_norm:.4f}")
        self.info("[TRAIN] " + " | ".join(parts))

    def close(self) -> None:
        self._handle.close()


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def compute_point_loss(
    student_z: torch.Tensor,
    teacher_scores: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(torch.sigmoid(student_z), teacher_scores)


def compute_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Standard autoregressive cross-entropy on target tokens.

    Args:
        logits: (1, seq_len, vocab_size)
        labels: (1, seq_len) with -100 for masked (prompt) positions
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    # If prompt fills the entire sequence, all labels are -100 and
    # cross_entropy(reduction="mean") returns NaN. Return 0 instead.
    if (shift_labels != -100).sum() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------
def save_model_bundle(
    model: Any,
    tokenizer: Any,
    cfg: TrainConfig,
    save_dir: Path,
    accelerator: Accelerator,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    with open(save_dir / "train_config.resolved.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False, allow_unicode=False)
    accelerator.print(f"Saved checkpoint to {save_dir}")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class RerankerTrainer:
    def __init__(
        self, cfg: TrainConfig, *, config_path: str | Path | None = None
    ) -> None:
        self.cfg = cfg
        self.config_path = Path(config_path) if config_path else None
        # Map config dtype name to accelerate mixed_precision key
        _MP_MAP = {"bfloat16": "bf16", "float16": "fp16"}
        mixed_precision = _MP_MAP.get(cfg.model.dtype or "", "no")

        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.training.grad_accum_steps,
            mixed_precision=mixed_precision,
        )
        set_seed(cfg.training.seed)

        self.output_dir = Path(cfg.output.dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger: TrainLogger | None = None
        self.wandb_run: Any | None = None
        self._init_tracking()

        # ---- Model ----
        self.accelerator.print(f"Loading model from {cfg.model.path} ...")
        self.model, self.tokenizer = load_model_and_tokenizer(cfg)
        self.accelerator.print("Model loaded.")

        # ---- Training data ----
        sft_ds: FlatDataset | None = None
        pw_ds: FlatDataset | None = None
        if cfg.data.sft_data_file:
            sft_ds = FlatDataset(cfg.data.sft_data_file, cfg.data.train_samples)
        if cfg.data.point_wise_data_file:
            pw_ds = FlatDataset(cfg.data.point_wise_data_file, cfg.data.train_samples)

        self.train_dataset = InterleavedDataset(
            sft_dataset=sft_ds,
            point_wise_dataset=pw_ds,
            sft_ratio=cfg.data.sft_ratio,
            seed=cfg.training.seed,
        )

        self.accelerator.print(f"Training samples: {len(self.train_dataset)}")
        if self.logger:
            self.logger.info(f"Training samples loaded: {len(self.train_dataset)}")

        # ---- DataLoader ----
        train_collate = make_train_collate_fn(self.tokenizer, cfg.model.max_seq_length)
        pin_memory = cfg.data.pin_memory and torch.cuda.is_available()

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=train_collate,
            num_workers=cfg.data.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        # ---- Optimizer & Scheduler ----
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

        queries_per_optimizer_update = (
            cfg.training.grad_accum_steps * self.accelerator.num_processes
        )
        self.optimizer_updates_per_epoch = max(
            1,
            math.ceil(len(self.train_dataset) / queries_per_optimizer_update),
        )
        self.scheduler_steps_per_epoch = self.optimizer_updates_per_epoch
        self.total_optimizer_updates = (
            self.optimizer_updates_per_epoch * cfg.training.num_epochs
        )
        self.total_scheduler_steps = (
            self.scheduler_steps_per_epoch * cfg.training.num_epochs
        )
        self.scheduler = get_scheduler(
            name=cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.warmup_steps,
            num_training_steps=self.total_scheduler_steps,
        )

        self.accelerator.print(
            "Optimizer updates: "
            f"{self.total_optimizer_updates} | "
            f"Scheduler steps: {self.total_scheduler_steps} "
            f"(epochs={cfg.training.num_epochs}, "
            f"grad_accum={cfg.training.grad_accum_steps}, "
            f"world_size={self.accelerator.num_processes})"
        )
        if self.logger:
            self.logger.info(f"Optimizer updates: {self.total_optimizer_updates}")
            self.logger.info(f"Scheduler steps: {self.total_scheduler_steps}")
            self.logger.info("Training started.")

        # ---- Prepare with accelerator ----
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.scheduler,
        )

        self.model.train()
        self.metrics = MetricTracker()
        self.state = TrainingState(
            next_save_at=cfg.output.save_interval_samples,
        )

    # ---- Tracking init ----
    def _init_tracking(self) -> None:
        if not self.accelerator.is_main_process:
            return
        if self.config_path is not None:
            shutil.copy2(self.config_path, self.output_dir / "train_config.yaml")
        self.logger = TrainLogger(self.output_dir / "train_log.txt")
        self.logger.log_config(self.cfg)
        if self.cfg.logging.wandb_mode == "disabled":
            return
        os.environ["WANDB_MODE"] = self.cfg.logging.wandb_mode
        os.environ["WANDB_DIR"] = str(self.output_dir)
        self.wandb_run = wandb.init(
            project=self.cfg.logging.wandb_project,
            name=self.cfg.output.run_name,
            dir=str(self.output_dir),
            config=self.cfg.to_flat_dict(),
        )

    def _log_wandb(self, metrics: dict[str, float | int], step: int) -> None:
        if self.accelerator.is_main_process and self.wandb_run is not None:
            wandb.log(metrics, step=step)

    def _advance_sample_counters(self) -> None:
        remaining = max(len(self.train_dataset) - self.state.epoch_samples_seen, 0)
        newly_seen = min(self.accelerator.num_processes, remaining)
        self.state.epoch_samples_seen += newly_seen
        self.state.global_samples_seen += newly_seen

    # ---- Train step ----
    def _run_train_step(
        self,
        batch: dict[str, Any],
    ) -> tuple[LossBreakdown, float | None]:
        with self.accelerator.accumulate(self.model):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            loss_type: str = batch["loss_type"]

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (1, seq_len, vocab_size)

            zero = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            loss_point = zero
            loss_sft = zero

            if "point-wise" in loss_type:
                if "sft" in loss_type:
                    pos = batch["prompt_length"] - 1
                else:
                    pos = -1
                last_logits = logits[:, pos, :]
                student_z = last_logits[:, YES_TOKEN_ID] - last_logits[:, NO_TOKEN_ID]
                teacher_score = batch["teacher_score"].to(student_z.device)
                loss_point = compute_point_loss(student_z, teacher_score)

            if "sft" in loss_type:
                labels = batch["labels"].to(logits.device)
                loss_sft = compute_sft_loss(logits, labels)

            total = (
                self.cfg.loss.gamma_point * loss_point
                + self.cfg.loss.gamma_sft * loss_sft
            )

            self.accelerator.backward(total)

            grad_norm: float | None = None
            if self.accelerator.sync_gradients and self.cfg.training.max_grad_norm > 0:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.training.max_grad_norm,
                ).item()

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return (
            LossBreakdown(total=total, pointwise=loss_point, sft=loss_sft),
            grad_norm,
        )

    # ---- Metrics emission ----
    def _emit_train_metrics(
        self,
        *,
        epoch: int,
        grad_norm: float | None,
        progress: tqdm | None,
    ) -> None:
        if self.metrics.count_total == 0:
            return

        averaged = self.metrics.distributed_averages(self.accelerator)
        current_lr = self.scheduler.get_last_lr()[0]
        payload: dict[str, float | int] = {
            **averaged,
            "lr": current_lr,
            "samples_seen": self.state.global_samples_seen,
            "optimizer_step": self.state.optimizer_steps,
        }
        if grad_norm is not None:
            payload["grad_norm"] = grad_norm

        if self.accelerator.is_main_process:
            self._log_wandb(payload, step=self.state.optimizer_steps)
            if self.logger:
                self.logger.log_step(
                    step=self.state.optimizer_steps,
                    epoch=epoch,
                    samples_seen=self.state.global_samples_seen,
                    learning_rate=current_lr,
                    losses=averaged,
                    grad_norm=grad_norm,
                )
            if progress is not None:
                progress.set_postfix(
                    loss=f"{averaged['loss_total']:.4f}",
                    lr=f"{current_lr:.2e}",
                )

        self.metrics.reset()

    # ---- Periodic checkpoint saving ----
    def _maybe_save_checkpoint(self) -> None:
        while self.state.global_samples_seen >= self.state.next_save_at:
            self.state.next_save_at += self.cfg.output.save_interval_samples
            if self.accelerator.is_main_process:
                save_model_bundle(
                    self.model,
                    self.tokenizer,
                    self.cfg,
                    self.output_dir / f"samples-{self.state.global_samples_seen}",
                    self.accelerator,
                )
            self.accelerator.wait_for_everyone()

    # ---- Main training loop ----
    def train(self) -> None:
        for epoch_index in range(self.cfg.training.num_epochs):
            epoch_number = epoch_index + 1
            self.state.epoch_samples_seen = 0
            progress = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch_number}/{self.cfg.training.num_epochs}",
                disable=not self.accelerator.is_main_process,
            )
            last_grad_norm: float | None = None

            for batch in progress:
                losses, last_grad_norm = self._run_train_step(batch)
                self.metrics.update(losses, batch["loss_type"])
                self._advance_sample_counters()

                if self.accelerator.sync_gradients:
                    self.state.optimizer_steps += 1
                    if (
                        self.state.optimizer_steps % self.cfg.logging.log_interval_steps
                        == 0
                    ):
                        self._emit_train_metrics(
                            epoch=epoch_number,
                            grad_norm=last_grad_norm,
                            progress=progress,
                        )

                self._maybe_save_checkpoint()

            self._emit_train_metrics(
                epoch=epoch_number,
                grad_norm=last_grad_norm,
                progress=progress,
            )

        # ---- Final save & cleanup ----
        if self.accelerator.is_main_process:
            if self.wandb_run is not None:
                wandb.finish()
            if self.logger:
                self.logger.info(
                    f"Training complete. "
                    f"Total samples={self.state.global_samples_seen} | "
                    f"Total steps={self.state.optimizer_steps}"
                )
                self.logger.close()

        self.accelerator.wait_for_everyone()
        self.accelerator.print("Training complete.")
