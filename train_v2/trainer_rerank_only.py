"""Trainer variant that restricts SFT supervision to the single yes/no token.

和 ``RerankerTrainer`` 的唯一差异在 ``_run_train_step`` 里的 SFT 部分：

- ``point-wise`` 样本：完全不变（sigmoid(yes_logit - no_logit) 对 teacher_score 做 MSE）
- ``sft`` 样本：只对 prompt 之后的第一个 token（即 ``yes`` / ``no``）计算交叉熵，
  ``contribution_evidence`` 部分被全部 mask 掉
- ``point-wise;sft`` 样本：point-wise 照旧，SFT 部分同上

保存 / dev 评估 / Excel 记录都由基类统一提供，两种 trainer 行为一致。
"""

from __future__ import annotations

from typing import Any

from train_v2.trainer import (
    LossBreakdown,
    RerankerTrainer,
    compute_chunked_sft_loss,
    compute_point_loss,
)


class RerankerOnlyTrainer(RerankerTrainer):
    """只监督 yes/no 单 token 的 reranker 训练器。"""

    def _run_train_step(
        self,
        batch: dict[str, Any],
    ) -> tuple[LossBreakdown, float | None]:
        with self.accelerator.accumulate(self.model):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            loss_type: str = batch["loss_type"]

            with self.accelerator.autocast():
                outputs = self.transformer(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                hidden_states = outputs[0]

                zero = (hidden_states * 0.0).sum()
                loss_point = zero
                loss_sft = zero

                if "point-wise" in loss_type:
                    if "sft" in loss_type:
                        pos = batch["prompt_length"] - 1
                        sliced = hidden_states[:, pos : pos + 1, :]
                    else:
                        sliced = hidden_states[:, -1:, :]
                    pos_logits = self.lm_head(sliced).squeeze(1)
                    student_z = (
                        pos_logits[:, self.yes_token_id]
                        - pos_logits[:, self.no_token_id]
                    )
                    teacher_score = batch["teacher_score"].to(student_z.device)
                    loss_point = compute_point_loss(student_z, teacher_score)

                if "sft" in loss_type:
                    labels = batch["labels"].to(hidden_states.device).clone()
                    prompt_length: int = batch["prompt_length"]
                    # 只保留 prompt 之后第一个 token（yes/no）的监督
                    labels[:, prompt_length + 1 :] = -100
                    loss_sft = compute_chunked_sft_loss(
                        hidden_states, labels, self.lm_head
                    )

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
