from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from train.constants import DEFAULT_LORA_TARGET_MODULES

ALLOWED_DTYPES = {None, "bfloat16", "float16", "float32"}
ALLOWED_SCHEDULERS = {"cosine", "linear"}
ALLOWED_WANDB_MODES = {"offline", "online", "disabled"}


@dataclass
class ModelConfig:
    path: str = ""
    max_seq_length: int = 1024
    load_in_4bit: bool = False
    dtype: str | None = "bfloat16"
    attn_implementation: str | None = "flash_attention_2"
    gradient_checkpointing: bool = True


@dataclass
class LoraConfig:
    enabled: bool = True
    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: list(DEFAULT_LORA_TARGET_MODULES)
    )
    use_rslora: bool = False


@dataclass
class DataConfig:
    train_file: str = ""
    dev_file: str = ""
    train_samples: int | None = None
    eval_samples: int | None = 1000
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class TrainingConfig:
    num_epochs: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    lr_scheduler: str = "cosine"
    grad_accum_steps: int = 4
    max_grad_norm: float = 1.0
    seed: int = 42


@dataclass
class LossConfig:
    alpha_rank: float = 1.0
    beta_list: float = 1.0
    gamma_point: float = 0.5
    temperature: float = 2.0
    hard_neg_scale: float = 5.0


@dataclass
class EvaluationConfig:
    interval_samples: int = 5000


@dataclass
class OutputConfig:
    dir: str = "./train/output"
    run_name: str = "reranker-distill"
    save_every_eval: bool = True
    save_last_checkpoint: bool = False


@dataclass
class LoggingConfig:
    wandb_project: str = "prism-reranker-v1"
    wandb_mode: str = "offline"
    log_interval_steps: int = 10


SECTION_TYPES = {
    "model": ModelConfig,
    "lora": LoraConfig,
    "data": DataConfig,
    "training": TrainingConfig,
    "loss": LossConfig,
    "evaluation": EvaluationConfig,
    "output": OutputConfig,
    "logging": LoggingConfig,
}

LEGACY_KEY_MAP: dict[str, tuple[str, str]] = {
    "model_path": ("model", "path"),
    "max_seq_length": ("model", "max_seq_length"),
    "load_in_4bit": ("model", "load_in_4bit"),
    "dtype": ("model", "dtype"),
    "attn_implementation": ("model", "attn_implementation"),
    "gradient_checkpointing": ("model", "gradient_checkpointing"),
    "use_lora": ("lora", "enabled"),
    "lora_r": ("lora", "r"),
    "lora_alpha": ("lora", "alpha"),
    "lora_dropout": ("lora", "dropout"),
    "lora_target_modules": ("lora", "target_modules"),
    "use_rslora": ("lora", "use_rslora"),
    "train_file": ("data", "train_file"),
    "dev_file": ("data", "dev_file"),
    "train_samples": ("data", "train_samples"),
    "eval_samples": ("data", "eval_samples"),
    "num_workers": ("data", "num_workers"),
    "pin_memory": ("data", "pin_memory"),
    "num_epochs": ("training", "num_epochs"),
    "learning_rate": ("training", "learning_rate"),
    "weight_decay": ("training", "weight_decay"),
    "warmup_steps": ("training", "warmup_steps"),
    "lr_scheduler": ("training", "lr_scheduler"),
    "grad_accum_steps": ("training", "grad_accum_steps"),
    "max_grad_norm": ("training", "max_grad_norm"),
    "seed": ("training", "seed"),
    "alpha_rank": ("loss", "alpha_rank"),
    "beta_list": ("loss", "beta_list"),
    "gamma_point": ("loss", "gamma_point"),
    "temperature": ("loss", "temperature"),
    "hard_neg_scale": ("loss", "hard_neg_scale"),
    "eval_interval": ("evaluation", "interval_samples"),
    "output_dir": ("output", "dir"),
    "run_name": ("output", "run_name"),
    "save_last_checkpoint": ("output", "save_last_checkpoint"),
    "wandb_project": ("logging", "wandb_project"),
    "wandb_mode": ("logging", "wandb_mode"),
    "log_interval": ("logging", "log_interval_steps"),
}

SECTION_FIELD_NAMES = {
    name: set(section.__dataclass_fields__) for name, section in SECTION_TYPES.items()
}


def _assign_field(
    target: dict[str, dict[str, Any]],
    section_name: str,
    field_name: str,
    value: Any,
    source_key: str,
) -> None:
    if field_name in target[section_name]:
        raise ValueError(
            f"Duplicate config value for {section_name}.{field_name} via {source_key}"
        )
    target[section_name][field_name] = value


def _normalize_raw_config(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalized = {name: {} for name in SECTION_TYPES}

    for top_key, value in raw.items():
        if top_key in SECTION_TYPES:
            if not isinstance(value, dict):
                raise TypeError(f"Config section '{top_key}' must be a mapping.")
            valid_fields = SECTION_FIELD_NAMES[top_key]
            for field_name, field_value in value.items():
                if field_name not in valid_fields:
                    raise ValueError(f"Unknown config key: {top_key}.{field_name}")
                _assign_field(
                    normalized,
                    top_key,
                    field_name,
                    field_value,
                    f"{top_key}.{field_name}",
                )
            continue

        legacy_target = LEGACY_KEY_MAP.get(top_key)
        if legacy_target is None:
            raise ValueError(f"Unknown config key: {top_key}")
        section_name, field_name = legacy_target
        _assign_field(normalized, section_name, field_name, value, top_key)

    return normalized


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, dotted))
        else:
            flat[dotted] = value
    return flat


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        if not isinstance(raw, dict):
            raise TypeError("The YAML config must be a top-level mapping.")

        normalized = _normalize_raw_config(raw)
        config = cls(
            model=ModelConfig(**normalized["model"]),
            lora=LoraConfig(**normalized["lora"]),
            data=DataConfig(**normalized["data"]),
            training=TrainingConfig(**normalized["training"]),
            loss=LossConfig(**normalized["loss"]),
            evaluation=EvaluationConfig(**normalized["evaluation"]),
            output=OutputConfig(**normalized["output"]),
            logging=LoggingConfig(**normalized["logging"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.model.path:
            raise ValueError("model.path is required.")
        if not self.data.train_file:
            raise ValueError("data.train_file is required.")
        if not self.data.dev_file:
            raise ValueError("data.dev_file is required.")
        if self.model.dtype not in ALLOWED_DTYPES:
            raise ValueError(f"Unsupported model.dtype: {self.model.dtype}")
        if self.training.lr_scheduler not in ALLOWED_SCHEDULERS:
            raise ValueError(
                f"training.lr_scheduler must be one of {sorted(ALLOWED_SCHEDULERS)}."
            )
        if self.logging.wandb_mode not in ALLOWED_WANDB_MODES:
            raise ValueError(
                f"logging.wandb_mode must be one of {sorted(ALLOWED_WANDB_MODES)}."
            )
        if self.model.load_in_4bit and not self.lora.enabled:
            raise ValueError("model.load_in_4bit requires lora.enabled=true.")
        if self.model.max_seq_length <= 0:
            raise ValueError("model.max_seq_length must be > 0.")
        if self.data.num_workers < 0:
            raise ValueError("data.num_workers must be >= 0.")
        if self.data.train_samples is not None and self.data.train_samples <= 0:
            raise ValueError("data.train_samples must be > 0 or null.")
        if self.data.eval_samples is not None and self.data.eval_samples <= 0:
            raise ValueError("data.eval_samples must be > 0 or null.")
        if self.training.num_epochs <= 0:
            raise ValueError("training.num_epochs must be > 0.")
        if self.training.learning_rate <= 0:
            raise ValueError("training.learning_rate must be > 0.")
        if self.training.warmup_steps < 0:
            raise ValueError("training.warmup_steps must be >= 0.")
        if self.training.grad_accum_steps <= 0:
            raise ValueError("training.grad_accum_steps must be > 0.")
        if self.training.max_grad_norm < 0:
            raise ValueError("training.max_grad_norm must be >= 0.")
        if self.evaluation.interval_samples <= 0:
            raise ValueError("evaluation.interval_samples must be > 0.")
        if self.logging.log_interval_steps <= 0:
            raise ValueError("logging.log_interval_steps must be > 0.")
        if self.lora.enabled and self.lora.r <= 0:
            raise ValueError("lora.r must be > 0.")
        if self.lora.enabled and self.lora.alpha <= 0:
            raise ValueError("lora.alpha must be > 0.")
        if self.lora.enabled and not self.lora.target_modules:
            raise ValueError("lora.target_modules must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_flat_dict(self) -> dict[str, Any]:
        return _flatten_dict(self.to_dict())
