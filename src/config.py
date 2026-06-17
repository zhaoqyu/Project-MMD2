from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset_name: str = "ailsntua/QEvasion"
    dataset_config: str = "default"
    question_column: str = "question"
    answer_column: str = "interview_answer"
    task1_label_column: str = "clarity_label"
    task2_label_column: str = "evasion_label"
    annotator_columns: list[str] = field(
        default_factory=lambda: ["annotator1", "annotator2", "annotator3"]
    )
    id_column: str = "index"
    eval_size: float = 0.2
    max_length: int = 256
    input_template: str = "default"
    custom_template: str | None = None
    overwrite_split: bool = False


@dataclass
class ModelConfig:
    model_name: str = "bert-base-uncased"
    trust_remote_code: bool = False
    gradient_checkpointing: bool = False
    tokenizer_use_fast: bool = True


@dataclass
class LossConfig:
    name: str = "cross_entropy"
    use_class_weights: bool = False
    focal_gamma: float = 2.0


@dataclass
class TrainingConfig:
    num_train_epochs: float = 5.0
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = False
    logging_steps: int = 25
    save_total_limit: int = 1
    early_stopping_patience: int = 2
    dataloader_num_workers: int = 0
    metric_for_best_model: str = "macro_f1"
    greater_is_better: bool = True
    report_to: str = "none"
    resume_from_checkpoint: str | None = None


@dataclass
class MultiTaskConfig:
    lambda_task2: float = 1.0
    primary_metric: str = "task2_macro_f1"


@dataclass
class ExperimentConfig:
    experiment_name: str = "qevasion_experiment"
    task: str = "task1"
    seeds: list[int] = field(default_factory=lambda: [42])
    output_dir: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    multitask: MultiTaskConfig = field(default_factory=MultiTaskConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = ExperimentConfig()
    _update_dataclass(cfg, raw)
    if cfg.task == "multitask":
        cfg.training.metric_for_best_model = cfg.multitask.primary_metric
    return cfg


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    return asdict(cfg)


def _update_dataclass(obj: Any, values: dict[str, Any]) -> None:
    if not is_dataclass(obj):
        raise TypeError(f"Expected dataclass instance, got {type(obj)!r}")

    fields = obj.__dataclass_fields__
    for key, value in values.items():
        if key not in fields:
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        else:
            setattr(obj, key, value)

