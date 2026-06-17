from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from .config import ExperimentConfig


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: list[float] | None = None, focal_gamma: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
        weight = None
        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, dtype=logits.dtype, device=logits.device)
        loss = _classification_loss(logits, labels, weight, self.focal_gamma)
        return (loss, outputs) if return_outputs else loss


class MultiTaskTrainer(Trainer):
    def __init__(
        self,
        *args,
        lambda_task2: float = 1.0,
        task1_class_weights: list[float] | None = None,
        task2_class_weights: list[float] | None = None,
        focal_gamma: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_task2 = lambda_task2
        self.task1_class_weights = task1_class_weights
        self.task2_class_weights = task2_class_weights
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels_task1 = inputs.pop("labels_task1")
        labels_task2 = inputs.pop("labels_task2")
        outputs = model(**inputs)
        logits_task1 = outputs["logits_task1"]
        logits_task2 = outputs["logits_task2"]
        weight1 = _weight_tensor(self.task1_class_weights, logits_task1)
        weight2 = _weight_tensor(self.task2_class_weights, logits_task2)
        loss_task1 = _classification_loss(logits_task1, labels_task1, weight1, self.focal_gamma)
        loss_task2 = _classification_loss(logits_task2, labels_task2, weight2, self.focal_gamma)
        loss = loss_task1 + self.lambda_task2 * loss_task2
        outputs["loss_task1"] = loss_task1.detach()
        outputs["loss_task2"] = loss_task2.detach()
        return (loss, outputs) if return_outputs else loss


def build_training_arguments(cfg: ExperimentConfig, run_dir: str | Path) -> TrainingArguments:
    args = {
        "output_dir": str(Path(run_dir) / "checkpoints"),
        "per_device_train_batch_size": cfg.training.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.training.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "learning_rate": cfg.training.learning_rate,
        "weight_decay": cfg.training.weight_decay,
        "warmup_ratio": cfg.training.warmup_ratio,
        "num_train_epochs": cfg.training.num_train_epochs,
        "max_grad_norm": cfg.training.max_grad_norm,
        "logging_steps": cfg.training.logging_steps,
        "disable_tqdm": True,
        "logging_first_step": True,
        "save_total_limit": cfg.training.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": cfg.training.metric_for_best_model,
        "greater_is_better": cfg.training.greater_is_better,
        "fp16": cfg.training.fp16,
        "bf16": cfg.training.bf16,
        "dataloader_num_workers": cfg.training.dataloader_num_workers,
        "report_to": [] if cfg.training.report_to == "none" else [cfg.training.report_to],
        "seed": cfg.seeds[0],
        "data_seed": cfg.seeds[0],
        "remove_unused_columns": cfg.task != "multitask",
    }
    params = inspect.signature(TrainingArguments).parameters
    if "eval_strategy" in params:
        args["eval_strategy"] = "epoch"
    else:
        args["evaluation_strategy"] = "epoch"
    args["save_strategy"] = "epoch"
    if "logging_strategy" in params:
        args["logging_strategy"] = "steps"
    if "label_names" in params and cfg.task == "multitask":
        args["label_names"] = ["labels_task1", "labels_task2"]
    return TrainingArguments(**args)


def early_stopping_callbacks(cfg: ExperimentConfig) -> list[EarlyStoppingCallback]:
    if cfg.training.early_stopping_patience <= 0:
        return []
    return [EarlyStoppingCallback(early_stopping_patience=cfg.training.early_stopping_patience)]


def focal_gamma_from_config(cfg: ExperimentConfig) -> float | None:
    return cfg.loss.focal_gamma if cfg.loss.name == "focal_loss" else None


def _classification_loss(logits, labels, weight=None, focal_gamma: float | None = None):
    ce = F.cross_entropy(logits, labels, weight=weight, reduction="none")
    if focal_gamma is not None and focal_gamma > 0:
        pt = torch.exp(-ce)
        ce = ((1.0 - pt) ** focal_gamma) * ce
    return ce.mean()


def _weight_tensor(weights: list[float] | None, logits):
    if weights is None:
        return None
    return torch.tensor(weights, dtype=logits.dtype, device=logits.device)

