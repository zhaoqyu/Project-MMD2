from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification


def build_sequence_classifier(
    model_name: str,
    num_labels: int,
    label2id: dict[str, int],
    id2label: dict[int, str],
    trust_remote_code: bool = False,
    gradient_checkpointing: bool = False,
):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        trust_remote_code=trust_remote_code,
    )
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model


class MultiTaskSequenceClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_task1_labels: int,
        num_task2_labels: int,
        lambda_task2: float = 1.0,
        trust_remote_code: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.config = self.encoder.config
        hidden_size = getattr(self.config, "hidden_size")
        dropout_prob = getattr(self.config, "classifier_dropout", None)
        if dropout_prob is None:
            dropout_prob = getattr(self.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.task1_classifier = nn.Linear(hidden_size, num_task1_labels)
        self.task2_classifier = nn.Linear(hidden_size, num_task2_labels)
        self.lambda_task2 = lambda_task2

        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
            if hasattr(self.config, "use_cache"):
                self.config.use_cache = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels_task1=None,
        labels_task2=None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        encoder_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits_task1 = self.task1_classifier(pooled)
        logits_task2 = self.task2_classifier(pooled)
        return {
            "logits_task1": logits_task1,
            "logits_task2": logits_task2,
        }

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_directory / "pytorch_model.bin")
        self.config.save_pretrained(save_directory)


def build_multitask_classifier(
    model_name: str,
    num_task1_labels: int,
    num_task2_labels: int,
    lambda_task2: float,
    trust_remote_code: bool = False,
    gradient_checkpointing: bool = False,
) -> MultiTaskSequenceClassifier:
    return MultiTaskSequenceClassifier(
        model_name=model_name,
        num_task1_labels=num_task1_labels,
        num_task2_labels=num_task2_labels,
        lambda_task2=lambda_task2,
        trust_remote_code=trust_remote_code,
        gradient_checkpointing=gradient_checkpointing,
    )


def build_config_only(model_name: str, trust_remote_code: bool = False):
    return AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
