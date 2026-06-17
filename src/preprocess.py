from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .config import DataConfig


def format_question_answer(row: Mapping[str, Any], data_cfg: DataConfig) -> str:
    question = _clean_text(row.get(data_cfg.question_column, ""))
    answer = _clean_text(row.get(data_cfg.answer_column, ""))

    if data_cfg.custom_template:
        return data_cfg.custom_template.format(question=question, answer=answer)
    if data_cfg.input_template == "politician_answer":
        return f"Question: {question} Politician's answer: {answer}"
    if data_cfg.input_template != "default":
        raise ValueError(f"Unknown input_template: {data_cfg.input_template}")
    return f"Question: {question} Answer: {answer}"


def tokenize_dataset(dataset, tokenizer, data_cfg: DataConfig, remove_columns=None):
    def _tokenize(batch):
        texts = [
            format_question_answer(
                {
                    data_cfg.question_column: question,
                    data_cfg.answer_column: answer,
                },
                data_cfg,
            )
            for question, answer in zip(
                batch[data_cfg.question_column], batch[data_cfg.answer_column]
            )
        ]
        return tokenizer(
            texts,
            truncation=True,
            max_length=data_cfg.max_length,
            padding=False,
        )

    return dataset.map(_tokenize, batched=True, remove_columns=remove_columns)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())

