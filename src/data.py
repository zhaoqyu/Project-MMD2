from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

from .config import DataConfig
from .utils import dataset_slug, ensure_dir, load_json, save_json


TASK1_LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
TASK2_LABELS = [
    "Explicit",
    "Implicit",
    "Dodging",
    "General",
    "Deflection",
    "Partial/half-answer",
    "Declining to answer",
    "Claims ignorance",
    "Clarification",
]

CLARITY_ALIASES = {
    "Ambiguous": "Ambivalent",
    "Ambivalent Reply": "Ambivalent",
}


def load_qevasion_dataset(data_cfg: DataConfig) -> DatasetDict:
    return load_dataset(data_cfg.dataset_name, data_cfg.dataset_config)


def validate_expected_columns(dataset: DatasetDict, data_cfg: DataConfig) -> dict[str, Any]:
    expected = [
        data_cfg.question_column,
        data_cfg.answer_column,
        data_cfg.task1_label_column,
        data_cfg.task2_label_column,
        data_cfg.id_column,
        *data_cfg.annotator_columns,
    ]
    schema = {}
    for split, ds in dataset.items():
        missing = [column for column in expected if column not in ds.column_names]
        if missing:
            raise ValueError(f"Split {split!r} is missing columns: {missing}")
        schema[split] = {
            "num_rows": len(ds),
            "columns": list(ds.column_names),
            "features": {name: str(feature) for name, feature in ds.features.items()},
        }
    return schema


def task_label_column(task: str, data_cfg: DataConfig) -> str:
    if task == "task1":
        return data_cfg.task1_label_column
    if task == "task2":
        return data_cfg.task2_label_column
    raise ValueError(f"Single-task label column requested for unsupported task: {task}")


def normalize_label(label: Any, task: str) -> str | None:
    if label is None:
        return None
    label = str(label).strip()
    if not label:
        return None
    if task == "task1":
        return CLARITY_ALIASES.get(label, label)
    return label


def build_label_mapping(train_dataset: Dataset, task: str, data_cfg: DataConfig) -> tuple[list[str], dict[str, int], dict[int, str]]:
    if task == "task1":
        preferred = TASK1_LABELS
        label_column = data_cfg.task1_label_column
    elif task == "task2":
        preferred = TASK2_LABELS
        label_column = data_cfg.task2_label_column
    else:
        raise ValueError(f"Unsupported task for label mapping: {task}")

    observed = [
        normalize_label(value, task)
        for value in train_dataset[label_column]
        if normalize_label(value, task) is not None
    ]
    observed_set = set(observed)
    labels = [label for label in preferred if label in observed_set]
    labels.extend(sorted(observed_set.difference(labels)))
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return labels, label2id, id2label


def split_file_path(output_dir: str | Path, data_cfg: DataConfig, task: str, seed: int) -> Path:
    name = f"{dataset_slug(data_cfg.dataset_name)}_{task}_seed{seed}_eval{data_cfg.eval_size:g}.json"
    return Path(output_dir) / "splits" / name


def get_or_create_split(
    train_dataset: Dataset,
    data_cfg: DataConfig,
    task: str,
    seed: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    path = split_file_path(output_dir, data_cfg, task, seed)
    if path.exists() and not data_cfg.overwrite_split:
        split = load_json(path)
        if split.get("num_rows") != len(train_dataset):
            raise ValueError(
                f"Existing split {path} was built for {split.get('num_rows')} rows, "
                f"but current train split has {len(train_dataset)} rows."
            )
        return split

    ensure_dir(path.parent)
    label_column = data_cfg.task1_label_column if task == "multitask" else task_label_column(task, data_cfg)
    labels = [normalize_label(value, "task1" if task == "multitask" else task) for value in train_dataset[label_column]]
    indices = np.arange(len(train_dataset))
    stratify = _stratify_values(labels)

    try:
        from sklearn.model_selection import train_test_split

        train_idx, eval_idx = train_test_split(
            indices,
            test_size=data_cfg.eval_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
        stratified = stratify is not None
    except Exception:
        rng = np.random.default_rng(seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        eval_count = int(round(len(indices) * data_cfg.eval_size))
        eval_idx = shuffled[:eval_count]
        train_idx = shuffled[eval_count:]
        stratified = False

    train_idx = sorted(int(i) for i in train_idx)
    eval_idx = sorted(int(i) for i in eval_idx)
    split = {
        "dataset_name": data_cfg.dataset_name,
        "dataset_config": data_cfg.dataset_config,
        "task": task,
        "seed": seed,
        "eval_size": data_cfg.eval_size,
        "num_rows": len(train_dataset),
        "stratified": stratified,
        "stratify_column": label_column if stratified else None,
        "train_indices": train_idx,
        "eval_indices": eval_idx,
        "train_ids": _ids_for_indices(train_dataset, train_idx, data_cfg.id_column),
        "eval_ids": _ids_for_indices(train_dataset, eval_idx, data_cfg.id_column),
    }
    save_json(split, path)
    return split


def select_split(train_dataset: Dataset, split: dict[str, Any]) -> tuple[Dataset, Dataset]:
    return (
        train_dataset.select(split["train_indices"]),
        train_dataset.select(split["eval_indices"]),
    )


def add_single_task_labels(dataset: Dataset, label_column: str, label2id: dict[str, int], task: str) -> Dataset:
    def _map(row):
        label = normalize_label(row[label_column], task)
        if label not in label2id:
            raise ValueError(f"Unknown {task} label: {label!r}")
        return {"labels": label2id[label]}

    return dataset.map(_map)


def add_multitask_labels(
    dataset: Dataset,
    task1_label2id: dict[str, int],
    task2_label2id: dict[str, int],
    data_cfg: DataConfig,
) -> Dataset:
    def _map(row):
        clarity = normalize_label(row[data_cfg.task1_label_column], "task1")
        evasion = normalize_label(row[data_cfg.task2_label_column], "task2")
        return {
            "labels_task1": task1_label2id[clarity],
            "labels_task2": task2_label2id[evasion],
        }

    return dataset.map(_map)


def class_weights_from_labels(label_ids: list[int], num_labels: int) -> list[float]:
    counts = Counter(label_ids)
    total = sum(counts.values())
    if not total:
        return [1.0] * num_labels
    return [total / (num_labels * max(counts.get(i, 0), 1)) for i in range(num_labels)]


def _stratify_values(labels: list[str | None]) -> list[str] | None:
    if any(label is None for label in labels):
        return None
    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 2:
        return None
    return [str(label) for label in labels]


def _ids_for_indices(dataset: Dataset, indices: list[int], id_column: str) -> list[Any]:
    if id_column not in dataset.column_names:
        return indices
    values = dataset[id_column]
    return [values[i] for i in indices]

